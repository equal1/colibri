import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing necessary utilities and operations for the architecture search
from darts_search.operations import *
from darts_search.genotypes import Genotype, PRIMITIVES

from ml_tools.models.model_utils import drop_path

# MixedOp is used to combine multiple operations (like convolution, pooling, etc.) based on their weights.
class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        # Initializing a list of operations based on the primitives defined in the search space.
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            # If the operation is a pooling operation, add batch normalization after it.
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)
    
    # Forward pass computes the weighted sum of all operations.
    def forward(self, x, weights):
        # w is the operation mixing weights. see equation 2 in the original paper.
        return sum(w * op(x) for w, op in zip(weights, self._ops))

# AuxiliaryNetwork is a helper network used for regularization during training.
class AuxiliaryNetwork(nn.Module):
    def __init__(self, C, num_classes):
        super(AuxiliaryNetwork, self).__init__()
        # Define the feature extraction layers.
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        # Classifier layer to predict the class labels.
        self.classifier = nn.Linear(768*3*3, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

# Cell represents the basic building block of the architecture.
class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, genotype):
        super(Cell, self).__init__()
        self.reduction = reduction

        # Preprocessing layers to transform the input tensors.
        # If the previous cell was a reduction cell, use a FactorizedReduce layer.
        # Otherwise, use a ReLUConvBN layer.
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        
        # Apply a ReLUConvBN to the second input state
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        # If a genotype is provided, compile the cell based on the genotype.
        # Otherwise, initialize the cell with a set of MixedOps.
        if genotype is not None:
            if reduction:
                op_names, indices = zip(*genotype.reduce)
                concat = genotype.reduce_concat
            else:
                op_names, indices = zip(*genotype.normal)
                concat = genotype.normal_concat
            self._compile(C, op_names, indices, concat, reduction)
        else:
            self._steps = steps
            self._multiplier = multiplier
            self._ops = nn.ModuleList()
            for i in range(self._steps):
                for j in range(2 + i):
                    stride = 2 if reduction and j < 2 else 1
                    op = MixedOp(C, stride)
                    self._ops.append(op)

    def _compile(self, C, op_names, indices, concat, reduction):
        """Compile the cell based on the provided genotype."""
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, weights=None, drop_prob=0.):
        """Forward pass of the cell."""
        # Preprocess the input tensors.
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        # If the cell was compiled using a genotype, use the operations and indices from the genotype.
        # Otherwise, use the MixedOps and weights.
        if hasattr(self, '_indices'):
            states = [s0, s1]
            for i in range(self._steps):
                h1 = states[self._indices[2 * i]]
                h2 = states[self._indices[2 * i + 1]]
                op1 = self._ops[2 * i]
                op2 = self._ops[2 * i + 1]
                h1 = op1(h1)
                h2 = op2(h2)
                if drop_prob > 0.:
                    if not isinstance(op1, Identity):
                        h1 = drop_path(h1, drop_prob)
                    if not isinstance(op2, Identity):
                        h2 = drop_path(h2, drop_prob)
                s = h1 + h2
                states += [s]
            return torch.cat([states[i] for i in self._concat], dim=1)
        else:
            states = [s0, s1]
            offset = 0
            for i in range(self._steps):
                s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
                offset += len(states)
                states.append(s)
            return torch.cat(states[-self._multiplier:], dim=1)


# Network represents the main neural network architecture.
class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion=None, steps=4, multiplier=4, stem_multiplier=3, auxiliary=False, genotype=None):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._stem_multiplier = stem_multiplier
        self._auxiliary = auxiliary
        self.genotype = genotype

        # Initial convolutional layer
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        # Initialize the previous and current channels for the cells
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        # Define positions in the network where reduction cells are placed
        
        if layers in [2]:
            reduction_cell_position = [1]
        else:
            reduction_cell_position = [layers // 3, 2 * layers // 3]

        # Create the cells based on the number of layers and the reduction positions
        for i in range(layers):
            if i in reduction_cell_position:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, genotype)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            if i == 2 * layers // 3 and auxiliary:
                C_to_auxiliary = C_prev

        # If auxiliary head is enabled, add an auxiliary classifier
        if auxiliary:
            self.auxiliary_head = AuxiliaryNetwork(C_prev, num_classes)
        
        # Global pooling and classifier layers
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        # If no genotype is provided, initialize the architecture parameters
        if genotype is None:
            self._initialize_alphas()

    def forward(self, input):
        """Forward pass of the network."""
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if self.genotype is None:
                if cell.reduction:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
                s0, s1 = s1, cell(s0, s1, weights)
            else:
                s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3 and self._auxiliary:
                logits_aux = self.auxiliary_head(s1)

        # Apply global pooling and classifier
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        return logits, logits_aux

    def new(self):
        """Create a new network with the same architecture parameters."""
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_alphas(self):
        """Initialize the architecture parameters (alphas) which determine the importance of each operation in the MixedOp."""
        k = sum(1 for i in range(self._steps) for _ in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        """Return the architecture parameters."""
        return self._arch_parameters

    def get_genotype(self):
        """Decode the learned architecture parameters to get the final architecture (genotype)."""

        # Helper functions for parsing the weights
        def _isCNNStructure(k_best):
            return k_best >= 4

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            cnn_structure_count = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k

                    if _isCNNStructure(k_best):
                        cnn_structure_count += 1
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene, cnn_structure_count

        # Parse the architecture parameters to get the genotype
        with torch.no_grad():
            gene_normal, cnn_structure_count_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
            gene_reduce, cnn_structure_count_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

            concat = range(2 + self._steps - self._multiplier, self._steps + 2)
            genotype = Genotype(
                normal=gene_normal, normal_concat=concat,
                reduce=gene_reduce, reduce_concat=concat
            )
        return genotype, cnn_structure_count_normal, cnn_structure_count_reduce

