from collections import OrderedDict

import numpy as np 
import torch  
from torch.autograd import Variable 

"""
This file contains the code for the DARTS architecture search algorithm.
Due to the math-heavy nature of the algorithm, we have provided a very verbose explanation of the algorithm in the comments.
"""

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])  # Concatenates tensors in the list 'xs' after flattening them

 # The Architect class is used to update the architecture parameters of the model
class Architect(object): 

    def __init__(self, model, criterion, config):  
        self.network_momentum = config['momentum']  # Setting network momentum from the config
        self.network_weight_decay = config['weight_decay']  # Setting weight decay from the config
        self.model = model  # Setting the model
        self.criterion = criterion  # Setting the criterion (loss function)

        arch_parameters = self.model.arch_parameters()  # Getting architecture parameters from the model
        
        self.optimizer = torch.optim.Adam( # Initialize the Adam optimizer for updating architecture parameters.
            arch_parameters,
            lr=config['arch_learning_rate'], betas=(0.5, 0.999),
            weight_decay=config['arch_weight_decay'])

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):  # Computes the unrolled model for bi-level optimization
        logits, _ = self.model(input)          # Initialize the Adam optimizer for updating architecture parameters.
        loss = self.criterion(logits, target)  # Calculates the loss using the predictions and true labels.

        # Concatenate and flatten the model parameters.
        theta = _concat(self.model.parameters()).data
        
        try:
            # Attempt to compute the momentum for the model parameters.
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            # If momentum doesn't exist, initialize it with zeros.
            moment = torch.zeros_like(theta)

        # Compute the gradient of the loss with respect to the model parameters and add weight decay.
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        
        # Construct a new model using the updated parameters.
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))

        return unrolled_model
        

    # DARTS
    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):  # Step function for updating architecture parameters
        self.optimizer.zero_grad()  # Zero out the gradients
        if unrolled:
             # If using the unrolled model, compute gradients using the unrolled model.
             self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
             # Otherwise, compute gradients using the current model.
             self._backward_step(input_valid, target_valid)

        self.optimizer.step()  # Apply the gradient updates to the architecture parameters.

    def _backward_step(self, input_valid, target_valid):  # Compute gradients using the current model
        logits, _ = self.model(input_valid)  # Computes the model's predictions on validation data.
        loss = self.criterion(logits, target_valid)  # Calculates the loss using the predictions and true labels.
        
        loss.backward()  # Backpropagates the loss to compute gradients.

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):  # Compute gradients using the unrolled model
                # Retrieve the unrolled model.
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        
        logits, _ = unrolled_model(input_valid)  # Computes the unrolled model's predictions on validation data.
        unrolled_loss = self.criterion(logits, target_valid)  # Calculates the loss using the predictions and true labels.
        
        unrolled_loss.backward()  # Backpropagates the loss to compute gradients for the unrolled model.

        # Retrieve gradients for the architecture parameters of the unrolled model.
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        
        # Retrieve gradients for the model parameters of the unrolled model.
        vector = [v.grad.data for v in unrolled_model.parameters()]
        
        # Compute the Hessian-vector product.
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)
        
        # Update the architecture gradients using the Hessian-vector product.
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        # Retrieve the current architecture parameters.
        arch_parameters = self.model.arch_parameters()
        
        # Update the architecture parameters using the computed gradients.
        for v, g in zip(arch_parameters, dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)  # If gradient doesn't exist for a parameter, initialize it.
            else:
                v.grad.data.copy_(g.data)  # Otherwise, copy the computed gradient to the parameter's gradient.

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()  # Create a new instance of the model
        model_dict = self.model.state_dict()  # Get the state dictionary of the current model
    
        params, offset = {}, 0
        named_parameters = self.model.named_parameters()
        for k, v in named_parameters:  # Iterate over the named parameters of the model
            v_length = np.prod(v.size())  # Compute the total number of elements in the parameter tensor
            params[k] = theta[offset: offset + v_length].view(v.size())  # Extract the corresponding parameters from theta and reshape them
            offset += v_length  # Update the offset
    
        assert offset == len(theta)  # Ensure that the entire theta has been processed
        model_dict.update(params)  # Update the model's state dictionary with the new parameters
        new_state_dict = model_dict  # Assign the updated state dictionary to a new variable
        model_new.load_state_dict(new_state_dict)  # Load the new state dictionary into the new model instance
        return model_new.cuda()  # Return the new model instance moved to the GPU
    
    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        # This function computes the Hessian-vector product, which is used to approximate second-order derivatives.
        # The Hessian-vector product is a way to efficiently compute the product of the Hessian matrix (second-order derivatives) and a vector without explicitly computing the Hessian matrix.
    
        R = r / _concat(vector).norm()  # Compute the scaling factor R
        parameters = self.model.parameters()  # Get the parameters of the model
        for p, v in zip(parameters, vector):  # Iterate over the model parameters and the vector
            p.data.add_(R, v)  # Perturb the model parameters by adding a scaled version of the vector (w+ in equation (8))
    
        # Compute the gradient of the architecture parameters with respect to the loss using the perturbed model parameters (w+)
        logits, _ = self.model(input)  # Forward pass through the model
        loss = self.criterion(logits, target)  # Compute the loss
    
        arch_parameters = self.model.arch_parameters()  # Get the architecture parameters of the model
        grads_p = torch.autograd.grad(loss, arch_parameters)  # Compute the gradient of the loss with respect to the architecture parameters
    
        for p, v in zip(parameters, vector):  # Iterate over the model parameters and the vector again
            p.data.sub_(2 * R, v)  # Perturb the model parameters by subtracting a scaled version of the vector (w- in equation (8))
    
        # Compute the gradient of the architecture parameters with respect to the loss using the perturbed model parameters (w-)
        logits, _ = self.model(input)  # Forward pass through the model
        loss = self.criterion(logits, target)  # Compute the loss
    
        arch_parameters =  self.model.arch_parameters()  # Get the architecture parameters of the model again
        grads_n = torch.autograd.grad(loss, arch_parameters)  # Compute the gradient of the loss with respect to the architecture parameters
    
        # Restore the original model parameters by adding back the scaled vector
        for p, v in zip(parameters, vector):
            p.data.add_(R, v)
    
        # Compute the Hessian-vector product by taking the difference between the gradients computed using w+ and w- and dividing by (2 * R)
        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
    