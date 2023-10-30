from collections import namedtuple

# Define a named tuple for Genotype which will represent the structure of the network.
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# Define the primitive operations that can be used in the network.
PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]


# Cell structes for different purposes found by the NAS algorithm.
CHARGE_DETECTION_5 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('skip_connect', 3), ('max_pool_3x3', 1), ('skip_connect', 3), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
CHARGE_DETECTION_2 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_3x3', 4), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_3x3', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))


