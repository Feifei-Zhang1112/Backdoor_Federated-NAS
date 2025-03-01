from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
"""
PRIMITIVES = [
    'none',
    'fire_2',
    'skip_connect',
    'group_conv_3x3',
    'depth_conv_5x5',
    'group_conv_7x7',
    'ghost_conv_7x7',
    'fire_1'
]
"""
"""
PRIMITIVES = [
    'none',
    'skip_connect',
    'dil_conv_3x3_2',
    'dil_conv_5x5_1',
    'dil_conv_5x5_2',
    'sep_conv_3x3',
    'eca_3',
    'eca_5',
    'eca_7'
]
"""
"""
PRIMITIVES = [
    'none',
    'skip_connect',
    'group_conv_3x3',
    'depth_conv_5x5',
    'ghost_conv_7x7',
    'sep_conv_3x3',
    'eca_3',
    'eca_5',
    'eca_7'
]"""
PRIMITIVES = [
    'none',
    'group_3x3_4',
    'group_3x3_2',
    'group_7x7_32',
    'group_5x5_8',
    'group_7x7_16',
    'group_5x5_32',
    'sep_conv_7x7',
    'skip_connect'
]


NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

# DARTS_V1 = Genotype(
#     normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
#             ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
#     reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0),
#             ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
# DARTS_V2 = Genotype(
#     normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
#             ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
#     reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
#             ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
#
# # poisoned
# DARTS_V3 = Genotype(normal=[('sep_conv_3x3', 1), ('eca_5', 0), ('sep_conv_3x3', 2),
#                             ('sep_conv_3x3', 1), ('eca_7', 0), ('sep_conv_3x3', 2),
#                             ('sep_conv_3x3', 4), ('eca_7', 0)], normal_concat=range(2, 6),
#                     reduce=[('dil_conv_5x5_2', 0), ('skip_connect', 1), ('sep_conv_3x3', 0),
#                             ('dil_conv_3x3_2', 2), ('skip_connect', 1), ('skip_connect', 0),
#                             ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))


A3FL_LOW = Genotype(normal=[('group_3x3_2', 1), ('group_7x7_32', 0), ('group_7x7_32', 0), ('group_3x3_2', 2),
                            ('group_5x5_32', 1), ('group_3x3_2', 3), ('group_5x5_32', 0), ('group_7x7_16', 1)],
                    normal_concat=range(2, 6),
                    reduce=[('group_3x3_2', 1), ('group_5x5_32', 0), ('group_5x5_32', 1), ('group_3x3_2', 2),
                            ('group_3x3_4', 0), ('group_7x7_32', 2), ('group_3x3_4', 3), ('group_3x3_4', 1)],
                    reduce_concat=range(2, 6))

A3FL_HIGH = Genotype(normal=[('group_7x7_16', 0), ('group_7x7_16', 1), ('group_7x7_16', 0), ('group_5x5_32', 1),
                            ('group_5x5_32', 2), ('group_5x5_32', 3), ('group_7x7_32', 1), ('group_5x5_32', 0)],
                    normal_concat=range(2, 6), reduce=[('group_3x3_2', 1), ('group_3x3_2', 0), ('group_3x3_2', 0),
                                                       ('skip_connect', 2), ('skip_connect', 2), ('group_7x7_32', 3),
                                                       ('group_3x3_4', 4), ('skip_connect', 2)],
                    reduce_concat=range(2, 6))

CERP_HIGH = Genotype(
    normal=[('group_7x7_32', 0), ('group_5x5_32', 1), ('group_5x5_32', 2), ('group_7x7_32', 0), ('group_7x7_32', 0),
            ('group_5x5_32', 3), ('group_5x5_32', 3), ('group_7x7_32', 0)], normal_concat=range(2, 6),
    reduce=[('group_3x3_2', 0), ('group_5x5_32', 1), ('group_3x3_2', 0), ('skip_connect', 2), ('group_3x3_4', 0),
            ('group_5x5_32', 1), ('group_5x5_32', 1), ('group_5x5_8', 0)], reduce_concat=range(2, 6))

CERP_LOW = Genotype(normal=[('group_5x5_32', 0), ('group_3x3_4', 1), ('group_5x5_32', 0), ('group_3x3_4', 2), ('group_5x5_32', 0),
                 ('group_3x3_4', 2), ('group_7x7_32', 0), ('group_3x3_4', 2)], normal_concat=range(2, 6),
         reduce=[('group_5x5_8', 0), ('group_7x7_32', 1), ('group_5x5_32', 1), ('group_7x7_16', 0),
                 ('group_3x3_4', 1), ('group_3x3_2', 3), ('group_3x3_4', 0), ('group_3x3_2', 3)],
         reduce_concat=range(2, 6))

DBA_LOW = Genotype(normal=[('group_7x7_32', 0), ('group_5x5_32', 1), ('group_7x7_32', 0), ('group_5x5_32', 1),
                 ('group_5x5_32', 0), ('group_3x3_4', 2), ('group_7x7_16', 0), ('group_3x3_4', 2)],
         normal_concat=range(2, 6), reduce=[('group_3x3_2', 0), ('group_3x3_2', 1), ('skip_connect', 2),
                                            ('group_3x3_2', 0), ('skip_connect', 2), ('group_7x7_32', 3),
                                            ('group_5x5_32', 4), ('group_5x5_8', 1)], reduce_concat=range(2, 6))

NO_ATTACK = Genotype(normal=[('group_7x7_16', 0), ('group_5x5_32', 1), ('group_7x7_32', 0), ('group_5x5_32', 2),
                             ('group_7x7_32', 0), ('group_3x3_4', 3), ('group_3x3_4', 2), ('group_5x5_32', 1)],
                     normal_concat=range(2, 6), reduce=[('group_3x3_2', 0), ('sep_conv_7x7', 1), ('group_3x3_2', 1),
                                                        ('skip_connect', 0), ('group_3x3_4', 3), ('group_5x5_32', 0),
                                                        ('group_3x3_2', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))
DARTS = DBA_LOW

FedNAS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1),
            ('sep_conv_5x5', 3), ('dil_conv_5x5', 3), ('sep_conv_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0),
            ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
