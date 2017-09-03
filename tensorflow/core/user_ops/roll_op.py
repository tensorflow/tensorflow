from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import sys
# import numpy as np
#
# # from tensorflow.python.eager import context
# # from tensorflow.python.framework import common_shapes
# from tensorflow.python.framework import constant_op
# from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import ops
# from tensorflow.python.framework import sparse_tensor
# from tensorflow.python.framework import tensor_shape
# from tensorflow.python.framework import tensor_util
# # 'Constant' gets imported in the module 'array_ops'.
# from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import gen_array_ops
# from tensorflow.python.ops import gen_math_ops
# # go/tf-wildcard-import
# # pylint: disable=wildcard-import
# from tensorflow.python.ops.gen_array_ops import *
# from tensorflow.python.util import deprecation
# from tensorflow.python.util.deprecation import deprecated
# # pylint: enable=wildcard-import

def roll(input, shift, axis):
    shifts = {}
    for s,a in zip(shift, axis):
        shifts[a] = shifts.get(a, 0) + s
    shifts = sorted([(a,s) for a,s in shifts.items()])

    def roll_inner(input, shifts):
        if shifts:
            a,s = shifts.pop(0)
            print(a,s)
            gen_array_ops._split_v(
                value=value,
                size_splits=size_splits,
                split_dim=axis,
                num_split=num,
                name=name)
            roll_inner(input, shifts.copy())
            roll_inner(input, shifts.copy())

    roll_inner(input, shifts)



roll([], [2,1,-3,2,-3], [2,1,0,1,2])
# help(gen_array_ops)
