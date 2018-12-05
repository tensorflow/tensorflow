"""Ragged Tensors.

This package defines the `tf.RaggedTensor` class, which
represents tensors with non-uniform shapes.  In particular, each `RaggedTensor`
has one or more *ragged dimensions*, which are dimensions whose slices may have
different lengths.  For example, the inner (column) dimension of
`rt=[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is ragged, since the column slices
(`rt[0, :]`, ..., `rt[4, :]`) have different lengths.  For a more detailed
description of ragged tensors, see the `tf.RaggedTensor`
class documentation.

## `RaggedTensor` Operations

### `RaggedTensor` Factory ops

* `tf.ragged.constant`
* `tf.ragged.from_row_splits`
* `tf.ragged.from_row_splits`
* `tf.ragged.from_row_lengths`
* `tf.ragged.from_row_starts`
* `tf.ragged.from_row_limits`
* `tf.ragged.from_value_rowids`
* `tf.ragged.from_nested_row_splits`
* `tf.ragged.from_nested_value_rowids`

### `RaggedTensor` Conversion ops

* `tf.ragged.from_tensor`
* `tf.ragged.to_tensor`
* `tf.ragged.from_sparse`
* `tf.ragged.to_sparse`
* `tf.ragged.from_variant`
* `tf.ragged.to_variant`
* `tf.ragged.convert_to_tensor_or_ragged_tensor`

### `RaggedTensor` Shape ops

* `tf.ragged.row_splits`
* `tf.ragged.row_lengths`
* `tf.ragged.row_starts`
* `tf.ragged.row_limits`
* `tf.ragged.value_rowids`
* `tf.ragged.nrows`
* `tf.ragged.nested_row_splits`
* `tf.ragged.row_splits_to_segment_ids`
* `tf.ragged.segment_ids_to_row_splits`
* `tf.ragged.bounding_shape`

### Functional ops
* `tf.ragged.map_inner_values`


<!-- Ragged Classes & related helper functions -->
@@RaggedTensor
@@RaggedTensorType
@@RaggedTensorValue
@@is_ragged

<!-- Factory Ops -->
@@constant
@@constant_value
@@from_row_splits
@@from_row_lengths
@@from_row_starts
@@from_row_limits
@@from_value_rowids
@@from_nested_row_splits
@@from_nested_value_rowids
@@convert_to_tensor_or_ragged_tensor

<!-- Conversion Ops -->
@@from_tensor
@@to_tensor
@@from_sparse
@@to_sparse
@@row_splits_to_segment_ids
@@segment_ids_to_row_splits

<!-- Array Ops -->
@@row_splits
@@row_lengths
@@row_starts
@@row_limits
@@value_rowids
@@nrows
@@nested_row_splits
@@bounding_shape
@@gather
@@batch_gather
@@gather_nd
@@boolean_mask
@@concat
@@stack
@@tile
@@expand_dims
@@where

<!-- Math Ops -->
@@range

@@segment_sum
@@segment_prod
@@segment_min
@@segment_max
@@segment_mean
@@segment_sqrt_n

@@reduce_sum
@@reduce_prod
@@reduce_min
@@reduce_max
@@reduce_mean
@@reduce_all
@@reduce_any

<!-- Functional Ops -->
@@map_inner_values
@@map_fn

<!-- Shape & broadcasting -->
@@RaggedTensorDynamicShape
@@broadcast_to
@@broadcast_dynamic_shape
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.ragged import ragged_dispatch
from tensorflow.python.ops.ragged import ragged_operators
from tensorflow.python.ops.ragged import ragged_string_ops

from tensorflow.python.ops.ragged.ragged_array_ops import batch_gather
from tensorflow.python.ops.ragged.ragged_array_ops import boolean_mask
from tensorflow.python.ops.ragged.ragged_array_ops import bounding_shape
from tensorflow.python.ops.ragged.ragged_array_ops import concat
from tensorflow.python.ops.ragged.ragged_array_ops import expand_dims
from tensorflow.python.ops.ragged.ragged_array_ops import gather
from tensorflow.python.ops.ragged.ragged_array_ops import gather_nd
from tensorflow.python.ops.ragged.ragged_array_ops import nrows
from tensorflow.python.ops.ragged.ragged_array_ops import row_lengths
from tensorflow.python.ops.ragged.ragged_array_ops import row_limits
from tensorflow.python.ops.ragged.ragged_array_ops import row_starts
from tensorflow.python.ops.ragged.ragged_array_ops import stack
from tensorflow.python.ops.ragged.ragged_array_ops import tile
from tensorflow.python.ops.ragged.ragged_array_ops import value_rowids
from tensorflow.python.ops.ragged.ragged_array_ops import where

from tensorflow.python.ops.ragged.ragged_conversion_ops import from_sparse
from tensorflow.python.ops.ragged.ragged_conversion_ops import from_tensor
from tensorflow.python.ops.ragged.ragged_conversion_ops import to_sparse
from tensorflow.python.ops.ragged.ragged_conversion_ops import to_tensor

from tensorflow.python.ops.ragged.ragged_factory_ops import constant
from tensorflow.python.ops.ragged.ragged_factory_ops import constant_value
from tensorflow.python.ops.ragged.ragged_factory_ops import convert_to_tensor_or_ragged_tensor
from tensorflow.python.ops.ragged.ragged_factory_ops import from_nested_row_splits
from tensorflow.python.ops.ragged.ragged_factory_ops import from_nested_value_rowids
from tensorflow.python.ops.ragged.ragged_factory_ops import from_row_lengths
from tensorflow.python.ops.ragged.ragged_factory_ops import from_row_limits
from tensorflow.python.ops.ragged.ragged_factory_ops import from_row_splits
from tensorflow.python.ops.ragged.ragged_factory_ops import from_row_starts
from tensorflow.python.ops.ragged.ragged_factory_ops import from_value_rowids

from tensorflow.python.ops.ragged.ragged_functional_ops import map_inner_values

from tensorflow.python.ops.ragged.ragged_map_ops import map_fn

from tensorflow.python.ops.ragged.ragged_math_ops import range  # pylint: disable=redefined-builtin

from tensorflow.python.ops.ragged.ragged_math_ops import reduce_all
from tensorflow.python.ops.ragged.ragged_math_ops import reduce_any
from tensorflow.python.ops.ragged.ragged_math_ops import reduce_max
from tensorflow.python.ops.ragged.ragged_math_ops import reduce_mean
from tensorflow.python.ops.ragged.ragged_math_ops import reduce_min
from tensorflow.python.ops.ragged.ragged_math_ops import reduce_prod
from tensorflow.python.ops.ragged.ragged_math_ops import reduce_sum

from tensorflow.python.ops.ragged.ragged_math_ops import segment_max
from tensorflow.python.ops.ragged.ragged_math_ops import segment_mean
from tensorflow.python.ops.ragged.ragged_math_ops import segment_min
from tensorflow.python.ops.ragged.ragged_math_ops import segment_prod
from tensorflow.python.ops.ragged.ragged_math_ops import segment_sqrt_n
from tensorflow.python.ops.ragged.ragged_math_ops import segment_sum

from tensorflow.python.ops.ragged.ragged_tensor import is_ragged
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorType

from tensorflow.python.ops.ragged.ragged_tensor_shape import broadcast_dynamic_shape
from tensorflow.python.ops.ragged.ragged_tensor_shape import broadcast_to
from tensorflow.python.ops.ragged.ragged_tensor_shape import RaggedTensorDynamicShape

from tensorflow.python.ops.ragged.ragged_tensor_value import RaggedTensorValue

from tensorflow.python.ops.ragged.segment_id_ops import row_splits_to_segment_ids
from tensorflow.python.ops.ragged.segment_id_ops import segment_ids_to_row_splits

from tensorflow.python.util import all_util as _all_util


# Register OpDispatchers that override standard TF ops to work w/ RaggedTensors.
__doc__ += ragged_dispatch.register_dispatchers()  # pylint: disable=redefined-builtin

# Any symbol that is not referenced (with "@@name") in the module docstring
# above will be removed.
_all_util.remove_undocumented(__name__)
