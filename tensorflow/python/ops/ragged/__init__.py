"""Ragged Tensors.

This package defines the [`RaggedTensor`](ragged/RaggedTensor.md) class, which
represents tensors with non-uniform shapes.  In particular, each `RaggedTensor`
has one or more *ragged dimensions*, which are dimensions whose slices may have
different lengths.  For example, the inner (column) dimension of
`rt=[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is ragged, since the column slices
(`rt[0, :]`, ..., `rt[4, :]`) have different lengths.  For a more detailed
description of ragged tensors, see the [`RaggedTensor`](ragged/RaggedTensor.md)
class documentation.

## RaggedTensor Operations

This package also defines a collection of operations for manipulating
ragged tensors.

### RaggedTensor Versions of Standard Tensor Operations

Many of the operations defined by this package are analogous to
[`Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor)
operations, but they accept `RaggedTensor`s as input and can return
`RaggedTensor`s as output.  For example, `ragged.add` performs elementwise
addition just like `tf.add`, but can be used on `RaggedTensor`s.

These `RaggedTensor` versions of the standard `Tensor` operations can also be
used with standard `Tensors`; and for the most part, they will return the same
value that the standard `Tensor` operation would return.  However, there are
a few notable exceptions:

* For [`ragged.stack(...)`](ragged/stack.md) and
  [`ragged.concat(...)`](ragged/concat.md), the input tensors are not required
  to have matching shapes.  In the returned tensor, all dimensions up to the
  `axis` dimension will be ragged.

### Ragged-Tensor Specific Operations

The following operations are specific to ragged tensors:

* **Factory ops**:
  [`constant(...)`](ragged/constant.md),
  [`from_row_splits(...)`](ragged/from_row_splits.md),
  [`from_row_lengths(...)`](ragged/from_row_lengths.md),
  [`from_row_starts(...)`](ragged/from_row_starts.md),
  [`from_row_limits(...)`](ragged/from_row_limits.md),
  [`from_value_rowids(...)`](ragged/from_value_rowids.md),
  [`from_nested_row_splits(...)`](ragged/from_nested_row_splits.md),
  [`from_nested_value_rowids(...)`](ragged/from_nested_value_rowids.md).

* **Conversion ops**:
  [`from_tensor(...)`](ragged/from_tensor.md),
  [`to_tensor(...)`](ragged/to_tensor.md),
  [`from_sparse(...)`](ragged/from_sparse.md),
  [`to_sparse(...)`](ragged/to_sparse.md),
  [`from_variant(...)`](ragged/from_variant.md),
  [`to_variant(...)`](ragged/to_variant.md),
  [`convert_to_tensor_or_ragged_tensor(...)`](
  ragged/convert_to_tensor_or_ragged_tensor.md).

* **Shape ops**:
  [`row_splits(...)`](ragged/row_splits.md),
  [`row_lengths(...)`](ragged/row_lengths.md),
  [`row_starts(...)`](ragged/row_starts.md),
  [`row_limits(...)`](ragged/row_limits.md),
  [`value_rowids(...)`](ragged/value_rowids.md),
  [`nrows(...)`](ragged/nrows.md),
  [`nested_row_splits(...)`](ragged/nested_row_splits.md),
  [`row_splits_to_segment_ids(...)`](ragged/row_splits_to_segment_ids.md),
  [`segment_ids_to_row_splits(...)`](ragged/segment_ids_to_row_splits.md),
  [`bounding_shape(...)`](ragged/bounding_shape.md).

* **Functional ops**:
  [`map_inner_values(...)`](ragged/map_inner_values.md),
  [`make_elementwise_op(...)`](ragged/make_elementwise_op.md).


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

<!-- Elementwise Ops -->
@@make_elementwise_op

<!-- Symbols from  ragged_elementwise_ops._symbols_to_export are whitelisted -->
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.ragged import ragged_operators

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

# pylint: disable=protected-access, wildcard-import
from tensorflow.python.ops.ragged.ragged_elementwise_ops import *
from tensorflow.python.ops.ragged.ragged_elementwise_ops import _symbols_to_export as _elementwise_ops
# pylint: enable=protected-access, wildcard-import

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

from tensorflow.python.ops.ragged.ragged_tensor_value import RaggedTensorValue

from tensorflow.python.ops.ragged.segment_id_ops import row_splits_to_segment_ids
from tensorflow.python.ops.ragged.segment_id_ops import segment_ids_to_row_splits

from tensorflow.python.util import all_util as _all_util

# Any symbol that is not referenced (with "@@name") in the module docstring
# above, or included in the "_elementwise_ops" whitelist, will be removed.
_all_util.remove_undocumented(__name__, _elementwise_ops)
