/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;

REGISTER_OP("SetSize")
    .Input("set_indices: int64")
    .Input("set_values: T")
    .Input("set_shape: int64")
    .Attr("validate_indices: bool = true")
    .Attr("T: {int8, int16, int32, int64, uint8, uint16, string}")
    .Output("size: int32")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Number of unique elements along last dimension of input `set`.

Input `set` is a `SparseTensor` represented by `set_indices`, `set_values`,
and `set_shape`. The last dimension contains values in a set, duplicates are
allowed but ignored.

If `validate_indices` is `True`, this op validates the order and range of `set`
indices.

set_indices: 2D `Tensor`, indices of a `SparseTensor`.
set_values: 1D `Tensor`, values of a `SparseTensor`.
set_shape: 1D `Tensor`, shape of a `SparseTensor`.
size: For `set` ranked `n`, this is a `Tensor` with rank `n-1`, and the same 1st
    `n-1` dimensions as `set`. Each value is the number of unique elements in
    the corresponding `[0...n-1]` dimension of `set`.
)doc");

REGISTER_OP("DenseToDenseSetOperation")
    .Input("set1: T")
    .Input("set2: T")
    .Attr("set_operation: string")
    .Attr("validate_indices: bool = true")
    .Attr("T: {int8, int16, int32, int64, uint8, uint16, string}")
    .Output("result_indices: int64")
    .Output("result_values: T")
    .Output("result_shape: int64")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), 2));
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->Vector(c->UnknownDim()));
      return Status::OK();
    })
    .Doc(R"doc(
Applies set operation along last dimension of 2 `Tensor` inputs.

See SetOperationOp::SetOperationFromContext for values of `set_operation`.

Output `result` is a `SparseTensor` represented by `result_indices`,
`result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
dimension contains the result of `set_operation` applied to the corresponding
`[0...n-1]` dimension of `set`.

set1: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set2`.
    Dimension `n` contains values in a set, duplicates are allowed but ignored.
set2: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set1`.
    Dimension `n` contains values in a set, duplicates are allowed but ignored.
result_indices: 2D indices of a `SparseTensor`.
result_values: 1D values of a `SparseTensor`.
result_shape: 1D `Tensor` shape of a `SparseTensor`. `result_shape[0...n-1]` is
    the same as the 1st `n-1` dimensions of `set1` and `set2`, `result_shape[n]`
    is the max result set size across all `0...n-1` dimensions.
)doc");

REGISTER_OP("DenseToSparseSetOperation")
    .Input("set1: T")
    .Input("set2_indices: int64")
    .Input("set2_values: T")
    .Input("set2_shape: int64")
    .Attr("set_operation: string")
    .Attr("validate_indices: bool = true")
    .Attr("T: {int8, int16, int32, int64, uint8, uint16, string}")
    .Output("result_indices: int64")
    .Output("result_values: T")
    .Output("result_shape: int64")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), 2));
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->Vector(c->UnknownDim()));
      return Status::OK();
    })
    .Doc(R"doc(
Applies set operation along last dimension of `Tensor` and `SparseTensor`.

See SetOperationOp::SetOperationFromContext for values of `set_operation`.

Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
ignored.

If `validate_indices` is `True`, this op validates the order and range of `set2`
indices.

Output `result` is a `SparseTensor` represented by `result_indices`,
`result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
dimension contains the result of `set_operation` applied to the corresponding
`[0...n-1]` dimension of `set`.

set1: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set2`.
    Dimension `n` contains values in a set, duplicates are allowed but ignored.
set2_indices: 2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
    order.
set2_values: 1D `Tensor`, values of a `SparseTensor`. Must be in row-major
    order.
set2_shape: 1D `Tensor`, shape of a `SparseTensor`. `set2_shape[0...n-1]` must
    be the same as the 1st `n-1` dimensions of `set1`, `result_shape[n]` is the
    max set size across `n-1` dimensions.
result_indices: 2D indices of a `SparseTensor`.
result_values: 1D values of a `SparseTensor`.
result_shape: 1D `Tensor` shape of a `SparseTensor`. `result_shape[0...n-1]` is
    the same as the 1st `n-1` dimensions of `set1` and `set2`, `result_shape[n]`
    is the max result set size across all `0...n-1` dimensions.
)doc");

REGISTER_OP("SparseToSparseSetOperation")
    .Input("set1_indices: int64")
    .Input("set1_values: T")
    .Input("set1_shape: int64")
    .Input("set2_indices: int64")
    .Input("set2_values: T")
    .Input("set2_shape: int64")
    .Attr("set_operation: string")
    .Attr("validate_indices: bool = true")
    .Attr("T: {int8, int16, int32, int64, uint8, uint16, string}")
    .Output("result_indices: int64")
    .Output("result_values: T")
    .Output("result_shape: int64")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Matrix(c->UnknownDim(), 2));
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->Vector(c->UnknownDim()));
      return Status::OK();
    })
    .Doc(R"doc(
Applies set operation along last dimension of 2 `SparseTensor` inputs.

See SetOperationOp::SetOperationFromContext for values of `set_operation`.

If `validate_indices` is `True`, `SparseToSparseSetOperation` validates the
order and range of `set1` and `set2` indices.

Input `set1` is a `SparseTensor` represented by `set1_indices`, `set1_values`,
and `set1_shape`. For `set1` ranked `n`, 1st `n-1` dimensions must be the same
as `set2`. Dimension `n` contains values in a set, duplicates are allowed but
ignored.

Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
ignored.

If `validate_indices` is `True`, this op validates the order and range of `set1`
and `set2` indices.

Output `result` is a `SparseTensor` represented by `result_indices`,
`result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
dimension contains the result of `set_operation` applied to the corresponding
`[0...n-1]` dimension of `set`.

set1_indices: 2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
    order.
set1_values: 1D `Tensor`, values of a `SparseTensor`. Must be in row-major
    order.
set1_shape: 1D `Tensor`, shape of a `SparseTensor`. `set1_shape[0...n-1]` must
    be the same as `set2_shape[0...n-1]`, `set1_shape[n]` is the
    max set size across `0...n-1` dimensions.
set2_indices: 2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
    order.
set2_values: 1D `Tensor`, values of a `SparseTensor`. Must be in row-major
    order.
set2_shape: 1D `Tensor`, shape of a `SparseTensor`. `set2_shape[0...n-1]` must
    be the same as `set1_shape[0...n-1]`, `set2_shape[n]` is the
    max set size across `0...n-1` dimensions.
result_indices: 2D indices of a `SparseTensor`.
result_values: 1D values of a `SparseTensor`.
result_shape: 1D `Tensor` shape of a `SparseTensor`. `result_shape[0...n-1]` is
    the same as the 1st `n-1` dimensions of `set1` and `set2`, `result_shape[n]`
    is the max result set size across all `0...n-1` dimensions.
)doc");

}  // namespace tensorflow
