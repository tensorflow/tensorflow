/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("Variable")
    .Output("ref: Ref(dtype)")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Holds state in the form of a tensor that persists across steps.

Outputs a ref to the tensor state so it may be read or modified.
TODO(zhifengc/mrry): Adds a pointer to a more detail document
about sharing states in tensorflow.

ref: A reference to the variable tensor.
shape: The shape of the variable tensor.
dtype: The type of elements in the variable tensor.
container: If non-empty, this variable is placed in the given container.
        Otherwise, a default container is used.
shared_name: If non-empty, this variable is named in the given bucket
             with this shared_name. Otherwise, the node name is used instead.
)doc");

REGISTER_OP("IsVariableInitialized")
    .Input("ref: Ref(dtype)")
    .Output("is_initialized: bool")
    .Attr("dtype: type")
    .SetAllowsUninitializedInput()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Checks whether a tensor has been initialized.

Outputs boolean scalar indicating whether the tensor has been initialized.

ref: Should be from a `Variable` node. May be uninitialized.
dtype: The type of elements in the variable tensor.
)doc");

REGISTER_OP("TemporaryVariable")
    .Output("ref: Ref(dtype)")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("var_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      TensorShapeProto shape_proto;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape_proto));
      ShapeHandle output;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeProto(shape_proto, &output));
      c->set_output(0, output);
      return Status::OK();
    })
    .Doc(R"doc(
Returns a tensor that may be mutated, but only persists within a single step.

This is an experimental op for internal use only and it is possible to use this
op in unsafe ways.  DO NOT USE unless you fully understand the risks.

It is the caller's responsibility to ensure that 'ref' is eventually passed to a
matching 'DestroyTemporaryVariable' op after all other uses have completed.

Outputs a ref to the tensor state so it may be read or modified.

  E.g.
      var = state_ops._temporary_variable([1, 2], types.float_)
      var_name = var.op.name
      var = state_ops.assign(var, [[4.0, 5.0]])
      var = state_ops.assign_add(var, [[6.0, 7.0]])
      final = state_ops._destroy_temporary_variable(var, var_name=var_name)

ref: A reference to the variable tensor.
shape: The shape of the variable tensor.
dtype: The type of elements in the variable tensor.
var_name: Overrides the name used for the temporary variable resource. Default
value is the name of the 'TemporaryVariable' op (which is guaranteed unique).
)doc");

REGISTER_OP("DestroyTemporaryVariable")
    .Input("ref: Ref(T)")
    .Output("value: T")
    .Attr("T: type")
    .Attr("var_name: string")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Destroys the temporary variable and returns its final value.

Sets output to the value of the Tensor pointed to by 'ref', then destroys
the temporary variable called 'var_name'.
All other uses of 'ref' *must* have executed before this op.
This is typically achieved by chaining the ref through each assign op, or by
using control dependencies.

Outputs the final value of the tensor pointed to by 'ref'.

ref: A reference to the temporary variable tensor.
var_name: Name of the temporary variable, usually the name of the matching
'TemporaryVariable' op.
)doc");

REGISTER_OP("Assign")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("validate_shape: bool = true")
    .Attr("use_locking: bool = true")
    .SetAllowsUninitializedInput()
    .SetShapeFn([](InferenceContext* c) {
      bool validate_shape;
      TF_RETURN_IF_ERROR(c->GetAttr("validate_shape", &validate_shape));
      if (validate_shape) {
        return shape_inference::MergeBothInputsShapeFn(c);
      }

      c->set_output(0, c->input(1));
      return Status::OK();
    })
    .Doc(R"doc(
Update 'ref' by assigning 'value' to it.

This operation outputs "ref" after the assignment is done.
This makes it easier to chain operations that need to use the reset value.

ref: Should be from a `Variable` node. May be uninitialized.
value: The value to be assigned to the variable.
validate_shape: If true, the operation will validate that the shape
  of 'value' matches the shape of the Tensor being assigned to.  If false,
  'ref' will take on the shape of 'value'.
use_locking: If True, the assignment will be protected by a lock;
  otherwise the behavior is undefined, but may exhibit less contention.
output_ref:= Same as "ref".  Returned as a convenience for operations that want
  to use the new value after the variable has been reset.
)doc");

REGISTER_OP("AssignAdd")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn)
    .Doc(R"doc(
Update 'ref' by adding 'value' to it.

This operation outputs "ref" after the update is done.
This makes it easier to chain operations that need to use the reset value.

ref: Should be from a `Variable` node.
value: The value to be added to the variable.
use_locking: If True, the addition will be protected by a lock;
  otherwise the behavior is undefined, but may exhibit less contention.
output_ref:= Same as "ref".  Returned as a convenience for operations that want
  to use the new value after the variable has been updated.
)doc");

REGISTER_OP("AssignSub")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn)
    .Doc(R"doc(
Update 'ref' by subtracting 'value' from it.

This operation outputs "ref" after the update is done.
This makes it easier to chain operations that need to use the reset value.

ref: Should be from a `Variable` node.
value: The value to be subtracted to the variable.
use_locking: If True, the subtraction will be protected by a lock;
  otherwise the behavior is undefined, but may exhibit less contention.
output_ref:= Same as "ref".  Returned as a convenience for operations that want
  to use the new value after the variable has been updated.
)doc");

namespace {

Status ScatterUpdateShape(InferenceContext* c) {
  ShapeHandle var_shape = c->input(0);
  ShapeHandle indices_shape = c->input(1);

  ShapeHandle unused_updates_shape;
  ShapeHandle concat;
  ShapeHandle var_subshape;
  TF_RETURN_IF_ERROR(c->Subshape(var_shape, 1, &var_subshape));
  TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, var_subshape, &concat));
  TF_RETURN_IF_ERROR(c->Merge(c->input(2), concat, &unused_updates_shape));

  c->set_output(0, var_shape);
  return Status::OK();
}

}  // namespace

REGISTER_OP("ScatterUpdate")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = true")
    .SetShapeFn(ScatterUpdateShape)
    .Doc(R"doc(
Applies sparse updates to a variable reference.

This operation computes

    # Scalar indices
    ref[indices, ...] = updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] = updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]

This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.

If values in `ref` is to be updated more than once, because there are
duplicate entires in `indices`, the order at which the updates happen
for each value is undefined.

Requires `updates.shape = indices.shape + ref.shape[1:]`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/ScatterUpdate.png" alt>
</div>

ref: Should be from a `Variable` node.
indices: A tensor of indices into the first dimension of `ref`.
updates: A tensor of updated values to store in `ref`.
output_ref:= Same as `ref`.  Returned as a convenience for operations that want
  to use the updated values after the update is done.
use_locking: If True, the assignment will be protected by a lock;
  otherwise the behavior is undefined, but may exhibit less contention.
)doc");

REGISTER_OP("ScatterAdd")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterUpdateShape)
    .Doc(R"doc(
Adds sparse updates to a variable reference.

This operation computes

    # Scalar indices
    ref[indices, ...] += updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] += updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]

This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions add.

Requires `updates.shape = indices.shape + ref.shape[1:]`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/ScatterAdd.png" alt>
</div>

ref: Should be from a `Variable` node.
indices: A tensor of indices into the first dimension of `ref`.
updates: A tensor of updated values to add to `ref`.
output_ref:= Same as `ref`.  Returned as a convenience for operations that want
  to use the updated values after the update is done.
use_locking: If True, the addition will be protected by a lock;
  otherwise the behavior is undefined, but may exhibit less contention.
)doc");

REGISTER_OP("ScatterSub")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterUpdateShape)
    .Doc(R"doc(
Subtracts sparse updates to a variable reference.

    # Scalar indices
    ref[indices, ...] -= updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] -= updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]

This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their (negated) contributions add.

Requires `updates.shape = indices.shape + ref.shape[1:]`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/ScatterSub.png" alt>
</div>

ref: Should be from a `Variable` node.
indices: A tensor of indices into the first dimension of `ref`.
updates: A tensor of updated values to subtract from `ref`.
output_ref:= Same as `ref`.  Returned as a convenience for operations that want
  to use the updated values after the update is done.
use_locking: If True, the subtraction will be protected by a lock;
  otherwise the behavior is undefined, but may exhibit less contention.
)doc");

REGISTER_OP("ScatterMul")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterUpdateShape)
    .Doc(R"doc(
Multiplies sparse updates into a variable reference.

This operation computes

    # Scalar indices
    ref[indices, ...] *= updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] *= updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] *= updates[i, ..., j, ...]

This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions multiply.

Requires `updates.shape = indices.shape + ref.shape[1:]`.

ref: Should be from a `Variable` node.
indices: A tensor of indices into the first dimension of `ref`.
updates: A tensor of updated values to multiply to `ref`.
output_ref:= Same as `ref`.  Returned as a convenience for operations that want
  to use the updated values after the update is done.
use_locking: If True, the operation will be protected by a lock;
  otherwise the behavior is undefined, but may exhibit less contention.
)doc");

REGISTER_OP("ScatterDiv")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterUpdateShape)
    .Doc(R"doc(
Divides a variable reference by sparse updates.

This operation computes

    # Scalar indices
    ref[indices, ...] /= updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] /= updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] /= updates[i, ..., j, ...]

This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions divide.

Requires `updates.shape = indices.shape + ref.shape[1:]`.

ref: Should be from a `Variable` node.
indices: A tensor of indices into the first dimension of `ref`.
updates: A tensor of values that `ref` is divided by.
output_ref:= Same as `ref`.  Returned as a convenience for operations that want
  to use the updated values after the update is done.
use_locking: If True, the operation will be protected by a lock;
  otherwise the behavior is undefined, but may exhibit less contention.
)doc");

REGISTER_OP("CountUpTo")
    .Input("ref: Ref(T)")
    .Output("output: T")
    .Attr("limit: int")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle output;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &output));
      c->set_output(0, output);
      return Status::OK();
    })
    .Doc(R"doc(
Increments 'ref' until it reaches 'limit'.

This operation outputs "ref" after the update is done.  This makes it
easier to chain operations that need to use the updated value.

ref: Should be from a scalar `Variable` node.
limit: If incrementing ref would bring it above limit, instead generates an
  'OutOfRange' error.
output: A copy of the input before increment. If nothing else modifies the
  input, the values produced will all be distinct.
)doc");

}  // namespace tensorflow
