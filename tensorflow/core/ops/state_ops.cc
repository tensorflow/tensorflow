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

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("VariableV2")
    .Output("ref: Ref(dtype)")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      TensorShapeProto shape_proto;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape_proto));
      ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(
          c->MakeShapeFromShapeProto(shape_proto, &output_shape));
      c->set_output(0, output_shape);
      return Status::OK();
    })
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

REGISTER_OP("Variable")
    .Output("ref: Ref(dtype)")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      PartialTensorShape shape;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape));

      // Variable has legacy behavior where we cannot tell the difference
      // between a scalar shape attribute and 'unknown shape'.  So if the shape
      // is a scalar, we return an unknown shape.
      if (shape.dims() <= 0) {
        return shape_inference::UnknownShape(c);
      }

      TensorShapeProto shape_proto;
      shape.AsProto(&shape_proto);
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeProto(shape_proto, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Holds state in the form of a tensor that persists across steps.

Outputs a ref to the tensor state so it may be read or modified.
TODO(zhifengc/mrry): Adds a pointer to a more detail document
about sharing states in tensorflow.

ref: A reference to the variable tensor.
shape: The shape of the variable tensor, where scalar shapes are
  treated as undefined.
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
duplicate entries in `indices`, the order at which the updates happen
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

namespace {

Status ScatterNdUpdateShape(InferenceContext* c) {
  ShapeHandle ref_shape = c->input(0);
  ShapeHandle indices_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &indices_shape));
  ShapeHandle updates_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 1, &updates_shape));

  if (c->RankKnown(indices_shape) && c->RankKnown(updates_shape)) {
    const int64 outer_dims = c->Rank(indices_shape) - 1;
    const DimensionHandle ixdim = c->Dim(indices_shape, -1);

    // We can only do more validation if the last dimension of indices
    // is a known value.
    if (c->ValueKnown(ixdim)) {
      int64 ix = c->Value(ixdim);
      ShapeHandle unused;
      ShapeHandle prefix_indices;
      TF_RETURN_IF_ERROR(
          c->Subshape(indices_shape, 0, outer_dims, &prefix_indices));
      ShapeHandle prefix_updates;
      TF_RETURN_IF_ERROR(
          c->Subshape(updates_shape, 0, outer_dims, &prefix_updates));

      Status s = c->Merge(prefix_indices, prefix_updates, &unused);
      if (!s.ok()) {
        return errors::InvalidArgument(
            "The outer ", outer_dims, " dimensions of indices.shape=",
            c->DebugString(indices_shape), "must match the outer ", outer_dims,
            " dimensions of updates.shape=", c->DebugString(updates_shape),
            ": ", s.error_message());
      }

      ShapeHandle suffix_ref;
      TF_RETURN_IF_ERROR(c->Subshape(ref_shape, ix, &suffix_ref));
      ShapeHandle suffix_updates;
      TF_RETURN_IF_ERROR(
          c->Subshape(updates_shape, outer_dims, &suffix_updates));
      s = c->Merge(suffix_ref, suffix_updates, &unused);
      if (!s.ok()) {
        return errors::InvalidArgument(
            "The inner ", c->Rank(ref_shape) - ix, " dimensions of ref.shape=",
            c->DebugString(ref_shape), "must match the inner ",
            c->Rank(updates_shape) - outer_dims,
            " dimensions of updates.shape=", c->DebugString(updates_shape),
            ": ", s.error_message());
      }
    }
  }

  c->set_output(0, ref_shape);
  return Status::OK();
}

}  // namespace

REGISTER_OP("ScatterNdUpdate")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = true")
    .SetShapeFn(ScatterNdUpdateShape)
    .Doc(R"doc(
Applies sparse `updates` to individual values or slices within a given
variable according to `indices`.

`ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

`indices` must be integer tensor, containing indices into `ref`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `ref`.

`updates` is `Tensor` of rank `Q-1+P-K` with shape:

```
[d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
```

For example, say we want to update 4 scattered elements to a rank-1 tensor to
8 elements. In Python, that update would look like this:

    ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
    indices = tf.constant([[4], [3], [1] ,[7]])
    updates = tf.constant([9, 10, 11, 12])
    update = tf.scatter_nd_update(ref, indices, updates)
    with tf.Session() as sess:
      print sess.run(update)

The resulting update to ref would look like this:

    [1, 11, 3, 10, 9, 6, 7, 12]

See [tf.scatter_nd](#scatter_nd) for more details about how to make updates to
slices.

ref: A mutable Tensor. Should be from a Variable node.
indices: A Tensor. Must be one of the following types: int32, int64.
  A tensor of indices into ref.
updates: A Tensor. Must have the same type as ref. A tensor of updated
  values to add to ref.
use_locking: An optional bool. Defaults to True. If True, the assignment will
  be protected by a lock; otherwise the behavior is undefined,
  but may exhibit less contention.
output_ref: Same as ref. Returned as a convenience for operations that want to
  use the updated values after the update is done.
)doc");

REGISTER_OP("ScatterNdAdd")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterNdUpdateShape)
    .Doc(R"doc(
Applies sparse addition between `updates` and individual values or slices
within a given variable according to `indices`.

`ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

`indices` must be integer tensor, containing indices into `ref`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `ref`.

`updates` is `Tensor` of rank `Q-1+P-K` with shape:

```
[d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
```

For example, say we want to add 4 scattered elements to a rank-1 tensor to 8
elements. In Python, that addition would look like this:

    ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    add = tf.scatter_nd_add(ref, indices, updates)
    with tf.Session() as sess:
      print sess.run(add)

The resulting update to ref would look like this:

    [1, 13, 3, 14, 14, 6, 7, 20]

See [tf.scatter_nd](#scatter_nd) for more details about how to make updates to
slices.

ref: A mutable Tensor. Should be from a Variable node.
indices: A Tensor. Must be one of the following types: int32, int64.
  A tensor of indices into ref.
updates: A Tensor. Must have the same type as ref. A tensor of updated values
  to add to ref.
use_locking: An optional bool. Defaults to True. If True, the assignment will
  be protected by a lock; otherwise the behavior is undefined,
  but may exhibit less contention.
output_ref: Same as ref. Returned as a convenience for operations that want
  to use the updated values after the update is done.
)doc");

REGISTER_OP("ScatterNdSub")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterNdUpdateShape)
    .Doc(R"doc(
Applies sparse subtraction between `updates` and individual values or slices
within a given variable according to `indices`.

`ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

`indices` must be integer tensor, containing indices into `ref`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `ref`.

`updates` is `Tensor` of rank `Q-1+P-K` with shape:

```
[d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
```

For example, say we want to subtract 4 scattered elements from a rank-1 tensor
with 8 elements. In Python, that subtraction would look like this:

    ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    sub = tf.scatter_nd_sub(ref, indices, updates)
    with tf.Session() as sess:
      print sess.run(sub)

The resulting update to ref would look like this:

    [1, -9, 3, -6, -4, 6, 7, -4]

See [tf.scatter_nd](#scatter_nd) for more details about how to make updates to
slices.

ref: A mutable Tensor. Should be from a Variable node.
indices: A Tensor. Must be one of the following types: int32, int64.
  A tensor of indices into ref.
updates: A Tensor. Must have the same type as ref. A tensor of updated values
  to subtract from ref.
use_locking: An optional bool. Defaults to True. If True, the assignment will
  be protected by a lock; otherwise the behavior is undefined,
  but may exhibit less contention.
output_ref: Same as ref. Returned as a convenience for operations that want
  to use the updated values after the update is done.
)doc");

// TODO(simister): Re-enable once these additional ops do not dramatically
// increase binary size.

// REGISTER_OP("ScatterNdMul")
//     .Input("ref: Ref(T)")
//     .Input("indices: Tindices")
//     .Input("updates: T")
//     .Output("output_ref: Ref(T)")
//     .Attr("T: numbertype")
//     .Attr("Tindices: {int32, int64}")
//     .Attr("use_locking: bool = false")
//     .SetShapeFn(ScatterNdUpdateShape)
//     .Doc(
//         R"doc(Applies sparse subtraction between `updates` and individual
//         values or slices within a given variable according to `indices`.

// `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

// `indices` must be integer tensor, containing indices into `ref`.
// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

// The innermost dimension of `indices` (with length `K`) corresponds to
// indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
// dimension of `ref`.

// `updates` is `Tensor` of rank `Q-1+P-K` with shape:

// ```
// [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
// ```

// For example, say we want to multiply 4 scattered elements with a rank-1
// tensor with 8 elements. In Python, that multiplication would look like this:

//     ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
//     indices = tf.constant([[4], [3], [1], [7]])
//     updates = tf.constant([9, 10, 11, 12])
//     sub = tf.scatter_nd_mul(ref, indices, updates)
//     with tf.Session() as sess:
//       print sess.run(sub)

// The resulting update to ref would look like this:

//     [1, 22, 3, 40, 45, 6, 7, 96]

// See [tf.scatter_nd](#scatter_nd) for more details about how to make updates
// to slices.

// ref: A mutable Tensor. Should be from a Variable node.
// indices: A Tensor. Must be one of the following types: int32, int64. A tensor
// of indices into ref.
// updates: A Tensor. Must have the same type as ref. A tensor of updated values
// to subtract from ref.
// use_locking: An optional bool. Defaults to True. If True, the assignment will
// be protected by a lock; otherwise the behavior is undefined, but may exhibit
// less contention.
// output_ref: Same as ref. Returned as a convenience for operations that want
// to use the updated values after the update is done.)doc");

// REGISTER_OP("ScatterNdDiv")
//     .Input("ref: Ref(T)")
//     .Input("indices: Tindices")
//     .Input("updates: T")
//     .Output("output_ref: Ref(T)")
//     .Attr("T: numbertype")
//     .Attr("Tindices: {int32, int64}")
//     .Attr("use_locking: bool = false")
//     .SetShapeFn(ScatterNdUpdateShape)
//     .Doc(
//         R"doc(Applies sparse subtraction between `updates` and individual
//         values or slices within a given variable according to `indices`.

// `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

// `indices` must be integer tensor, containing indices into `ref`.
// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

// The innermost dimension of `indices` (with length `K`) corresponds to
// indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
// dimension of `ref`.

// `updates` is `Tensor` of rank `Q-1+P-K` with shape:

// ```
// [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
// ```

// For example, say we want to divide a rank-1 tensor with 8 elements by 4
// scattered elements. In Python, that division would look like this:

//     ref = tf.Variable([10, 20, 30, 40, 50, 60, 70, 80])
//     indices = tf.constant([[4], [3], [1], [7]])
//     updates = tf.constant([2, 3, 4, 5])
//     sub = tf.scatter_nd_div(ref, indices, updates)
//     with tf.Session() as sess:
//       print sess.run(sub)

// The resulting update to ref would look like this:

//     [10, 5, 30, 13, 25, 60, 70, 16]

// See [tf.scatter_nd](#scatter_nd) for more details about how to make updates
// to slices.

// ref: A mutable Tensor. Should be from a Variable node.
// indices: A Tensor. Must be one of the following types: int32, int64. A tensor
// of indices into ref.
// updates: A Tensor. Must have the same type as ref. A tensor of updated values
// to subtract from ref.
// use_locking: An optional bool. Defaults to True. If True, the assignment will
// be protected by a lock; otherwise the behavior is undefined, but may exhibit
// less contention.
// output_ref: Same as ref. Returned as a convenience for operations that want
// to use the updated values after the update is done.)doc");

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

ref: Should be from a scalar `Variable` node.
limit: If incrementing ref would bring it above limit, instead generates an
  'OutOfRange' error.
output: A copy of the input before increment. If nothing else modifies the
  input, the values produced will all be distinct.
)doc");

}  // namespace tensorflow
