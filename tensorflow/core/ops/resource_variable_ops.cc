// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

namespace tensorflow {

REGISTER_OP("VarHandleOp")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Output("resource: resource")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      DataType t;
      c->GetAttr("dtype", &t);
      c->set_output_handle_dtype(0, t);
      TensorShapeProto p;
      c->GetAttr("shape", &p);
      shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeProto(p, &s));
      c->set_output_handle_shape(0, s);
      return Status::OK();
    })
    .Doc(R"(
Creates a handle to a Variable resource.

container: the container this variable is placed in.
shared_name: the name by which this variable is referred to.
dtype: the type of this variable. Must agree with the dtypes
  of all ops using this variable.
shape: The (possibly partially specified) shape of this variable.
)");

REGISTER_OP("ReadVariableOp")
    .Input("resource: resource")
    .Output("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn([](InferenceContext* c) {
      DataType handle_dtype = c->input_handle_dtype(0);
      DataType value_dtype;
      c->GetAttr("dtype", &value_dtype);
      if (handle_dtype != value_dtype) {
        return errors::InvalidArgument(
            "Trying to read variable with wrong dtype. "
            "Expected ",
            handle_dtype, " got ", value_dtype);
      }
      c->set_output(0, c->input_handle_shape(0));
      return Status::OK();
    })
    .Doc(R"(
Reads the value of a variable.

The tensor returned by this operation is immutable.

The value returned by this operation is guaranteed to be influenced by all the
writes on which this operation depends directly or indirectly, and to not be
influenced by any of the writes which depend directly or indirectly on this
operation.

resource: handle to the resource in which to store the variable.
dtype: the dtype of the value.
)");

REGISTER_OP("DestroyResourceOp")
    .Input("resource: resource")
    .Attr("ignore_lookup_error: bool = true")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"(
Deletes the resource specified by the handle.

All subsequent operations using the resource will result in a NotFound
error status.

resource: handle to the resource to delete.
ignore_lookup_error: whether to ignore the error when the resource
  doesn't exist.
)");

Status CreateAssignShapeFn(InferenceContext* c) {
  DataType handle_dtype = c->input_handle_dtype(0);
  DataType value_dtype;
  c->GetAttr("dtype", &value_dtype);
  if (handle_dtype != value_dtype) {
    return errors::InvalidArgument(
        "Trying to initialize handle for variable with wrong dtype. "
        "Expected ",
        handle_dtype, " got ", value_dtype);
  }
  ShapeHandle s = c->input_handle_shape(0);
  ShapeHandle value_shape = c->input(1);
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(s, value_shape, &unused));
  return Status::OK();
}

REGISTER_OP("AssignVariableOp")
    .Input("resource: resource")
    .Input("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn(CreateAssignShapeFn)
    .Doc(R"(
Assigns a new value to a variable.

Any ReadVariableOp with a control dependency on this op is guaranteed to return
this value or a subsequent newer value of the variable.

resource: handle to the resource in which to store the variable.
value: the value to set the new tensor to use.
dtype: the dtype of the value.
)");

REGISTER_OP("AssignAddVariableOp")
    .Input("resource: resource")
    .Input("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn(CreateAssignShapeFn)
    .Doc(R"(
Adds a value to the current value of a variable.

Any ReadVariableOp which depends directly or indirectly on this assign is
guaranteed to see the incremented value or a subsequent newer one.

Outputs the incremented value, which can be used to totally order the
increments to this variable.

resource: handle to the resource in which to store the variable.
value: the value by which the variable will be incremented.
dtype: the dtype of the value.
)");

REGISTER_OP("AssignSubVariableOp")
    .Input("resource: resource")
    .Input("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn(CreateAssignShapeFn)
    .Doc(R"(
Subtracts a value from the current value of a variable.

Any ReadVariableOp which depends directly or indirectly on this assign is
guaranteed to see the incremented value or a subsequent newer one.

Outputs the incremented value, which can be used to totally order the
increments to this variable.

resource: handle to the resource in which to store the variable.
value: the value by which the variable will be incremented.
dtype: the dtype of the value.
)");

REGISTER_OP("VarIsInitializedOp")
    .Input("resource: resource")
    .Output("is_initialized: bool")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Checks whether a resource handle-based variable has been initialized.

resource: the input resource handle.
is_initialized: a scalar boolean which is true if the variable has been
initialized.
)doc");

REGISTER_OP("ResourceGather")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Attr("validate_indices: bool = true")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn([](InferenceContext* c) {
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("dtype", &dtype));
      if (c->input_handle_dtype(0) != dtype) {
        return errors::InvalidArgument(
            "Trying to gather from a variable with the wrong dtype.");
      }
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(
          c->WithRankAtLeast(c->input_handle_shape(0), 1, &unused));
      ShapeHandle params_subshape;
      TF_RETURN_IF_ERROR(
          c->Subshape(c->input_handle_shape(0), 1, &params_subshape));
      ShapeHandle indices_shape = c->input(1);
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, params_subshape, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Gather slices from the variable pointed to by `resource` according to `indices`.

`indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
Produces an output tensor with shape `indices.shape + params.shape[1:]` where:

```python
    # Scalar indices
    output[:, ..., :] = params[indices, :, ... :]

    # Vector indices
    output[i, :, ..., :] = params[indices[i], :, ... :]

    # Higher rank indices
    output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
```

)doc");

REGISTER_OP("ResourceScatterAdd")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle var_shape = c->input_handle_shape(0);
      ShapeHandle indices_shape = c->input(1);

      ShapeHandle unused_updates_shape;
      ShapeHandle concat;
      ShapeHandle var_subshape;
      TF_RETURN_IF_ERROR(c->Subshape(var_shape, 1, &var_subshape));
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, var_subshape, &concat));
      TF_RETURN_IF_ERROR(c->Merge(c->input(2), concat, &unused_updates_shape));
      return Status::OK();
    })
    .Doc(R"doc(
Adds sparse updates to the variable referenced by `resource`.

This operation computes

    # Scalar indices
    ref[indices, ...] += updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] += updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions add.

Requires `updates.shape = indices.shape + ref.shape[1:]`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/ScatterAdd.png" alt>
</div>

resource: Should be from a `Variable` node.
indices: A tensor of indices into the first dimension of `ref`.
updates: A tensor of updated values to add to `ref`.
)doc");

}  // namespace tensorflow
