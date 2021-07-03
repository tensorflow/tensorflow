/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/c/experimental/ops/array_ops.h"

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/tracing_utils.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"

using tensorflow::tracing::MaybeSetOpName;

namespace tensorflow {
namespace ops {

// Op: Identity()
// Summary: Return a tensor with the same shape and contents as the input tensor
// or value.
//
// Description:
Status Identity(AbstractContext* ctx, AbstractTensorHandle* const input,
                AbstractTensorHandle** output, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Identity", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(input));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(output, 1), &num_retvals);
}

// Op: IdentityN()
// Summary: Returns a list of tensors with the same shapes and contents as the
// input
//
// Description:
//   tensors.
//
//   This op can be used to override the gradient for complicated functions. For
//   example, suppose y = f(x) and we wish to apply a custom function g for
//   backprop such that dx = g(dy). In Python,
//
//   ```python
//   with tf.get_default_graph().gradient_override_map(
//       {'IdentityN': 'OverrideGradientWithG'}):
//     y, _ = identity_n([f(x), x])
//
//   @tf.RegisterGradient('OverrideGradientWithG')
//   def ApplyG(op, dy, _):
//     return [None, g(dy)]  # Do not backprop to f(x).
//   ```
Status IdentityN(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> input,
                 absl::Span<AbstractTensorHandle*> output, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("IdentityN", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInputList(input));
  int num_retvals = output.size();
  return op_ptr->Execute(output, &num_retvals);
}

// Op: ZerosLike()
// Summary: Returns a tensor of zeros with the same shape and type as x.
//
// Description:
Status ZerosLike(AbstractContext* ctx, AbstractTensorHandle* const x,
                 AbstractTensorHandle** y, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("ZerosLike", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(y, 1), &num_retvals);
}

// Op: Shape()
// Summary: Returns the shape of a tensor.
//
// Description:
//   This operation returns a 1-D integer tensor representing the shape of
//   `input`.
//
//   For example:
//
//   ```
//   # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
//   shape(t) ==> [2, 2, 3]
//   ```
Status Shape(AbstractContext* ctx, AbstractTensorHandle* const input,
             AbstractTensorHandle** output, DataType out_type,
             const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Shape", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(input));
  TF_RETURN_IF_ERROR(op_ptr->SetAttrType("out_type", out_type));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(output, 1), &num_retvals);
}

// Op: ExpandDims()
// Summary: Inserts a dimension of 1 into a tensor's shape.
//
// Description:
//   Given a tensor `input`, this operation inserts a dimension of 1 at the
//   dimension index `axis` of `input`'s shape. The dimension index `axis`
//   starts at zero; if you specify a negative number for `axis` it is counted
//   backward from the end.
//
//   This operation is useful if you want to add a batch dimension to a single
//   element. For example, if you have a single image of shape `[height, width,
//   channels]`, you can make it a batch of 1 image with `expand_dims(image,
//   0)`, which will make the shape `[1, height, width, channels]`.
//
//   Other examples:
//
//   ```
//   # 't' is a tensor of shape [2]
//   shape(expand_dims(t, 0)) ==> [1, 2]
//   shape(expand_dims(t, 1)) ==> [2, 1]
//   shape(expand_dims(t, -1)) ==> [2, 1]
//
//   # 't2' is a tensor of shape [2, 3, 5]
//   shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
//   shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
//   shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
//   ```
//
//   This operation requires that:
//
//   `-1-input.dims() <= dim <= input.dims()`
//
//   This operation is related to `squeeze()`, which removes dimensions of
//   size 1.
Status ExpandDims(AbstractContext* ctx, AbstractTensorHandle* const input,
                  AbstractTensorHandle* const dim,
                  AbstractTensorHandle** output, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("ExpandDims", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(input));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(dim));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(output, 1), &num_retvals);
}

// Op: OnesLike()
// Summary: Returns a tensor of ones with the same shape and type as x.
//
// Description:
Status OnesLike(AbstractContext* ctx, AbstractTensorHandle* const x,
                AbstractTensorHandle** y, const char* name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("OnesLike", /*raw_device_name=*/nullptr));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(y, 1), &num_retvals);
}

}  // namespace ops
}  // namespace tensorflow
