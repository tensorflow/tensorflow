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

#include "tensorflow/c/experimental/ops/nn_ops.h"

#include <cstring>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/tracing_utils.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/platform/status.h"

using tensorflow::tracing::MaybeSetOpName;

namespace tensorflow {
namespace ops {

// Op: SparseSoftmaxCrossEntropyWithLogits()
// Summary: Computes softmax cross entropy cost and gradients to backpropagate.
//
// Description:
//   Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
//   a matrix of label probabilities, but rather a single label per row
//   of features.  This label is considered to have probability 1.0 for the
//   given row.
//
//   Inputs are the logits, not probabilities.
absl::Status SparseSoftmaxCrossEntropyWithLogits(
    AbstractContext* ctx, AbstractTensorHandle* const features,
    AbstractTensorHandle* const labels, AbstractTensorHandle** loss,
    AbstractTensorHandle** backprop, const char* name,
    const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      op_ptr->Reset("SparseSoftmaxCrossEntropyWithLogits", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(features));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(labels));
  int num_retvals = 2;
  AbstractTensorHandle* temp_outputs[2];
  absl::Status status = op_ptr->Execute(temp_outputs, &num_retvals);
  *loss = temp_outputs[0];
  *backprop = temp_outputs[1];
  return status;
}

// Op: ReluGrad()
// Summary: Computes rectified linear gradients for a Relu operation.
//
// Description:
absl::Status ReluGrad(AbstractContext* ctx,
                      AbstractTensorHandle* const gradients,
                      AbstractTensorHandle* const features,
                      AbstractTensorHandle** backprops, const char* name,
                      const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("ReluGrad", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(gradients));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(features));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(backprops, 1), &num_retvals);
}

// Op: Relu()
// Summary: Computes rectified linear: `max(features, 0)`.
//
// Description:
//   See: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
//   Example usage:
//   >>> tf.nn.relu([-2., 0., 3.]).numpy()
//   array([0., 0., 3.], dtype=float32)
absl::Status Relu(AbstractContext* ctx, AbstractTensorHandle* const features,
                  AbstractTensorHandle** activations, const char* name,
                  const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Relu", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(features));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(activations, 1), &num_retvals);
}

// Op: BiasAdd()
// Summary: Adds `bias` to `value`.
//
// Description:
//   This is a special case of `tf.add` where `bias` is restricted to be 1-D.
//   Broadcasting is supported, so `value` may have any number of dimensions.
absl::Status BiasAdd(AbstractContext* ctx, AbstractTensorHandle* const value,
                     AbstractTensorHandle* const bias,
                     AbstractTensorHandle** output, const char* data_format,
                     const char* name, const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("BiasAdd", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(value));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(bias));
  TF_RETURN_IF_ERROR(
      op_ptr->SetAttrString("data_format", data_format, strlen(data_format)));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(output, 1), &num_retvals);
}

// Op: BiasAddGrad()
// Summary: The backward operation for "BiasAdd" on the "bias" tensor.
//
// Description:
//   It accumulates all the values from out_backprop into the feature dimension.
//   For NHWC data format, the feature dimension is the last. For NCHW data
//   format, the feature dimension is the third-to-last.
absl::Status BiasAddGrad(AbstractContext* ctx,
                         AbstractTensorHandle* const out_backprop,
                         AbstractTensorHandle** output, const char* data_format,
                         const char* name, const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("BiasAddGrad", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(out_backprop));
  TF_RETURN_IF_ERROR(
      op_ptr->SetAttrString("data_format", data_format, strlen(data_format)));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(output, 1), &num_retvals);
}

}  // namespace ops
}  // namespace tensorflow
