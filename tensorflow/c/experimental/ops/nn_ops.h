/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_C_EXPERIMENTAL_OPS_NN_OPS_H_
#define TENSORFLOW_C_EXPERIMENTAL_OPS_NN_OPS_H_

#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"

namespace tensorflow {
namespace ops {

Status SparseSoftmaxCrossEntropyWithLogits(
    AbstractContext* ctx, AbstractTensorHandle* const features,
    AbstractTensorHandle* const labels,
    absl::Span<AbstractTensorHandle*> outputs, const char* name);

Status ReluGrad(AbstractContext* ctx, AbstractTensorHandle* const gradients,
                AbstractTensorHandle* const features,
                absl::Span<AbstractTensorHandle*> backprops, const char* name);

Status Relu(AbstractContext* ctx, AbstractTensorHandle* const features,
            absl::Span<AbstractTensorHandle*> activations, const char* name);

Status BiasAdd(AbstractContext* ctx, AbstractTensorHandle* const value,
               AbstractTensorHandle* const bias,
               absl::Span<AbstractTensorHandle*> output, const char* name,
               const char* data_format = "NHWC");

Status BiasAddGrad(AbstractContext* ctx,
                   AbstractTensorHandle* const out_backprop,
                   absl::Span<AbstractTensorHandle*> output, const char* name,
                   const char* data_format = "NHWC");

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_OPS_NN_OPS_H_
