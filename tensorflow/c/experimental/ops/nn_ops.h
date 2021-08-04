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

#ifndef TENSORFLOW_C_EXPERIMENTAL_OPS_NN_OPS_H_
#define TENSORFLOW_C_EXPERIMENTAL_OPS_NN_OPS_H_

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"

namespace tensorflow {
namespace ops {

Status SparseSoftmaxCrossEntropyWithLogits(AbstractContext* ctx,
                                           AbstractTensorHandle* const features,
                                           AbstractTensorHandle* const labels,
                                           AbstractTensorHandle** loss,
                                           AbstractTensorHandle** backprop,
                                           const char* name = nullptr);

Status ReluGrad(AbstractContext* ctx, AbstractTensorHandle* const gradients,
                AbstractTensorHandle* const features,
                AbstractTensorHandle** backprops, const char* name = nullptr);

Status Relu(AbstractContext* ctx, AbstractTensorHandle* const features,
            AbstractTensorHandle** activations, const char* name = nullptr);

Status BiasAdd(AbstractContext* ctx, AbstractTensorHandle* const value,
               AbstractTensorHandle* const bias, AbstractTensorHandle** output,
               const char* data_format = "NHWC", const char* name = nullptr);

Status BiasAddGrad(AbstractContext* ctx,
                   AbstractTensorHandle* const out_backprop,
                   AbstractTensorHandle** output,
                   const char* data_format = "NHWC",
                   const char* name = nullptr);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_OPS_NN_OPS_H_
