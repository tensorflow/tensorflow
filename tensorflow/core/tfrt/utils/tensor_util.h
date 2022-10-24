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
#ifndef TENSORFLOW_CORE_TFRT_UTILS_TENSOR_UTIL_H_
#define TENSORFLOW_CORE_TFRT_UTILS_TENSOR_UTIL_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/statusor.h"
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace tfrt {

// Converts a tfrt::Tensor to tensorflow::Tensor.
llvm::Expected<tensorflow::Tensor> TFRTTensorToTFTensor(const Tensor& tensor,
                                                        HostContext* host);

// Converts a tensorflow::Tensor to tfrt::TensorHandle.
AsyncValueRef<TensorHandle> TFTensorToTFRTTensorHandle(
    const tensorflow::Tensor& tf_tensor, HostContext* host_ctx);

// Creates a TFRT TensorHandle using the shape and data in a tensorflow tensor.
tensorflow::StatusOr<TensorHandle> CreateTensorHandleFromTFTensor(
    const tensorflow::Tensor& tensor, HostContext* host);

// Creates a tensorflow tensor using the shape and data in a TFRT tensorhandle.
tensorflow::StatusOr<tensorflow::Tensor> CreateTFTensorFromTensorHandle(
    const TensorHandle& tensor_handle);

}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_UTILS_TENSOR_UTIL_H_
