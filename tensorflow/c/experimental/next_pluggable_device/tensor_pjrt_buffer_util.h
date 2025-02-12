/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_C_EXPERIMENTAL_NEXT_PLUGGABLE_DEVICE_TENSOR_PJRT_BUFFER_UTIL_H_
#define TENSORFLOW_C_EXPERIMENTAL_NEXT_PLUGGABLE_DEVICE_TENSOR_PJRT_BUFFER_UTIL_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

absl::StatusOr<PJRT_Buffer*> GetPjRtCBufferFromTensor(const Tensor* tensor);

absl::Status SetPjRtCBufferToTensor(PJRT_Buffer* c_buffer,
                                    xla::PjRtCApiClient* c_api_client,
                                    Tensor* tensor);

absl::StatusOr<xla::PjRtCApiClient*> GetPjRtCApiClient(
    const DeviceType& device_type);

absl::Status ResetPjRtClient(const DeviceType& device_type);

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_NEXT_PLUGGABLE_DEVICE_TENSOR_PJRT_BUFFER_UTIL_H_
