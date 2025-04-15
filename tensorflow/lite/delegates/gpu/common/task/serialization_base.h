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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_SERIALIZATION_BASE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_SERIALIZATION_BASE_H_

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tflite_serialization_base_generated.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

flatbuffers::Offset<data::Int2> Encode(const int2& v,
                                       flatbuffers::FlatBufferBuilder* builder);

flatbuffers::Offset<data::Int3> Encode(const int3& v,
                                       flatbuffers::FlatBufferBuilder* builder);

flatbuffers::Offset<data::TensorDescriptor> Encode(
    const TensorDescriptor& desc, flatbuffers::FlatBufferBuilder* builder);
void Decode(const data::TensorDescriptor* fb_desc, TensorDescriptor* desc);

flatbuffers::Offset<data::GPUOperation> Encode(
    const GPUOperation& op, flatbuffers::FlatBufferBuilder* builder);
absl::Status Decode(const data::GPUOperation* fb_op, GPUOperation* op);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_SERIALIZATION_BASE_H_
