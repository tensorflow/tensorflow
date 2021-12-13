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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_SERIALIZATION_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_SERIALIZATION_H_

#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"
#include "tensorflow/lite/delegates/gpu/cl/program_cache.h"
#include "tensorflow/lite/delegates/gpu/cl/serialization_generated.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

class InferenceContext;

flatbuffers::Offset<data::GpuModel> Encode(
    const GpuModel& gpu_model, flatbuffers::FlatBufferBuilder* builder);

absl::Status Decode(const data::GpuModel* fb_gpu_model, GpuModel* gpu_model);

flatbuffers::Offset<data::InferenceContext> Encode(
    const CLDevice& device, const InferenceContext& inference,
    const ProgramCache& program_cache,
    flatbuffers::Offset<data::GpuModel> gpu_model_fb,
    flatbuffers::FlatBufferBuilder* builder);

absl::Status Decode(const CLContext& context, const CLDevice& device,
                    ProgramCache* program_cache,
                    const data::InferenceContext* fb_inference,
                    InferenceContext* inference);

absl::Status GetInOutRefs(const absl::Span<const uint8_t> serialized_model,
                          std::vector<int64_t>* in_refs,
                          std::vector<int64_t>* out_refs);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_SERIALIZATION_H_
