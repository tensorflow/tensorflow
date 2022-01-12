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

#include "tensorflow/lite/delegates/gpu/cl/serialization.h"

#include <cstdint>
#include <set>
#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/cl/gpu_object.h"
#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"
#include "tensorflow/lite/delegates/gpu/cl/serialization_generated.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/task/arguments.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/serialization_base.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_linear_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"

namespace tflite {
namespace gpu {
namespace cl {
flatbuffers::Offset<data::InferenceContext> Encode(
    const CLDevice& device, const InferenceContext& inference,
    const ProgramCache& program_cache,
    flatbuffers::Offset<tflite::gpu::data::GpuModel> gpu_model_fb,
    flatbuffers::FlatBufferBuilder* builder) {
  std::vector<flatbuffers::Offset<tflite::gpu::data::Int3>> work_groups_fb;
  for (int i = 0; i < inference.nodes_.size(); ++i) {
    auto work_group_fb =
        Encode(inference.nodes_[i].cl_operation.GetWorkGroupSize(), builder);
    work_groups_fb.push_back(work_group_fb);
  }
  auto work_groups_fb_vec = builder->CreateVector(work_groups_fb);
  std::vector<uint64_t> node_fingerprints(inference.nodes_.size());
  for (int i = 0; i < inference.nodes_.size(); ++i) {
    node_fingerprints[i] =
        inference.nodes_[i].cl_operation.GetKernelFingerprint();
  }
  auto node_fingerprints_fb = builder->CreateVector(node_fingerprints);

  std::set<uint64_t> fingerprints;
  for (const auto& node : inference.nodes_) {
    fingerprints.insert(node.cl_operation.GetKernelFingerprint());
  }
  std::vector<flatbuffers::Offset<data::BinaryProgram>> binary_programs_fb;
  for (auto fingerprint : fingerprints) {
    std::vector<uint8_t> program_binary;
    program_cache.GetProgramBinary(fingerprint, &program_binary).IgnoreError();
    auto binary_fb = builder->CreateVector(program_binary);
    data::BinaryProgramBuilder program_builder(*builder);
    program_builder.add_fingerprint(fingerprint);
    program_builder.add_binary(binary_fb);
    binary_programs_fb.push_back(program_builder.Finish());
  }
  auto binary_programs_fb_vec = builder->CreateVector(binary_programs_fb);
  auto driver_version = builder->CreateString(device.GetPlatformVersion());

  data::InferenceContextBuilder inf_builder(*builder);
  inf_builder.add_gpu_model(gpu_model_fb);
  inf_builder.add_driver_version(driver_version);
  inf_builder.add_binary_programs(binary_programs_fb_vec);
  inf_builder.add_tuned_work_group_sizes_per_node(work_groups_fb_vec);
  inf_builder.add_fingerprints_per_node(node_fingerprints_fb);
  return inf_builder.Finish();
}

absl::Status Decode(const CLContext& context, const CLDevice& device,
                    ProgramCache* program_cache,
                    const data::InferenceContext* fb_inference,
                    InferenceContext* inference) {
  std::string platform_version(fb_inference->driver_version()->c_str(),
                               fb_inference->driver_version()->size());
  if (device.GetPlatformVersion() != platform_version) {
    return absl::InvalidArgumentError(
        "OpenCL driver changed, model respresentation invalid, must be "
        "regenerated.");
  }

  GpuModel gpu_model;
  RETURN_IF_ERROR(Decode(fb_inference->gpu_model(), &gpu_model));
  inference->CopyFromGpuModel(&gpu_model);

  for (auto binary_program_fb : *fb_inference->binary_programs()) {
    RETURN_IF_ERROR(program_cache->AddProgramBinary(
        context, device, binary_program_fb->fingerprint(),
        absl::MakeSpan(binary_program_fb->binary()->data(),
                       binary_program_fb->binary()->size())));
  }

  for (int i = 0; i < inference->nodes_.size(); ++i) {
    uint64_t fingerprint = (*fb_inference->fingerprints_per_node())[i];
    RETURN_IF_ERROR(inference->nodes_[i].cl_operation.InitFromCache(
        fingerprint, *program_cache));

    int3 wg_size;
    wg_size.x = (*fb_inference->tuned_work_group_sizes_per_node())[i]->x();
    wg_size.y = (*fb_inference->tuned_work_group_sizes_per_node())[i]->y();
    wg_size.z = (*fb_inference->tuned_work_group_sizes_per_node())[i]->z();
    inference->nodes_[i].cl_operation.SetWorkGroupSize(wg_size);
  }
  return absl::OkStatus();
}

absl::Status GetInOutRefs(const absl::Span<const uint8_t> serialized_model,
                          std::vector<int64_t>* in_refs,
                          std::vector<int64_t>* out_refs) {
  flatbuffers::Verifier verifier(serialized_model.data(),
                                 serialized_model.size());
  if (!data::VerifyInferenceContextBuffer(verifier)) {
    return absl::DataLossError("Deserialization failed.");
  }
  auto fb_inference = data::GetInferenceContext(serialized_model.data());
  if (in_refs) {
    in_refs->clear();
    for (auto in_fb : *fb_inference->gpu_model()->input_refs()) {
      in_refs->push_back(in_fb);
    }
  }
  if (out_refs) {
    out_refs->clear();
    for (auto out_fb : *fb_inference->gpu_model()->output_refs()) {
      out_refs->push_back(out_fb);
    }
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
