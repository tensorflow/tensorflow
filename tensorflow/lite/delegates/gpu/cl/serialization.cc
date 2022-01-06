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

flatbuffers::Offset<data::TensorDescWithId> Encode(
    const TensorDescriptor& desc, const ValueId& id,
    flatbuffers::FlatBufferBuilder* builder) {
  auto desc_fb = Encode(desc, builder);
  data::TensorDescWithIdBuilder desc_builder(*builder);
  desc_builder.add_desc(desc_fb);
  desc_builder.add_id(id);
  return desc_builder.Finish();
}

void Decode(const data::TensorDescWithId* fb_desc, TensorDescriptor* desc,
            ValueId* id) {
  Decode(fb_desc->desc(), desc);
  *id = fb_desc->id();
}

flatbuffers::Offset<data::GpuNode> Encode(
    const GpuNode& node, flatbuffers::FlatBufferBuilder* builder) {
  auto op_fb = Encode(*node.gpu_operation, builder);
  std::vector<int32_t> in_ids(node.inputs.size());
  for (int i = 0; i < in_ids.size(); ++i) {
    in_ids[i] = node.inputs[i];
  }
  std::vector<int32_t> out_ids(node.outputs.size());
  for (int i = 0; i < out_ids.size(); ++i) {
    out_ids[i] = node.outputs[i];
  }
  auto in_ids_fb = builder->CreateVector(in_ids);
  auto out_ids_fb = builder->CreateVector(out_ids);
  auto name_fb = builder->CreateString(node.name);
  data::GpuNodeBuilder node_builder(*builder);
  node_builder.add_gpu_op(op_fb);
  node_builder.add_input_ids(in_ids_fb);
  node_builder.add_output_ids(out_ids_fb);
  node_builder.add_name(name_fb);
  return node_builder.Finish();
}

absl::Status Decode(const data::GpuNode* fb_node, GpuNode* node) {
  GPUOperation op;
  RETURN_IF_ERROR(Decode(fb_node->gpu_op(), &op));
  node->gpu_operation = absl::make_unique<GPUOperation>(std::move(op));
  for (auto in_fb : *fb_node->input_ids()) {
    node->inputs.push_back(in_fb);
  }
  for (auto out_fb : *fb_node->output_ids()) {
    node->outputs.push_back(out_fb);
  }
  node->name = std::string(fb_node->name()->c_str(), fb_node->name()->size());

  return absl::OkStatus();
}

flatbuffers::Offset<data::GpuModel> Encode(
    const GpuModel& gpu_model, flatbuffers::FlatBufferBuilder* builder) {
  std::vector<int32_t> in_ids(gpu_model.input_ids_and_refs.size());
  std::vector<int64_t> in_refs(gpu_model.input_ids_and_refs.size());
  for (int i = 0; i < in_ids.size(); ++i) {
    in_ids[i] = gpu_model.input_ids_and_refs[i].first;
    in_refs[i] = gpu_model.input_ids_and_refs[i].second;
  }
  auto in_ids_fb = builder->CreateVector(in_ids);
  auto in_refs_fb = builder->CreateVector(in_refs);

  std::vector<int32_t> out_ids(gpu_model.output_ids_and_refs.size());
  std::vector<int64_t> out_refs(gpu_model.output_ids_and_refs.size());
  for (int i = 0; i < out_ids.size(); ++i) {
    out_ids[i] = gpu_model.output_ids_and_refs[i].first;
    out_refs[i] = gpu_model.output_ids_and_refs[i].second;
  }
  auto out_ids_fb = builder->CreateVector(out_ids);
  auto out_refs_fb = builder->CreateVector(out_refs);

  std::vector<flatbuffers::Offset<data::GpuNode>> nodes_fb;
  for (int i = 0; i < gpu_model.nodes.size(); ++i) {
    auto node_fb = Encode(gpu_model.nodes[i], builder);
    nodes_fb.push_back(node_fb);
  }
  auto nodes_fb_vec = builder->CreateVector(nodes_fb);

  std::vector<flatbuffers::Offset<data::TensorDescWithId>> tensors_fb;
  for (const auto& tensor : gpu_model.tensors) {
    auto tensor_fb = Encode(tensor.second, tensor.first, builder);
    tensors_fb.push_back(tensor_fb);
  }
  auto tensors_fb_vec = builder->CreateVector(tensors_fb);

  std::vector<flatbuffers::Offset<data::TensorDescWithId>> const_tensors_fb;
  for (const auto& tensor : gpu_model.const_tensors) {
    auto tensor_fb = Encode(tensor.second, tensor.first, builder);
    const_tensors_fb.push_back(tensor_fb);
  }
  auto const_tensors_fb_vec = builder->CreateVector(const_tensors_fb);

  std::vector<flatbuffers::Offset<data::PairOfValueIds>>
      variable_ids_and_refs_fb;
  for (auto& pair : gpu_model.variable_ids_and_refs) {
    data::PairOfValueIdsBuilder pair_builder(*builder);
    pair_builder.add_first(pair.first);
    pair_builder.add_second(pair.second);
    variable_ids_and_refs_fb.push_back(pair_builder.Finish());
  }
  auto variable_ids_and_refs_fb_vec =
      builder->CreateVector(variable_ids_and_refs_fb);

  data::GpuModelBuilder gpu_model_builder(*builder);
  gpu_model_builder.add_nodes(nodes_fb_vec);
  gpu_model_builder.add_tensors(tensors_fb_vec);
  gpu_model_builder.add_const_tensors(const_tensors_fb_vec);
  gpu_model_builder.add_input_ids(in_ids_fb);
  gpu_model_builder.add_output_ids(out_ids_fb);
  gpu_model_builder.add_variable_ids_and_refs(variable_ids_and_refs_fb_vec);
  gpu_model_builder.add_input_refs(in_refs_fb);
  gpu_model_builder.add_output_refs(out_refs_fb);
  return gpu_model_builder.Finish();
}

absl::Status Decode(const data::GpuModel* fb_gpu_model, GpuModel* gpu_model) {
  gpu_model->nodes.resize(fb_gpu_model->nodes()->size());
  int counter = 0;
  for (auto node_fb : *fb_gpu_model->nodes()) {
    RETURN_IF_ERROR(Decode(node_fb, &gpu_model->nodes[counter]));
    counter++;
  }

  for (const auto& tensor_fb : *fb_gpu_model->tensors()) {
    TensorDescriptor desc;
    Decode(tensor_fb->desc(), &desc);
    gpu_model->tensors[tensor_fb->id()] = std::move(desc);
  }
  for (const auto& tensor_fb : *fb_gpu_model->const_tensors()) {
    TensorDescriptor desc;
    Decode(tensor_fb->desc(), &desc);
    gpu_model->const_tensors[tensor_fb->id()] = std::move(desc);
  }
  for (int i = 0; i < fb_gpu_model->input_ids()->size(); ++i) {
    gpu_model->input_ids_and_refs.push_back(
        {(*fb_gpu_model->input_ids())[i], (*fb_gpu_model->input_refs())[i]});
  }
  for (int i = 0; i < fb_gpu_model->output_ids()->size(); ++i) {
    gpu_model->output_ids_and_refs.push_back(
        {(*fb_gpu_model->output_ids())[i], (*fb_gpu_model->output_refs())[i]});
  }

  for (auto variable_id : *fb_gpu_model->variable_ids_and_refs()) {
    gpu_model->variable_ids_and_refs.push_back(
        {variable_id->first(), variable_id->second()});
  }
  return absl::OkStatus();
}

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
