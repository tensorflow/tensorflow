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

#include "tensorflow/lite/delegates/gpu/common/gpu_model.h"

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/serialization_base.h"
#include "tensorflow/lite/delegates/gpu/common/task/storage_type_util.h"

namespace tflite {
namespace gpu {

namespace {
bool IsReady(const absl::flat_hash_set<ValueId>& ready_tensors,
             const GpuNode& node) {
  for (const ValueId in_id : node.inputs) {
    if (ready_tensors.find(in_id) == ready_tensors.end()) {
      return false;
    }
  }
  return true;
}

absl::Status MergeGpuNodes(GpuNode* src, GpuNode* dst) {
  for (int j = 1; j < src->inputs.size(); ++j) {
    dst->inputs.push_back(src->inputs[j]);
  }
  dst->outputs[0] = src->outputs[0];
  dst->name += " linked : " + src->name;
  return dst->gpu_operation->AddOperation(src->gpu_operation.get());
}

flatbuffers::Offset<data::TensorDescWithId> Encode(
    const TensorDescriptor& desc, const ValueId& id,
    flatbuffers::FlatBufferBuilder* builder) {
  auto desc_fb = Encode(desc, builder);
  data::TensorDescWithIdBuilder desc_builder(*builder);
  desc_builder.add_desc(desc_fb);
  desc_builder.add_id(id);
  return desc_builder.Finish();
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

}  // namespace

bool IsAssociativeLinkableOp(const Node& node,
                             const std::vector<Value*>& inputs,
                             const std::vector<Value*>& outputs) {
  if (inputs.size() == 1) {
    return false;
  }
  const OperationType op_type = OperationTypeFromString(node.operation.type);
  if (op_type != OperationType::ADD && op_type != OperationType::MUL) {
    return false;
  }

  const auto dst_shape = outputs[0]->tensor.shape;
  for (int i = 0; i < inputs.size(); ++i) {
    const auto src_shape = inputs[i]->tensor.shape;
    if (dst_shape.b != src_shape.b && src_shape.b == 1) {
      return false;
    }
    if (dst_shape.h != src_shape.h && src_shape.h == 1) {
      return false;
    }
    if (dst_shape.w != src_shape.w && src_shape.w == 1) {
      return false;
    }
    if (dst_shape.c != src_shape.c && src_shape.c == 1) {
      return false;
    }
  }
  return true;
}

absl::Status CheckExternalTensorDescription(const GpuInfo& gpu_info,
                                            const TensorDescriptor& tensor_desc,
                                            const BHWC& shape,
                                            DataType data_type) {
  if (tensor_desc.data_type != data_type) {
    return absl::InvalidArgumentError(
        "Global precision and precision of predefined/external tensors must be "
        "synchronized.");
  }
  const bool tensor_supported_layout = tensor_desc.layout == Layout::HWDC ||
                                       tensor_desc.layout == Layout::BHWDC ||
                                       tensor_desc.layout == Layout::HWC ||
                                       tensor_desc.layout == Layout::BHWC;
  if (!tensor_supported_layout) {
    return absl::InvalidArgumentError(
        "Currently no support of this layouts for spatial tensors.");
  }
  const bool has_depth =
      tensor_desc.layout == Layout::HWDC || tensor_desc.layout == Layout::BHWDC;
  if (has_depth) {
    return absl::InvalidArgumentError(
        "Currently no support of Depth dimension in predefined/external "
        "tensors.");
  }
  const bool has_batch =
      tensor_desc.layout == Layout::BHWC || tensor_desc.layout == Layout::BHWDC;
  if (has_batch && shape.b == 1) {
    return absl::InvalidArgumentError("Wrong layout, batch mismatch.");
  }
  if (!has_batch && shape.b != 1) {
    return absl::InvalidArgumentError("Wrong layout, batch mismatch.");
  }
  if (!CanCreateTensorWithShape(gpu_info, shape, tensor_desc).ok()) {
    return absl::UnavailableError(
        "Current device can not allocate tensor with this shape for "
        "predefined/external descriptor.");
  }
  return absl::OkStatus();
}

absl::Status MergeNodes(GpuModel* gpu_model) {
  absl::flat_hash_set<ValueId> ready_tensors;
  for (const auto& input : gpu_model->input_ids_and_refs) {
    ready_tensors.insert(input.first);
  }
  auto& nodes = gpu_model->nodes;
  for (int i = 0; i < nodes.size(); ++i) {
    auto& node = nodes[i];
    for (const auto& out_id : node.outputs) {
      ready_tensors.insert(out_id);
    }
    if (node.outputs.size() != 1) {
      continue;
    }
    std::vector<int> next_nodes;
    int link_index = 0;
    for (int j = i + 1; j < nodes.size(); ++j) {
      for (int k = 0; k < nodes[j].inputs.size(); ++k) {
        if (nodes[j].inputs[k] == node.outputs[0]) {
          next_nodes.push_back(j);
          link_index = k;
        }
      }
    }
    if (next_nodes.size() != 1 || link_index != 0) {
      continue;
    }
    auto& linkable_node = nodes[next_nodes[0]];
    if (!linkable_node.gpu_operation->IsLinkable() ||
        linkable_node.outputs.size() != 1 ||
        !IsReady(ready_tensors, linkable_node)) {
      continue;
    }
    const auto& original_dst_def =
        node.gpu_operation->GetDefinition().dst_tensors[0];
    const auto& link_dst_def =
        linkable_node.gpu_operation->GetDefinition().dst_tensors[0];
    if (original_dst_def != link_dst_def) {
      continue;
    }
    RETURN_IF_ERROR(MergeGpuNodes(&linkable_node, &node));
    nodes.erase(nodes.begin() + next_nodes[0]);
    i -= 1;
  }
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

}  // namespace gpu
}  // namespace tflite
