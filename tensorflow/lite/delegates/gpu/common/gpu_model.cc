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

#include <algorithm>
#include <any>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/operation_selector.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/special_selector.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/subgraph.h"
#include "tensorflow/lite/delegates/gpu/common/task/serialization_base.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/add_bias.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/global_pooling_to_reduce_op.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/merge_padding_with.h"

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

absl::Status MergeGpuNodes(const GpuInfo& gpu_info, GpuNode* src,
                           GpuNode* dst) {
  for (int j = 1; j < src->inputs.size(); ++j) {
    dst->inputs.push_back(src->inputs[j]);
  }
  dst->outputs[0] = src->outputs[0];
  dst->name += " -> " + src->name;
  return dst->gpu_operation->AddOperation(gpu_info, src->gpu_operation.get());
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
  node->gpu_operation = std::make_unique<GPUOperation>(std::move(op));
  for (auto in_fb : *fb_node->input_ids()) {
    node->inputs.push_back(in_fb);
  }
  for (auto out_fb : *fb_node->output_ids()) {
    node->outputs.push_back(out_fb);
  }
  node->name = std::string(fb_node->name()->c_str(), fb_node->name()->size());

  return absl::OkStatus();
}

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
  if (tensor_desc.GetDataType() != data_type) {
    return absl::InvalidArgumentError(
        "Global precision and precision of predefined/external tensors must be "
        "synchronized.");
  }
  if (tensor_desc.HasAxis(Axis::DEPTH)) {
    return absl::InvalidArgumentError(
        "Currently no support of Depth dimension in predefined/external "
        "tensors.");
  }
  if (tensor_desc.HasAxis(Axis::BATCH) && shape.b == 1) {
    return absl::InvalidArgumentError("Wrong layout, batch mismatch.");
  }
  if (!tensor_desc.HasAxis(Axis::BATCH) && shape.b != 1) {
    return absl::InvalidArgumentError("Wrong layout, batch mismatch.");
  }
  if (!tensor_desc.CanCreateTensorWithShape(gpu_info, shape).ok()) {
    return absl::UnavailableError(
        "Current device can not allocate tensor with this shape for "
        "predefined/external descriptor.");
  }
  return absl::OkStatus();
}

// Helper class for creating descriptors for appropriate tensors from
// GraphFloat32
// Also allows to create descriptors for new tensors(not present in
// GraphFloat32)
class TensorReserver {
 public:
  TensorReserver() : next_(0) {}
  ValueId Add(const TensorDescriptor& dummy) {
    reservations_[next_] = dummy;
    return next_++;
  }
  void Add(ValueId id, const TensorDescriptor& dummy) {
    reservations_[id] = dummy;
  }
  ValueId GetNewId() { return next_++; }
  void SetNext(ValueId id) { next_ = id; }
  TensorDescriptor Get(ValueId id) { return reservations_[id]; }

 public:
  absl::flat_hash_map<ValueId, TensorDescriptor> reservations_;
  ValueId next_;
};

absl::Status ReserveGraphTensors(const CreateGpuModelInfo& create_info,
                                 const GpuInfo& gpu_info,
                                 const GraphFloat32& graph,
                                 TensorReserver* tensor_reserver) {
  ValueId max_id = 0;
  auto tensors = graph.values();
  for (auto& t : tensors) {
    auto data_type = DeduceDataTypeFromPrecision(create_info.precision);
    if (t->tensor.type != DataType::FLOAT32 &&
        t->tensor.type != DataType::FLOAT16) {
      data_type = t->tensor.type;
    }
    const auto shape = graph.GetValue(t->id)->tensor.shape;
    auto it_predefined = create_info.predefined.find(t->id);
    auto it_immutable_external =
        create_info.external_immutable_tensors.find(t->id);
    auto it_mutable_external = create_info.external_mutable_tensors.find(t->id);
    int external_categories_count = 0;
    TensorDescriptor tensor_desc;
    if (it_predefined != create_info.predefined.end()) {
      external_categories_count++;
      tensor_desc = it_predefined->second;
    }
    if (it_immutable_external != create_info.external_immutable_tensors.end()) {
      external_categories_count++;
      tensor_desc = it_immutable_external->second->GetDescriptor();
    }
    if (it_mutable_external != create_info.external_mutable_tensors.end()) {
      external_categories_count++;
      tensor_desc = it_mutable_external->second;
    }
    if (external_categories_count > 1) {
      return absl::InvalidArgumentError(
          "Tensors ids from predefined / external_immutable_tensors / "
          "external_mutable_tensors should not intersect.");
    }
    if (external_categories_count == 1) {
      if (!(graph.IsGraphInput(t->id) || graph.IsGraphOutput(t->id))) {
        return absl::InvalidArgumentError(
            "Currently external can be used only for graph inputs/outputs");
      }
      RETURN_IF_ERROR(CheckExternalTensorDescription(gpu_info, tensor_desc,
                                                     shape, data_type));
    } else {
      TensorStorageType storage_type = create_info.storage_type;
      Layout layout = shape.b == 1 ? Layout::HWC : Layout::BHWC;
      const bool can_use_single_texture =
          storage_type == TensorStorageType::TEXTURE_2D ||
          storage_type == TensorStorageType::TEXTURE_3D ||
          storage_type == TensorStorageType::TEXTURE_ARRAY;
      if (shape.c < 4 && can_use_single_texture &&
          TensorDescriptor{data_type, TensorStorageType::SINGLE_TEXTURE_2D,
                           layout}
              .CanCreateTensorWithShape(gpu_info, shape)
              .ok()) {
        storage_type = TensorStorageType::SINGLE_TEXTURE_2D;
      }
      tensor_desc = TensorDescriptor{data_type, storage_type, layout};
      RETURN_IF_ERROR(
          tensor_desc.UpdateToSupportedStorageType(gpu_info, shape));
      if (gpu_info.IsApiMetal() &&
          storage_type == TensorStorageType::TEXTURE_2D) {
        tensor_desc.SetUseBufferForWriteOnlyTexture2d(true);
      }
    }
    tensor_desc.SetBHWCShape(shape);
    tensor_reserver->Add(t->id, tensor_desc);
    max_id = std::max(max_id, t->id);
  }
  tensor_reserver->SetNext(max_id + 1);
  return absl::OkStatus();
}

absl::Status ConvertOperations(const GpuInfo& gpu_info,
                               const GraphFloat32& graph,
                               const CreateGpuModelInfo& create_info,
                               TensorReserver* tensor_reserver,
                               GpuModel* gpu_model) {
  std::map<ValueId, TensorDescriptor> tensor_descriptors;
  const auto values = graph.values();
  for (auto value : values) {
    tensor_descriptors[value->id] = tensor_reserver->Get(value->id);
  }
  std::set<NodeId> consumed_nodes;
  std::vector<Node*> graph_nodes = graph.nodes();
  std::map<ValueId, int>
      tensor_usages;  // keeps latest index of operation that updated tensor
  for (const auto& input : gpu_model->input_ids_and_refs) {
    tensor_usages[input.first] = -1;  // so as inputs "updated" before operation
                                      // 0, we will mark them with -1
  }
  std::vector<SharedWeightsConvDesc> shared_conv_weights;
  std::vector<SharedWeightsConvDesc>* shared_conv_weights_ptr =
      create_info.hints.Check(ModelHints::kReuseConvWeights)
          ? &shared_conv_weights
          : nullptr;
  for (int i = 0; i < graph_nodes.size(); ++i) {
    const Node& node = *graph_nodes[i];
    if (consumed_nodes.find(node.id) != consumed_nodes.end()) {
      continue;
    }
    auto op_type = OperationTypeFromString(node.operation.type);
    if (op_type == OperationType::CONSTANT) {
      auto attr =
          absl::any_cast<ConstTensorAttributes>(node.operation.attributes);
      auto outputs = graph.FindOutputs(node.id);
      gpu_model->const_tensors[outputs[0]->id] =
          tensor_reserver->Get(outputs[0]->id);
      gpu_model->const_tensors[outputs[0]->id].UploadData(attr.tensor);
      continue;
    }
    GPUOperationsSubgraph gpu_subgraph;
    if (GPUSubgraphFromGraph(create_info.hints, gpu_info, create_info.precision,
                             graph, node.id, tensor_descriptors,
                             &consumed_nodes, &gpu_subgraph)
            .ok()) {
      // Mapping of subgraph (set of nodes) to GPU operations. Should happen
      // before straigtforward mapping.
    } else {
      // Straigtforward mapping of one graph node to GPU operations.
      auto inputs = graph.FindInputs(node.id);
      auto outputs = graph.FindOutputs(node.id);
      // Reordering of input ids and updating of temporary tensors_usage struct.
      // To have better linking we need linking tensor(latest written during
      // linear execution) on first position.
      if (IsAssociativeLinkableOp(node, inputs, outputs)) {
        int latest_written_tensor_index = 0;
        int last_usage = tensor_usages[inputs[0]->id];
        for (int j = 1; j < inputs.size(); ++j) {
          if (tensor_usages[inputs[j]->id] > last_usage) {
            last_usage = tensor_usages[inputs[j]->id];
            latest_written_tensor_index = j;
          }
        }
        std::swap(inputs[0], inputs[latest_written_tensor_index]);
      }
      consumed_nodes.insert(node.id);
      OperationDef op_def;
      op_def.precision = create_info.precision;
      for (int j = 0; j < inputs.size(); ++j) {
        op_def.src_tensors.push_back(tensor_reserver->Get(inputs[j]->id));
      }
      for (int j = 0; j < outputs.size(); ++j) {
        op_def.dst_tensors.push_back(tensor_reserver->Get(outputs[j]->id));
      }
      RETURN_IF_ERROR(GPUOperationFromNode(
          gpu_info, op_def, create_info.hints, inputs, outputs, node,
          shared_conv_weights_ptr, &gpu_subgraph));
    }
    absl::flat_hash_map<int, ValueId> mapping_to_global_ids;
    for (int j = 0; j < gpu_subgraph.new_tensors.size(); ++j) {
      const auto& t = gpu_subgraph.new_tensors[j];
      if (!t.GetData().empty()) {  // constant tensor
        auto global_id = tensor_reserver->GetNewId();
        gpu_model->const_tensors[global_id] =
            std::move(gpu_subgraph.new_tensors[j]);
        mapping_to_global_ids[j] = global_id;
      } else {
        auto global_id = tensor_reserver->Add(t);
        mapping_to_global_ids[j] = global_id;
      }
    }
    if (!shared_conv_weights.empty() && !mapping_to_global_ids.empty()) {
      shared_conv_weights.back().RemapIds(mapping_to_global_ids);
    }
    for (auto& gpu_op : gpu_subgraph.operations) {
      GpuNode gpu_node;
      gpu_node.gpu_operation = std::move(gpu_op.operation);
      gpu_node.inputs.resize(gpu_op.input_ids.size());
      for (int j = 0; j < gpu_op.input_ids.size(); ++j) {
        int id = gpu_op.input_ids[j];
        if (id >= 0) {
          gpu_node.inputs[j] = id;
        } else {
          gpu_node.inputs[j] = mapping_to_global_ids[-(id + 1)];
        }
      }
      gpu_node.outputs.resize(gpu_op.output_ids.size());
      for (int j = 0; j < gpu_op.output_ids.size(); ++j) {
        int id = gpu_op.output_ids[j];
        if (id >= 0) {
          gpu_node.outputs[j] = id;
          tensor_usages[id] = i;
        } else {
          gpu_node.outputs[j] = mapping_to_global_ids[-(id + 1)];
        }
      }
      gpu_node.name = gpu_op.name;
      gpu_model->nodes.push_back(std::move(gpu_node));
    }
  }

  return absl::OkStatus();
}

absl::Status MergeElementwiseNodes(const GpuInfo& gpu_info,
                                   GpuModel* gpu_model) {
  auto& nodes = gpu_model->nodes;
  for (int elem_root_index = 1; elem_root_index < nodes.size();
       ++elem_root_index) {
    auto& elem_root = nodes[elem_root_index];
    if (!(elem_root.inputs.size() == 1 || elem_root.inputs.size() == 2) ||
        elem_root.outputs.size() != 1 ||
        !elem_root.gpu_operation->IsLinkable()) {
      continue;
    }
    // key is elem_root input index, value is node index
    std::map<int, int> prev_nodes;
    for (int j = elem_root_index - 1; j >= 0; --j) {
      for (int k = 0; k < elem_root.inputs.size(); ++k) {
        if (elem_root.inputs[k] == nodes[j].outputs[0]) {
          prev_nodes[k] = j;
          break;
        }
      }
    }
    // TYPE_0
    //    input       input
    //      |           |
    //    elem0         |
    //      |    -->  elem
    //  elem_root       |
    //      |           |
    //    output      output
    if (prev_nodes.size() == 1) {
      if (elem_root.inputs.size() != 1) {
        continue;
      }
      const int prev_first_node_index = prev_nodes[0];
      auto& prev_node = nodes[prev_first_node_index];
      if (prev_node.inputs.size() != 1 || prev_node.outputs.size() != 1 ||
          !prev_node.gpu_operation->IsLinkable()) {
        continue;
      }
      int consumers_count = 0;
      for (const auto& node : nodes) {
        for (const auto& input : node.inputs) {
          if (input == elem_root.inputs[0]) {
            consumers_count++;
          }
        }
      }
      if (consumers_count != 1) {
        continue;
      }
      prev_node.outputs[0] = elem_root.outputs[0];
      prev_node.name += " -> " + elem_root.name;
      RETURN_IF_ERROR(prev_node.gpu_operation->FuseSimpleElemWithSimpleElem(
          gpu_info, elem_root.gpu_operation.get()));
      nodes.erase(nodes.begin() + elem_root_index);
      elem_root_index = prev_first_node_index;
      continue;
    }

    // check TYPE_1/2/3
    if (prev_nodes.size() == 2) {
      if (elem_root.inputs.size() != 2) {
        continue;
      }
      const int prev_first_node_index = prev_nodes[0];
      const int prev_second_node_index = prev_nodes[1];
      auto& prev_first_node = nodes[prev_first_node_index];
      auto& prev_second_node = nodes[prev_second_node_index];

      // check TYPE_1
      // TYPE_1
      //      input           input
      //     /    \             |
      //   elem0   |            |
      //     \    /      -->  elem
      //   elem_root            |
      //       |                |
      //     output           output
      if (prev_first_node.gpu_operation->IsLinkable() &&
          !prev_second_node.gpu_operation->IsLinkable() &&
          prev_second_node.outputs.size() == 1 &&
          prev_first_node.inputs.size() == 1 &&
          prev_first_node.outputs.size() == 1) {
        int first_node_parent_index = -1;
        for (int j = prev_first_node_index - 1; j >= 0; --j) {
          if (nodes[j].outputs[0] == prev_first_node.inputs[0]) {
            first_node_parent_index = j;
            break;
          }
        }
        if (first_node_parent_index == -1 ||
            first_node_parent_index != prev_second_node_index) {
          continue;
        }
        int consumers_count = 0;
        for (const auto& node : nodes) {
          for (const auto& input : node.inputs) {
            if (input == elem_root.inputs[0]) {
              consumers_count++;
            }
          }
        }
        if (consumers_count != 1) {
          continue;
        }

        prev_first_node.outputs[0] = elem_root.outputs[0];
        prev_first_node.name += " -> " + elem_root.name;
        RETURN_IF_ERROR(prev_first_node.gpu_operation
                            ->Fuse2InputElemWithSimpleElemAsFirstInput(
                                gpu_info, elem_root.gpu_operation.get()));
        nodes.erase(nodes.begin() + elem_root_index);
        elem_root_index = prev_first_node_index;
        continue;
      }

      // check TYPE_2
      // TYPE_2
      //      input           input
      //     /    \             |
      //    |    elem0          |
      //     \    /      -->  elem
      //   elem_root            |
      //       |                |
      //     output           output
      if (!prev_first_node.gpu_operation->IsLinkable() &&
          prev_second_node.gpu_operation->IsLinkable() &&
          prev_first_node.outputs.size() == 1 &&
          prev_second_node.inputs.size() == 1 &&
          prev_second_node.outputs.size() == 1) {
        int second_node_parent_index = -1;
        for (int j = prev_second_node_index - 1; j >= 0; --j) {
          if (nodes[j].outputs[0] == prev_second_node.inputs[0]) {
            second_node_parent_index = j;
            break;
          }
        }
        if (second_node_parent_index == -1 ||
            second_node_parent_index != prev_first_node_index) {
          continue;
        }
        int consumers_count = 0;
        for (const auto& node : nodes) {
          for (const auto& input : node.inputs) {
            if (input == elem_root.inputs[1]) {
              consumers_count++;
            }
          }
        }
        if (consumers_count != 1) {
          continue;
        }

        prev_second_node.outputs[0] = elem_root.outputs[0];
        prev_second_node.name += " -> " + elem_root.name;
        RETURN_IF_ERROR(prev_second_node.gpu_operation
                            ->Fuse2InputElemWithSimpleElemAsSecondInput(
                                gpu_info, elem_root.gpu_operation.get()));
        nodes.erase(nodes.begin() + elem_root_index);
        elem_root_index = prev_second_node_index;
        continue;
      }

      // check TYPE_3
      // TYPE_3
      //      input           input
      //     /    \             |
      //  elem0  elem1          |
      //     \    /      -->  elem
      //   elem_root            |
      //       |                |
      //     output           output
      if (prev_first_node.gpu_operation->IsLinkable() &&
          prev_second_node.gpu_operation->IsLinkable() &&
          prev_first_node.inputs.size() == 1 &&
          prev_first_node.outputs.size() == 1 &&
          prev_second_node.inputs.size() == 1 &&
          prev_second_node.outputs.size() == 1) {
        int first_node_parent_index = -1;
        for (int j = prev_first_node_index - 1; j >= 0; --j) {
          if (nodes[j].outputs[0] == prev_first_node.inputs[0]) {
            first_node_parent_index = j;
            break;
          }
        }
        int second_node_parent_index = -1;
        for (int j = prev_second_node_index - 1; j >= 0; --j) {
          if (nodes[j].outputs[0] == prev_second_node.inputs[0]) {
            second_node_parent_index = j;
            break;
          }
        }
        if (first_node_parent_index == -1 || second_node_parent_index == -1 ||
            first_node_parent_index != second_node_parent_index) {
          continue;
        }

        int consumers_count = 0;
        for (const auto& node : nodes) {
          for (const auto& input : node.inputs) {
            if (input == elem_root.inputs[1]) {
              consumers_count++;
            }
          }
        }
        if (consumers_count != 1) {
          continue;
        }

        consumers_count = 0;
        for (const auto& node : nodes) {
          for (const auto& input : node.inputs) {
            if (input == elem_root.inputs[0]) {
              consumers_count++;
            }
          }
        }
        if (consumers_count != 1) {
          continue;
        }

        GPUOperation new_operation;
        RETURN_IF_ERROR(Fuse2InputElemWith2SimpleElem(
            gpu_info, std::move(*prev_first_node.gpu_operation.get()),
            std::move(*prev_second_node.gpu_operation.get()),
            std::move(*elem_root.gpu_operation.get()), &new_operation));
        GpuNode new_node;
        new_node.inputs.push_back(prev_first_node.inputs[0]);
        new_node.outputs.push_back(elem_root.outputs[0]);
        new_node.name = prev_first_node.name + " -> " + prev_second_node.name +
                        " -> " + elem_root.name;
        new_node.gpu_operation =
            std::make_unique<GPUOperation>(std::move(new_operation));

        // prev_first_node_index and prev_second_node_index ordered relative to
        // elem_root inputs.
        // first_prev_node_index and second_prev_node_index ordered relative to
        // nodes.
        int first_prev_node_index =
            std::min(prev_first_node_index, prev_second_node_index);
        int second_prev_node_index =
            std::max(prev_first_node_index, prev_second_node_index);
        nodes.erase(nodes.begin() + elem_root_index);
        nodes.erase(nodes.begin() + second_prev_node_index);
        nodes[first_prev_node_index] = std::move(new_node);
        elem_root_index = first_prev_node_index - 1;
        continue;
      }
    }
  }
  return absl::OkStatus();
}

absl::Status MergeNodes(const GpuInfo& gpu_info, GpuModel* gpu_model) {
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
    RETURN_IF_ERROR(MergeGpuNodes(gpu_info, &linkable_node, &node));
    nodes.erase(nodes.begin() + next_nodes[0]);
    i -= 1;
  }
  return absl::OkStatus();
}

void CopyExternals(const GraphFloat32& graph, GpuModel* gpu_model) {
  const auto inputs = graph.inputs();
  for (const auto& value : inputs) {
    gpu_model->input_ids_and_refs.push_back({value->id, value->tensor.ref});
  }

  const auto variable_inputs = graph.variable_inputs();
  for (const auto& value : variable_inputs) {
    gpu_model->variable_ids_and_refs.push_back({value->id, value->tensor.ref});
  }

  const auto outputs = graph.outputs();
  for (const auto& value : outputs) {
    gpu_model->output_ids_and_refs.push_back({value->id, value->tensor.ref});
  }
}

// Removing tensors that was fused in complex operations
void RemoveUnusedTensors(GpuModel* gpu_model) {
  absl::flat_hash_set<ValueId> used_tensors;
  for (const auto& node : gpu_model->nodes) {
    for (const auto& id : node.inputs) {
      used_tensors.insert(id);
    }
    for (const auto& id : node.outputs) {
      used_tensors.insert(id);
    }
  }
  for (const auto& inputs : gpu_model->input_ids_and_refs) {
    used_tensors.insert(inputs.first);
  }
  for (const auto& outputs : gpu_model->output_ids_and_refs) {
    used_tensors.insert(outputs.first);
  }
  for (auto it = gpu_model->tensors.begin(); it != gpu_model->tensors.end();) {
    if (used_tensors.find(it->first) == used_tensors.end()) {
      gpu_model->tensors.erase(it++);
    } else {
      ++it;
    }
  }
}

// Serialized model will lose polymorphic properties for GpuOperations.
// Here we will retrieve some information needed for generic execution of
// GpuOperations. Specifically, BindArguments and RecalculateGridSize must be
// executed.
absl::Status ResolvePolymorphicArgs(GpuModel* gpu_model) {
  class DummySpatialTensor : public GpuSpatialTensor {
   public:
    DummySpatialTensor() = default;
    explicit DummySpatialTensor(const BHWDC& shape,
                                const TensorDescriptor& tensor_desc)
        : shape_(shape), tensor_desc_(tensor_desc) {}
    ~DummySpatialTensor() override = default;

    int Width() const override { return shape_.w; }
    int Height() const override { return shape_.h; }
    int Depth() const override { return shape_.d; }
    int Channels() const override { return shape_.c; }
    int Slices() const override { return DivideRoundUp(shape_.c, 4); }
    int Batch() const override { return shape_.b; }

    TensorDescriptor GetDescriptor() const override { return tensor_desc_; }

   private:
    BHWDC shape_;
    TensorDescriptor tensor_desc_;
  };

  for (auto& node : gpu_model->nodes) {
    std::vector<DummySpatialTensor> src_tensors(node.inputs.size());
    for (int i = 0; i < node.inputs.size(); ++i) {
      const auto& tensor_desc = gpu_model->tensors[node.inputs[i]];
      src_tensors[i] =
          DummySpatialTensor(tensor_desc.GetBHWDCShape(), tensor_desc);
      node.gpu_operation->SetSrc(&src_tensors[i], i);
    }
    std::vector<DummySpatialTensor> dst_tensors(node.outputs.size());
    for (int i = 0; i < node.outputs.size(); ++i) {
      const auto& tensor_desc = gpu_model->tensors[node.outputs[i]];
      dst_tensors[i] =
          DummySpatialTensor(tensor_desc.GetBHWDCShape(), tensor_desc);
      node.gpu_operation->SetDst(&dst_tensors[i], i);
    }
    RETURN_IF_ERROR(
        node.gpu_operation->BindArguments(&node.gpu_operation->args_));
    node.gpu_operation->RecalculateGridSize();
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status GraphToGpuModel(const GraphFloat32& graph,
                             const CreateGpuModelInfo& create_info,
                             const GpuInfo& gpu_info, GpuModel* gpu_model) {
  TensorReserver tensor_reserver;
  RETURN_IF_ERROR(
      ReserveGraphTensors(create_info, gpu_info, graph, &tensor_reserver));
  CopyExternals(graph, gpu_model);
  RETURN_IF_ERROR(ConvertOperations(gpu_info, graph, create_info,
                                    &tensor_reserver, gpu_model));
  // MergeElementwise fuse only elemntwise nodes, MergeNodes fuse elementwise to
  // usual nodes
  RETURN_IF_ERROR(MergeElementwiseNodes(gpu_info, gpu_model));
  RETURN_IF_ERROR(MergeNodes(gpu_info, gpu_model));
  gpu_model->tensors = std::move(tensor_reserver.reservations_);
  RemoveUnusedTensors(gpu_model);

  for (auto& node : gpu_model->nodes) {
    RETURN_IF_ERROR(node.gpu_operation->AssembleCode(gpu_info));
  }

  return ResolvePolymorphicArgs(gpu_model);
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

absl::Status RunGraphTransformsForGpuModel(GraphFloat32* graph) {
  auto merge_padding_transform = NewMergePaddingWithAdd();
  auto add_bias_transform = NewAddBias();
  auto pooling_to_reduce_op = NewGlobalPoolingToReduceOp();
  ModelTransformer transformer(graph);
  if (!transformer.Apply("add_bias", add_bias_transform.get())) {
    return absl::InternalError("Invalid add_bias transform");
  }
  if (!transformer.Apply("merge_padding", merge_padding_transform.get())) {
    return absl::InternalError("Invalid merge_padding transform");
  }
  if (!transformer.Apply("global pooling to mean",
                         pooling_to_reduce_op.get())) {
    return absl::InternalError("Invalid global pooling to mean transform");
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
