/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/serialization.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/operation_selector.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/special_selector.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/storage_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/add_bias.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/global_pooling_to_reduce_op.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/merge_padding_with.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {

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

std::vector<std::pair<ValueId, TensorDescriptor>> GetCLNodeTensors(
    const CLNode& node) {
  std::vector<std::pair<ValueId, TensorDescriptor>> result;
  result.reserve(node.inputs.size() + node.outputs.size());
  const OperationDef op_def = node.cl_operation.GetDefinition();
  for (int j = 0; j < node.inputs.size(); ++j) {
    result.push_back({node.inputs[j], op_def.src_tensors[j]});
  }
  for (int j = 0; j < node.outputs.size(); ++j) {
    result.push_back({node.outputs[j], op_def.dst_tensors[j]});
  }

  return result;
}

absl::Status MergeGpuNodes(GpuNode* src, GpuNode* dst) {
  for (int j = 1; j < src->inputs.size(); ++j) {
    dst->inputs.push_back(src->inputs[j]);
  }
  dst->outputs[0] = src->outputs[0];
  dst->name += " linked : " + src->name;
  return dst->gpu_operation->AddOperation(src->gpu_operation.get());
}

void AddUsage(ValueId id, int task_index,
              std::map<ValueId, int2>* usage_records) {
  auto it = usage_records->find(id);
  if (it == usage_records->end()) {
    (*usage_records)[id].x = task_index;
    (*usage_records)[id].y = task_index;
  } else {
    (*usage_records)[id].y = task_index;
  }
}

// returns true if actual memory for this storage type will be allocated with
// clCreateBuffer.
bool IsBufferBased(const GpuInfo& gpu_info, const TensorStorageType& type) {
  const bool image2d_based_buffer =
      (type == TensorStorageType::TEXTURE_2D ||
       type == TensorStorageType::SINGLE_TEXTURE_2D) &&
      gpu_info.opencl_info.IsImage2dFromBufferSupported();
  return type == TensorStorageType::BUFFER ||
         type == TensorStorageType::IMAGE_BUFFER || image2d_based_buffer;
}

bool IsAssociativeLinkableOp(const tflite::gpu::Node& node,
                             const std::vector<tflite::gpu::Value*>& inputs,
                             const std::vector<tflite::gpu::Value*>& outputs) {
  if (inputs.size() == 1) {
    return false;
  }
  const tflite::gpu::OperationType op_type =
      tflite::gpu::OperationTypeFromString(node.operation.type);
  if (op_type != tflite::gpu::OperationType::ADD &&
      op_type != tflite::gpu::OperationType::MUL) {
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

// Calculates the total size of the assignment.
size_t TotalSize(const ObjectsAssignment<size_t>& assignment) {
  return std::accumulate(assignment.object_sizes.begin(),
                         assignment.object_sizes.end(), static_cast<size_t>(0));
}

// Checks if sub-buffer image 2D mapping is supported.
bool CanUseSubBuffer(const GpuInfo& gpu_info) {
  if (!gpu_info.IsCL11OrHigher()) {
    return false;
  }
  if (gpu_info.IsPowerVR()) {
    return false;
  }
  if (gpu_info.IsMali() &&
      (gpu_info.mali_info.IsBifrost() || gpu_info.mali_info.IsMidgard())) {
    // Known driver issue on some G72 (Bifrost), G76 (Bifrost), T830 (Midgard),
    // and T880 (Midgard) devices.
    return false;
  }
  return true;
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
  void SetNext(ValueId id) { next_ = id; }
  TensorDescriptor Get(ValueId id) { return reservations_[id]; }

 public:
  absl::flat_hash_map<ValueId, TensorDescriptor> reservations_;
  ValueId next_;
};

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

absl::Status ReserveGraphTensors(const CreateGpuModelInfo& create_info,
                                 const GpuInfo& gpu_info,
                                 const GraphFloat32& graph,
                                 TensorReserver* tensor_reserver) {
  ValueId max_id = 0;
  auto tensors = graph.values();
  auto data_type = DeduceDataTypeFromPrecision(create_info.precision);
  for (auto& t : tensors) {
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
      if (graph.IsGraphInput(t->id) || graph.IsGraphOutput(t->id)) {
        if (shape.c < 4 &&
            CanCreateTensorWithShape(
                gpu_info, shape,
                TensorDescriptor{data_type,
                                 TensorStorageType::SINGLE_TEXTURE_2D, layout})
                .ok()) {
          storage_type = TensorStorageType::SINGLE_TEXTURE_2D;
        }
      }
      RETURN_IF_ERROR(SelectBestStorageType(gpu_info, shape, storage_type,
                                            data_type, layout, &storage_type));
      tensor_desc = TensorDescriptor{data_type, storage_type, layout};
    }
    tensor_desc.shape = BHWDC(shape.b, shape.h, shape.w, 1, shape.c);
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
    if (create_info.hints.Check(ModelHints::kAllowSpecialKernels) &&
        GPUSubgraphFromGraph(gpu_info, create_info.precision, graph, node.id,
                             tensor_descriptors, &consumed_nodes, &gpu_subgraph)
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
      RETURN_IF_ERROR(GPUOperationFromNode(gpu_info, op_def, create_info.hints,
                                           inputs, outputs, node,
                                           &gpu_subgraph));
    }
    absl::flat_hash_map<int, ValueId> mapping_to_global_ids;
    for (int j = 0; j < gpu_subgraph.new_tensors.size(); ++j) {
      const auto& t = gpu_subgraph.new_tensors[j];
      TensorDescriptor td = t.second;
      td.shape = BHWDC(t.first.b, t.first.h, t.first.w, 1, t.first.c);
      auto global_id = tensor_reserver->Add(td);
      mapping_to_global_ids[j] = global_id;
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

absl::Status Merge(GpuModel* gpu_model) {
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
      src_tensors[i] = DummySpatialTensor(tensor_desc.shape, tensor_desc);
      node.gpu_operation->SetSrc(&src_tensors[i], i);
    }
    std::vector<DummySpatialTensor> dst_tensors(node.outputs.size());
    for (int i = 0; i < node.outputs.size(); ++i) {
      const auto& tensor_desc = gpu_model->tensors[node.outputs[i]];
      dst_tensors[i] = DummySpatialTensor(tensor_desc.shape, tensor_desc);
      node.gpu_operation->SetDst(&dst_tensors[i], i);
    }
    RETURN_IF_ERROR(
        node.gpu_operation->BindArguments(&node.gpu_operation->args_));
    node.gpu_operation->RecalculateGridSize();
  }
  return absl::OkStatus();
}

absl::Status GraphToGpuModel(const CreateGpuModelInfo& create_info,
                             const GraphFloat32& graph, const GpuInfo& gpu_info,
                             GpuModel* gpu_model) {
  TensorReserver tensor_reserver;
  RETURN_IF_ERROR(
      ReserveGraphTensors(create_info, gpu_info, graph, &tensor_reserver));
  CopyExternals(graph, gpu_model);
  RETURN_IF_ERROR(ConvertOperations(gpu_info, graph, create_info,
                                    &tensor_reserver, gpu_model));
  RETURN_IF_ERROR(Merge(gpu_model));
  gpu_model->tensors = std::move(tensor_reserver.reservations_);

  for (auto& node : gpu_model->nodes) {
    RETURN_IF_ERROR(node.gpu_operation->AssembleCode(gpu_info));
  }

  return ResolvePolymorphicArgs(gpu_model);
}

}  // namespace

void InferenceContext::ExecutionHints::Init(const GpuInfo& gpu_info) {
  if (gpu_info.IsMali()) {
    need_flush = true;
    need_manual_release = gpu_info.mali_info.IsValhall() ? false : true;

    flush_periodically = true;
    flush_period = 24;
  }
  if (gpu_info.IsPowerVR()) {
    need_flush = true;
    flush_periodically = true;
    flush_period = 16;
  }
}

absl::Status InferenceContext::InitFromGraph(
    const CreateGpuModelInfo& create_info, const GraphFloat32& graph,
    Environment* env, std::vector<uint8_t>* serialized_model) {
  GpuModel gpu_model;
  RETURN_IF_ERROR(GraphToGpuModel(create_info, graph,
                                  env->GetDevicePtr()->GetInfo(), &gpu_model));

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<data::GpuModel> gpu_model_fb;
  if (serialized_model) {
    gpu_model_fb = Encode(gpu_model, &builder);
  }
  CopyFromGpuModel(&gpu_model);

  CreationContext creation_context;
  creation_context.device = env->GetDevicePtr();
  creation_context.context = &env->context();
  creation_context.queue = env->queue();
  creation_context.cache = env->program_cache();
  for (const auto& external_tensor : create_info.external_immutable_tensors) {
    auto* cl_spatial_tensor = dynamic_cast<Tensor*>(external_tensor.second);
    if (!cl_spatial_tensor) {
      return absl::InvalidArgumentError("Expected CLSpatialTensor.");
    }
    external_immutable_tensors_[external_tensor.first] = cl_spatial_tensor;
  }
  std::map<ValueId, Tensor> temp_external_tensors;
  for (const auto& external_tensor : create_info.external_mutable_tensors) {
    RETURN_IF_ERROR(CreateTensor(
        env->context(), tensors_descs_[external_tensor.first].shape,
        tensors_descs_[external_tensor.first],
        &temp_external_tensors[external_tensor.first]));
    external_mutable_tensors_[external_tensor.first] =
        &temp_external_tensors[external_tensor.first];
  }
  PrepareExternal();
  execution_hints_.Init(env->device().GetInfo());
  RETURN_IF_ERROR(
      AllocateMemory(creation_context.GetGpuInfo(), creation_context.context));
  BindMemoryToOperations();
  RETURN_IF_ERROR(Compile(creation_context));
  RETURN_IF_ERROR(UpdateParams());

  TuningType tuning_type = TuningType::kExhaustive;
  if (create_info.hints.Check(ModelHints::kFastTuning)) {
    tuning_type = TuningType::kFast;
  }
  if (env->device().GetInfo().IsMali()) {
    const MaliInfo& info = env->device().GetInfo().mali_info;
    if (info.IsMaliT6xx()) {
      // Mali T628 hangs forever in clFinish when used profiling queue
      // TuningType::FAST does not use profiling queue.
      tuning_type = TuningType::kFast;
    }
  }
  RETURN_IF_ERROR(
      Tune(tuning_type, env->device().GetInfo(), env->profiling_queue()));
  if (external_mutable_tensors_.empty()) {
    // using recordable queue only when no mutable external tensors
    InitRecordableQueue(env);
  }

  for (auto& external_tensor : external_mutable_tensors_) {
    external_tensor.second = nullptr;
  }

  gpu_info_ = env->device().GetInfo();

  if (serialized_model) {
    auto encoded_fb = Encode(*env->GetDevicePtr(), *this, *env->program_cache(),
                             gpu_model_fb, &builder);
    data::FinishInferenceContextBuffer(builder, encoded_fb);
    serialized_model->resize(builder.GetSize());
    std::memcpy(serialized_model->data(), builder.GetBufferPointer(),
                builder.GetSize());
  }
  ReleaseCPURepresentation();
  return absl::OkStatus();
}

absl::Status InferenceContext::RestoreDeserialized(
    const absl::Span<const uint8_t> serialized_model, Environment* env,
    CreateGpuModelInfo* create_info) {
  flatbuffers::Verifier verifier(serialized_model.data(),
                                 serialized_model.size());
  if (!data::VerifyInferenceContextBuffer(verifier)) {
    return absl::DataLossError("Deserialization failed.");
  }
  auto decoded_fb = data::GetInferenceContext(serialized_model.data());
  RETURN_IF_ERROR(Decode(env->context(), *env->GetDevicePtr(),
                         env->program_cache(), decoded_fb, this));

  CreationContext creation_context;
  creation_context.device = env->GetDevicePtr();
  creation_context.context = &env->context();
  creation_context.queue = env->queue();
  creation_context.cache = env->program_cache();
  std::map<ValueId, Tensor> temp_external_tensors;
  if (create_info) {
    for (const auto& external_tensor :
         create_info->external_immutable_tensors) {
      auto* cl_spatial_tensor = dynamic_cast<Tensor*>(external_tensor.second);
      if (!cl_spatial_tensor) {
        return absl::InvalidArgumentError("Expected CLSpatialTensor.");
      }
      external_immutable_tensors_[external_tensor.first] = cl_spatial_tensor;
    }
    for (const auto& external_tensor : create_info->external_mutable_tensors) {
      RETURN_IF_ERROR(CreateTensor(
          env->context(), tensors_descs_[external_tensor.first].shape,
          tensors_descs_[external_tensor.first],
          &temp_external_tensors[external_tensor.first]));
      external_mutable_tensors_[external_tensor.first] =
          &temp_external_tensors[external_tensor.first];
    }
  }
  PrepareExternal();

  execution_hints_.Init(env->device().GetInfo());

  RETURN_IF_ERROR(
      AllocateMemory(creation_context.GetGpuInfo(), creation_context.context));
  BindMemoryToOperations();
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.cl_operation.RestoreDeserialized(creation_context));
  }
  RETURN_IF_ERROR(UpdateParams());
  if (external_mutable_tensors_.empty()) {
    // using recordable queue only when no mutable external tensors
    InitRecordableQueue(env);
  }
  for (auto& external_tensor : external_mutable_tensors_) {
    external_tensor.second = nullptr;
  }
  ReleaseCPURepresentation();
  return absl::OkStatus();
}

void InferenceContext::CopyFromGpuModel(GpuModel* gpu_model) {
  for (const auto& input : gpu_model->input_ids_and_refs) {
    input_ids_.push_back(input.first);
  }
  for (const auto& variable_input : gpu_model->variable_ids_and_refs) {
    variable_ids_and_refs_[variable_input.first] = variable_input.second;
  }
  for (const auto& output : gpu_model->output_ids_and_refs) {
    output_ids_.push_back(output.first);
  }
  nodes_.resize(gpu_model->nodes.size());
  for (int i = 0; i < gpu_model->nodes.size(); ++i) {
    nodes_[i].cl_operation.Init(std::move(gpu_model->nodes[i].gpu_operation));
    nodes_[i].inputs = gpu_model->nodes[i].inputs;
    nodes_[i].outputs = gpu_model->nodes[i].outputs;
    nodes_[i].name = gpu_model->nodes[i].name;
  }
  const_tensors_descs_ = std::move(gpu_model->const_tensors);
  tensors_descs_ = std::move(gpu_model->tensors);
}

void InferenceContext::InitRecordableQueue(Environment* env) {
  std::vector<ClOperation*> ops(nodes_.size());
  for (int i = 0; i < nodes_.size(); ++i) {
    ops[i] = &nodes_[i].cl_operation;
  }
  recordable_queue_ = CreateRecordableQueue(ops, env->device(), env->context());
}

absl::Status InferenceContext::InitFromGraphWithTransforms(
    const CreateGpuModelInfo& create_info, GraphFloat32* graph,
    Environment* env, std::vector<uint8_t>* serialized_model) {
  RETURN_IF_ERROR(RunGraphTransforms(graph));
  RETURN_IF_ERROR(InitFromGraph(create_info, *graph, env, serialized_model));
  return absl::OkStatus();
}

void InferenceContext::GetUsages(const std::function<bool(ValueId)>& functor,
                                 std::map<ValueId, int2>* usages) {
  for (ValueId in_id : input_ids_) {
    if (functor(in_id)) {
      AddUsage(in_id, 0, usages);
    }
  }
  for (int op_index = 0; op_index < nodes_.size(); ++op_index) {
    auto tensors = GetCLNodeTensors(nodes_[op_index]);
    for (auto& tensor : tensors) {
      if (functor(tensor.first)) {
        AddUsage(tensor.first, op_index, usages);
      }
    }
  }
  for (ValueId out_id : output_ids_) {
    if (functor(out_id)) {
      AddUsage(out_id, nodes_.size(), usages);
    }
  }
}

InferenceContext::TensorMemoryType InferenceContext::GetTensorMemoryType(
    const GpuInfo& gpu_info, ValueId id) {
  if (external_immutable_tensors_.find(id) !=
      external_immutable_tensors_.end()) {
    return TensorMemoryType::kExternal;
  } else if (external_mutable_tensors_.find(id) !=
             external_mutable_tensors_.end()) {
    return TensorMemoryType::kExternal;
  } else if (const_tensors_.find(id) != const_tensors_.end()) {
    return TensorMemoryType::kConst;
  } else if (variable_ids_and_refs_.find(id) != variable_ids_and_refs_.end()) {
    return TensorMemoryType::kVariable;
  } else if (IsBufferBased(gpu_info, tensors_descs_[id].storage_type)) {
    return TensorMemoryType::kBuffer;
  } else {
    return TensorMemoryType::kStrongShape;
  }
}

absl::Status InferenceContext::AllocateMemory(const GpuInfo& gpu_info,
                                              CLContext* context) {
  RETURN_IF_ERROR(AllocateMemoryForConstTensors(context));
  RETURN_IF_ERROR(AllocateMemoryForVariableTensors(context));
  RETURN_IF_ERROR(AllocateMemoryForBuffers(gpu_info, context));
  RETURN_IF_ERROR(AllocateMemoryForStrongShapes(gpu_info, context));
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateMemoryForConstTensors(
    CLContext* context) {
  for (auto& description : const_tensors_descs_) {
    RETURN_IF_ERROR(const_tensors_[description.first].CreateFromDescriptor(
        description.second, context));
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateMemoryForVariableTensors(
    CLContext* context) {
  std::map<ValueId, int> ref_value_to_tensor_index;

  for (auto value_and_ref_value : variable_ids_and_refs_) {
    if (ref_value_to_tensor_index.find(value_and_ref_value.second) ==
        ref_value_to_tensor_index.end()) {
      const auto& t = tensors_descs_[value_and_ref_value.first];
      const auto& shape = t.shape;
      const auto& descriptor = t;

      RETURN_IF_ERROR(
          CreateTensor(*context, shape, descriptor,
                       &variable_tensors_[value_and_ref_value.second]));
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateMemoryForBuffers(const GpuInfo& gpu_info,
                                                        CLContext* context) {
  std::map<ValueId, int2> buffer_usages;
  GetUsages(
      [this, &gpu_info](ValueId id) {
        return GetTensorMemoryType(gpu_info, id) == TensorMemoryType::kBuffer;
      },
      &buffer_usages);

  std::vector<TensorUsageRecord<size_t>> buffer_usage_records;
  for (auto& usage : buffer_usages) {
    const auto& t = tensors_descs_[usage.first];
    const auto& shape = t.shape;
    const auto& descriptor = t;
    const size_t element_size = SizeOf(descriptor.data_type);
    size_t buffer_size;
    if (descriptor.storage_type == TensorStorageType::TEXTURE_2D ||
        descriptor.storage_type == TensorStorageType::SINGLE_TEXTURE_2D) {
      const size_t bytes_per_pixel =
          element_size *
          (descriptor.storage_type == TensorStorageType::TEXTURE_2D ? 4
                                                                    : shape.c);
      const size_t width = shape.b * shape.w;
      const size_t height = shape.h * DivideRoundUp(shape.c, 4);
      size_t width_pixel_alignment = gpu_info.opencl_info.image_pitch_alignment;
      if (gpu_info.IsAdreno() && width_pixel_alignment % bytes_per_pixel == 0) {
        width_pixel_alignment /= bytes_per_pixel;
      }
      const size_t width_aligned = AlignByN(width, width_pixel_alignment);
      buffer_size = width_aligned * bytes_per_pixel * height;
    } else {
      buffer_size =
          shape.b * shape.w * shape.h * AlignByN(shape.c, 4) * element_size;
    }
    graph_ids_to_shared_buffer_tensors_[usage.first] =
        buffer_usage_records.size();
    buffer_usage_records.push_back({buffer_size,
                                    static_cast<TaskId>(usage.second.x),
                                    static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<size_t> buffer_assignment;
  RETURN_IF_ERROR(AssignObjectsToTensors(
      buffer_usage_records, MemoryStrategy::GREEDY_BEST, &buffer_assignment));

  size_t base_align_bytes =
      std::max<size_t>(gpu_info.opencl_info.base_addr_align_in_bits >> 3, 1);
  bool use_offset_assignment = false;

  OffsetsAssignment offset_assignment;
  if (CanUseSubBuffer(gpu_info)) {
    RETURN_IF_ERROR(AssignOffsetsToTensors(
        buffer_usage_records, MemoryStrategy::GREEDY_BY_SIZE,
        &offset_assignment, base_align_bytes));
    if (offset_assignment.total_size < TotalSize(buffer_assignment) &&
        offset_assignment.total_size <= gpu_info.GetMaxBufferSize()) {
      use_offset_assignment = true;
    }
  }

  if (use_offset_assignment) {
    shared_buffers_.resize(offset_assignment.offsets.size());
    RETURN_IF_ERROR(CreateReadWriteBuffer(offset_assignment.total_size, context,
                                          &shared_buffers_parent_));
    for (int i = 0; i < offset_assignment.offsets.size(); ++i) {
      RETURN_IF_ERROR(CreateReadWriteSubBuffer(
          shared_buffers_parent_, offset_assignment.offsets[i],
          buffer_usage_records[i].tensor_size, context, &shared_buffers_[i]));
    }
  } else {
    shared_buffers_.resize(buffer_assignment.object_sizes.size());
    for (int i = 0; i < buffer_assignment.object_sizes.size(); ++i) {
      RETURN_IF_ERROR(CreateReadWriteBuffer(buffer_assignment.object_sizes[i],
                                            context, &shared_buffers_[i]));
    }
  }

  std::vector<bool> created_tensors(buffer_usage_records.size(), false);
  shared_buffer_tensors_.resize(buffer_usage_records.size());
  for (auto& node : nodes_) {
    auto tensors = GetCLNodeTensors(node);
    for (auto& t : tensors) {
      if (GetTensorMemoryType(gpu_info, t.first) != TensorMemoryType::kBuffer)
        continue;
      const int tensor_index = graph_ids_to_shared_buffer_tensors_[t.first];
      if (created_tensors[tensor_index]) continue;
      const auto& shape_5d = tensors_descs_[t.first].shape;
      const auto shape = BHWC(shape_5d.b, shape_5d.h, shape_5d.w, shape_5d.c);
      const int buffer_index = use_offset_assignment
                                   ? tensor_index
                                   : buffer_assignment.object_ids[tensor_index];
      if (t.second.storage_type == TensorStorageType::TEXTURE_2D ||
          t.second.storage_type == TensorStorageType::SINGLE_TEXTURE_2D) {
        const size_t bytes_per_pixel =
            SizeOf(t.second.data_type) *
            (t.second.storage_type == TensorStorageType::TEXTURE_2D ? 4
                                                                    : shape.c);
        size_t width_pixel_alignment =
            gpu_info.opencl_info.image_pitch_alignment;
        if (gpu_info.IsAdreno() &&
            width_pixel_alignment % bytes_per_pixel == 0) {
          width_pixel_alignment /= bytes_per_pixel;
        }
        RETURN_IF_ERROR(CreateSharedImage2DBufferTensor(
            *context, shared_buffers_[buffer_index].GetMemoryPtr(), shape,
            t.second, width_pixel_alignment,
            &shared_buffer_tensors_[tensor_index]));
      } else {
        RETURN_IF_ERROR(CreateSharedTensor(
            *context, shared_buffers_[buffer_index].GetMemoryPtr(), shape,
            t.second, &shared_buffer_tensors_[tensor_index]));
      }
      created_tensors[tensor_index] = true;
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateMemoryForStrongShapes(
    const GpuInfo& gpu_info, CLContext* context) {
  std::map<ValueId, int2> usages;
  GetUsages(
      [this, &gpu_info](ValueId id) {
        return GetTensorMemoryType(gpu_info, id) ==
               TensorMemoryType::kStrongShape;
      },
      &usages);

  struct TensorDescComparator {
    TensorDescriptor tensor_desc;

    bool operator==(const TensorDescComparator& t) const {
      return tensor_desc.data_type == t.tensor_desc.data_type &&
             tensor_desc.storage_type == t.tensor_desc.storage_type &&
             tensor_desc.layout == t.tensor_desc.layout &&
             tensor_desc.shape == t.tensor_desc.shape;
    }
  };

  std::vector<TensorUsageRecord<TensorDescComparator>> usage_records;
  std::map<ValueId, ValueId> remap_from_graph_ids;
  for (auto& usage : usages) {
    remap_from_graph_ids[usage.first] = usage_records.size();
    usage_records.push_back({{tensors_descs_[usage.first]},
                             static_cast<TaskId>(usage.second.x),
                             static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<TensorDescComparator> assignment;
  RETURN_IF_ERROR(AssignObjectsToTensors(
      usage_records, MemoryStrategy::EQUALITY, &assignment));

  for (auto& node : nodes_) {
    auto tensors = GetCLNodeTensors(node);
    for (auto& t : tensors) {
      if (GetTensorMemoryType(gpu_info, t.first) !=
          TensorMemoryType::kStrongShape) {
        continue;
      }
      const auto& shape = tensors_descs_[t.first].shape;
      const auto id = assignment.object_ids[remap_from_graph_ids[t.first]];
      graph_ids_to_strong_shape_tensors_[t.first] = id;
      const auto& it = strong_shape_tensors_.find(id);
      if (it == strong_shape_tensors_.end()) {
        RETURN_IF_ERROR(CreateTensor(*context, shape, t.second,
                                     &strong_shape_tensors_[id]));
      }
    }
  }
  return absl::OkStatus();
}

void InferenceContext::BindMemoryToOperations() {
  for (auto& node : nodes_) {
    for (int i = 0; i < node.inputs.size(); ++i) {
      node.cl_operation.GetGpuOperation().SetSrc(GetTensor(node.inputs[i]), i);
    }
    for (int i = 0; i < node.outputs.size(); ++i) {
      node.cl_operation.GetGpuOperation().SetDst(GetTensor(node.outputs[i]), i);
    }
  }
}

absl::Status InferenceContext::Compile(
    const CreationContext& creation_context) {
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.cl_operation.Compile(creation_context));
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::Tune(TuningType tuning_type,
                                    const GpuInfo& gpu_info,
                                    ProfilingCommandQueue* profiling_queue) {
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(
        node.cl_operation.Tune(tuning_type, gpu_info, profiling_queue));
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::UpdateParams() {
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.cl_operation.UpdateParams());
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::SetTensor(const ValueId& tensor_id,
                                         Tensor* tensor_ptr) {
  auto it = external_mutable_tensors_.find(tensor_id);
  if (it == external_mutable_tensors_.end()) {
    return absl::InvalidArgumentError("No external tensor with this id.");
  }
  external_mutable_tensors_[tensor_id] = tensor_ptr;
  for (int node_index : external_tensor_to_nodes_[tensor_id]) {
    auto& node = nodes_[node_index];
    for (int i = 0; i < node.inputs.size(); ++i) {
      if (node.inputs[i] == tensor_id) {
        RETURN_IF_ERROR(node.cl_operation.SetSrcTensor(i, tensor_ptr));
      }
    }
    for (int i = 0; i < node.outputs.size(); ++i) {
      if (node.outputs[i] == tensor_id) {
        RETURN_IF_ERROR(node.cl_operation.SetDstTensor(i, tensor_ptr));
      }
    }
  }
  return absl::OkStatus();
}

void InferenceContext::PrepareExternal() {
  for (auto& external : external_mutable_tensors_) {
    for (int i = 0; i < nodes_.size(); ++i) {
      bool has_tensor = false;
      const auto& src_ids = nodes_[i].inputs;
      for (int i = 0; i < src_ids.size(); ++i) {
        if (src_ids[i] == external.first) {
          has_tensor = true;
        }
      }
      const auto& dst_ids = nodes_[i].outputs;
      for (int i = 0; i < dst_ids.size(); ++i) {
        if (dst_ids[i] == external.first) {
          has_tensor = true;
        }
      }
      if (has_tensor) {
        external_tensor_to_nodes_[external.first].push_back(i);
      }
    }
  }
}

absl::Status InferenceContext::AddToQueue(CLCommandQueue* queue) {
  if (recordable_queue_ && recordable_queue_->IsSupported()) {
    return recordable_queue_->Execute(queue);
  }
  if (execution_hints_.need_manual_release) {
    if (execution_hints_.prev_enqueue_start_point.is_valid()) {
      execution_hints_.prev_enqueue_start_point.Wait();
    }
    RETURN_IF_ERROR(
        queue->EnqueueEvent(&execution_hints_.prev_enqueue_start_point));
  }
  int counter = 0;
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.cl_operation.AddToQueue(queue));
    counter++;
    if (execution_hints_.flush_periodically &&
        counter % execution_hints_.flush_period == 0) {
      clFlush(queue->queue());
    }
  }
  if (execution_hints_.need_flush) {
    clFlush(queue->queue());
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::ProfileTime(ProfilingCommandQueue* queue,
                                           ProfilingInfo* result) {
  queue->ResetMeasurements();
  for (auto& node : nodes_) {
    queue->SetEventsLabel(node.name);
    RETURN_IF_ERROR(node.cl_operation.AddToQueue(queue));
  }
  RETURN_IF_ERROR(queue->WaitForCompletion());
  *result = queue->GetProfilingInfo();

  if (!(gpu_info_.IsMali() || gpu_info_.IsPowerVR())) {
    return absl::OkStatus();
  }

  if (gpu_info_.IsMali()) {
    queue->ResetMeasurements();
    for (int i = 0; i < nodes_.size(); ++i) {
      queue->SetEventsLabel(nodes_[i].name);
      const double times =
          16.0 / absl::ToDoubleMilliseconds(result->dispatches[i].duration);
      const int n = std::min(256.0, std::max(2.0, times));
      RETURN_IF_ERROR(nodes_[i].cl_operation.AddToQueueNTimes(queue, n));
    }
    RETURN_IF_ERROR(queue->WaitForCompletion());
    *result = queue->GetProfilingInfo();
    return absl::OkStatus();
  }

  if (gpu_info_.IsPowerVR()) {
    queue->ResetMeasurements();
    for (int i = 0; i < nodes_.size(); ++i) {
      queue->SetEventsLabel(nodes_[i].name);
      const double times =
          32.0 / absl::ToDoubleMilliseconds(result->dispatches[i].duration);
      const int n = std::min(64.0, std::max(4.0, times));
      RETURN_IF_ERROR(nodes_[i].cl_operation.AddToQueueNTimes(queue, n));
    }
    RETURN_IF_ERROR(queue->WaitForCompletion());
    *result = queue->GetProfilingInfo();

    queue->ResetMeasurements();
    for (int i = 0; i < nodes_.size(); ++i) {
      queue->SetEventsLabel(nodes_[i].name);
      const double times =
          128.0 / absl::ToDoubleMilliseconds(result->dispatches[i].duration);
      const int n = std::min(1024.0, std::max(4.0, times));
      RETURN_IF_ERROR(nodes_[i].cl_operation.AddToQueueNTimes(queue, n));
    }
    RETURN_IF_ERROR(queue->WaitForCompletion());
    *result = queue->GetProfilingInfo();
    return absl::OkStatus();
  }

  return absl::OkStatus();
}

absl::Status InferenceContext::Profile(ProfilingCommandQueue* queue,
                                       ProfilingInfo* result) {
  RETURN_IF_ERROR(ProfileTime(queue, result));
  for (int i = 0; i < nodes_.size(); ++i) {
    uint64_t read_size = 0;
    for (auto& src_id : nodes_[i].inputs) {
      read_size += GetTensor(src_id)->GetMemorySizeInBytes();
    }
    const auto& gpu_op = nodes_[i].cl_operation.GetGpuOperation();
    read_size += gpu_op.const_args_size_;
    uint64_t write_size = 0;
    for (auto& dst_id : nodes_[i].outputs) {
      write_size += GetTensor(dst_id)->GetMemorySizeInBytes();
    }
    result->dispatches[i].flops = gpu_op.flops_;
    result->dispatches[i].read_mem_size = read_size;
    result->dispatches[i].write_mem_size = write_size;
  }

  return absl::OkStatus();
}

uint64_t InferenceContext::GetSizeOfMemoryAllocatedForIntermediateTensors()
    const {
  uint64_t total_memory = 0;
  for (const auto& t : strong_shape_tensors_) {
    total_memory += t.second.GetMemorySizeInBytes();
  }
  for (const auto& b : shared_buffers_) {
    // Sub-buffers do not allocate memory. Count the size of the parent buffer
    // object instead.
    if (!b.IsSubBuffer()) {
      total_memory += b.GetMemorySizeInBytes();
    }
  }
  for (const auto& t : variable_tensors_) {
    total_memory += t.second.GetMemorySizeInBytes();
  }
  total_memory += shared_buffers_parent_.GetMemorySizeInBytes();

  return total_memory;
}

Tensor* InferenceContext::GetTensor(ValueId id) {
  if (external_immutable_tensors_.find(id) !=
      external_immutable_tensors_.end()) {
    return external_immutable_tensors_[id];
  } else if (external_mutable_tensors_.find(id) !=
             external_mutable_tensors_.end()) {
    return external_mutable_tensors_[id];
  } else if (const_tensors_.find(id) != const_tensors_.end()) {
    return &const_tensors_[id];
  } else if (variable_ids_and_refs_.find(id) != variable_ids_and_refs_.end()) {
    return &variable_tensors_[variable_ids_and_refs_[id]];
  } else if (graph_ids_to_shared_buffer_tensors_.find(id) !=
             graph_ids_to_shared_buffer_tensors_.end()) {
    return &shared_buffer_tensors_[graph_ids_to_shared_buffer_tensors_[id]];
  } else {
    return &strong_shape_tensors_[graph_ids_to_strong_shape_tensors_[id]];
  }
}

absl::Status InferenceContext::SetInputTensor(ValueId id,
                                              const TensorFloat32& tensor,
                                              CLCommandQueue* queue) {
  return GetTensor(id)->WriteData(queue, tensor);
}

absl::Status InferenceContext::GetOutputTensor(ValueId id,
                                               CLCommandQueue* queue,
                                               TensorFloat32* result) {
  const auto& gpu_tensor = *GetTensor(id);
  const auto dst_shape = BHWC(gpu_tensor.Batch(), gpu_tensor.Height(),
                              gpu_tensor.Width(), gpu_tensor.Channels());
  result->id = id;
  result->shape = dst_shape;
  result->data.resize(dst_shape.DimensionsProduct());
  return gpu_tensor.ReadData(queue, result);
}

void InferenceContext::ReleaseCPURepresentation() {
  for (auto& node : nodes_) {
    node.cl_operation.GetGpuOperation().args_.ReleaseCPURepresentation();
  }
  const_tensors_descs_.clear();
}

absl::Status RunGraphTransforms(GraphFloat32* graph) {
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

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
