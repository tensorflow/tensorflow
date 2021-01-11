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

#include "tensorflow/lite/delegates/gpu/metal/inference_context.h"

#include <map>
#include <string>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"
#include "tensorflow/lite/delegates/gpu/metal/selectors/operation_selector.h"
#include "tensorflow/lite/delegates/gpu/metal/selectors/subgraph.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {
void AddUsage(ValueId id, int task_index,
              std::map<ValueId, int2>* usage_records) {
  auto it = usage_records->find(id);
  if (it == usage_records->end()) {
    // initializing start index(.x) and end index(.y)
    (*usage_records)[id].x = task_index;
    (*usage_records)[id].y = task_index;
  } else {
    // updating end index(.y)
    (*usage_records)[id].y = task_index;
  }
}

// Generic add is add that have several runtime inputs and they are not
// broadcasted, i.e. pointwise add for N tensors where N > 1.
bool IsGenericAdd(const Node& node, const std::vector<Value*>& inputs,
                  const std::vector<Value*>& outputs) {
  if (inputs.size() == 1) {
    return false;
  }
  const OperationType op_type = OperationTypeFromString(node.operation.type);
  if (op_type != OperationType::ADD) {
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

// Allows to get result about the graph compilation to validate graph. This
// information helps to find a cause of performance degradation, like misfusing.
struct OptimizationInfo {
  // Initial operations count before compilation.
  int operations_count;
  // GPU tasks count after fusion and splitting complex operations into few GPU
  // subtasks.
  int gpu_tasks_count;
  // Some operations are not used due to dependencies of the graph.
  std::vector<int> unused_operations;
  // Used inputs.
  std::vector<ValueId> input_buffer_ids;
  // Unused inputs. Requested outputs do not require this inputs to be used.
  std::vector<ValueId> unused_input_buffer_ids;
  // The outputs are deducted by the graph but not requested by user.
  std::vector<ValueId> extra_output_buffer_ids;
  // Outputs that are requested but can't be calculated by the graph.
  std::vector<ValueId> missing_output_buffer_ids;
};

using FusionSequence = std::vector<NodeDescriptor>;

bool Contains(const std::vector<ValueId>& container, ValueId value) {
  return std::find(container.begin(), container.end(), value) !=
         container.end();
}

template <class T>
bool Contains(const std::vector<T>& container, ValueId value) {
  for (const auto& buffer : container) {
    if (buffer.id == value) {
      return true;
    }
  }
  return false;
}

// Checks if all elements of the narrow vector exist in the wide vector. Vectors
// are expected to be unsorted.
bool Contains(const std::vector<ValueId>& wide,
              const std::vector<ValueId>& narrow) {
  if (narrow.empty() || narrow.size() > wide.size()) {
    return false;
  }
  std::set<ValueId> wide_sorted;
  wide_sorted.insert(wide.begin(), wide.end());
  for (auto element : narrow) {
    if (std::find(wide.begin(), wide.end(), element) == wide.end()) {
      return false;
    }
  }
  return true;
}

uint32_t BufferUseCount(ValueId id,
                        const std::list<NodeDescriptor>& descriptors,
                        std::list<FusionSequence>* chains) {
  uint32_t use_count = 0;
  // Buffer may be read by both processed and not processed operations.
  for (auto& desc : descriptors) {
    if (Contains(desc.src_tensors_ids, id)) {
      use_count++;
    }
  }

  for (auto& chain : *chains) {
    for (auto& desc : chain) {
      if (Contains(desc.src_tensors_ids, id)) {
        use_count++;
      }
    }
  }
  return use_count;
}

// Examines if the second operation can be linked to the first one. Linking may
// be skipped in the situation when conflict may happen: if first operation's
// output is used by more than 1 other operation.
bool CanFuseOperations(const NodeDescriptor& first,
                       const NodeDescriptor& second,
                       const std::vector<ValueId>& output_ids,
                       const std::list<NodeDescriptor>& descriptors,
                       std::list<FusionSequence>* chains) {
  return second.task->is_linkable &&
         !Contains(output_ids, first.dst_tensors_ids[0]) &&
         BufferUseCount(first.dst_tensors_ids[0], descriptors, chains) == 1;
}

// Takes an unsorted list of task descriptors, builds a list of chains. Each
// chain is a list of task descriptors that can be fused into a single GPU task.
// Building is started from the input IDs and building statistic is filled.
void BuildFusableChains(const std::vector<ValueId>& input_ids,
                        const std::vector<ValueId>& output_ids,
                        std::list<NodeDescriptor>* descriptors,
                        std::list<FusionSequence>* chains,
                        std::vector<int>* unused_ids) {
  // Proxy tasks for inputs - only output is valid on this elements.
  for (auto input_id : input_ids) {
    auto desc = std::make_shared<ComputeTaskDescriptor>();
    desc->is_linkable = true;
    desc->AddDstTensor("", {});
    NodeDescriptor node;
    node.task = desc;
    node.dst_tensors_ids = {input_id};
    chains->push_back({node});
  }

  if (descriptors->empty()) return;
  // Get all possible operations - grow-up chains.
  bool added;
  do {
    // At least one element must be added to any chain at this step.
    added = false;
    for (auto it = descriptors->begin(); it != descriptors->end();) {
      const NodeDescriptor& task_descriptor = *it;

      // Gather all outputs of all chains to check with.
      std::vector<ValueId> ready_buffer_ids;
      ready_buffer_ids.reserve(chains->size());
      for (const auto& chain : *chains) {
        ready_buffer_ids.push_back(chain.back().dst_tensors_ids[0]);
      }

      // Check if all inputs of this operation are ready.
      if (Contains(ready_buffer_ids, task_descriptor.src_tensors_ids)) {
        // Now find a chain to fuse with.
        bool fused = false;
        for (auto& chain : *chains) {
          // We can fuse only single output for now.
          const bool can_link = task_descriptor.src_tensors_ids[0] ==
                                chain.back().dst_tensors_ids[0];
          if (can_link && CanFuseOperations(chain.back(), task_descriptor,
                                            output_ids, *descriptors, chains)) {
            chain.push_back(task_descriptor);
            fused = true;
            break;
          }
        }
        if (!fused) {
          chains->push_back({task_descriptor});
        }

        // Remove operation from original list and start from the beginning.
        descriptors->erase(it);
        added = true;
        break;
      } else {
        ++it;
      }
    }
  } while (!descriptors->empty() && added);

  unused_ids->reserve(descriptors->size());
  for (const auto& desc : *descriptors) {
    unused_ids->push_back(desc.id);
  }
}

// Accepts unsorted list of chains and returns sorted list with the order of GPU
// task execution.
std::list<FusionSequence> SortChains(
    const std::vector<ValueId>& graph_input_ids,
    std::list<FusionSequence>* chains) {
  std::list<FusionSequence> sorted_chains;
  while (!chains->empty()) {
    // Collect ready buffers.
    std::vector<ValueId> ready_buffer_ids;
    ready_buffer_ids.reserve(graph_input_ids.size() + sorted_chains.size());
    ready_buffer_ids.insert(ready_buffer_ids.begin(), graph_input_ids.begin(),
                            graph_input_ids.end());
    for (auto& chain : sorted_chains) {
      ready_buffer_ids.push_back(chain.back().dst_tensors_ids[0]);
    }

    for (auto it = chains->begin(); it != chains->end();) {
      const FusionSequence& chain = *it;

      // If the input is also is the output in the same chain - eliminate
      // because it used internally inside sthis chain only.
      std::vector<ValueId> elements_output_buffer_ids;
      elements_output_buffer_ids.reserve(chain.size());
      for (const auto& element : chain) {
        elements_output_buffer_ids.push_back(element.dst_tensors_ids[0]);
      }

      // Collect all inputs also for linked operations.
      std::vector<ValueId> elements_input_buffer_ids;
      for (const auto& element : chain) {
        for (const auto& id : element.src_tensors_ids) {
          if (!Contains(elements_output_buffer_ids, id)) {
            elements_input_buffer_ids.push_back(id);
          }
        }
      }

      if (Contains(ready_buffer_ids, elements_input_buffer_ids)) {
        // All input buffers for all elements of this chain are ready.
        sorted_chains.push_back(chain);
        it = chains->erase(it);
      } else {
        ++it;
      }
    }
  }
  return sorted_chains;
}

// If a graph structure contains unused outputs then it can lead to unused
// operations and unused input buffers. It's not an error but some sort of
// warning.
std::vector<ValueId> GetUsedInputBufferIds(
    const std::list<FusionSequence>& sorted_chains) {
  // Match requested outputs with all outputs and intermediate buffers.
  std::vector<ValueId> output_and_intermediate_ids;
  output_and_intermediate_ids.reserve(sorted_chains.size());
  std::set<ValueId> input_and_intermediate_ids;
  for (auto it = sorted_chains.begin(); it != sorted_chains.end(); ++it) {
    output_and_intermediate_ids.push_back(it->back().dst_tensors_ids[0]);
    for (const auto& id : it->front().src_tensors_ids) {
      input_and_intermediate_ids.insert(id);
    }
  }
  std::vector<ValueId> input_ids;
  for (ValueId id : input_and_intermediate_ids) {
    if (!Contains(output_and_intermediate_ids, id)) {
      input_ids.push_back(id);
    }
  }
  return input_ids;
}

// If a buffer is requested as output from the graph but the graph structure
// can't provide this buffer by output (can't deduct), that means the graph
// structure is incorrect.
std::vector<ValueId> GetMissingOutputBufferIds(
    const std::vector<ValueId>& output_ids,
    const std::list<FusionSequence>& sorted_chains) {
  // Match requested outputs with all output and intermediate buffers.
  std::vector<ValueId> output_and_intermediate_ids;
  output_and_intermediate_ids.reserve(sorted_chains.size());
  for (auto it = sorted_chains.begin(); it != sorted_chains.end(); ++it) {
    output_and_intermediate_ids.push_back(it->back().dst_tensors_ids[0]);
  }
  std::vector<ValueId> missing_output_ids;
  for (ValueId id : output_ids) {
    if (!Contains(output_and_intermediate_ids, id)) {
      missing_output_ids.push_back(id);
    }
  }
  return missing_output_ids;
}

// Graph may contain leafs with outputs that are not requested. It wastes GPU
// computations.
std::vector<ValueId> DeductOutputBufferIds(
    const std::vector<ValueId>& output_ids,
    const std::list<FusionSequence>& sorted_chains) {
  std::vector<ValueId> extra_output_ids;
  // Detect all unused output buffers - all outputs.
  for (auto it1 = sorted_chains.begin(); it1 != sorted_chains.end(); ++it1) {
    bool found_as_input = false;
    for (auto it2 = sorted_chains.begin(); it2 != sorted_chains.end(); ++it2) {
      if (it1 != it2) {
        std::vector<ValueId> input_ids;
        for (const auto& element : *it2) {
          for (const auto& id : element.src_tensors_ids) {
            input_ids.push_back(id);
          }
        }
        if (Contains(input_ids, it1->back().dst_tensors_ids[0])) {
          found_as_input = true;
          break;
        }
      }
    }
    if (!found_as_input) {
      if (!Contains(output_ids, it1->back().dst_tensors_ids[0])) {
        extra_output_ids.push_back(it1->back().dst_tensors_ids[0]);
      }
    }
  }
  return extra_output_ids;
}

// Delete all unused task descriptors that have non-requested outputs.
// !delete not the whole chain but only the last element, then others.!
std::vector<int> DeleteUnusedTasks(const std::vector<ValueId>& output_ids,
                                   std::list<FusionSequence>* chains) {
  std::vector<int> unused_operations;
  for (auto it1 = chains->rbegin(); it1 != chains->rend();) {
    // Don't delete if output is requested.
    if (Contains(output_ids, it1->back().dst_tensors_ids[0])) {
      ++it1;
      continue;
    }

    // Don't delete if some operation uses the output.
    bool output_used = false;
    for (auto it2 = chains->rbegin(); it2 != chains->rend(); ++it2) {
      std::vector<ValueId> input_ids;
      for (const auto& element : *it2) {
        for (const auto& id : element.src_tensors_ids) {
          input_ids.push_back(id);
        }
      }
      if (Contains(input_ids, it1->back().dst_tensors_ids[0])) {
        output_used = true;
        break;
      }
    }
    if (output_used) {
      ++it1;
      continue;
    }
    // Delete if not used.
    unused_operations.push_back(it1->back().id);
    it1 = decltype(it1){chains->erase(std::next(it1).base())};
  }
  return unused_operations;
}

// Returns unused input buffer IDs.
void RemoveInputProxies(std::list<FusionSequence>* chains) {
  // Remove input proxy and sort items.
  for (auto it = chains->begin(); it != chains->end();) {
    auto& chain = *it;
    // Remove input proxy-operations.
    if (chain.front().src_tensors_ids.empty()) {
      chain.erase(chain.begin());
    }
    if (chain.empty()) {
      // Input proxy operation has been deleted and the chain is empty due to
      // unused input buffer.
      it = chains->erase(it);
    } else {
      ++it;
    }
  }
}

absl::Status MergeNodes(const NodeDescriptor* src, NodeDescriptor* dst) {
  for (int j = 1; j < src->src_tensors_ids.size(); ++j) {
    dst->src_tensors_ids.push_back(src->src_tensors_ids[j]);
  }
  dst->dst_tensors_ids[0] = src->dst_tensors_ids[0];
  dst->description += " linked : " + src->description;
  return dst->task->AddTask(src->task.get());
}

absl::Status FuseChain(const FusionSequence& chain, NodeDescriptor* node_desc) {
  *node_desc = chain.front();
  for (int j = 1; j < chain.size(); ++j) {
    RETURN_IF_ERROR(MergeNodes(&chain[j], node_desc));
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status InferenceContext::InitFromGraph(
    const CreateInferenceInfo& create_info, const GraphFloat32& graph,
    id<MTLDevice> device) {
  const auto inputs = graph.inputs();
  for (const auto& input : inputs) {
    input_ids_.push_back(input->id);
  }

  const auto outputs = graph.outputs();
  for (const auto& output : outputs) {
    output_ids_.push_back(output->id);
  }

  std::string device_name = std::string([[device name] UTF8String]);
  GpuInfo gpu_info;
  GetGpuInfoFromDeviceDescription(device_name, GpuApi::kMetal, &gpu_info);
  CompiledModel compiled_model;
  RETURN_IF_ERROR(
      Compile(graph, gpu_info, create_info.precision, &compiled_model));
  CompiledModel optimized_model;
  RETURN_IF_ERROR(ValidateOptimizeModel(input_ids_, output_ids_, compiled_model,
                                        &optimized_model));

  RETURN_IF_ERROR(CompileModelWithDevice(device, optimized_model, input_ids_,
                                         output_ids_, create_info.precision));
  return absl::OkStatus();
}

absl::Status InferenceContext::Compile(const GraphFloat32& graph,
                                       const GpuInfo& gpu_info,
                                       CalculationsPrecision precision,
                                       CompiledModel* compiled_model) {
  if (!IsBatchMatchesForAllValues(graph)) {
    return absl::InvalidArgumentError(
        "Only identical batch dimension is supported");
  }
  int last_value_id = 0;
  for (const auto& value : graph.values()) {
    compiled_model->tensor_shapes[value->id] = value->tensor.shape;
    last_value_id = std::max(last_value_id, static_cast<int>(value->id));
  }
  std::map<ValueId, int>
      tensor_usages;  // keeps latest index of operation that updated tensor
  for (const auto& input_id : input_ids_) {
    tensor_usages[input_id] = -1;  // so as inputs "updated" before operation 0,
                                   // we will mark them with -1
  }
  std::vector<Node*> graph_nodes = graph.nodes();
  for (int i = 0; i < graph_nodes.size(); ++i) {
    const Node& node = *graph_nodes[i];
    auto inputs = graph.FindInputs(node.id);
    auto outputs = graph.FindOutputs(node.id);
    // Reordering of input ids and updating of temporary tensors_usage struct.
    // This stage is necessary because we are building OperationDef that rely
    // on order of input ids. But we also should have input id on first
    // position that potentially can be "linking" tensor and as result
    // eliminated(unused) We apply it only for ADD operation, because of ADD
    // associativity and ADD can be linked. In current approach "linking"
    // tensor can be only latest written tensor(during linear order of
    // execution) among input tensors.
    // TODO(b/176397043) sorokin, check failure on segmentation model
    // if (IsGenericAdd(node, inputs, outputs)) {
    //   int latest_written_tensor_index = 0;
    //   int last_usage = tensor_usages[inputs[0]->id];
    //   for (int j = 1; j < inputs.size(); ++j) {
    //     if (tensor_usages[inputs[j]->id] > last_usage) {
    //       last_usage = tensor_usages[inputs[j]->id];
    //       latest_written_tensor_index = j;
    //     }
    //   }
    //   std::swap(inputs[0], inputs[latest_written_tensor_index]);
    // }
    DataType data_type = DeduceDataTypeFromPrecision(precision);
    TensorDescriptor tensor_descriptor =
        TensorDescriptor{data_type, TensorStorageType::BUFFER, Layout::HWC};
    OperationDef op_def;
    op_def.precision = precision;
    for (int j = 0; j < inputs.size(); ++j) {
      op_def.src_tensors.push_back(tensor_descriptor);
    }
    for (int j = 0; j < outputs.size(); ++j) {
      op_def.dst_tensors.push_back(tensor_descriptor);
    }
    GPUOperationsSubgraph gpu_subgraph;
    RETURN_IF_ERROR(GPUOperationFromNode(gpu_info, op_def, inputs, outputs,
                                         node, &gpu_subgraph));
    std::map<int, ValueId> mapping_to_global_ids;
    for (int j = 0; j < gpu_subgraph.new_tensors.size(); ++j) {
      const auto& t = gpu_subgraph.new_tensors[j];
      last_value_id++;
      compiled_model->tensor_shapes[last_value_id] = t.first;
      mapping_to_global_ids[j] = last_value_id;
    }
    for (auto& gpu_op : gpu_subgraph.operations) {
      NodeDescriptor metal_node;
      metal_node.task = std::move(gpu_op.operation);
      metal_node.src_tensors_ids.resize(gpu_op.input_ids.size());
      for (int j = 0; j < gpu_op.input_ids.size(); ++j) {
        int id = gpu_op.input_ids[j];
        if (id >= 0) {
          metal_node.src_tensors_ids[j] = id;
        } else {
          metal_node.src_tensors_ids[j] = mapping_to_global_ids[-(id + 1)];
        }
      }
      metal_node.dst_tensors_ids.resize(gpu_op.output_ids.size());
      for (int j = 0; j < gpu_op.output_ids.size(); ++j) {
        int id = gpu_op.output_ids[j];
        if (id >= 0) {
          metal_node.dst_tensors_ids[j] = id;
          tensor_usages[id] = i;
        } else {
          metal_node.dst_tensors_ids[j] = mapping_to_global_ids[-(id + 1)];
        }
      }
      metal_node.description =
          node.operation.type + " " + std::to_string(node.id);
      metal_node.id = i;
      compiled_model->nodes.push_back(std::move(metal_node));
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::ValidateOptimizeModel(
    const std::vector<ValueId>& input_buffers,
    const std::vector<ValueId>& output_buffers,
    const CompiledModel& input_model, CompiledModel* output_model) {
  std::list<NodeDescriptor> input;
  input.insert(input.end(), input_model.nodes.begin(), input_model.nodes.end());
  OptimizationInfo info;
  info.operations_count = static_cast<int>(input.size());

  // A chain is a sequence of fusable operations. All internal outputs are
  // consumed with the next element of the chain. The last element of each chain
  // contains outputs which are ready to be used as inputs. if a chain can't be
  // extended with linkable element then new chain is created.
  std::list<FusionSequence> unsorted_chains;
  BuildFusableChains(input_buffers, output_buffers, &input, &unsorted_chains,
                     &info.unused_operations);

  RemoveInputProxies(&unsorted_chains);
  std::list<FusionSequence> sorted_chains =
      SortChains(input_buffers, &unsorted_chains);

  info.extra_output_buffer_ids =
      DeductOutputBufferIds(output_buffers, sorted_chains);
  info.unused_operations = DeleteUnusedTasks(output_buffers, &sorted_chains);
  info.input_buffer_ids = GetUsedInputBufferIds(sorted_chains);
  // find provided input buffers that has not being used
  for (ValueId id : input_buffers) {
    if (!Contains(info.input_buffer_ids, id)) {
      info.unused_input_buffer_ids.push_back(id);
    }
  }
  info.missing_output_buffer_ids =
      GetMissingOutputBufferIds(output_buffers, sorted_chains);
  info.gpu_tasks_count = static_cast<int>(sorted_chains.size());
  if (sorted_chains.empty()) {
    const std::string message =
        "No valid operations in the graph.\nInput operations count " +
        std::to_string(info.operations_count) + "\nUnused operations " +
        std::to_string(info.unused_operations.size()) + "\nUnused inputs " +
        std::to_string(info.unused_input_buffer_ids.size()) +
        "\nMissing output buffers " +
        std::to_string(info.missing_output_buffer_ids.size());
    return absl::InternalError(message);
  }
  for (const auto& chain : sorted_chains) {
    NodeDescriptor fused_node;
    RETURN_IF_ERROR(FuseChain(chain, &fused_node));
    output_model->nodes.push_back(std::move(fused_node));
  }
  output_model->tensor_shapes = input_model.tensor_shapes;
  return absl::OkStatus();
}

absl::Status InferenceContext::CompileModelWithDevice(
    id<MTLDevice> device, const CompiledModel& compiled_model,
    const std::vector<ValueId>& input_ids,
    const std::vector<ValueId>& output_ids, CalculationsPrecision precision) {
  input_ids_ = input_ids;
  output_ids_ = output_ids;
  precision_ = precision;
  // Metal resources are created here.
  for (const auto& node : compiled_model.nodes) {
    ComputeTask task;
    RETURN_IF_ERROR(task.CompileWithDevice(device, node, precision_));
    task.SetDescription(node.description);
    compute_tasks_.emplace_back(std::move(task));
  }
  tensor_shapes_ = compiled_model.tensor_shapes;
  for (auto& task : compute_tasks_) {
    // The same device must be used here as well as on shader compilation stage.
    RETURN_IF_ERROR(task.UpdateParamsWithDevice(device, tensor_shapes_));
  }
  RETURN_IF_ERROR(AllocateTensors(device));
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateTensors(id<MTLDevice> device) {
  std::set<ValueId> preallocated_ids;
  for (auto tensor_id : input_ids_) {
    preallocated_ids.insert(tensor_id);
  }
  for (const auto& outputId : output_ids_) {
    preallocated_ids.insert(outputId);
  }
  for (int i = 0; i < compute_tasks_.size(); ++i) {
    auto& task = compute_tasks_[i];
    if (task.HasInOutIds(preallocated_ids)) {
      task_ids_with_preallocated_tensors_.push_back(i);
    }
  }

  const bool f32_storage = precision_ == CalculationsPrecision::F32;
  for (auto& tensor_id : preallocated_ids) {
    BHWC shape = tensor_shapes_[tensor_id];
    TensorDescriptor descriptor;
    descriptor.storage_type = TensorStorageType::BUFFER;
    descriptor.data_type = f32_storage ? DataType::FLOAT32 : DataType::FLOAT16;
    descriptor.layout = Layout::HWC;
    preallocated_tensors_[tensor_id] =
        CreateSharedBufferTensor(nil, shape, descriptor);
  }

  RETURN_IF_ERROR(AllocateMemoryForBuffers(device));
  BindTensorsToOperations();
  return absl::OkStatus();
}

MetalSpatialTensor* InferenceContext::GetTensor(ValueId tensor_id) {
  if (preallocated_tensors_.find(tensor_id) != preallocated_tensors_.end()) {
    return &preallocated_tensors_[tensor_id];
  } else if (graph_ids_to_shared_buffer_tensors_.find(tensor_id) !=
             graph_ids_to_shared_buffer_tensors_.end()) {
    return &shared_buffer_tensors_
        [graph_ids_to_shared_buffer_tensors_[tensor_id]];
  }
  return nullptr;
}

void InferenceContext::BindTensorsToOperations() {
  for (auto& task : compute_tasks_) {
    const auto& src_ids = task.GetInputIds();
    for (int i = 0; i < src_ids.size(); ++i) {
      MetalSpatialTensor* tensor = GetTensor(src_ids[i]);
      task.SetSrcTensor(*tensor, i);
    }
    const auto& dst_ids = task.GetOutputIds();
    for (int i = 0; i < dst_ids.size(); ++i) {
      MetalSpatialTensor* tensor = GetTensor(dst_ids[i]);
      task.SetDstTensor(*tensor, i);
    }
  }
}

void InferenceContext::GetUsages(std::map<ValueId, int2>* usages) {
  for (ValueId in_id : input_ids_) {
    if (preallocated_tensors_.find(in_id) == preallocated_tensors_.end()) {
      AddUsage(in_id, 0, usages);
    }
  }
  for (int op_index = 0; op_index < compute_tasks_.size(); ++op_index) {
    for (auto& tensor_id : compute_tasks_[op_index].GetInputIds()) {
      if (preallocated_tensors_.find(tensor_id) ==
          preallocated_tensors_.end()) {
        AddUsage(tensor_id, op_index, usages);
      }
    }
    for (auto& tensor_id : compute_tasks_[op_index].GetOutputIds()) {
      if (preallocated_tensors_.find(tensor_id) ==
          preallocated_tensors_.end()) {
        AddUsage(tensor_id, op_index, usages);
      }
    }
  }
  for (ValueId out_id : output_ids_) {
    if (preallocated_tensors_.find(out_id) == preallocated_tensors_.end()) {
      AddUsage(out_id, compute_tasks_.size(), usages);
    }
  }
}

absl::Status InferenceContext::AllocateMemoryForBuffers(id<MTLDevice> device) {
  std::map<ValueId, int2> buffer_usages;
  GetUsages(&buffer_usages);

  std::vector<TensorUsageRecord<size_t>> buffer_usage_records;
  for (auto& usage : buffer_usages) {
    const auto& shape = tensor_shapes_[usage.first];
    const size_t buffer_size =
        shape.b * shape.w * shape.h * AlignByN(shape.c, 4);
    graph_ids_to_shared_buffer_tensors_[usage.first] =
        buffer_usage_records.size();
    buffer_usage_records.push_back({buffer_size,
                                    static_cast<TaskId>(usage.second.x),
                                    static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<size_t> buffer_assignment;
  RETURN_IF_ERROR(AssignObjectsToTensors(
      buffer_usage_records, MemoryStrategy::GREEDY_BEST, &buffer_assignment));

  const bool f32_storage = precision_ == CalculationsPrecision::F32;
  size_t dataTypeSize = f32_storage ? sizeof(float) : sizeof(HalfBits);
  shared_buffers_.resize(buffer_assignment.object_sizes.size());
  for (int i = 0; i < buffer_assignment.object_sizes.size(); ++i) {
    // Initialize metal buffer
    NSUInteger bufferSize = dataTypeSize * buffer_assignment.object_sizes[i];

#if (defined(__MAC_10_14) &&                               \
     __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_14) ||    \
    (defined(__IPHONE_12_0) &&                             \
     __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_12_0) || \
    (defined(__TVOS_12_0) && __TV_OS_VERSION_MIN_REQUIRED >= __TVOS_12_0)
    if (bufferSize > [device maxBufferLength]) {
      std::string error("Tensor id: ");
      error += std::to_string(buffer_assignment.object_ids[i]) +
               " with size: " + std::to_string(bufferSize) +
               " exceeds MTLDevice maxBufferLength: " +
               std::to_string([device maxBufferLength]);
      return absl::ResourceExhaustedError(error);
    }
#endif
#if defined(__MAC_10_12) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_12
    if ([device currentAllocatedSize] + bufferSize >
        [device recommendedMaxWorkingSetSize]) {
      std::string error(
          "Out of memory in MTLBuffer allocation. Currently allocated: ");
      error += std::to_string([device currentAllocatedSize]);
      return absl::ResourceExhaustedError(error);
    }
#endif

    shared_buffers_[i] =
        [device newBufferWithLength:bufferSize
                            options:MTLResourceStorageModeShared];
  }

  std::vector<bool> created_tensors(buffer_usage_records.size(), false);
  shared_buffer_tensors_.resize(buffer_usage_records.size());
  TensorDescriptor descriptor;
  descriptor.storage_type = TensorStorageType::BUFFER;
  descriptor.data_type = f32_storage ? DataType::FLOAT32 : DataType::FLOAT16;
  descriptor.layout = Layout::HWC;
  for (auto& task : compute_tasks_) {
    const std::vector<ValueId> input_ids = task.GetInputIds();
    const std::vector<ValueId> output_ids = task.GetOutputIds();
    std::vector<ValueId> all_ids = input_ids;
    all_ids.insert(all_ids.end(), output_ids.begin(), output_ids.end());
    for (auto& tensor_id : all_ids) {
      if (preallocated_tensors_.find(tensor_id) != preallocated_tensors_.end())
        continue;
      const int tensor_index = graph_ids_to_shared_buffer_tensors_[tensor_id];
      if (created_tensors[tensor_index]) continue;
      const auto& shape = tensor_shapes_[tensor_id];
      const int buffer_index = buffer_assignment.object_ids[tensor_index];
      shared_buffer_tensors_[tensor_index] = CreateSharedBufferTensor(
          shared_buffers_[buffer_index], shape, descriptor);
      created_tensors[tensor_index] = true;
    }
  }
  return absl::OkStatus();
}

void InferenceContext::EncodeWithEncoder(
    id<MTLComputeCommandEncoder> command_encoder,
    const std::map<ValueId, id<MTLBuffer>>& in_out_buffers) {
  UpdatePreallocatedTensors(in_out_buffers);
  for (int i = 0; i < compute_tasks_.size(); ++i) {
    auto& task = compute_tasks_[i];
    task.EncodeWithEncoder(command_encoder);
  }
}

void InferenceContext::EncodeWithCommandBuffer(
    id<MTLCommandBuffer> command_buffer,
    const std::map<ValueId, id<MTLBuffer>>& in_out_buffers) {
  UpdatePreallocatedTensors(in_out_buffers);
  for (int i = 0; i < compute_tasks_.size(); ++i) {
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    auto& task = compute_tasks_[i];
    task.EncodeWithEncoder(encoder);
    [encoder endEncoding];
  }
}

void InferenceContext::EncodeWithCommandQueue(
    id<MTLCommandQueue> command_queue,
    const std::map<ValueId, id<MTLBuffer>>& in_out_buffers, int flush_period) {
  UpdatePreallocatedTensors(in_out_buffers);
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  for (int i = 0; i < compute_tasks_.size(); ++i) {
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    auto& task = compute_tasks_[i];
    task.EncodeWithEncoder(encoder);
    [encoder endEncoding];
    if (i % flush_period == (flush_period - 1)) {
      [command_buffer commit];
      command_buffer = [command_queue commandBuffer];
    }
  }
  [command_buffer commit];
}

void InferenceContext::UpdatePreallocatedTensors(
    const std::map<ValueId, id<MTLBuffer>>& preallocated) {
  for (const auto& it : preallocated) {
    preallocated_tensors_[it.first].SetBufferHandle(it.second);
  }
  for (auto& task_index : task_ids_with_preallocated_tensors_) {
    auto& task = compute_tasks_[task_index];
    const auto& src_ids = task.GetInputIds();
    for (int i = 0; i < src_ids.size(); ++i) {
      const auto& it = preallocated_tensors_.find(src_ids[i]);
      if (it != preallocated_tensors_.end()) {
        task.SetSrcTensor(it->second, i);
      }
    }
    const auto& dst_ids = task.GetOutputIds();
    for (int i = 0; i < dst_ids.size(); ++i) {
      const auto& it = preallocated_tensors_.find(dst_ids[i]);
      if (it != preallocated_tensors_.end()) {
        task.SetDstTensor(it->second, i);
      }
    }
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
