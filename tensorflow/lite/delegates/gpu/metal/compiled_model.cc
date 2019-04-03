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

#include "tensorflow/lite/delegates/gpu/metal/compiled_model.h"

#include <algorithm>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

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

using FusionSequence = std::vector<ComputeTaskDescriptorPtr>;

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

// Checks if all elements of the narrow vector exist in the wide vector. Vectors
// are expected to be unsorted.
bool Contains(
    const std::vector<ValueId>& wide,
    const std::vector<ComputeTaskDescriptor::InputBufferDescriptor>& buffers) {
  if (buffers.empty() || buffers.size() > wide.size()) {
    return false;
  }
  std::set<ValueId> wide_sorted(wide.begin(), wide.end());
  for (const auto& buffer : buffers) {
    if (!std::binary_search(wide_sorted.begin(), wide_sorted.end(),
                            buffer.id)) {
      return false;
    }
  }
  return true;
}

// Examines if the second operation can be linked to the first one. Linking may
// be skipped in the situation when conflic may happen: if first operation's
// output is used by more than 1 other operation.
bool CanFuseOperations(const ComputeTaskDescriptorPtr first,
                       const ComputeTaskDescriptorPtr second,
                       const std::vector<ValueId>& output_ids,
                       const std::list<ComputeTaskDescriptorPtr>& descriptors) {
  int use_count = 0;
  if (second->is_linkable && !Contains(output_ids, first->output_buffer.id)) {
    for (auto& desc : descriptors) {
      if (Contains(desc->input_buffers, first->output_buffer.id)) {
        use_count++;
      }
    }
  }
  return (use_count == 1);
}

// Takes an unsorted list of task descriptors, builds a list of chains. Each
// chain is a list of task descriptors that can be fused into a single GPU task.
// Building is started from the input IDs and building statistic is filled.
void BuildFusableChains(const std::vector<ValueId>& input_ids,
                        const std::vector<ValueId>& output_ids,
                        std::list<ComputeTaskDescriptorPtr>* descriptors,
                        std::list<FusionSequence>* chains,
                        std::vector<int>* unused_ids) {
  // Proxy tasks for inputs - only output is valid on this elements.
  for (auto input_id : input_ids) {
    auto desc = std::make_shared<ComputeTaskDescriptor>();
    desc->id = 0;
    desc->is_linkable = true;
    desc->output_buffer = {input_id};
    chains->push_back({desc});
  }

  if (descriptors->empty()) return;
  // Get all possible operations - grow-up chains.
  bool added;
  do {
    // At least one element must be added to any chain at this step.
    added = false;
    for (auto it = descriptors->begin(); it != descriptors->end();) {
      const ComputeTaskDescriptorPtr task_descriptor = *it;

      // Gather all outputs of all chains to check with.
      std::vector<ValueId> ready_buffer_ids;
      ready_buffer_ids.reserve(chains->size());
      for (const auto& chain : *chains) {
        ready_buffer_ids.push_back(chain.back()->output_buffer.id);
      }

      // Check if all inputs of this operation are ready.
      if (Contains(ready_buffer_ids, task_descriptor->input_buffers)) {
        // Now find a chain to fuse with.
        for (auto& chain : *chains) {
          // We can fuse only single output for now.
          if (Contains(task_descriptor->input_buffers,
                       chain.back()->output_buffer.id)) {
            if (CanFuseOperations(chain.back(), task_descriptor, output_ids,
                                  *descriptors)) {
              chain.push_back(task_descriptor);
            } else {
              // Start new chain.
              chains->push_back({task_descriptor});
            }
            break;
          }
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
    unused_ids->push_back(desc->id);
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
      ready_buffer_ids.push_back(chain.back()->output_buffer.id);
    }

    for (auto it = chains->begin(); it != chains->end();) {
      const FusionSequence& chain = *it;

      // If the input is also is the output in the same chain - eliminate
      // because it used internally inside this chain only.
      std::vector<ValueId> elements_output_buffer_ids;
      elements_output_buffer_ids.reserve(chain.size());
      for (const ComputeTaskDescriptorPtr& element : chain) {
        elements_output_buffer_ids.push_back(element->output_buffer.id);
      }

      // Collect all inputs also for linked operations.
      std::vector<ValueId> elements_input_buffer_ids;
      for (auto element : chain) {
        for (const auto& buffer : element->input_buffers) {
          if (!Contains(elements_output_buffer_ids, buffer.id)) {
            elements_input_buffer_ids.push_back(buffer.id);
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
    output_and_intermediate_ids.push_back(it->back()->output_buffer.id);
    for (const auto& buffer : it->front()->input_buffers) {
      input_and_intermediate_ids.insert(buffer.id);
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
    output_and_intermediate_ids.push_back(it->back()->output_buffer.id);
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
        for (auto element : *it2) {
          for (const auto& buffer : element->input_buffers) {
            input_ids.push_back(buffer.id);
          }
        }
        if (Contains(input_ids, it1->back()->output_buffer.id)) {
          found_as_input = true;
          break;
        }
      }
    }
    if (!found_as_input) {
      if (!Contains(output_ids, it1->back()->output_buffer.id)) {
        extra_output_ids.push_back(it1->back()->output_buffer.id);
      }
    }
  }
  return extra_output_ids;
}

// Delete all unused task descriptors that have non-requested outputs.
// TODO(chirkov): delete not the whole chain but only the last element, then
// others.
std::vector<int> DeleteUnusedTasks(const std::vector<ValueId>& output_ids,
                                   std::list<FusionSequence>* chains) {
  std::vector<int> unused_operations;
  for (auto it1 = chains->rbegin(); it1 != chains->rend();) {
    // Don't delete if output is requested.
    if (Contains(output_ids, it1->back()->output_buffer.id)) {
      ++it1;
      continue;
    }

    // Don't delete if some operation uses the output.
    bool output_used = false;
    for (auto it2 = chains->rbegin(); it2 != chains->rend(); ++it2) {
      std::vector<ValueId> input_ids;
      for (auto element : *it2) {
        for (const auto& buffer : element->input_buffers) {
          input_ids.push_back(buffer.id);
        }
      }
      if (Contains(input_ids, it1->back()->output_buffer.id)) {
        output_used = true;
        break;
      }
    }
    if (output_used) {
      ++it1;
      continue;
    }
    // Delete if not used.
    unused_operations.push_back(it1->back()->id);
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
    if (chain.front()->input_buffers.empty()) {
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

ComputeTaskDescriptorPtr NonLinkableStub(int operation_id, ValueId input_id,
                                         ValueId output_id) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = operation_id;
  desc->is_linkable = false;
  desc->shader_source = R"(
    #include <metal_stdlib>
    using namespace metal;
    $0
    kernel void ComputeFunction(
                                $1
                                uint3 gid[[thread_position_in_grid]]) {
      if (int(gid.x) >= size.x || int(gid.y) >= size.y) {
        return;
      }
      const int linear_index = (gid.z * size.y + gid.y) * size.x + gid.x;
      FLT4 value = input_buffer[linear_index];
      $2
      output_buffer[linear_index] = value;
    }
  )";

  desc->input_buffers = {
      {input_id, "device FLT4* const input_buffer"},
  };

  desc->output_buffer = {output_id, "device FLT4* output_buffer",
                         [input_id](const std::map<ValueId, BHWC>& buffers) {
                           return buffers.find(input_id)->second;
                         }};

  desc->uniform_buffers = {
      {"constant int2& size",
       [input_id](const std::map<ValueId, BHWC>& buffers) {
         const auto& dimension = buffers.find(input_id)->second;
         return VectorToUint8Vector(std::vector<int>{dimension.w, dimension.h});
       }},
  };

  desc->resize_function = [input_id](const std::map<ValueId, BHWC>& buffers) {
    const auto& dimension = buffers.find(input_id)->second;
    uint3 groups_size{16, 16, 1};
    uint3 groups_count{AlignByN(dimension.w, groups_size.x),
                       AlignByN(dimension.h, groups_size.y),
                       AlignByN(dimension.c, 4)};
    return std::make_pair(groups_size, groups_count);
  };

  return {desc};
}

ComputeTaskDescriptorPtr FuseChain(const FusionSequence& chain) {
  auto fused_desciptor = std::make_shared<ComputeTaskDescriptor>();
  // The id of fused descriptor is the id of the first descriptor in the list.
  fused_desciptor->id = chain.front()->id;
  FusionSequence sequence;
  if (chain.front()->is_linkable) {
    // The first task is linkable so it contains only linkable code. Insert
    // unlinkable meta-task with remaining shader code.
    sequence.push_back(NonLinkableStub(-1, chain.front()->input_buffers[0].id,
                                       chain.front()->input_buffers[0].id));
  }
  sequence.insert(sequence.end(), chain.begin(), chain.end());

  // Count buffers to calculate proper indices then.
  int num_outputs = 1;
  int num_inputs = 0;
  int num_immutables = 0;
  bool invalid_id = true;
  ValueId fused_id;
  for (const auto& desc : sequence) {
    for (const auto& buffer : desc->input_buffers) {
      if (invalid_id || buffer.id != fused_id) {
        num_inputs++;
      }
    }
    fused_id = desc->output_buffer.id;
    invalid_id = false;
    num_immutables += desc->immutable_buffers.size();
  }

  int output_index = 0;
  int input_index = num_outputs;
  int immutable_index = num_outputs + num_inputs;
  int uniform_index = num_outputs + num_inputs + num_immutables;

  int function_index = 0;
  std::string function_code;
  std::string buffer_declarations;
  std::string call_code;
  invalid_id = true;
  for (const auto& desc : sequence) {
    if (desc->is_linkable) {
      function_code +=
          absl::Substitute(desc->shader_source, function_index) + "\n";
    } else {
      // Declare output buffer only for the first unlinkable task.
      buffer_declarations +=
          desc->output_buffer.declaration + "[[buffer(0)]],\n";
      output_index++;
    }

    std::string call_arguments;
    for (const auto& buffer : desc->input_buffers) {
      if (invalid_id || buffer.id != fused_id) {
        std::string index = std::to_string(input_index);
        std::string name = (desc->is_linkable ? (" buffer" + index) : "");
        buffer_declarations +=
            buffer.declaration + name + "[[buffer(" + index + ")]],\n";
        call_arguments += ", buffer" + index;
        input_index++;
        fused_desciptor->input_buffers.push_back({buffer.id, ""});
      }
    }
    // We have an output id that is the input for the next task.
    fused_id = desc->output_buffer.id;
    invalid_id = false;

    for (auto buffer : desc->immutable_buffers) {
      std::string index = std::to_string(immutable_index);
      std::string name = (desc->is_linkable ? (" buffer" + index) : "");
      buffer_declarations +=
          buffer.declaration + name + "[[buffer(" + index + ")]],\n";
      call_arguments += ", buffer" + index;
      immutable_index++;
      fused_desciptor->immutable_buffers.push_back(buffer);
    }

    for (auto buffer : desc->uniform_buffers) {
      std::string index = std::to_string(uniform_index);
      std::string name = (desc->is_linkable ? (" buffer" + index) : "");
      buffer_declarations +=
          buffer.declaration + name + "[[buffer(" + index + ")]],\n";
      call_arguments += ", buffer" + index;
      uniform_index++;
      fused_desciptor->uniform_buffers.push_back({"", buffer.data_function});
    }

    if (desc->is_linkable) {
      call_code +=
          absl::Substitute("value = linkable$0(value, linear_index, gid$1);\n",
                           function_index, call_arguments);
      function_index++;
    }
  }

  ComputeTaskDescriptorPtr non_linkable = sequence.front();
  fused_desciptor->shader_source =
      absl::Substitute(non_linkable->shader_source, function_code,
                       buffer_declarations, call_code);
  std::vector<ValueId> alias;
  alias.reserve(chain.size() - 1);
  for (int i = 0; i < chain.size() - 1; i++) {
    alias.push_back(chain[i]->output_buffer.id);
  }
  fused_desciptor->output_buffer = {
      fused_id, "", non_linkable->output_buffer.dimensions_function, alias};
  fused_desciptor->resize_function = non_linkable->resize_function;
  return fused_desciptor;
}

}  // namespace

Status ValidateOptimizeModel(const std::vector<ValueId>& input_buffers,
                             const std::vector<ValueId>& output_buffers,
                             const CompiledModel& input_vector,
                             CompiledModel* output) {
  std::list<ComputeTaskDescriptorPtr> input;
  input.insert(input.end(), input_vector.begin(), input_vector.end());
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
    return InternalError("Empty chains");
  }
  for (const auto& chain : sorted_chains) output->push_back(FuseChain(chain));
  return OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
