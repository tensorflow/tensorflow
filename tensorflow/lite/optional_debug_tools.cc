/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/optional_debug_tools.h"

#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

namespace {
// Just forward declarations.
const char* AllocTypeName(TfLiteAllocationType type);

void PrintIntVector(const std::vector<int>& v,
                    bool collapse_consecutives = true,
                    bool add_newline = false);

// A class to represent the information of a memory arena that's used in TfLite
// runtime for holding allocated memory of tensors. The information includes
// the following:
// 1. The memory allocation type.
// 2. The tensor id of the tensor that has the most amount of memory allocated,
// and the memory size.
// 3. The estimated memory boundary and size of the arena.
class MemoryArenaInfo {
 public:
  explicit MemoryArenaInfo(TfLiteAllocationType type)
      : allocation_type_(type) {}

  void Update(size_t tensor_index, const TfLiteTensor& tensor) {
    if (tensor.allocation_type != allocation_type_) return;
    if (tensor.data.data == nullptr) return;
    if (tensor.bytes > max_tensor_mem_bytes_) {
      max_tensor_mem_bytes_ = tensor.bytes;
      max_tensor_id_ = tensor_index;
    }

    size_t current_start_addr = reinterpret_cast<size_t>(tensor.data.data);

    size_t current_end_addr = current_start_addr + tensor.bytes;
    if (current_start_addr < min_tensor_start_addr_) {
      min_tensor_start_addr_ = current_start_addr;
    }
    if (current_end_addr > max_tensor_end_addr_) {
      max_tensor_end_addr_ = current_end_addr;
    }

    TensorAllocInfo info;
    info.tensor_id = tensor_index;
    info.start_addr = current_start_addr;
    info.bytes = tensor.bytes;
    const auto result = alloc_info_.insert(info);
    // Simply check that the insertion succeeds.
    assert(result.second);
    (void)result;  // suppress the "unused variable" compilation error.
  }

  size_t GetArenaStartingAddress() const { return min_tensor_start_addr_; }

  void Print() const {
    printf("%s Info: ", AllocTypeName(allocation_type_));
    if (max_tensor_end_addr_ == 0) {
      printf("not holding any allocation.\n");
      return;
    }
    printf("\nTensor %zu has the max size %zu bytes (%.3f MB).\n",
           max_tensor_id_, max_tensor_mem_bytes_,
           static_cast<float>(max_tensor_mem_bytes_) / (1 << 20));
    const size_t arena_size = max_tensor_end_addr_ - min_tensor_start_addr_;
    printf(
        "This memory arena is estimated as[0x%zx, 0x%zx), taking %zu bytes "
        "(%.3f MB).\n",
        max_tensor_end_addr_, min_tensor_start_addr_, arena_size,
        static_cast<float>(arena_size) / (1 << 20));

    std::vector<const TensorAllocInfo*> arena_increase_trace;
    size_t last_end_addr = 0;
    for (const auto& info : alloc_info_) {
      if (info.start_addr >= last_end_addr) {
        arena_increase_trace.emplace_back(&info);
        last_end_addr = info.start_addr + info.bytes;
      }
    }
    printf(
        "One possible set of tensors that have non-overlapping memory spaces "
        "with each other, and they take up the whole arena:\n");
    printf("Tensor ");
    for (int i = 0; i < arena_increase_trace.size() - 1; ++i) {
      printf("%zu -> ", arena_increase_trace[i]->tensor_id);
    }
    printf("%zu.\n", arena_increase_trace.back()->tensor_id);
  }

 private:
  struct TensorAllocInfo {
    size_t tensor_id;
    size_t start_addr;
    size_t bytes;
  };

  // Compare first according to 'start_addr' in increasing order, then secondly
  // according to 'bytes' in decreasing order and finally according to
  // 'tensor_id' in increasing order.
  struct TensorAllocInfoCompare {
    bool operator()(const TensorAllocInfo& lhs,
                    const TensorAllocInfo& rhs) const {
      if (lhs.start_addr < rhs.start_addr) return true;
      if (lhs.start_addr == rhs.start_addr) {
        if (lhs.bytes > rhs.bytes) return true;
        if (lhs.bytes == rhs.bytes) return lhs.tensor_id < rhs.tensor_id;
        return false;
      }
      return false;
    }
  };

  const TfLiteAllocationType allocation_type_;
  size_t max_tensor_mem_bytes_ = 0;
  // the index of the tensor that has the max memory size.
  size_t max_tensor_id_ = -1;
  size_t min_tensor_start_addr_ = std::numeric_limits<size_t>::max();
  size_t max_tensor_end_addr_ = 0;
  std::set<TensorAllocInfo, TensorAllocInfoCompare> alloc_info_;
};

class DynamicMemoryInfo {
 public:
  void Update(size_t tensor_index, const TfLiteTensor& tensor) {
    if (tensor.allocation_type != kTfLiteDynamic) return;
    if (tensor.data.data == nullptr) return;
    if (tensor.bytes > max_tensor_mem_bytes_) {
      max_tensor_mem_bytes_ = tensor.bytes;
      max_tensor_ids_.clear();
      max_tensor_ids_.push_back(tensor_index);
    } else if (tensor.bytes == max_tensor_mem_bytes_) {
      max_tensor_ids_.push_back(static_cast<int>(tensor_index));
    }
    total_mem_bytes_ += tensor.bytes;
    num_total_tensors_++;
  }

  void Print() const {
    printf("kTfLiteDynamic Info: ");
    if (total_mem_bytes_ == 0) {
      printf("not holding any allocation.\n");
      return;
    }
    printf("\n%zu Tensors ", max_tensor_ids_.size());
    PrintIntVector(max_tensor_ids_, /*collapse_consecutives*/ false);
    printf(" have the max size %zu bytes (%.3f MB).\n", max_tensor_mem_bytes_,
           static_cast<float>(max_tensor_mem_bytes_) / (1 << 20));
    printf("There are %d dynamic tensors, taking %zu bytes (%.3f MB).\n",
           num_total_tensors_, total_mem_bytes_,
           static_cast<float>(total_mem_bytes_) / (1 << 20));
  }

 private:
  size_t max_tensor_mem_bytes_ = 0;
  // the index list of the tensor that has the max memory size.
  std::vector<int> max_tensor_ids_;
  size_t total_mem_bytes_ = 0;
  int num_total_tensors_ = 0;
};

class ModelTensorMemoryInfo {
 public:
  ModelTensorMemoryInfo()
      : rw_info_(kTfLiteArenaRw),
        rw_persistent_info_(kTfLiteArenaRwPersistent),
        mmap_info_(kTfLiteMmapRo) {}

  void Update(size_t tensor_index, const TfLiteTensor& tensor) {
    rw_info_.Update(tensor_index, tensor);
    rw_persistent_info_.Update(tensor_index, tensor);
    mmap_info_.Update(tensor_index, tensor);
    dynamic_info_.Update(tensor_index, tensor);
  }

  // Get the offset from the beginning address of the memory arena for 'tensor'.
  // Returns -1 if not applicable. Otherwise, returns a non-negative value.
  int64_t GetOffsetFromArenaStart(const TfLiteTensor& tensor) const {
    if (tensor.data.data == nullptr) return -1;
    size_t tensor_address = reinterpret_cast<size_t>(tensor.data.data);
    if (tensor.allocation_type == kTfLiteArenaRw) {
      return static_cast<int64_t>(tensor_address -
                                  rw_info_.GetArenaStartingAddress());
    }
    if (tensor.allocation_type == kTfLiteArenaRwPersistent) {
      return static_cast<int64_t>(
          tensor_address - rw_persistent_info_.GetArenaStartingAddress());
    }
    if (tensor.allocation_type == kTfLiteMmapRo) {
      return static_cast<int64_t>(tensor_address -
                                  mmap_info_.GetArenaStartingAddress());
    }
    return -1;
  }

  void Print() const {
    printf("\n");
    rw_info_.Print();
    printf("\n");
    rw_persistent_info_.Print();
    printf("\n");
    mmap_info_.Print();
    printf("\n");
    dynamic_info_.Print();
    printf("\n");
  }

 private:
  MemoryArenaInfo rw_info_;
  MemoryArenaInfo rw_persistent_info_;
  MemoryArenaInfo mmap_info_;
  DynamicMemoryInfo dynamic_info_;
};

template <typename T>
void PrintTotalBytesOfTensors(const Subgraph& subgraph, const T& tensor_ids,
                              const std::string& prefix = " -> ") {
  size_t total = 0;
  for (const auto id : tensor_ids) {
    const TfLiteTensor* tensor = subgraph.tensor(id);
    if (tensor == nullptr) continue;
    total += tensor->bytes;
  }
  printf("%s%zuB (%.2fMB)\n", prefix.c_str(), total,
         static_cast<float>(total) / (1 << 20));
}

void PrintIntVector(const std::vector<int>& v, bool collapse_consecutives,
                    bool add_newline) {
  if (v.empty()) {
    printf("(null)");
    if (add_newline) {
      printf("\n");
    }
    return;
  }

  int range_start = v[0];
  int range_end = range_start;
  std::function<void(const char*)> print_range = [&](const char* suffix) {
    if (range_end == range_start) {
      printf("%d%s", range_start, suffix);
    } else if (range_end == range_start + 1) {
      printf("%d,%d%s", range_start, range_end, suffix);
    } else {
      printf("%d-%d%s", range_start, range_end, suffix);
    }
  };

  printf("[");
  for (int i = 1; i < v.size(); ++i) {
    int current = v[i];
    if (collapse_consecutives && (current == range_end + 1)) {
      range_end = current;
    } else {
      print_range(",");
      range_start = range_end = current;
    }
  }
  print_range("]");
  if (add_newline) {
    printf("\n");
  }
}

void PrintTfLiteIntVector(const TfLiteIntArray* v,
                          bool collapse_consecutives = true,
                          bool add_newline = false) {
  std::vector<int> tmp;
  if (!v || v->size <= 0) {
    PrintIntVector(tmp, collapse_consecutives, add_newline);
    return;
  }
  tmp.insert(tmp.end(), v->data, v->data + v->size);
  PrintIntVector(tmp, collapse_consecutives, add_newline);
}

const char* TensorTypeName(TfLiteType type) {
  switch (type) {
    case kTfLiteNoType:
      return "kTfLiteNoType";
    case kTfLiteFloat32:
      return "kTfLiteFloat32";
    case kTfLiteInt32:
      return "kTfLiteInt32";
    case kTfLiteUInt32:
      return "kTfLiteUInt32";
    case kTfLiteUInt8:
      return "kTfLiteUInt8";
    case kTfLiteInt8:
      return "kTfLiteInt8";
    case kTfLiteInt64:
      return "kTfLiteInt64";
    case kTfLiteUInt64:
      return "kTfLiteUInt64";
    case kTfLiteString:
      return "kTfLiteString";
    case kTfLiteBool:
      return "kTfLiteBool";
    case kTfLiteUInt16:
      return "kTfLiteUInt16";
    case kTfLiteInt16:
      return "kTfLiteInt16";
    case kTfLiteComplex64:
      return "kTfLiteComplex64";
    case kTfLiteComplex128:
      return "kTfLiteComplex128";
    case kTfLiteFloat16:
      return "kTfLiteFloat16";
    case kTfLiteFloat64:
      return "kTfLiteFloat64";
    case kTfLiteResource:
      return "kTfLiteResource";
    case kTfLiteVariant:
      return "kTfLiteVariant";
    case kTfLiteInt4:
      return "kTfLiteInt4";
  }
  return "(invalid)";
}

const char* AllocTypeName(TfLiteAllocationType type) {
  switch (type) {
    case kTfLiteMemNone:
      return "kTfLiteMemNone";
    case kTfLiteMmapRo:
      return "kTfLiteMmapRo";
    case kTfLiteDynamic:
      return "kTfLiteDynamic";
    case kTfLiteArenaRw:
      return "kTfLiteArenaRw";
    case kTfLiteArenaRwPersistent:
      return "kTfLiteArenaRwPersistent";
    case kTfLitePersistentRo:
      return "kTfLitePersistentRo";
    case kTfLiteCustom:
      return "kTfLiteCustom";
  }
  return "(invalid)";
}

std::string TruncateString(const char* str, int size_limit,
                           bool truncate_at_end = false) {
  if (str == nullptr) return "(nil)";

  std::string truncated(str);
  const size_t length = truncated.size();
  if (length <= size_limit) return truncated;

  if (size_limit <= 3) return std::string(size_limit, '.');

  if (truncate_at_end) {
    truncated.resize(size_limit);
    // Change the last 3 chars to  "..." to imply truncation.
    truncated.replace(size_limit - 3, 3, "...");
  } else {
    truncated.erase(0, length - size_limit);
    // Change the first 3 chars to  "..." to imply truncation.
    truncated.replace(0, 3, "...");
  }
  return truncated;
}

}  // namespace

// Prints a dump of what tensors and what nodes are in the interpreter.
void PrintInterpreterState(const Interpreter* interpreter) {
  const size_t num_subgraphs = interpreter->subgraphs_size();
  printf("Interpreter has %zu subgraphs.\n\n", num_subgraphs);

  for (int i = 0; i < num_subgraphs; ++i) {
    const Subgraph& subgraph = *(interpreter->subgraph(i));
    printf("-----------Subgraph-%d has %zu tensors and %zu nodes------------\n",
           i, subgraph.tensors_size(), subgraph.nodes_size());
    printf("%zu Inputs: ", subgraph.inputs().size());
    PrintIntVector(subgraph.inputs());
    PrintTotalBytesOfTensors(subgraph, subgraph.inputs());

    printf("%zu Outputs: ", subgraph.outputs().size());
    PrintIntVector(subgraph.outputs());
    PrintTotalBytesOfTensors(subgraph, subgraph.outputs());
    printf("\n");

    // Collect info about tensor memory allocation.
    ModelTensorMemoryInfo tensor_mem_info;
    for (size_t tensor_index = 0; tensor_index < subgraph.tensors_size();
         tensor_index++) {
      const TfLiteTensor* tensor =
          subgraph.tensor(static_cast<int>(tensor_index));
      tensor_mem_info.Update(tensor_index, *tensor);
    }

    printf("Tensor %3s %-25s %-15s %-18s %-18s %-10s %-16s\n", "ID", "Name",
           "Type", "AllocType", "Size (Bytes/MB)", "Shape", "MemAddr-Offset");
    for (size_t tensor_index = 0; tensor_index < subgraph.tensors_size();
         tensor_index++) {
      const TfLiteTensor* tensor =
          subgraph.tensor(static_cast<int>(tensor_index));
      printf("Tensor %3zu %-25s %-15s %-18s %-8zu / %.2f ", tensor_index,
             TruncateString(tensor->name, 25, /*truncate_at_end*/ true).c_str(),
             TruncateString(TensorTypeName(tensor->type), 15).c_str(),
             TruncateString(AllocTypeName(tensor->allocation_type), 18).c_str(),
             tensor->bytes, (static_cast<float>(tensor->bytes) / (1 << 20)));
      PrintTfLiteIntVector(tensor->dims, /*collapse_consecutives*/ false);
      const int64_t start_offset =
          tensor_mem_info.GetOffsetFromArenaStart(*tensor);
      const int64_t end_offset =
          start_offset == -1
              ? -1
              : start_offset + static_cast<int64_t>(tensor->bytes);
      printf(" [%" PRId64 ", %" PRId64 ")\n", start_offset, end_offset);
    }
    tensor_mem_info.Print();

    // Dumps debugging info provided by the underlying memory planner.
    // Note that this will output nothing unless the
    // ":simple_memory_arena_debug_dump" is added as an extra dependence.
    subgraph.DumpMemoryPlannerDebugInfo();

    // Going to print out all nodes (i.e. op kernels) in this subgraph.
    std::vector<bool> replaced_node_bits;
    std::vector<size_t> replaced_by_node;
    replaced_node_bits.resize(subgraph.nodes_size());
    replaced_by_node.resize(subgraph.nodes_size());
    bool has_delegate_applied = false;
    for (size_t node_index = 0; node_index < subgraph.nodes_size();
         node_index++) {
      replaced_node_bits[node_index] = false;
      const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg =
          subgraph.node_and_registration(static_cast<int>(node_index));
      const TfLiteNode& node = node_and_reg->first;
      auto* const delegate = node.delegate;
      if (delegate != nullptr) {
        has_delegate_applied = true;
        auto* params = static_cast<TfLiteDelegateParams*>(node.builtin_data);
        for (int nid : TfLiteIntArrayView(params->nodes_to_replace)) {
          replaced_node_bits[nid] = true;
          replaced_by_node[nid] = node_index;
        }
      }
    }
    for (size_t node_index = 0; node_index < subgraph.nodes_size();
         node_index++) {
      const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg =
          subgraph.node_and_registration(static_cast<int>(node_index));
      const TfLiteNode& node = node_and_reg->first;
      const TfLiteRegistration& reg = node_and_reg->second;

      std::string delegated_status;
      bool is_node_delegated = false;
      TfLiteIntArray empty_int_array;
      empty_int_array.size = 0;
      if (node.delegate == nullptr) {
        if (replaced_node_bits[node_index]) {
          delegated_status = "(delegated by node ";
          delegated_status.append(std::to_string(replaced_by_node[node_index]));
          delegated_status.append(")");
          is_node_delegated = true;
        } else {
          delegated_status = "(not delegated)";
        }
      }

      if (reg.custom_name != nullptr) {
        printf("Node %3zu Operator Custom Name %s %s\n", node_index,
               reg.custom_name, delegated_status.c_str());
      } else {
        printf("Node %3zu Operator Builtin Code %3d %s %s\n", node_index,
               reg.builtin_code, EnumNamesBuiltinOperator()[reg.builtin_code],
               delegated_status.c_str());
      }
      printf("  %d Input Tensors:",
             node.inputs != nullptr ? node.inputs->size : 0);
      if (node.inputs) {
        PrintTfLiteIntVector(
            node.inputs,
            /*collapse_consecutives=*/(node.delegate != nullptr));
        PrintTotalBytesOfTensors(
            subgraph, is_node_delegated ? TfLiteIntArrayView(&empty_int_array)
                                        : TfLiteIntArrayView(node.inputs));
      }

      printf("  %d Output Tensors:",
             node.outputs != nullptr ? node.outputs->size : 0);
      if (node.outputs) {
        PrintTfLiteIntVector(node.outputs);
        PrintTotalBytesOfTensors(
            subgraph, is_node_delegated ? TfLiteIntArrayView(&empty_int_array)
                                        : TfLiteIntArrayView(node.outputs));
      }

      if (node.intermediates && node.intermediates->size) {
        printf("  %d Intermediate Tensors:", node.intermediates->size);
        PrintTfLiteIntVector(node.intermediates);
        PrintTotalBytesOfTensors(subgraph,
                                 is_node_delegated
                                     ? TfLiteIntArrayView(&empty_int_array)
                                     : TfLiteIntArrayView(node.intermediates));
      }

      if (node.temporaries && node.temporaries->size) {
        printf("  %d Temporary Tensors:", node.temporaries->size);
        PrintTfLiteIntVector(node.temporaries);
        PrintTotalBytesOfTensors(
            subgraph, is_node_delegated ? TfLiteIntArrayView(&empty_int_array)
                                        : TfLiteIntArrayView(node.temporaries));
      }
    }

    printf("\nExecution plan as the list of %zu nodes invoked in-order: ",
           subgraph.execution_plan().size());
    PrintIntVector(subgraph.execution_plan(), /*collapse_consecutives=*/true,
                   /*add_newline=*/true);
    if (has_delegate_applied) {
      printf("Among these nodes in the execution plan:\n");
      for (int node_id : subgraph.execution_plan()) {
        const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg =
            subgraph.node_and_registration(node_id);
        const TfLiteNode& node = node_and_reg->first;
        auto* const delegate = node.delegate;
        if (delegate == nullptr) continue;
        const char* delegate_name = node_and_reg->second.custom_name;
        auto* delegate_params =
            static_cast<TfLiteDelegateParams*>(node.builtin_data);
        printf("  Node %d is a %s node (%p), which has delegated %d nodes: ",
               node_id, delegate_name == nullptr ? "[n/a]" : delegate_name,
               delegate, delegate_params->nodes_to_replace->size);
        PrintTfLiteIntVector(delegate_params->nodes_to_replace,
                             /*collapse_consecutives=*/true,
                             /*add_newline=*/true);
      }
    }

    printf("--------------Subgraph-%d dump has completed--------------\n\n", i);
  }
  printf("--------------Memory Arena Status Start--------------\n");
  size_t total_arena_memory_bytes = 0;
  size_t total_dynamic_memory_bytes = 0;
  size_t total_resource_bytes = 0;

  for (int i = 0; i < num_subgraphs; ++i) {
    const Subgraph& subgraph = *(interpreter->subgraph(i));
    Subgraph::SubgraphAllocInfo alloc_info;
    subgraph.GetMemoryAllocInfo(&alloc_info);
    total_arena_memory_bytes += alloc_info.arena_size;
    total_arena_memory_bytes += alloc_info.arena_persist_size;
    total_dynamic_memory_bytes += alloc_info.dynamic_size;
    // Resources are shared with all subgraphs. So calculate it only once.
    if (i == 0) {
      total_resource_bytes = alloc_info.resource_size;
    }
  }
  size_t total_memory_bytes = total_arena_memory_bytes +
                              total_dynamic_memory_bytes + total_resource_bytes;
  printf("Total memory usage: %zu bytes (%.3f MB)\n", total_memory_bytes,
         static_cast<float>(total_memory_bytes) / (1 << 20));
  printf("- Total arena memory usage: %zu bytes (%.3f MB)\n",
         total_arena_memory_bytes,
         static_cast<float>(total_arena_memory_bytes) / (1 << 20));
  printf("- Total dynamic memory usage: %zu bytes (%.3f MB)\n",
         total_dynamic_memory_bytes,
         static_cast<float>(total_dynamic_memory_bytes) / (1 << 20));
  if (total_resource_bytes) {
    printf("- Total resource memory usage: %zu bytes (%.3f MB)\n",
           total_resource_bytes,
           static_cast<float>(total_resource_bytes) / (1 << 20));
  }
  putchar('\n');

  for (int i = 0; i < num_subgraphs; ++i) {
    const Subgraph& subgraph = *(interpreter->subgraph(i));
    Subgraph::SubgraphAllocInfo alloc_info;
    subgraph.GetMemoryAllocInfo(&alloc_info);
    if (alloc_info.arena_size) {
      printf(
          "Subgraph#%-3d %-18s %10zu (%.2f%%)\n", i, "Arena (Normal)",
          alloc_info.arena_size,
          static_cast<float>(alloc_info.arena_size * 100) / total_memory_bytes);
    }
    if (alloc_info.arena_persist_size) {
      printf("Subgraph#%-3d %-18s %10zu (%.2f%%)\n", i, "Arena (Persistent)",
             alloc_info.arena_persist_size,
             static_cast<float>(alloc_info.arena_persist_size * 100) /
                 total_memory_bytes);
    }
    if (alloc_info.dynamic_size) {
      printf("Subgraph#%-3d %-18s %10zu (%.2f%%)\n", i, "Dyanmic Tensors",
             alloc_info.dynamic_size,
             static_cast<float>(alloc_info.dynamic_size * 100) /
                 total_memory_bytes);
    }
  }
  printf("--------------Memory Arena Status End--------------\n\n");
}

}  // namespace tflite
