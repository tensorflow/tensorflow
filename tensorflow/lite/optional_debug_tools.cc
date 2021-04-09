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

#include <stddef.h>
#include <stdio.h>

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

namespace {
// Just forward declarations.
const char* AllocTypeName(TfLiteAllocationType type);

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
  }

  void Print() const {
    printf("%s Info: ", AllocTypeName(allocation_type_));
    if (max_tensor_end_addr_ == 0) {
      printf("not holding any allocation.\n");
      return;
    }
    printf("\nTensor %zu has the max size %zu bytes (%.1f MB).\n",
           max_tensor_id_, max_tensor_mem_bytes_,
           static_cast<float>(max_tensor_mem_bytes_) / (1 << 20));
    printf("This memory arena is estimated as[0x%zx, 0x%zx), taking %.1f MB.\n",
           max_tensor_end_addr_, min_tensor_start_addr_,
           static_cast<float>(max_tensor_end_addr_ - min_tensor_start_addr_) /
               (1 << 20));
  }

 private:
  TfLiteAllocationType allocation_type_;
  size_t max_tensor_mem_bytes_ = 0;
  // the index of the tensor that has the max memory size.
  size_t max_tensor_id_ = -1;
  size_t min_tensor_start_addr_ = std::numeric_limits<size_t>::max();
  size_t max_tensor_end_addr_ = 0;
};

void PrintIntVector(const std::vector<int>& v) {
  if (v.empty()) {
    printf("(null)\n");
    return;
  }

  printf("[");
  for (int i = 0; i < v.size() - 1; ++i) {
    printf("%d,", v[i]);
  }
  printf("%d]\n", v.back());
}

void PrintTfLiteIntVector(const TfLiteIntArray* v) {
  if (!v || v->size <= 0) {
    printf("(null)\n");
    return;
  }
  printf("[");
  for (int k = 0; k < v->size - 1; k++) {
    printf("%d,", v->data[k]);
  }
  printf("%d]\n", v->data[v->size - 1]);
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
    // Change the the last 3 chars to  "..." to imply truncation.
    truncated.replace(size_limit - 3, 3, "...");
  } else {
    truncated.erase(0, length - size_limit);
    // Change the the first 3 chars to  "..." to imply truncation.
    truncated.replace(0, 3, "...");
  }
  return truncated;
}

}  // namespace

// Prints a dump of what tensors and what nodes are in the interpreter.
void PrintInterpreterState(Interpreter* interpreter) {
  const size_t num_subgraphs = interpreter->subgraphs_size();
  printf("Interpreter has %zu subgraphs.\n\n", num_subgraphs);

  for (int i = 0; i < num_subgraphs; ++i) {
    const Subgraph& subgraph = *(interpreter->subgraph(i));
    printf("-----------Subgraph-%d has %zu tensors and %zu nodes------------\n",
           i, subgraph.tensors_size(), subgraph.nodes_size());
    printf("Inputs: ");
    PrintIntVector(subgraph.inputs());
    printf("Outputs: ");
    PrintIntVector(subgraph.outputs());
    printf("\n");

    printf("Tensor %3s %-25s %-15s %-18s %10s\n", "ID", "Name", "Type",
           "AllocType", "Size");
    MemoryArenaInfo rw_info(kTfLiteArenaRw);
    MemoryArenaInfo rw_persistent_info(kTfLiteArenaRwPersistent);
    for (size_t tensor_index = 0; tensor_index < subgraph.tensors_size();
         tensor_index++) {
      const TfLiteTensor* tensor =
          subgraph.tensor(static_cast<int>(tensor_index));
      printf("Tensor %3zu %-25s %-15s %-18s %10zuB (%4.1f MB) ", tensor_index,
             TruncateString(tensor->name, 25, /*truncate_at_end*/ true).c_str(),
             TruncateString(TensorTypeName(tensor->type), 15).c_str(),
             TruncateString(AllocTypeName(tensor->allocation_type), 18).c_str(),
             tensor->bytes, (static_cast<float>(tensor->bytes) / (1 << 20)));
      PrintTfLiteIntVector(tensor->dims);
      rw_info.Update(tensor_index, *tensor);
      rw_persistent_info.Update(tensor_index, *tensor);
    }
    printf("\n");
    rw_info.Print();
    printf("\n");
    rw_persistent_info.Print();
    printf("\n");
    for (size_t node_index = 0; node_index < subgraph.nodes_size();
         node_index++) {
      const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg =
          subgraph.node_and_registration(static_cast<int>(node_index));
      const TfLiteNode& node = node_and_reg->first;
      const TfLiteRegistration& reg = node_and_reg->second;
      if (reg.custom_name != nullptr) {
        printf("Node %3zu Operator Custom Name %s\n", node_index,
               reg.custom_name);
      } else {
        printf("Node %3zu Operator Builtin Code %3d %s\n", node_index,
               reg.builtin_code, EnumNamesBuiltinOperator()[reg.builtin_code]);
      }
      printf("  Input Tensors:");
      PrintTfLiteIntVector(node.inputs);
      printf("  Output Tensors:");
      PrintTfLiteIntVector(node.outputs);
      if (node.intermediates && node.intermediates->size) {
        printf("  Intermediate Tensors:");
        PrintTfLiteIntVector(node.intermediates);
      }
      if (node.temporaries && node.temporaries->size) {
        printf("  Temporary Tensors:");
        PrintTfLiteIntVector(node.temporaries);
      }
    }
    printf("--------------Subgraph-%d dump has completed--------------\n\n", i);
  }
}

}  // namespace tflite
