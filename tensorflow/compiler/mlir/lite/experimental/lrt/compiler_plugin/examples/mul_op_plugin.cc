// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>

#include <cstddef>
#include <cstdlib>
#include <string>

#include "tensorflow/compiler/mlir/lite/experimental/lrt/api/graph_tools.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/api/lite_rt_model_api.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/compiler_plugin/lite_rt_compiler_plugin.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/lite_rt_common.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/lite_rt_op_code.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/lite_rt_support.h"

constexpr char kPluginNamespace[] = "mul_op_plugin";

struct LrtCompiledPartitionT {
  std::string byte_code;
  std::string name;
};

struct LrtCompilerPluginT {
  int num_partitions_compiled = 0;
};

LrtStatus PluginInit(LrtCompilerPlugin* compiler_plugin) {
  *compiler_plugin = new LrtCompilerPluginT;
  return StatusOk();
}

void PluginDestroy(LrtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

// Claims all mul ops.
LrtStatus PluginPartitionModel(LrtCompilerPlugin compiler_plugin,
                               LrtModel model, LrtOpList selected_ops) {
  LRT_ASSIGN_OR_RETURN_STATUS(auto subgraph, graph_tools::GetSubgraph(model));
  LRT_ASSIGN_OR_RETURN_STATUS(auto ops, graph_tools::GetSubgraphOps(subgraph));

  for (auto op : ops) {
    LrtOpCode op_code;
    LRT_RETURN_STATUS_IF_NOT_OK(GetOpCode(op, &op_code));
    if (op_code != kLrtOpCodeTflMul) {
      continue;
    }
    LRT_RETURN_STATUS_IF_NOT_OK(PushOp(selected_ops, op));
  }
  return StatusOk();
}

LrtStatus PluginCompilePartition(LrtCompilerPlugin compiler_plugin,
                                 LrtSubgraph partition,
                                 LrtCompiledPartition* compiled_partition) {
  LRT_ASSIGN_OR_RETURN_STATUS(auto ops, graph_tools::GetSubgraphOps(partition));

  int num_muls_in_partition = 0;
  for (auto op : ops) {
    LrtOpCode op_code;

    LRT_RETURN_STATUS_IF_NOT_OK(GetOpCode(op, &op_code));
    if (op_code != kLrtOpCodeTflMul) {
      return StatusCreate(kLrtStatusErrorUnsupported);
    }

    ++num_muls_in_partition;
  }

  auto* result = new LrtCompiledPartitionT();

  char* byte_code;
  asprintf(&byte_code, "partition_with_%d_muls", num_muls_in_partition);
  result->byte_code.assign(byte_code);
  free(byte_code);

  char* name;
  asprintf(&name, "partition_number_%d",
           compiler_plugin->num_partitions_compiled);
  result->name.assign(name);
  free(name);

  ++compiler_plugin->num_partitions_compiled;

  *compiled_partition = result;

  return StatusOk();
}

const char* PluginGetNamespace(LrtCompilerPlugin compiler_plugin) {
  return kPluginNamespace;
}

void PluginCompiledPartitionDestroy(LrtCompiledPartition compiled_partition) {
  delete compiled_partition;
}

LrtStatus PluginCompiledPartitionInit(
    LrtCompiledPartition* compiled_partition) {
  *compiled_partition = new LrtCompiledPartitionT;
  return StatusOk();
}

LrtStatus PluginCompiledPartitionGetByteCode(
    LrtCompiledPartition compiled_partition, const void** byte_code,
    size_t* byte_code_size) {
  *byte_code = compiled_partition->byte_code.data();
  *byte_code_size = compiled_partition->byte_code.size();
  return StatusOk();
}

LrtStatus PluginCompiledPartitionGetName(
    LrtCompiledPartition compiled_partition, const char** partition_name) {
  *partition_name = compiled_partition->name.data();
  return StatusOk();
}
