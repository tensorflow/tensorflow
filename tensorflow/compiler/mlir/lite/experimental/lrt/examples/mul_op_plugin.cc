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
#include <vector>

#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_compiler_plugin.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_op_code.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/core/graph_tools.h"

//
// Configurations
//

constexpr char kPluginMan[] = "ExampleSocManufacturer";
constexpr char kPluginModel[] = "DummyMulOp";

const char* LrtPluginSocManufacturer() { return kPluginMan; }

lrt_param_index_t LrtPluginNumSupportedSocModels(
    LrtCompilerPlugin compiler_plugin) {
  return 1;
}

LrtStatus LrtPluginGetSupportedSocModelId(LrtCompilerPlugin compiler_plugin,
                                          lrt_param_index_t config_idx,
                                          const char** config_id) {
  if (config_idx != 0) {
    return StatusCreate(kLrtStatusErrorUnsupported);
  }
  *config_id = kPluginModel;
  return StatusOk();
}

//
// Compiled Result Definition
//

struct LrtCompiledResultT {
  std::string byte_code;
  std::vector<std::string> per_op_data;
};

LrtStatus LrtCompiledResultGetByteCode(LrtCompiledResult compiled_result,
                                       const void** byte_code,
                                       size_t* byte_code_size) {
  *byte_code = compiled_result->byte_code.data();
  *byte_code_size = compiled_result->byte_code.size();
  return StatusOk();
}

LrtStatus LrtCompiledResultGetCallInfo(LrtCompiledResult compiled_result,
                                       lrt_param_index_t call_idx,
                                       const void** call_info,
                                       size_t* call_info_size) {
  if (call_idx >= compiled_result->per_op_data.size()) {
    return StatusCreate(kLrtParamIndexOOB);
  }

  *call_info = compiled_result->per_op_data.at(call_idx).data();
  *call_info_size = compiled_result->per_op_data.at(call_idx).size();

  return StatusOk();
}

LrtStatus LrtCompiledResultGetNumCalls(LrtCompiledResult compiled_result,
                                       lrt_param_index_t* num_calls) {
  *num_calls = compiled_result->per_op_data.size();
  return StatusOk();
}

void LrtCompiledResultDestroy(LrtCompiledResult compiled_result) {
  delete compiled_result;
}

//
// Plugin Definition
//

// Plugins can hold state.
struct LrtCompilerPluginT {
};

LrtStatus LrtPluginInit(LrtCompilerPlugin* compiler_plugin) {
  *compiler_plugin = new LrtCompilerPluginT;
  return StatusOk();
}

void LrtPluginDestroy(LrtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

LrtStatus LrtPluginPartitionModel(LrtCompilerPlugin compiler_plugin,
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

LrtStatus CompileSinglePartition(lrt_param_index_t partition_index,
                                 LrtSubgraph subgraph,
                                 LrtCompiledResultT& result) {
  LRT_ASSIGN_OR_RETURN_STATUS(auto ops, graph_tools::GetSubgraphOps(subgraph));

  int num_muls_in_partition = 0;
  for (auto op : ops) {
    LrtOpCode op_code;

    LRT_RETURN_STATUS_IF_NOT_OK(GetOpCode(op, &op_code));
    if (op_code != kLrtOpCodeTflMul) {
      return StatusCreate(kLrtStatusErrorUnsupported);
    }

    ++num_muls_in_partition;
  }

  {
    char* byte_code_append;
    (void)asprintf(&byte_code_append,
                   "Partition_%lu_with_%d_muls:", partition_index,
                   num_muls_in_partition);
    result.byte_code.append(byte_code_append);
    free(byte_code_append);
  }

  {
    char* per_op_data;
    (void)asprintf(&per_op_data, "Partition_%lu", partition_index);
    result.per_op_data.push_back(per_op_data);
    free(per_op_data);
  }

  return StatusOk();
}

LrtStatus LrtPluginCompile(LrtCompilerPlugin compiler_plugin,
                           LrtSubgraphArray partitions,
                           lrt_param_index_t num_partitions,
                           LrtCompiledResult* compiled_result) {
  LrtCompiledResult result = new LrtCompiledResultT;

  for (auto i = 0; i < num_partitions; ++i) {
    LRT_RETURN_STATUS_IF_NOT_OK(
        CompileSinglePartition(i, partitions[i], *result));
  }

  *compiled_result = result;

  return StatusOk();
}
