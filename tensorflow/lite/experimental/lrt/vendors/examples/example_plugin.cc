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

#include "tensorflow/lite/experimental/lrt/c/litert_common.h"
#include "tensorflow/lite/experimental/lrt/c/litert_model.h"
#include "tensorflow/lite/experimental/lrt/c/litert_op_code.h"
#include "tensorflow/lite/experimental/lrt/cc/litert_support.h"
#include "tensorflow/lite/experimental/lrt/core/graph_tools.h"
#include "tensorflow/lite/experimental/lrt/vendors/c/litert_compiler_plugin.h"

//
// Configurations
//

namespace {

constexpr char kPluginManufacturer[] = "ExampleSocManufacturer";
constexpr char kPluginSocModel[] = "ExampleSocModel";

}  // namespace

const char* LiteRtPluginSocManufacturer() { return kPluginManufacturer; }

LiteRtParamIndex LiteRtPluginNumSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin) {
  return 1;
}

LiteRtStatus LiteRtPluginGetSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (soc_model_idx != 0) {
    return kLiteRtStatusErrorUnsupported;
  }
  *soc_model_name = kPluginSocModel;
  return kLiteRtStatusOk;
}

//
// Compiled Result Definition
//

struct LiteRtCompiledResultT {
  std::string byte_code;
  std::vector<std::string> per_op_data;
};

LiteRtStatus LiteRtCompiledResultGetByteCode(
    LiteRtCompiledResult compiled_result, const void** byte_code,
    size_t* byte_code_size) {
  *byte_code = compiled_result->byte_code.data();
  *byte_code_size = compiled_result->byte_code.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledResultGetCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size) {
  if (call_idx >= compiled_result->per_op_data.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }

  *call_info = compiled_result->per_op_data.at(call_idx).data();
  *call_info_size = compiled_result->per_op_data.at(call_idx).size();

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledResultGetNumCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  *num_calls = compiled_result->per_op_data.size();
  return kLiteRtStatusOk;
}

void LiteRtCompiledResultDestroy(LiteRtCompiledResult compiled_result) {
  delete compiled_result;
}

//
// Plugin Definition
//

// Plugins can hold state.
struct LiteRtCompilerPluginT {};

LiteRtStatus LiteRtPluginInit(LiteRtCompilerPlugin* compiler_plugin) {
  *compiler_plugin = new LiteRtCompilerPluginT;
  return kLiteRtStatusOk;
}

void LiteRtPluginDestroy(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

LiteRtStatus LiteRtPluginPartitionModel(LiteRtCompilerPlugin compiler_plugin,
                                        LiteRtModel model,
                                        LiteRtOpList selected_ops) {
  LITERT_ASSIGN_OR_RETURN_STATUS(auto subgraph,
                                 graph_tools::GetSubgraph(model));
  LITERT_ASSIGN_OR_RETURN_STATUS(auto ops,
                                 graph_tools::GetSubgraphOps(subgraph));

  for (auto op : ops) {
    LiteRtOpCode op_code;
    LITERT_RETURN_STATUS_IF_NOT_OK(GetOpCode(op, &op_code));
    if (op_code != kLiteRtOpCodeTflMul) {
      continue;
    }
    LITERT_RETURN_STATUS_IF_NOT_OK(PushOp(selected_ops, op));
  }
  return kLiteRtStatusOk;
}

namespace {

LiteRtStatus CompileSinglePartition(LiteRtParamIndex partition_index,
                                    LiteRtSubgraph subgraph,
                                    LiteRtCompiledResultT& result) {
  LITERT_ASSIGN_OR_RETURN_STATUS(auto ops,
                                 graph_tools::GetSubgraphOps(subgraph));

  int num_muls_in_partition = 0;
  for (auto op : ops) {
    LiteRtOpCode op_code;

    LITERT_RETURN_STATUS_IF_NOT_OK(GetOpCode(op, &op_code));
    if (op_code != kLiteRtOpCodeTflMul) {
      return kLiteRtStatusErrorUnsupported;
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

  return kLiteRtStatusOk;
}

}  // namespace

LiteRtStatus LiteRtPluginCompile(LiteRtCompilerPlugin compiler_plugin,
                                 const char* soc_model,
                                 LiteRtSubgraphArray partitions,
                                 LiteRtParamIndex num_partitions,
                                 LiteRtCompiledResult* compiled_result) {
  LiteRtCompiledResult result = new LiteRtCompiledResultT;

  for (auto i = 0; i < num_partitions; ++i) {
    LITERT_RETURN_STATUS_IF_NOT_OK(
        CompileSinglePartition(i, partitions[i], *result));
  }

  *compiled_result = result;

  return kLiteRtStatusOk;
}
