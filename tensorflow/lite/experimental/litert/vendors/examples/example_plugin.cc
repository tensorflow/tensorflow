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

#include <cstdlib>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/vendors/examples/example_plugin_common.h"

// A simple compiler plugin example that implements everything directly.
// This plugin matches on mul ops, and emits "byte code" that is simply
// a string representative of the ops consumed.

// Plugins can hold state.
struct LiteRtCompilerPluginT {};

LiteRtStatus LiteRtCreateCompilerPlugin(LiteRtCompilerPlugin* compiler_plugin) {
  *compiler_plugin = new LiteRtCompilerPluginT;
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  ::litert::Subgraph main_subgraph(subgraph);
  for (const auto& op : main_subgraph.Ops()) {
    if (op.Code() != kLiteRtOpCodeTflMul) {
      continue;
    }
    LITERT_RETURN_STATUS_IF_NOT_OK(LiteRtPushOp(selected_ops, op.Get()));
  }
  return kLiteRtStatusOk;
}

namespace {

LiteRtStatus CompileSinglePartition(LiteRtParamIndex partition_index,
                                    LiteRtSubgraph subgraph,
                                    LiteRtCompiledResultT& result) {
  const litert::Subgraph sg(subgraph);
  int num_muls_in_partition = 0;
  for (const auto& op : sg.Ops()) {
    if (op.Code() != kLiteRtOpCodeTflMul) {
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

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtSubgraph* partitions, LiteRtParamIndex num_partitions,
    LiteRtCompiledResult* compiled_result) {
  LiteRtCompiledResult result = new LiteRtCompiledResultT;

  for (auto i = 0; i < num_partitions; ++i) {
    LITERT_RETURN_STATUS_IF_NOT_OK(
        CompileSinglePartition(i, partitions[i], *result));
  }

  *compiled_result = result;

  return kLiteRtStatusOk;
}
