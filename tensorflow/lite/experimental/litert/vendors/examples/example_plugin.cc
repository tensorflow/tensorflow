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
#include <memory>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_op_options.h"
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
    if (op.Code() == kLiteRtOpCodeTflMul) {
      LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 0));
    } else if (op.Code() == kLiteRtOpCodeTflSub) {
      LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 1));
    } else if (op.Code() == kLiteRtOpCodeShloComposite) {
      const auto opts =
          litert::GetOptionsAs<litert::CompositeOptions>(op.Get());
      if (!opts) {
        return opts.Error().Status();
      }
      if (opts->name == "odml.rms_norm") {
        LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 0));
      }
    }
  }
  return kLiteRtStatusOk;
}

namespace {

LiteRtStatus CompileSinglePartition(LiteRtParamIndex partition_index,
                                    LiteRtSubgraph subgraph,
                                    LiteRtCompiledResultT& result,
                                    int byte_code_idx) {
  const litert::Subgraph sg(subgraph);
  int num_muls_in_partition = 0;
  for (const auto& op : sg.Ops()) {
    if (op.Code() != kLiteRtOpCodeTflMul && op.Code() != kLiteRtOpCodeTflSub) {
      return kLiteRtStatusErrorUnsupported;
    }
    if (op.Code() == kLiteRtOpCodeTflMul) {
      ++num_muls_in_partition;
    }
  }

  {
    char* byte_code_append;
    (void)asprintf(&byte_code_append,
                   "Partition_%lu_with_%d_muls:", partition_index,
                   num_muls_in_partition);
    result.byte_code[byte_code_idx].append(byte_code_append);
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
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  auto model = litert::Model::CreateFromNonOwnedHandle(partitions);
  const auto num_partitions = model.NumSubgraphs();
  auto result = std::make_unique<LiteRtCompiledResultT>();
  result->byte_code.resize(num_partitions);
  for (auto i = 0; i < num_partitions; ++i) {
    LITERT_RETURN_IF_ERROR(
        CompileSinglePartition(i, model.Subgraph(i)->Get(), *result, i));
  }

  *compiled_result = result.release();

  return kLiteRtStatusOk;
}
