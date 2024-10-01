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
#include <iostream>
#include <string>
#include <vector>

#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_compiler_plugin.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_op_code.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/core/graph_tools.h"
#include "tensorflow/lite/experimental/lrt/qnn/qnn_manager.h"

using ::qnn::QnnManager;

//
// Configurations
//

constexpr char kPluginMan[] = "QNN";
constexpr char kPluginModel[] = "HTP_Reference";

const char* LrtPluginSocManufacturer() { return kPluginMan; }

lrt_param_index_t LrtPluginNumSupportedSocModels(
    LrtCompilerPlugin compiler_plugin) {
  return 1;
}

LrtStatus LrtPluginGetSupportedSocModelId(LrtCompilerPlugin compiler_plugin,
                                          lrt_param_index_t config_idx,
                                          const char** config_id) {
  if (config_idx != 0) {
    return kLrtStatusErrorUnsupported;
  }
  *config_id = kPluginModel;
  return kLrtStatusOk;
}

//
// Compiled Result Definition
//

struct LrtCompiledResultT {
  std::vector<char> context_bin;
  std::vector<std::string> graph_names;
};

LrtStatus LrtCompiledResultGetByteCode(LrtCompiledResult compiled_result,
                                       const void** byte_code,
                                       size_t* byte_code_size) {
  *byte_code = compiled_result->context_bin.data();
  *byte_code_size = compiled_result->context_bin.size();
  return kLrtStatusOk;
}

LrtStatus LrtCompiledResultGetCallInfo(LrtCompiledResult compiled_result,
                                       lrt_param_index_t call_idx,
                                       const void** call_info,
                                       size_t* call_info_size) {
  if (call_idx >= compiled_result->graph_names.size()) {
    return kLrtStatusParamIndexOOB;
  }

  *call_info = compiled_result->graph_names.at(call_idx).data();
  *call_info_size = compiled_result->graph_names.at(call_idx).size();

  return kLrtStatusOk;
}

LrtStatus LrtCompiledResultGetNumCalls(LrtCompiledResult compiled_result,
                                       lrt_param_index_t* num_calls) {
  *num_calls = compiled_result->graph_names.size();
  return kLrtStatusOk;
}

void LrtCompiledResultDestroy(LrtCompiledResult compiled_result) {
  delete compiled_result;
}

//
// Plugin Definition
//

// Plugins can hold state.
struct LrtCompilerPluginT {
  QnnManager qnn;
};

LrtStatus LrtPluginInit(LrtCompilerPlugin* compiler_plugin) {
  auto* plugin = new LrtCompilerPluginT;
  LRT_RETURN_STATUS_IF_NOT_OK(qnn::SetupAll(plugin->qnn));
  *compiler_plugin = plugin;
  return kLrtStatusOk;
}

void LrtPluginDestroy(LrtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

bool IsOpSupported(LrtOp op) {
  using TyInfo = graph_tools::RankedTypeInfo;

  // NOTE: Currently we are demoing by just mapping simple f32 mul ops.
  // In the limit this function withh want to leverage QNN SDK's getSuportedOps
  // feature (along with our op/type mappings).

  static const TyInfo supported_op_type = {kLrtElementTypeFloat32, {2, 2}};
  return graph_tools::MatchOpType(op, {supported_op_type, supported_op_type},
                                  {supported_op_type}, kLrtOpCodeTflMul);
}

LrtStatus LrtPluginPartitionModel(LrtCompilerPlugin compiler_plugin,
                                  LrtModel model, LrtOpList selected_ops) {
  LRT_ASSIGN_OR_RETURN_STATUS(auto subgraph, graph_tools::GetSubgraph(model));
  LRT_ASSIGN_OR_RETURN_STATUS(auto ops, graph_tools::GetSubgraphOps(subgraph));

  for (auto op : ops) {
    if (!IsOpSupported(op)) {
      continue;
    }

    LRT_RETURN_STATUS_IF_NOT_OK(PushOp(selected_ops, op));
  }

  return kLrtStatusOk;
}

// Composes a QNN graph with the context inside qnn from subgraph. On success,
// will write the QNN graph name (entry point) to output param.
LrtStatus ComposeGraph(QnnManager& qnn, LrtSubgraph subgraph,
                       std::string& qnn_graph_name) {
  // TODO: Implement this.
  qnn_graph_name = "Unimplemented_QNN_Graph";
  return kLrtStatusOk;
}

LrtStatus LrtPluginCompile(LrtCompilerPlugin compiler_plugin,
                           LrtSubgraphArray partitions,
                           lrt_param_index_t num_partitions,
                           LrtCompiledResult* compiled_result) {
  // NOTE: Currently we are demoing by just handling a simple case where
  // there is one partitions and the partitions is as follows:

  // func(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>)
  //   %0 = tfl.mul(%arg0, %arg1)
  //   return %0

  if (num_partitions != 1) {
    std::cerr << "Only 1 partition currently supported.\n";
    return kLrtStatusErrorUnsupported;
  }
  auto subgraph = partitions[0];

  LRT_ASSIGN_OR_RETURN_STATUS(auto inputs,
                              graph_tools::GetSubgraphInputs(subgraph));
  if (inputs.size() != 2) {
    std::cerr << "Only 2 inputs currently supported\n";
    return kLrtStatusErrorUnsupported;
  }

  LRT_ASSIGN_OR_RETURN_STATUS(auto outputs,
                              graph_tools::GetSubgraphOutputs(subgraph));
  if (outputs.size() != 1) {
    std::cerr << "Only 1 output currently supported\n";
    return kLrtStatusErrorUnsupported;
  }

  LRT_ASSIGN_OR_RETURN_STATUS(auto ops, graph_tools::GetSubgraphOps(subgraph));
  if (ops.size() != 1) {
    std::cerr << "Only one op subgraphs supported\n";
    return kLrtStatusErrorUnsupported;
  }

  LrtCompiledResult result = new LrtCompiledResultT;
  result->graph_names.reserve(num_partitions);

  LRT_RETURN_STATUS_IF_NOT_OK(ComposeGraph(compiler_plugin->qnn, subgraph,
                                           result->graph_names.emplace_back()));

  LRT_RETURN_STATUS_IF_NOT_OK(
      compiler_plugin->qnn.GenerateContextBin(result->context_bin));

  *compiled_result = result;

  return kLrtStatusOk;
}
