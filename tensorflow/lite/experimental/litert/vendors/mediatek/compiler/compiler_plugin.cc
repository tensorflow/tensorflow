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

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/compile_model.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/create_model.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter.h"

//
// Configurations
//

using litert::Error;
using litert::Expected;
using litert::mediatek::NEURON_NO_ERROR;
using litert::mediatek::NEURON_PREFER_SUSTAINED_SPEED;
using litert::mediatek::NEURON_PRIORITY_HIGH;
using litert::mediatek::NeuronAdapter;
using litert::mediatek::NeuronCompilation;
using litert::mediatek::NeuronCompilationPtr;
using litert::mediatek::NeuronModel;
using litert::mediatek::NeuronModelPtr;

namespace {

constexpr char kPluginManufacturer[] = "MediaTek";

// clang-format off
constexpr std::pair<const char*, const char*> kPluginSocModels[] = {
    {"mt6853", "mt6853"},
    {"mt6877", "mt6877"},
    {"mt6878", "mt6878"},
    {"mt6879", "mt6879"},
    {"mt6886", "mt6886"},
    {"mt6893", "mt6893"},
    {"mt6895", "mt6895"},
    {"mt6897", "mt6897"},
    {"mt6983", "mt6983"},
    {"mt6985", "mt6985"},
    {"mt6989", "mt6989"},
    {"mt6991", "mt6991"},
};

constexpr LiteRtOpCode kSupportedOps[] = {
    kLiteRtOpCodeTflAdd,
};
// clang-format on

constexpr auto kNumPluginSocModels =
    sizeof(kPluginSocModels) / sizeof(kPluginSocModels[0]);

std::optional<const char*> FindSocModel(absl::string_view soc_model_name) {
  std::optional<const char*> soc_model;
  for (auto i = 0; i < kNumPluginSocModels; ++i) {
    if (soc_model_name == kPluginSocModels[i].first) {
      soc_model = kPluginSocModels[i].second;
      break;
    }
  }
  return soc_model;
}

}  // namespace

LiteRtStatus LiteRtGetCompilerPluginVersion(LiteRtApiVersion* api_version) {
  if (api_version == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  api_version->major = LITERT_API_VERSION_MAJOR;
  api_version->minor = LITERT_API_VERSION_MINOR;
  api_version->patch = LITERT_API_VERSION_PATCH;
  return kLiteRtStatusOk;
}

const char* LiteRtGetCompilerPluginSocManufacturer() {
  return kPluginManufacturer;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware) {
  if (!compiler_plugin || !supported_hardware) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *supported_hardware = kLiteRtHwAccelatorNpu;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models) {
  if (!compiler_plugin || !num_supported_soc_models) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = kNumPluginSocModels;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (!compiler_plugin || !soc_model_name) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (soc_model_idx < 0 || soc_model_idx >= kNumPluginSocModels) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *soc_model_name = kPluginSocModels[soc_model_idx].first;
  return kLiteRtStatusOk;
}

//
// Compiled Result Definition
//

// TODO: Revisit this struct after we extend the compiler plugin API to return
// results with more than one single bytecode.
struct LiteRtCompiledResultT {
  using Bytecode = std::vector<uint8_t>;
  std::vector<Bytecode> bytecodes;
  std::vector<std::string> graph_names;
};

LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, const void** byte_code,
    size_t* byte_code_size) {
  if (!compiled_result || !byte_code || !byte_code_size) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (compiled_result->bytecodes.size() > 1) {
    // TODO: Revisit this struct after we extend the compiler plugin API to
    // return results with more than one single bytecode.
    LITERT_LOG(LITERT_ERROR, "CompilerPlugin API supports only 1 NPU bytecode");
    return kLiteRtStatusErrorIndexOOB;
  }
  *byte_code = compiled_result->bytecodes[0].data();
  *byte_code_size = compiled_result->bytecodes[0].size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size) {
  if (!compiled_result || !call_info || !call_info_size) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (call_idx >= compiled_result->graph_names.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }

  auto& graph_name = compiled_result->graph_names[call_idx];
  *call_info = graph_name.data();
  *call_info_size = graph_name.size();

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  if (!compiled_result || !num_calls) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_calls = compiled_result->bytecodes.size();
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompiledResult(LiteRtCompiledResult compiled_result) {
  delete compiled_result;
}

//
// Plugin Definition
//

// Plugins can hold state.
struct LiteRtCompilerPluginT {};

LiteRtStatus LiteRtCreateCompilerPlugin(LiteRtCompilerPlugin* compiler_plugin) {
  auto* plugin = new LiteRtCompilerPluginT;
  *compiler_plugin = plugin;
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

namespace {

// TODO update this function to match the new legalizations.
bool IsOpSupported(const litert::Op& op) {
  // NOTE: Currently we are demoing by just mapping simple f32 mul ops.  Use a
  // very loose guard for now -- only checking if op code is supported.
  for (auto supported_op : kSupportedOps) {
    if (op.Code() == supported_op) {
      return true;
    }
  }
  return false;
}

}  // namespace

LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  litert::Subgraph graph(subgraph);
  for (const auto& op : graph.Ops()) {
    if (!IsOpSupported(op)) {
      continue;
    }

    LITERT_RETURN_STATUS_IF_NOT_OK(LiteRtPushOp(selected_ops, op.Get()));
  }

  return kLiteRtStatusOk;
}

namespace {

Expected<std::vector<uint8_t>> CompilePartition(
    NeuronAdapter& neuron_adapter, const litert::Subgraph& partition,
    const std::string& graph_name, std::optional<std::string> soc_model) {
  auto model = CreateModel(neuron_adapter, partition, graph_name);
  if (!model) {
    return model.Error();
  }

  auto compilation = CompileModel(neuron_adapter, model->get(), soc_model);
  if (!compilation) {
    return compilation.Error();
  }

  size_t bytecode_size;
  if (neuron_adapter.api().compilation_get_compiled_network_size(
          compilation->get(), &bytecode_size) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to get compiled network size");
  }

  std::vector<uint8_t> bytecode(bytecode_size);
  if (neuron_adapter.api().compilation_store_compiled_network(
          compilation->get(), bytecode.data(), bytecode.size()) !=
      NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to get compiled network");
  }

  return bytecode;
}

}  // namespace

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtSubgraph* partitions, LiteRtParamIndex num_partitions,
    LiteRtCompiledResult* compiled_result) {
  LITERT_LOG(LITERT_INFO,
             "Starting MediaTek Compilation for %d subgraphs, soc_model=%s",
             num_partitions, soc_model);

  auto opt_soc_model = soc_model ? FindSocModel(soc_model) : std::nullopt;
  if (opt_soc_model) {
    LITERT_LOG(LITERT_ERROR, "Compiling for MediaTek architecture: %s",
               *opt_soc_model);
  } else if (soc_model) {
    LITERT_LOG(LITERT_ERROR, "Unexpected SoC model: %s", soc_model);
    return kLiteRtStatusErrorInvalidArgument;
  }

  // Initialize SDK and load qnn shared libraries.

  auto neuron_adapter =
      NeuronAdapter::Create(/*shared_library_dir=*/std::nullopt);
  if (!neuron_adapter) {
    return neuron_adapter.Error().Status();
  }

  auto result = std::make_unique<LiteRtCompiledResultT>();
  for (auto i = 0; i < num_partitions; ++i) {
    auto partition = litert::Subgraph(partitions[i]);
    auto graph_name = absl::StrFormat("Partition_%d", i);
    auto bytecode = CompilePartition(**neuron_adapter, partition, graph_name,
                                     opt_soc_model);
    if (!bytecode) {
      LITERT_LOG(LITERT_INFO, "%s", bytecode.Error().Message().data());
      return bytecode.Error().Status();
    }

    result->bytecodes.emplace_back(*bytecode);
    result->graph_names.emplace_back(graph_name);
  }

  *compiled_result = result.release();
  return kLiteRtStatusOk;
}
