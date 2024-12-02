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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "third_party/qairt/latest/include/QNN/HTP/QnnHtpDevice.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/qnn_compose_graph.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_manager.h"

using ::litert::qnn::QnnManager;

//
// Configurations
//

namespace {

constexpr char kPluginManufacturer[] = "Qualcomm";

// clang-format off
constexpr std::pair<const char*, QnnHtpDevice_Arch_t> kPluginSocModels[] = {
    {"V68", QNN_HTP_DEVICE_ARCH_V68},
    {"V69", QNN_HTP_DEVICE_ARCH_V69},
    {"V73", QNN_HTP_DEVICE_ARCH_V73},
    {"V75", QNN_HTP_DEVICE_ARCH_V75},
    {"V79", QNN_HTP_DEVICE_ARCH_V79},
};

constexpr LiteRtOpCode kSupportedOps[] = {
  kLiteRtOpCodeTflAdd,
  kLiteRtOpCodeTflDiv,
  kLiteRtOpCodeTflMul,
  kLiteRtOpCodeTflRsqrt,
  kLiteRtOpCodeTflSlice,
  kLiteRtOpCodeTflSelect,
  kLiteRtOpCodeTflSelectV2,
  kLiteRtOpCodeTflSub,
  kLiteRtOpCodeTflTanh,
  kLiteRtOpCodeTflBatchMatmul,
  kLiteRtOpCodeTflReshape,
  kLiteRtOpCodeTflSum,
  kLiteRtOpCodeTflConcatenation,
  kLiteRtOpCodeTflSoftmax,
  kLiteRtOpCodeTflCast,
  kLiteRtOpCodeTflTranspose,
  kLiteRtOpCodeTflSin,
  kLiteRtOpCodeTflCos,
  kLiteRtOpCodeTflFullyConnected,
};
// clang-format on

constexpr auto kNumPluginSocModels =
    sizeof(kPluginSocModels) / sizeof(kPluginSocModels[0]);

std::optional<QnnHtpDevice_Arch_t> FindSocModel(
    absl::string_view soc_model_name) {
  std::optional<QnnHtpDevice_Arch_t> soc_model;
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

struct LiteRtCompiledResultT {
  std::vector<char> context_bin;
  std::vector<std::string> graph_names;
};

LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, const void** byte_code,
    size_t* byte_code_size) {
  *byte_code = compiled_result->context_bin.data();
  *byte_code_size = compiled_result->context_bin.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size) {
  if (call_idx >= compiled_result->graph_names.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }

  *call_info = compiled_result->graph_names.at(call_idx).data();
  *call_info_size = compiled_result->graph_names.at(call_idx).size();

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  *num_calls = compiled_result->graph_names.size();
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
  // NOTE: Currently we are demoing by just mapping simple f32 mul ops.
  // In the limit this function withh want to leverage QNN SDK's getSuportedOps
  // feature (along with our op/type mappings).
  // Use a very loose guard for now -- only checking if op code is supported.

  for (auto supported_op : kSupportedOps) {
    if (op.Code() == supported_op) {
      return true;
    }
  }
  return false;
}

}  // namespace

LiteRtStatus LiteRtCompilerPluginPartitionModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtModel model,
    LiteRtOpList selected_ops) {
  auto m = litert::Model::CreateFromNonOwnedHandle(model);
  auto subgraph = m.MainSubgraph();
  if (!subgraph) {
    return subgraph.Error().Status();
  }

  for (const auto& op : subgraph->Ops()) {
    if (!IsOpSupported(op)) {
      continue;
    }

    LITERT_RETURN_STATUS_IF_NOT_OK(LiteRtPushOp(selected_ops, op.Get()));
  }

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtSubgraphArray partitions, LiteRtParamIndex num_partitions,
    LiteRtCompiledResult* compiled_result) {
  LITERT_LOG(LITERT_INFO, "Starting QNN Compilation for %d subgraphs",
             num_partitions);
  auto opt_soc_model = FindSocModel(soc_model);
  if (opt_soc_model.has_value()) {
    LITERT_LOG(LITERT_INFO, "For arch: %d", opt_soc_model.value());
  }

  // Initialize SDK and load qnn shared libraries.

  LITERT_LOG(LITERT_INFO, "%s", "Creating QNN manager");
  auto backend_configs = QnnManager::DefaultBackendConfigs();
  auto qnn_manager = QnnManager::Create(
      backend_configs, /*shared_library_dir=*/std::nullopt, opt_soc_model);
  if (!qnn_manager) {
    LITERT_LOG(LITERT_ERROR, "%s", qnn_manager.Error().Message().data());
    return qnn_manager.Error().Status();
  }
  LITERT_LOG(LITERT_INFO, "%s", "QNN manager created");

  // Initialize context.

  LITERT_LOG(LITERT_INFO, "%s", "Creating context handle");
  auto context_configs = QnnManager::DefaultContextConfigs();
  auto context_handle = (*qnn_manager)->CreateContextHandle(context_configs);
  if (!context_handle) {
    LITERT_LOG(LITERT_ERROR, "%s", context_handle.Error().Message().data());
    return context_handle.Error().Status();
  }
  LITERT_LOG(LITERT_INFO, "%s", "Context handle created");

  auto result = std::make_unique<LiteRtCompiledResultT>();

  // Compose graphs.

  LITERT_LOG(LITERT_INFO, "%s", "Composing graph(s)");
  // TODO: Support multiple partitions in QCC plugin compile.
  LITERT_ENSURE_SUPPORTED(num_partitions, 1);
  {
    std::string& entry_point_name = result->graph_names.emplace_back();
    entry_point_name = "qnn_partition_0";
    LITERT_RETURN_STATUS_IF_NOT_OK(litert::qnn::ComposeGraph(
        **qnn_manager, context_handle->get(), partitions[0], entry_point_name));
  }
  LITERT_LOG(LITERT_INFO, "%s", "Graph composed");

  // Generate context binary.

  LITERT_LOG(LITERT_INFO, "%s", "Generating context binary");
  LITERT_RETURN_STATUS_IF_NOT_OK(
      (*qnn_manager)
          ->GenerateContextBinary(context_handle->get(), result->context_bin));
  LITERT_LOG(LITERT_INFO, "%s", "Context binary generated");

  *compiled_result = result.release();

  return kLiteRtStatusOk;
}
