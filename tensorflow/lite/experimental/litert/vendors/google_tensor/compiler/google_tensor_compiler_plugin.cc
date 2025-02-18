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

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"
//
// Configurations
//

namespace {

constexpr char kPluginManufacturer[] = "GoogleTensor";

constexpr const char* kPluginSocModels[] = {
    "P25",
};  // get the name for plugin soc model

constexpr LiteRtOpCode kSupportedOps[] = {
    kLiteRtOpCodeTflMul,
};
// clang format on

constexpr auto kNumPluginSocModels =
    sizeof(kPluginSocModels) / sizeof(kPluginSocModels[0]);

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
  *supported_hardware = kLiteRtHwAcceleratorNpu;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models) {
  if (compiler_plugin == nullptr || num_supported_soc_models == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = kNumPluginSocModels;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (compiler_plugin == nullptr || soc_model_idx >= kNumPluginSocModels ||
      soc_model_name == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *soc_model_name = kPluginSocModels[soc_model_idx];
  return kLiteRtStatusOk;
}

//
// Compiled Result Definition
//

struct LiteRtCompiledResultT {
  std::string byte_code;
  std::vector<std::string> per_op_data;
};

LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, const void** byte_code,
    size_t* byte_code_size) {
  *byte_code = compiled_result->byte_code.data();
  *byte_code_size = compiled_result->byte_code.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size) {
  if (call_idx >= compiled_result->per_op_data.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }

  *call_info = compiled_result->per_op_data.at(call_idx).data();
  *call_info_size = compiled_result->per_op_data.at(call_idx).size();

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  *num_calls = compiled_result->per_op_data.size();
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
  *compiler_plugin = new LiteRtCompilerPluginT;
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  if (compiler_plugin == nullptr) {
    return;
  }
  delete compiler_plugin;
}

namespace {
//  TODO(abhirs): update the function to use the darwinn inbuilt way of
//  finding supportedops
bool IsOpSupported(const litert::Op& op) {
  for (auto supported_op : kSupportedOps) {
    if (supported_op == op.Code()) {
      return true;
    }
  }
  return false;
}

}  // namespace

LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  ::litert::Subgraph graph(subgraph);
  for (const auto& op : graph.Ops()) {
    if (!IsOpSupported(op)) {
      continue;
    }

    LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get()));
  }

  return kLiteRtStatusOk;
}

namespace {

absl::string_view convert_to_tfl(LiteRtSubgraph subgraph) {
  // TODO(abhirs): implement this
  // LiteRtModelT model;
  // model.EmplaceSubgraph(subgraph);
  // auto serialized = litert::internal::SerializeModel(std::move(model));
  // if (!serialized) {
  //   return "";
  // }
  // absl::string_view buffer(reinterpret_cast<const char*>(serialized->Data()),
  //                          serialized->Size());
  absl::string_view buffer = "";
  return buffer;
}

LiteRtStatus CompileSinglePartition(LiteRtParamIndex partition_index,
                                    LiteRtSubgraph subgraph,
                                    const char* soc_model,
                                    LiteRtCompiledResultT& result) {
  //  TODO(abhirs): implement this
  // 1. Convert the subgraph to a flatbuffer
  absl::string_view buffer = convert_to_tfl(subgraph);
  if (buffer.empty()) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  // 2. compile the flatbuffer using compiler_api_wrapper
  // 3. Get the bytecode from the compiled executable
  // 4. Get the per op data from the compiled executable
  // 5. Store the per op data in the result
  // 6. Store the bytecode in the result
  return kLiteRtStatusOk;
}
}  // namespace

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtSubgraph* partitions, LiteRtParamIndex num_partitions,
    LiteRtCompiledResult* compiled_result) {
  if (compiler_plugin == nullptr || soc_model == nullptr ||
      partitions == nullptr || compiled_result == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LiteRtCompiledResult result = new LiteRtCompiledResultT;
  for (auto i = 0; i < num_partitions; ++i) {
    LITERT_RETURN_IF_ERROR(
        CompileSinglePartition(i, partitions[i], soc_model, *result));
  }
  *compiled_result = result;
  return kLiteRtStatusOk;
}
