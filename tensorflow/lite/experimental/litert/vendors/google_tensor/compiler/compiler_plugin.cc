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
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/vendors/google_tensor/adapter.h"

//
// Configurations
//

namespace google_tensor {

constexpr char kPluginManufacturer[] = "GoogleTensor";

constexpr const char* kPluginSocModels[] = {
    "P25",
};  // get the name for plugin soc model

constexpr LiteRtOpCode kUnSupportedOps[] = {
    kLiteRtOpCodeTflAssignVariable,
    kLiteRtOpCodeTflBidirectionalSequenceLstm,
    kLiteRtOpCodeTflBroadcastArgs,
    kLiteRtOpCodeTflBucketize,
    kLiteRtOpCodeTflCallOnce,
    kLiteRtOpCodeTflComplexAbs,
    kLiteRtOpCodeTflConv3d,
    kLiteRtOpCodeTflConv3dTranspose,
    kLiteRtOpCodeTflDensify,
    kLiteRtOpCodeTflFakeQuant,
    kLiteRtOpCodeTflHashtable,
    kLiteRtOpCodeTflHashtableFind,
    kLiteRtOpCodeTflHashtableImport,
    kLiteRtOpCodeTflHashtableSize,
    kLiteRtOpCodeTflImag,
    kLiteRtOpCodeTflLocalResponseNormalization,
    kLiteRtOpCodeTflMatrixDiag,
    kLiteRtOpCodeTflMatrixSetDiag,
    kLiteRtOpCodeTflMultinomial,
    kLiteRtOpCodeTflNonMaxSuppressionV4,
    kLiteRtOpCodeTflNonMaxSuppressionV5,
    kLiteRtOpCodeTflRandomStandardNormal,
    kLiteRtOpCodeTflRandomUniform,
    kLiteRtOpCodeTflRank,
    kLiteRtOpCodeTflReadVariable,
    kLiteRtOpCodeTflReal,
    kLiteRtOpCodeTflReduceProd,
    kLiteRtOpCodeTflReverseSequence,
    kLiteRtOpCodeTflRfft2d,
    kLiteRtOpCodeTflSegmentSum,
    kLiteRtOpCodeTflShape,
    kLiteRtOpCodeTflSparseToDense,
    kLiteRtOpCodeTflSvdf,
    kLiteRtOpCodeTflUnidirectionalSequenceRnn,
    kLiteRtOpCodeTflUnique,
    kLiteRtOpCodeTflUnsortedSegmentMax,
    kLiteRtOpCodeTflUnsortedSegmentMin,
    kLiteRtOpCodeTflUnsortedSegmentProd,
    kLiteRtOpCodeTflUnsortedSegmentSum,
    kLiteRtOpCodeTflVarHandle,
    kLiteRtOpCodeTflWhere,
};
// clang format on

constexpr auto kNumPluginSocModels =
    sizeof(kPluginSocModels) / sizeof(kPluginSocModels[0]);

}  // namespace google_tensor

LiteRtStatus LiteRtGetCompilerPluginVersion(LiteRtApiVersion* api_version) {
  if (api_version == nullptr) {
    LITERT_LOG(LITERT_ERROR, "%s", "api_version is nullptr");
    return kLiteRtStatusErrorInvalidArgument;
  }
  api_version->major = LITERT_API_VERSION_MAJOR;
  api_version->minor = LITERT_API_VERSION_MINOR;
  api_version->patch = LITERT_API_VERSION_PATCH;
  return kLiteRtStatusOk;
}

const char* LiteRtGetCompilerPluginSocManufacturer() {
  return google_tensor::kPluginManufacturer;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware) {
  if (!compiler_plugin || !supported_hardware) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "compiler_plugin or supported_hardware is nullptr");
    return kLiteRtStatusErrorInvalidArgument;
  }
  *supported_hardware = kLiteRtHwAcceleratorNpu;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models) {
  if (compiler_plugin == nullptr || num_supported_soc_models == nullptr) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "compiler_plugin or num_supported_soc_models is nullptr");
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = google_tensor::kNumPluginSocModels;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (compiler_plugin == nullptr ||
      soc_model_idx >= google_tensor::kNumPluginSocModels ||
      soc_model_name == nullptr) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "compiler_plugin or soc_model_idx or soc_model_name is nullptr");
    return kLiteRtStatusErrorInvalidArgument;
  }
  *soc_model_name = google_tensor::kPluginSocModels[soc_model_idx];
  return kLiteRtStatusOk;
}

//
// Compiled Result Definition
//

// TODO (abhirs): Revisit this struct after updating the compiler api wrapper to
// return multiple bytecodes.
struct LiteRtCompiledResultT {
  std::string byte_code;
  std::vector<std::string> per_op_data;
};

LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  if (!compiled_result || !byte_code || !byte_code_size ||
      (byte_code_idx != 0)) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "compiled_result or byte_code or byte_code_size"
               "or byte_code_idx is nullptr");
    return kLiteRtStatusErrorInvalidArgument;
  }
  *byte_code = compiled_result->byte_code.data();
  *byte_code_size = compiled_result->byte_code.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code) {
  if (!compiled_result || !num_byte_code) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "compiled_result or num_byte_code is nullptr");
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_byte_code = 1;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx) {
  if (!compiled_result || !call_info || !call_info_size) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "compiled_result or call_info or call_info_size is nullptr");
    return kLiteRtStatusErrorInvalidArgument;
  } else if (call_idx >= compiled_result->per_op_data.size()) {
    LITERT_LOG(LITERT_ERROR, "%s", "call_idx is out of bounds");
    return kLiteRtStatusErrorIndexOOB;
  }

  *call_info = compiled_result->per_op_data.at(call_idx).data();
  *call_info_size = compiled_result->per_op_data.at(call_idx).size();
  *byte_code_idx = 0;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  if (!compiled_result || !num_calls) {
    LITERT_LOG(LITERT_ERROR, "%s", "compiled_result or num_calls is nullptr");
    return kLiteRtStatusErrorInvalidArgument;
  }
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

LiteRtStatus LiteRtCompilerPluginSetFlags(LiteRtCompilerPlugin compiler_plugin,
                                          LiteRtParamIndex num_flags,
                                          const char** keys,
                                          const char** values) {
  // IMPLEMENT ME
  return kLiteRtStatusOk;
}

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

namespace google_tensor {
//  TODO(abhirs): update the function to use the darwinn inbuilt way of
//  finding supportedops
bool IsOpSupported(const litert::Op& op) {
  for (auto unsupported_op : kUnSupportedOps) {
    if (unsupported_op == op.Code()) {
      return false;
    }
  }
  return true;
}

}  // namespace google_tensor

LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  ::litert::Subgraph graph(subgraph);
  for (const auto& op : graph.Ops()) {
    if (!google_tensor::IsOpSupported(op)) {
      continue;
    }

    LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 0));
  }

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  if (compiler_plugin == nullptr || soc_model == nullptr ||
      partitions == nullptr || compiled_result == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto model = litert::Model::CreateFromNonOwnedHandle(partitions);
  const auto num_partitions = model.NumSubgraphs();
  LITERT_LOG(LITERT_INFO,
             "Starting GoogleTensor Compilation for %d subgraphs, soc_model=%s",
             num_partitions, soc_model);

  // Serialize model.
  LITERT_LOG(LITERT_INFO, "%s", "Serializing model");
  litert::OwningBufferRef buf;
  auto [data, size, offset] = buf.GetWeak();
  const auto opts = litert::SerializationOptions::Defaults();
  LITERT_RETURN_IF_ERROR(
      LiteRtSerializeModel(partitions, &data, &size, &offset, false, opts));
  // TODO(abhirs): add support for serializing subgraphs

  absl::string_view buffer_str(reinterpret_cast<const char*>(buf.Data()),
                               buf.Size());

  // Loading Google Tensor Compiler Adapter
  LITERT_LOG(LITERT_INFO, "%s", "Loading Google Tensor Compiler Adapter");
  auto adapter_result = litert::google_tensor::Adapter::Create(
      /*shared_library_dir=*/std::nullopt);
  if (!adapter_result.HasValue()) {
    const auto& error_message = adapter_result.Error().Message();
    LITERT_LOG(LITERT_ERROR, "Failed to create adapter: %s",
               error_message.c_str());
    return kLiteRtStatusErrorRuntimeFailure;
  }

  // Compile model.
  LITERT_LOG(LITERT_INFO, "%s", "Compiling model...");
  // TODO(abhirs): add support for multiple bytecodes
  absl::string_view soc_model_view(soc_model);
  std::string compiled;
  auto compile_status = adapter_result.Value()->api().compile(
      buffer_str, soc_model_view, &compiled);

  if (!compile_status.ok()) {
    LITERT_LOG(
        LITERT_ERROR, "%s",
        absl::StrCat("Failed to compile model: ", compile_status.message())
            .c_str());
    return kLiteRtStatusErrorRuntimeFailure;
  }

  // Result
  auto result = std::make_unique<LiteRtCompiledResultT>();

  result->byte_code = std::string(compiled.data(), compiled.size());
  // Generate per_op_data.
  for (auto i = 0; i < num_partitions; ++i) {
    result->per_op_data.emplace_back(absl::StrFormat("Partition_%d", i));
  }
  *compiled_result = result.release();
  return kLiteRtStatusOk;
}
