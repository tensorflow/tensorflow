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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_API_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_API_H_

#include <stddef.h>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"

// Wrapper for dynamically loaded LiteRtCompilerPlugin library. See
// "litert_compiler_plugin.h".

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//
// Api Interface
//

typedef LiteRtStatus (*LiteRtGetCompilerPluginVersionT)(LiteRtApiVersion*);

typedef const char* (*LiteRtGetCompilerPluginSocManufacturerT)();

typedef LiteRtStatus (*LiteRtCreateCompilerPluginT)(LiteRtCompilerPlugin*);

typedef void (*LiteRtDestroyCompilerPluginT)(LiteRtCompilerPlugin);

typedef LiteRtStatus (*LiteRtGetCompilerPluginSupportedHardwareT)(
    LiteRtCompilerPlugin, LiteRtHwAccelerators*);

typedef LiteRtStatus (*LiteRtGetNumCompilerPluginSupportedSocModelsT)(
    LiteRtCompilerPlugin, LiteRtParamIndex*);

typedef LiteRtStatus (*LiteRtGetCompilerPluginSupportedSocModelT)(
    LiteRtCompilerPlugin, LiteRtParamIndex soc_model_idx,
    const char** soc_moel_idx);

typedef LiteRtStatus (*LiteRtCompilerPluginPartitionT)(
    LiteRtCompilerPlugin, const char* soc_model, LiteRtSubgraph subgraph,
    LiteRtOpList selected_ops);

typedef LiteRtStatus (*LiteRtCompilerPluginCompileT)(
    LiteRtCompilerPlugin, const char* soc_model, LiteRtModel partitions,
    LiteRtCompiledResult* compiled_result);

typedef void (*LiteRtDestroyCompiledResultT)(LiteRtCompiledResult);

typedef LiteRtStatus (*LiteRtGetCompiledResultByteCodeT)(
    LiteRtCompiledResult, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size);

typedef LiteRtStatus (*LiteRtCompiledResultNumByteCodeModulesT)(
    LiteRtCompiledResult, LiteRtParamIndex* num_byte_code);

typedef LiteRtStatus (*LiteRtGetCompiledResultCallInfoT)(
    LiteRtCompiledResult, LiteRtParamIndex call_idx, const void** call_info,
    size_t* call_info_size, LiteRtParamIndex* byte_code_idx);

typedef LiteRtStatus (*LiteRtGetNumCompiledResultCallsT)(
    LiteRtCompiledResult, LiteRtParamIndex* num_calls);

typedef LiteRtStatus (*LiteRtCompilerPluginSetFlagsT)(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex num_flags,
    const char** keys, const char** values);

//
// Function Pointer Container
//

// Wraps all resolved functions from api interface.
struct LiteRtCompilerPluginApi {
  LiteRtGetCompilerPluginVersionT get_compiler_plugin_version;
  LiteRtGetCompilerPluginSocManufacturerT get_compiler_plugin_soc_manufacturer;
  LiteRtCreateCompilerPluginT create_compiler_plugin;
  LiteRtDestroyCompilerPluginT destroy_compiler_plugin;

  LiteRtGetCompilerPluginSupportedHardwareT
      get_compiler_plugin_supported_hardware;
  LiteRtGetNumCompilerPluginSupportedSocModelsT
      get_num_compiler_plugin_supported_models;
  LiteRtGetCompilerPluginSupportedSocModelT
      get_compiler_plugin_supported_soc_model;

  LiteRtCompilerPluginPartitionT compiler_plugin_partition;
  LiteRtCompilerPluginCompileT compiler_plugin_compile;

  LiteRtDestroyCompiledResultT destroy_compiled_result;
  LiteRtGetCompiledResultByteCodeT get_compiled_result_byte_code;
  LiteRtCompiledResultNumByteCodeModulesT get_compiled_result_num_byte_code;
  LiteRtGetCompiledResultCallInfoT get_compiled_result_call_info;
  LiteRtGetNumCompiledResultCallsT get_compiled_result_num_calls;

  LiteRtCompilerPluginSetFlagsT set_flags;
};

#ifdef __cplusplus
}

#include "absl/strings/string_view.h"

static constexpr absl::string_view kLiteRtGetCompilerPluginVersion =
    "LiteRtGetCompilerPluginVersion";

static constexpr absl::string_view kLiteRtGetCompilerPluginSupportedHardware =
    "LiteRtGetCompilerPluginSupportedHardware";

static constexpr absl::string_view kLiteRtGetCompilerPluginSocManufacturer =
    "LiteRtGetCompilerPluginSocManufacturer";
static constexpr absl::string_view
    kLiteRtGetNumCompilerPluginSupportedSocModels =
        "LiteRtGetNumCompilerPluginSupportedSocModels";
static constexpr absl::string_view kLiteRtGetCompilerPluginSupportedSocModel =
    "LiteRtGetCompilerPluginSupportedSocModel";

static constexpr absl::string_view kLiteRtCreateCompilerPlugin =
    "LiteRtCreateCompilerPlugin";
static constexpr absl::string_view kLiteRtDestroyCompilerPlugin =
    "LiteRtDestroyCompilerPlugin";

static constexpr absl::string_view kLiteRtCompilerPluginPartition =
    "LiteRtCompilerPluginPartition";
static constexpr absl::string_view kLiteRtCompilerPluginCompile =
    "LiteRtCompilerPluginCompile";

static constexpr absl::string_view kLiteRtDestroyCompiledResult =
    "LiteRtDestroyCompiledResult";
static constexpr absl::string_view kLiteRtGetCompiledResultByteCode =
    "LiteRtGetCompiledResultByteCode";
static constexpr absl::string_view kLiteRtCompiledResultNumByteCodeModules =
    "LiteRtCompiledResultNumByteCodeModules";
static constexpr absl::string_view kLiteRtGetCompiledResultCallInfo =
    "LiteRtGetCompiledResultCallInfo";
static constexpr absl::string_view kLiteRtGetNumCompiledResultCalls =
    "LiteRtGetNumCompiledResultCalls";

static constexpr absl::string_view kLiteRtCompilerPluginSetFlags =
    "LiteRtCompilerPluginSetFlags";

#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_API_H_
