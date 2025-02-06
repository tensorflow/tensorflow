/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/jit/xla_compiler_options_util.h"

#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/framework/device_id_utils.h"
#include "tensorflow/core/framework/function.h"

namespace tensorflow {
namespace {
using XlaDeviceCompiler =
    DeviceCompiler<xla::LocalExecutable, xla::LocalClient>;
using PjRtDeviceCompiler =
    DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>;

inline void LogOptions(const XlaCompiler::Options& options) {
  VLOG(2) << "XlaCompiler::Options[device_type=" << options.device_type
          << ",device_ordinal=" << options.device_ordinal
          << ",client=" << options.client << ",flib_def=" << options.flib_def
          << ",graph_def_version=" << options.graph_def_version
          << ",options.shape_determination_fns.layout_preference_fn?="
          << (options.shape_determination_fns.layout_preference_fn != nullptr)
          << ",options.shape_determination_fns.shape_representation_fn?="
          << (options.shape_determination_fns.shape_representation_fn !=
              nullptr)
          << ",allow_cpu_custom_calls=" << options.allow_cpu_custom_calls
          << ",populate_resource_manager=" << options.populate_resource_manager
          << ",alias_passthrough_params=" << options.alias_passthrough_params
          << ",detailed_logging=" << options.detailed_logging << "]";
}
}  // namespace

XlaCompiler::Options GenerateCompilerOptions(
    const XlaDeviceCompiler& xla_device_compiler,
    const FunctionLibraryRuntime& function_library, DeviceBase* device,
    se::Stream* stream, const XlaPlatformInfo& platform_info,
    bool has_ref_vars) {
  XlaCompiler::Options options;
  options.client = static_cast<xla::LocalClient*>(xla_device_compiler.client());
  if (stream != nullptr) {
    options.device_ordinal = stream->parent()->device_ordinal();
  }
  options.device_type = xla_device_compiler.device_type();
  options.flib_def = function_library.GetFunctionLibraryDefinition();
  options.graph_def_version = function_library.graph_def_version();
  options.allow_cpu_custom_calls =
      (platform_info.platform_id() == se::host::kHostPlatformId);
  options.device_allocator = GetAllocator(device, stream, platform_info);
  if (platform_info.xla_device_metadata()) {
    options.shape_determination_fns =
        platform_info.xla_device_metadata()->default_shape_determination_fns();
  }
  // If reference variables are not present in the graph, we can safely alias
  // passthrough parameters without performing a copy.
  options.alias_passthrough_params =
      !has_ref_vars && !platform_info.is_on_xla_device();

  LogOptions(options);
  return options;
}

XlaCompiler::Options GenerateCompilerOptionsForTfrtTpu(
    const XlaDeviceCompiler& xla_device_compiler,
    const FunctionLibraryRuntime& function_library) {
  XlaCompiler::Options options;
  // TODO(b/238830423): consider device_ordinal and shape_determination_fns.
  options.device_type = xla_device_compiler.device_type();
  options.flib_def = function_library.GetFunctionLibraryDefinition();
  options.graph_def_version = function_library.graph_def_version();
  options.allow_cpu_custom_calls = false;
  options.alias_passthrough_params = false;
  return options;
}

XlaCompiler::Options GenerateCompilerOptionsForPjRt(
    const FunctionLibraryRuntime& function_library,
    const DeviceBase* device_base, const XlaPlatformInfo& platform_info,
    const DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>*
        pjrt_device_compiler) {
  return GenerateCompilerOptionsForPjRt(
      function_library.GetFunctionLibraryDefinition(),
      function_library.graph_def_version(), device_base, platform_info,
      pjrt_device_compiler);
}

XlaCompiler::Options GenerateCompilerOptionsForPjRt(
    const FunctionLibraryDefinition* function_library_def,
    int graph_def_version, const DeviceBase* device_base,
    const XlaPlatformInfo& platform_info,
    const PjRtDeviceCompiler* pjrt_device_compiler) {
  XlaCompiler::Options options;
  absl::StatusOr<int> platform_device_id =
      tsl::GetPlatformDeviceIdFromDeviceParsedName(
          device_base->parsed_name(),
          DeviceType(tensorflow::down_cast<const Device*>(device_base)
                         ->device_type()));
  if (platform_device_id.ok()) {
    options.device_ordinal = *platform_device_id;
  } else {
    options.device_ordinal = device_base->parsed_name().id;
  }
  options.flib_def = function_library_def;
  options.graph_def_version = graph_def_version;
  if (const auto* metadata = platform_info.xla_device_metadata();
      metadata != nullptr) {
    options.device_type = metadata->jit_device_type();
    options.shape_determination_fns =
        metadata->default_shape_determination_fns();
  } else if (const auto* metadata = platform_info.pjrt_device_metadata();
             metadata != nullptr) {
    options.device_type = metadata->jit_device_type();
    options.shape_determination_fns =
        metadata->default_shape_determination_fns();
  } else if (pjrt_device_compiler != nullptr) {
    options.device_type = pjrt_device_compiler->device_type();
  }
  // TODO(b/255826209): Confirm below options are correctly set after testing.
  options.allow_cpu_custom_calls = false;
  options.alias_passthrough_params = false;

  LogOptions(options);
  return options;
}

XlaCompiler::CompileOptions GenerateCompileOptions(
    bool has_ref_vars, bool may_alias_resource_update) {
  XlaCompiler::CompileOptions compile_options;
  compile_options.is_entry_computation = true;
  // Optimization: where possible, have the computation return a naked array
  // rather than a one-element tuple.
  compile_options.always_return_tuple = false;
  compile_options.alias_resource_update =
      !has_ref_vars && may_alias_resource_update;
  return compile_options;
}

}  // namespace tensorflow
