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

#include "tensorflow/compiler/jit/pjrt_compile_util.h"

#include <vector>

#include "tensorflow/compiler/jit/device_compilation_profiler.h"
#include "tensorflow/compiler/jit/device_compiler.h"
#include "tensorflow/compiler/jit/xla_compile_util.h"
#include "tensorflow/compiler/jit/xla_compiler_options_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/pjrt/pjrt_client.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

using PjRtDeviceCompiler =
    DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>;

absl::Status CompileToPjRtLoadedExecutable(
    const DeviceBase* device, const XlaPlatformInfo& platform_info,
    const NameAttrList& function,
    const std::vector<XlaCompiler::Argument>& args,
    DeviceCompileMode compile_mode, bool has_ref_vars,
    bool may_alias_resource_update, FunctionLibraryRuntime* flr,
    ResourceMgr* rm, const XlaCompiler::CompilationResult** compilation_result,
    xla::PjRtClient** client, xla::PjRtLoadedExecutable** executable) {
  PjRtDeviceCompiler* pjrt_device_compiler;
  DeviceCompilationProfiler* profiler;
  TF_RETURN_IF_ERROR(GetOrCreatePjRtDeviceCompilerAndProfiler(
      platform_info, rm, flr, &pjrt_device_compiler, &profiler));

  // Hold the reference to the PJRT device compiler and profiler during
  // evaluation. (We could probably free them sooner because the ResourceMgr
  // will retain references, but this is more obviously correct.)
  core::ScopedUnref pjrt_device_compiler_ref(pjrt_device_compiler);
  core::ScopedUnref profiler_ref(profiler);

  *client = pjrt_device_compiler->client();

  XlaCompiler::Options options = GenerateCompilerOptionsForPjRt(
      *flr, device, platform_info, pjrt_device_compiler);

  XlaCompiler::CompileOptions compile_options =
      GenerateCompileOptions(has_ref_vars, may_alias_resource_update);

  return pjrt_device_compiler->CompileIfNeeded(
      options, function, args, compile_options, compile_mode, profiler,
      compilation_result, executable);
}

absl::Status CompileToPjRtLoadedExecutable(
    const OpKernelContext& ctx, const XlaPlatformInfo& platform_info,
    const NameAttrList& function,
    const std::vector<XlaCompiler::Argument>& args,
    DeviceCompileMode compile_mode, bool has_ref_vars,
    bool may_alias_resource_update,
    const XlaCompiler::CompilationResult** compilation_result,
    xla::PjRtClient** client, xla::PjRtLoadedExecutable** executable) {
  TF_ASSIGN_OR_RETURN(ResourceMgr * rm, GetResourceMgrForDeviceCompiler(
                                            ctx, platform_info.device_type()));
  return CompileToPjRtLoadedExecutable(
      ctx.device(), platform_info, function, args, compile_mode, has_ref_vars,
      may_alias_resource_update, ctx.function_library(), rm, compilation_result,
      client, executable);
}

}  // namespace tensorflow
