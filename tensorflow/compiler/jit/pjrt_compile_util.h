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
#ifndef TENSORFLOW_COMPILER_JIT_PJRT_COMPILE_UTIL_H_
#define TENSORFLOW_COMPILER_JIT_PJRT_COMPILE_UTIL_H_

#include "tensorflow/compiler/jit/xla_compile_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/pjrt/pjrt_client.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// Compiles a `function` to PjRtLoadedExecutable `executable` with `ctx`.
// The compilation result is output in `compilation_result`. The PJRT client
// used for compilation is output in `client`. The PJRT executable is output in
// `executable`.
absl::Status CompileToPjRtLoadedExecutable(
    const OpKernelContext& ctx, const XlaPlatformInfo& platform_info,
    const NameAttrList& function,
    const std::vector<XlaCompiler::Argument>& args,
    DeviceCompileMode compile_mode, bool has_ref_vars,
    bool may_alias_resource_update,
    const XlaCompiler::CompilationResult** compilation_result,
    xla::PjRtClient** client, xla::PjRtLoadedExecutable** executable);

// Similar to the above function but it does not take a OpKernelContext.
// Instead, it takes the following arguments that are obtained from
// OpKernelContext in the above function.
// - `device`: the device used to compile the function.
// - `rm`: the resource manager for DeviceCompiler to store JIT-compiled XLA
// computation.
// - `flr`: the FunctionLibraryRuntime for the `function`.
absl::Status CompileToPjRtLoadedExecutable(
    const DeviceBase* device, const XlaPlatformInfo& platform_info,
    const NameAttrList& function,
    const std::vector<XlaCompiler::Argument>& args,
    DeviceCompileMode compile_mode, bool has_ref_vars,
    bool may_alias_resource_update, FunctionLibraryRuntime* flr,
    ResourceMgr* rm, const XlaCompiler::CompilationResult** compilation_result,
    xla::PjRtClient** client, xla::PjRtLoadedExecutable** executable);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_PJRT_COMPILE_UTIL_H_
