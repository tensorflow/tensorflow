/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// Defines the XlaCompileOnDemandOp.

#include "tensorflow/compiler/jit/xla_compile_on_demand_op.h"

#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/jit/device_compilation_cluster_signature.h"
#include "tensorflow/compiler/jit/device_compilation_profiler.h"
#include "tensorflow/compiler/jit/device_compiler.h"
#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/compiler/jit/variable_info_util.h"
#include "tensorflow/compiler/jit/xla_compiler_options_util.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {
namespace {
using PjRtDeviceCompiler =
    DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>;

XlaCompiler::CompileOptions GetCompileOptions() {
  XlaCompiler::CompileOptions compile_options;
  compile_options.is_entry_computation = true;
  compile_options.always_return_tuple = true;
  compile_options.use_tuple_arg = false;
  return compile_options;
}

// Gets `variables` from `ctx`, locks them and builds XlaCompiler::Arguments
// using them. Stores the arguments in `args`. `variables` and `args` passed in
// will be cleared before populating them.
absl::Status GetAndLockVariablesAndBuildXlaCompilerArguments(
    const OpKernelContext& ctx, const std::vector<const Tensor*>& inputs,
    const std::vector<int>& constant_indices,
    const std::vector<int>& variable_indices,
    std::vector<VariableInfo>* variables,
    std::vector<XlaCompiler::Argument>* args) {
  variables->clear();
  args->clear();
  TF_RETURN_IF_ERROR(GetVariableInfosFromInputs(ctx.resource_manager(),
                                                ctx.device(), inputs,
                                                variable_indices, variables));
  TF_RETURN_IF_ERROR(LockVariables(absl::MakeSpan(*variables)));
  TF_ASSIGN_OR_RETURN(*args,
                      XlaComputationLaunchContext::BuildXlaCompilerArguments(
                          constant_indices, inputs, *variables,
                          static_cast<Device*>(ctx.device())));
  return absl::OkStatus();
}
}  // namespace

XlaCompileOnDemandOp::XlaCompileOnDemandOp(OpKernelConstruction* ctx)
    : OpKernel(ctx),
      platform_info_(XlaPlatformInfoFromDevice(ctx->device())),
      function_(GetDeviceCompilerFunction(ctx->def())),
      canonical_function_(Canonicalize(function_)) {}

absl::Status XlaCompileOnDemandOp::Compile(
    const std::vector<XlaCompiler::Argument>& args, OpKernelContext* ctx,
    PjRtDeviceCompiler** pjrt_device_compiler,
    DeviceCompilationProfiler** profiler,
    const XlaCompiler::CompilationResult** result,
    xla::PjRtLoadedExecutable** executable) {
  TF_RETURN_IF_ERROR(GetOrCreatePjRtDeviceCompilerAndProfiler(
      *ctx, platform_info_, ctx->function_library(), pjrt_device_compiler,
      profiler));

  XlaCompiler::Options options =
      GenerateCompilerOptionsForPjRt(*(ctx->function_library()), ctx->device(),
                                     platform_info_, *pjrt_device_compiler);
  // No detailed logging for on demand op.
  options.detailed_logging = false;
  XlaCompiler::CompileOptions compile_options = GetCompileOptions();

  return (*pjrt_device_compiler)
      ->CompileSingleOpIfNeeded(options, function_, canonical_function_, args,
                                compile_options, ctx, *profiler, result,
                                executable);
}

void XlaCompileOnDemandOp::Compute(OpKernelContext* ctx) {
  const XlaCompiler::CompilationResult* result = nullptr;
  DeviceCompilationProfiler* profiler = nullptr;

  OP_REQUIRES(ctx, ctx->function_library(),
              absl::InternalError("Function library missing"));

  // Get constants, inputs and variables from the OpKernelContext.
  auto constant_indices_or = GetConstantInputIndicesFromContext(ctx);
  OP_REQUIRES_OK(ctx, constant_indices_or.status());
  std::vector<const Tensor*> inputs = InputsFromContext(ctx);
  std::vector<int> variable_indices =
      GetResourceVariableIndicesFromContext(ctx);

  std::vector<VariableInfo> variables;
  std::vector<XlaCompiler::Argument> args;
  // Lock variables for the whole duration of compile + execute.
  OP_REQUIRES_OK(ctx, GetAndLockVariablesAndBuildXlaCompilerArguments(
                          *ctx, inputs, *constant_indices_or, variable_indices,
                          &variables, &args));

  PjRtDeviceCompiler* pjrt_device_compiler = nullptr;
  xla::PjRtLoadedExecutable* pjrt_executable = nullptr;
  absl::Status status = Compile(args, ctx, &pjrt_device_compiler, &profiler,
                                &result, &pjrt_executable);
  // Hold the reference to the XLA device compiler and profiler during
  // evaluation. (We could probably free them sooner because the ResourceMgr
  // will retain references, but this is more obviously correct.)
  // We must also ensure these pointers are freed even if compilation fails.
  core::ScopedUnref pjrt_device_compiler_ref(pjrt_device_compiler);
  core::ScopedUnref profiler_ref(profiler);
  OP_REQUIRES_OK(ctx, status);

  VLOG(2) << "Compiled op with PJRT: " << ctx->status();
  VLOG(2) << "result != nullptr: " << (result != nullptr);
  VLOG(2) << "pjrt_executable != nullptr: " << (pjrt_executable != nullptr);
  VLOG(2) << "Executing with PJRT ...";

  OP_REQUIRES_OK(ctx, RunPjRtExecutable(inputs, variables, *result,
                                        pjrt_device_compiler->client(),
                                        pjrt_executable, ctx));

  VLOG(2) << "Completed executing with PJRT!";
}

}  // namespace tensorflow
