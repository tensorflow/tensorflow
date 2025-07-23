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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/jit/device_compilation_cluster_signature.h"
#include "tensorflow/compiler/jit/device_compilation_profiler.h"
#include "tensorflow/compiler/jit/device_compiler.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/compiler/jit/variable_info_util.h"
#include "tensorflow/compiler/jit/xla_compile_util.h"
#include "tensorflow/compiler/jit/xla_compiler_options_util.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/tf_pjrt_client.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/tfrt/common/pjrt_util.h"
#include "tsl/platform/errors.h"

namespace tensorflow {
namespace {
using XlaDeviceCompiler =
    DeviceCompiler<xla::LocalExecutable, xla::LocalClient>;
using PjRtDeviceCompiler =
    DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>;

XlaCompiler::CompileOptions GetCompileOptions(bool for_pjrt = false) {
  XlaCompiler::CompileOptions compile_options;
  compile_options.is_entry_computation = true;
  // Optimization: where possible, have the computation return a naked array
  // rather than a one-element tuple.
  compile_options.always_return_tuple = false;
  if (for_pjrt) {
    compile_options.use_tuple_arg = false;
    compile_options.always_return_tuple = true;
  }

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

absl::Status XlaCompileOnDemandOp::Run(
    const ResourceVarsSnapshot& variable_args,
    const XlaCompiler::CompilationResult* result,
    const XlaDeviceCompiler* xla_device_compiler,
    xla::LocalExecutable* executable, OpKernelContext* ctx) {
  xla::LocalClient* client =
      static_cast<xla::LocalClient*>(xla_device_compiler->client());

  se::Stream* stream =
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;
  std::shared_ptr<se::DeviceMemoryAllocator> allocator_ptr =
      GetAllocator(ctx->device(), stream, platform_info_);
  se::DeviceMemoryAllocator* allocator = allocator_ptr.get();
  XlaComputationLaunchContext launch_context(
      client, allocator, client->default_device_ordinal(),
      /*allocate_xla_tensors=*/platform_info_.xla_device_metadata() != nullptr,
      platform_info_.xla_device_metadata()
          ? platform_info_.xla_device_metadata()->UseMultipleStreams()
          : false);

  absl::flat_hash_map<int, const Tensor*> snapshot_ptrs;
  for (auto& p : variable_args) {
    snapshot_ptrs.emplace(p.first,
                          p.second.has_value() ? &p.second.value() : nullptr);
  }

  const xla::HloInputOutputAliasConfig& input_output_alias =
      executable->executable()->module().input_output_alias_config();
  absl::StatusOr<std::vector<xla::ExecutionInput>> execution_inputs =
      launch_context.PopulateInputs(ctx, result, snapshot_ptrs,
                                    /*missing_ctx_input_prefix=*/0,
                                    input_output_alias);
  TF_RETURN_IF_ERROR(execution_inputs.status());

  VLOG(2) << "Executing computation: " << name();
  xla::ExecutableRunOptions run_options;
  xla::gpu::GpuExecutableRunOptions gpu_options;
  xla::DeviceAssignment device_assignment;
  if (result->collective_info) {
    TF_RETURN_IF_ERROR(ResolveDeviceAssignment(ctx, *result->collective_info,
                                               run_options, device_assignment,
                                               gpu_options));
  }

  run_options.set_stream(stream);
  run_options.set_allocator(allocator);
  run_options.set_intra_op_thread_pool(&ctx->eigen_cpu_device());
  run_options.set_rng_seed(GetXLARandomSeed());

  absl::StatusOr<xla::ExecutionOutput> run_result =
      executable->Run(std::move(execution_inputs).value(), run_options);
  TF_RETURN_IF_ERROR(run_result.status());
  xla::ExecutionOutput execution_output = std::move(run_result).value();
  absl::StatusOr<std::vector<VariableInfo>> variable_infos =
      GatherVariableInfo(ctx, *result, 0);
  TF_RETURN_IF_ERROR(variable_infos.status());
  TF_RETURN_IF_ERROR(LockVariables(absl::MakeSpan(*variable_infos)));
  TF_RETURN_IF_ERROR(launch_context.PopulateOutputs(
      ctx, result, execution_output.ConsumeResult(),
      /*missing_ctx_input_prefix=*/0, absl::MakeSpan(*variable_infos),
      input_output_alias, snapshot_ptrs));
  return absl::OkStatus();
}

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
  XlaCompiler::CompileOptions compile_options = GetCompileOptions(true);

  return (*pjrt_device_compiler)
      ->CompileSingleOpIfNeeded(options, function_, canonical_function_, args,
                                compile_options, ctx, *profiler, result,
                                executable);
}

absl::Status XlaCompileOnDemandOp::Compile(
    const std::vector<XlaCompiler::Argument>& args, OpKernelContext* ctx,
    XlaDeviceCompiler** xla_device_compiler,
    DeviceCompilationProfiler** profiler,
    const XlaCompiler::CompilationResult** result,
    xla::LocalExecutable** executable) {
  // We store information about the JIT-compiled XLA computation
  // in the ResourceMgr.
  ResourceMgr* rm = ctx->resource_manager();
  CHECK(rm);

  TF_ASSIGN_OR_RETURN(DeviceType compilation_device_type,
                      GetCompilationDeviceType(platform_info_.device_type()));

  TF_RETURN_IF_ERROR(rm->LookupOrCreate<XlaDeviceCompiler>(
      rm->default_container(), "xla_device_compiler", xla_device_compiler,
      [&](XlaDeviceCompiler** xla_device_compiler) {
        return BuildXlaDeviceCompiler(ctx->device(), ctx->function_library(),
                                      platform_info_, compilation_device_type,
                                      xla_device_compiler);
      }));

  TF_RETURN_IF_ERROR(rm->LookupOrCreate<DeviceCompilationProfiler>(
      rm->default_container(), "device_compilation_profiler", profiler,
      [](DeviceCompilationProfiler** profiler) {
        *profiler = new DeviceCompilationProfiler();
        return absl::OkStatus();
      }));

  XlaCompiler::Options options = GenerateCompilerOptions(
      **xla_device_compiler, *ctx->function_library(), ctx->device(),
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr,
      platform_info_, /*has_ref_vars=*/true);
  // No detailed logging from on demand op.
  options.detailed_logging = false;
  XlaCompiler::CompileOptions compile_options = GetCompileOptions();

  return (*xla_device_compiler)
      ->CompileSingleOpIfNeeded(options, function_, canonical_function_, args,
                                compile_options, ctx, *profiler, result,
                                executable);
}

void XlaCompileOnDemandOp::Compute(OpKernelContext* ctx) {
  const XlaCompiler::CompilationResult* result;
  DeviceCompilationProfiler* profiler;

  OP_REQUIRES(ctx, ctx->function_library(),
              errors::Internal("Function library missing"));

  // Get constants, inputs and variables from the OpKernelContext.
  auto constant_indices_or = GetConstantInputIndicesFromContext(ctx);
  OP_REQUIRES_OK(ctx, constant_indices_or.status());
  std::vector<const Tensor*> inputs = InputsFromContext(ctx);
  std::vector<int> variable_indices =
      GetResourceVariableIndicesFromContext(ctx);

  bool use_pjrt =
      GetXlaOpsCommonFlags()
          ->tf_xla_use_device_api.IsEnabledInXlaCompileOnDemandForDevice(
              platform_info_.device_type());
  if (use_pjrt) {
    std::vector<VariableInfo> variables;
    std::vector<XlaCompiler::Argument> args;
    // Lock variables for the whole duration of compile + execute.
    OP_REQUIRES_OK(ctx, GetAndLockVariablesAndBuildXlaCompilerArguments(
                            *ctx, inputs, *constant_indices_or,
                            variable_indices, &variables, &args));

    PjRtDeviceCompiler* pjrt_device_compiler;
    xla::PjRtLoadedExecutable* pjrt_executable;
    OP_REQUIRES_OK(ctx, Compile(args, ctx, &pjrt_device_compiler, &profiler,
                                &result, &pjrt_executable));
    // Hold the reference to the XLA device compiler and profiler during
    // evaluation. (We could probably free them sooner because the ResourceMgr
    // will retain references, but this is more obviously correct.)
    core::ScopedUnref pjrt_device_compiler_ref(pjrt_device_compiler);
    core::ScopedUnref profiler_ref(profiler);

    VLOG(2) << "Compiled op with PJRT: " << ctx->status();
    VLOG(2) << "result != nullptr: " << (result != nullptr);
    VLOG(2) << "pjrt_executable != nullptr: " << (pjrt_executable != nullptr);
    VLOG(2) << "Executing with PJRT ...";

    OP_REQUIRES_OK(ctx, RunPjRtExecutable(inputs, variables, *result,
                                          pjrt_device_compiler->client(),
                                          pjrt_executable, ctx));

    VLOG(2) << "Completed executing with PJRT!";
  } else {
    ResourceVarsSnapshot variable_args;
    std::vector<XlaCompiler::Argument> args;
    // Lock variables only for generating XlaCompiler::Arguments and then
    // release them.
    {
      std::vector<VariableInfo> variables;
      OP_REQUIRES_OK(ctx, GetAndLockVariablesAndBuildXlaCompilerArguments(
                              *ctx, inputs, *constant_indices_or,
                              variable_indices, &variables, &args));
      OP_REQUIRES_OK(ctx, SnapshotResourceVariables(ctx, variable_indices,
                                                    variables, &variable_args));
    }

    XlaDeviceCompiler* xla_device_compiler;
    xla::LocalExecutable* executable;
    OP_REQUIRES_OK(ctx, Compile(args, ctx, &xla_device_compiler, &profiler,
                                &result, &executable));
    // Hold the reference to the XLA device compiler and profiler during
    // evaluation. (We could probably free them sooner because the ResourceMgr
    // will retain references, but this is more obviously correct.)
    core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);
    core::ScopedUnref profiler_ref(profiler);

    // Locks are acquired again when populating the `ctx` outputs.
    OP_REQUIRES_OK(
        ctx, Run(variable_args, result, xla_device_compiler, executable, ctx));
  }
}

}  // namespace tensorflow
