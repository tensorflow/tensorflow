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

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/jit/device_compilation_profiler.h"
#include "tensorflow/compiler/jit/xla_compiler_options_util.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {
namespace {
using XlaDeviceCompiler =
    DeviceCompiler<xla::LocalExecutable, xla::LocalClient>;
}  // namespace

Status XlaCompileOnDemandOp::Run(OpKernelContext* ctx,
                                 XlaDeviceCompiler* xla_device_compiler,
                                 const XlaCompiler::CompilationResult* result,
                                 xla::LocalExecutable* executable,
                                 const ResourceVarsSnapshot& variable_args) {
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

  std::map<int, const Tensor*> snapshot_ptrs;
  for (auto& p : variable_args) {
    snapshot_ptrs.emplace(p.first,
                          p.second.has_value() ? &p.second.value() : nullptr);
  }

  const xla::HloInputOutputAliasConfig& input_output_alias =
      executable->executable()->module().input_output_alias_config();
  StatusOr<std::vector<xla::ExecutionInput>> execution_inputs =
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

  StatusOr<xla::ExecutionOutput> run_result =
      executable->Run(std::move(execution_inputs).value(), run_options);
  TF_RETURN_IF_ERROR(run_result.status());
  xla::ExecutionOutput execution_output = std::move(run_result).value();
  StatusOr<std::vector<VariableInfo>> variable_infos =
      GatherVariableInfo(ctx, *result, 0);
  TF_RETURN_IF_ERROR(variable_infos.status());
  TF_RETURN_IF_ERROR(LockVariables(absl::MakeSpan(*variable_infos)));
  TF_RETURN_IF_ERROR(launch_context.PopulateOutputs(
      ctx, result, execution_output.ConsumeResult(),
      /*missing_ctx_input_prefix=*/0, absl::MakeSpan(*variable_infos),
      input_output_alias, snapshot_ptrs));
  return OkStatus();
}

Status XlaCompileOnDemandOp::Compile(
    OpKernelContext* ctx, const XlaCompiler::CompilationResult** result,
    XlaDeviceCompiler** xla_device_compiler,
    DeviceCompilationProfiler** profiler, ResourceVarsSnapshot* variable_args,
    xla::LocalExecutable** executable) {
  TF_ASSIGN_OR_RETURN(std::vector<int> constant_input_indices,
                      GetConstantInputIndicesFromContext(ctx));
  std::vector<const Tensor*> inputs = InputsFromContext(ctx);

  // We store information about the JIT-compiled XLA computation
  // in the ResourceMgr.
  ResourceMgr* rm = ctx->resource_manager();
  CHECK(rm);

  TF_RETURN_IF_ERROR(rm->LookupOrCreate<XlaDeviceCompiler>(
      rm->default_container(), "xla_device_compiler", xla_device_compiler,
      [&](XlaDeviceCompiler** xla_device_compiler) {
        return BuildXlaDeviceCompiler(ctx->device(), ctx->function_library(),
                                      platform_info_, xla_device_compiler);
      }));

  TF_RETURN_IF_ERROR(rm->LookupOrCreate<DeviceCompilationProfiler>(
      rm->default_container(), "device_compilation_profiler", profiler,
      [](DeviceCompilationProfiler** profiler) {
        *profiler = new DeviceCompilationProfiler();
        return OkStatus();
      }));

  XlaCompiler::Options options = GenerateCompilerOptions(
      **xla_device_compiler, *ctx->function_library(), ctx->device(),
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr,
      platform_info_, /*has_ref_vars=*/true);
  // No detailed logging from on demand op.
  options.detailed_logging = false;
  XlaCompiler::CompileOptions compile_options;
  compile_options.is_entry_computation = true;
  // Optimization: where possible, have the computation return a naked array
  // rather than a one-element tuple.
  compile_options.always_return_tuple = false;

  std::vector<int> variables_indices =
      GetResourceVariableIndicesFromContext(ctx);
  StatusOr<std::vector<XlaCompiler::Argument>> args;
  {
    std::vector<VariableInfo> variable_infos;
    TF_RETURN_IF_ERROR(
        GetVariableInfosFromInputs(ctx->resource_manager(), ctx->device(),
                                   inputs, variables_indices, &variable_infos));

    TF_RETURN_IF_ERROR(LockVariables(absl::MakeSpan(variable_infos)));
    TF_RETURN_IF_ERROR(SnapshotResourceVariables(
        ctx, variables_indices, variable_infos, variable_args));

    args = XlaComputationLaunchContext::BuildXlaCompilerArguments(
        constant_input_indices, inputs, variable_infos,
        static_cast<Device*>(ctx->device()));
    TF_RETURN_IF_ERROR(args.status());
  }

  return (*xla_device_compiler)
      ->CompileSingleOpIfNeeded(options, *args, compile_options, ctx, *profiler,
                                result, executable);
}

void XlaCompileOnDemandOp::Compute(OpKernelContext* ctx) {
  const XlaCompiler::CompilationResult* result;
  xla::LocalExecutable* executable;
  ResourceVarsSnapshot variable_args;
  XlaDeviceCompiler* xla_device_compiler;
  DeviceCompilationProfiler* profiler;
  OP_REQUIRES(ctx, ctx->function_library(),
              errors::Internal("Function library missing"));
  OP_REQUIRES_OK(ctx, Compile(ctx, &result, &xla_device_compiler, &profiler,
                              &variable_args, &executable));

  // Hold the reference to the XLA device compiler and profiler during
  // evaluation. (We could probably free them sooner because the ResourceMgr
  // will retain references, but this is more obviously correct.)
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);
  core::ScopedUnref profiler_ref(profiler);
  OP_REQUIRES_OK(
      ctx, Run(ctx, xla_device_compiler, result, executable, variable_args));
}

}  // namespace tensorflow
