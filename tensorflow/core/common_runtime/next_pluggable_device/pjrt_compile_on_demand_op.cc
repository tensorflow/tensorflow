/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/next_pluggable_device/pjrt_compile_on_demand_op.h"

#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "tensorflow/compiler/jit/device_compiler_client.h"
#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/compiler/jit/variable_info_util.h"
#include "tensorflow/compiler/jit/xla_compiler_options_util.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/common/create_pjrt_client_util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {

Status PjRtCompileOnDemandOp::Compile(
    OpKernelContext* ctx, xla::PjRtClient* pjrt_client,
    const std::vector<XlaCompiler::Argument>& args,
    XlaCompiler::CompilationResult* compilation_result,
    std::unique_ptr<xla::PjRtLoadedExecutable>* executable) {
  // TODO(b/260798754): use caching when it is ready.
  const XlaCompiler::Options options =
      GenerateCompilerOptionsForPjRt(*ctx->function_library(), ctx->device(),
                                     XlaPlatformInfoFromDevice(ctx->device()));
  XlaCompiler::CompileOptions compile_options;
  compile_options.is_entry_computation = true;
  compile_options.use_tuple_arg = false;
  compile_options.always_return_tuple = true;
  XlaCompiler compiler(options);
  TF_RETURN_IF_ERROR(compiler.CompileSingleOp(
      compile_options, XlaCompiler::SingleOpCompileArgument(*ctx), args,
      compilation_result));
  xla::ExecutableBuildOptions build_options =
      GetExecutableBuildOptions(options, *compilation_result, -1);
  xla::CompileOptions pjrt_compile_options;
  pjrt_compile_options.executable_build_options = build_options;
  pjrt_compile_options.compile_portable_executable = true;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtLoadedExecutable> pjrt_executable,
      pjrt_client->Compile(*compilation_result->computation,
                           pjrt_compile_options));
  VLOG(2) << "Compiled PJRT executable " << pjrt_executable->name()
          << " num_replicas " << pjrt_executable->num_replicas()
          << " num_partitions " << pjrt_executable->num_partitions();
  *executable = std::move(pjrt_executable);
  return OkStatus();
}

void PjRtCompileOnDemandOp::Compute(OpKernelContext* ctx) {
  OP_REQUIRES_VALUE(xla::PjRtClient * pjrt_client, ctx,
                    GetOrCreatePjRtClient(GetDeviceType(ctx)));
  OP_REQUIRES(ctx, ctx->function_library(),
              errors::Internal("Function library missing"));

  OP_REQUIRES_VALUE(std::vector<int> constant_input_indices, ctx,
                    GetConstantInputIndicesFromContext(ctx));
  const std::vector<int> variables_indices =
      GetResourceVariableIndicesFromContext(ctx);
  const std::vector<const Tensor*> inputs = InputsFromContext(ctx);
  std::vector<VariableInfo> variables;
  variables.reserve(variables_indices.size());
  OP_REQUIRES_OK(
      ctx, GetVariableInfosFromInputs(ctx->resource_manager(), ctx->device(),
                                      inputs, variables_indices, &variables));
  OP_REQUIRES_OK(ctx, LockVariables(absl::MakeSpan(variables)));

  // Compile
  XlaCompiler::CompilationResult result;
  std::unique_ptr<xla::PjRtLoadedExecutable> executable;
  OP_REQUIRES_VALUE(std::vector<XlaCompiler::Argument> args, ctx,
                    XlaComputationLaunchContext::BuildXlaCompilerArguments(
                        constant_input_indices, inputs, variables,
                        static_cast<Device*>(ctx->device())));
  OP_REQUIRES_OK(ctx, Compile(ctx, pjrt_client, args, &result, &executable));

  // Execute
  OP_REQUIRES_OK(ctx, RunPjRtExecutable(*pjrt_client, inputs, variables, result,
                                        executable.get(), ctx));

  ctx->SetStatus(OkStatus());
  VLOG(1) << "PjRtCompileOnDemandOp::Compute: " << ctx->op_kernel().name()
          << " on device " << ctx->device()->name() << " Done.";
}

void RegisterPjRtCompileOnDemand(const char* device, const char* jit_device) {
  // Any op assigned to the device that isn't rewritten by the graph rewriter
  // gets executed by a PjRtCompileOnDemandOp, which compiles it and executes
  // it just-in-time.
  auto factory = [](OpKernelConstruction* context) -> OpKernel* {
    return new PjRtCompileOnDemandOp(context);
  };
  XlaOpRegistry::RegisterCompilationKernels();
  static XlaDeviceOpRegistrations* registrations = RegisterXlaDeviceKernels(
      device, jit_device, factory, "PjRtCompileOnDemandOp");
  (void)registrations;
}

}  // namespace tensorflow
