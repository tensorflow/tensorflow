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

#include "absl/memory/memory.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {

namespace {
std::map<int, OptionalTensor> GetVariables(OpKernelContext* ctx) {
  std::map<int, OptionalTensor> variables;
  for (int64 i = 0; i < ctx->num_inputs(); ++i) {
    if (ctx->input(i).dtype() == DT_RESOURCE) {
      core::RefCountPtr<Var> variable;
      ResourceHandle handle = HandleFromInput(ctx, i);
      OptionalTensor& optional = variables[i];
      optional.name = handle.name();
      if (LookupResource(ctx, handle, &variable).ok()) {
        tf_shared_lock lock(*variable->mu());
        optional.present = true;
        optional.value = *variable->tensor();
      }
    }
  }
  return variables;
}
}  // namespace

Status XlaCompileOnDemandOp::Run(OpKernelContext* ctx,
                                 const XlaDevice::Metadata& metadata,
                                 const XlaCompiler::CompilationResult* result,
                                 xla::LocalExecutable* executable) {
  std::map<int, OptionalTensor> variables = GetVariables(ctx);

  xla::LocalClient* client = metadata.client();

  // Builds an XLA allocator for the device.
  XlaComputationLaunchContext launch_context(
      client, client->backend().memory_allocator(),
      /*allocate_xla_tensors=*/true,
      /*use_multiple_streams=*/metadata.UseMultipleStreams());

  launch_context.PopulateInputs(ctx, result, variables,
                                /*missing_ctx_input_prefix=*/0);

  se::Stream* stream =
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;
  TF_RET_CHECK(stream);

  VLOG(2) << "Executing computation: " << name();
  for (const xla::ShapedBuffer* arg : launch_context.arguments()) {
    VLOG(2) << name() << ": " << *arg;
  }
  xla::ExecutableRunOptions run_options;
  run_options.set_stream(stream);
  run_options.set_allocator(client->backend().memory_allocator());
  run_options.set_intra_op_thread_pool(&ctx->eigen_cpu_device());
  run_options.set_rng_seed(GetXLARandomSeed());

  xla::StatusOr<xla::ScopedShapedBuffer> run_result =
      executable->Run(launch_context.arguments(), run_options);
  TF_RETURN_IF_ERROR(run_result.status());

  const xla::HloInputOutputAliasConfig& input_output_alias =
      executable->executable()->module().input_output_alias_config();
  TF_RETURN_IF_ERROR(launch_context.PopulateOutputs(
      ctx, result, run_result.ConsumeValueOrDie(),
      /*missing_ctx_input_prefix=*/0, input_output_alias, variables));
  return Status::OK();
}

Status XlaCompileOnDemandOp::MustArgumentBeConstant(
    const OpKernel* op_kernel, int64 argument_idx,
    FunctionLibraryRuntime* flib_runtime, bool* result) {
  *result = false;

  // TODO(jmolloy): This could be expensive, so memoize.
  std::vector<int> constant_input_indices;
  TF_RETURN_IF_ERROR(GetCompileTimeConstInputs(
      op_kernel, &constant_input_indices, flib_runtime));
  *result = absl::c_binary_search(constant_input_indices, argument_idx);
  return Status::OK();
}

// TODO(ycao): Remove the need to call ShouldArgumentBeConstant. Its benefit is
// not clear yet and it causes heavy constant analysis to run twice.
Status XlaCompileOnDemandOp::ShouldArgumentBeConstant(
    const OpKernel* op_kernel, int64 argument_idx,
    FunctionLibraryRuntime* flib_runtime, bool* result) {
  return MustArgumentBeConstant(op_kernel, argument_idx, flib_runtime, result);
}

Status XlaCompileOnDemandOp::Compile(
    OpKernelContext* ctx, const XlaDevice::Metadata& metadata,
    const XlaCompiler::CompilationResult** result,
    xla::LocalExecutable** executable) {
  std::map<int, Tensor> constant_arguments;
  for (int64 i = 0; i < ctx->num_inputs(); ++i) {
    const Tensor& device_tensor = ctx->input(i);
    if (const XlaTensor* xla_tensor = XlaTensor::FromTensor(&device_tensor)) {
      if (xla_tensor->has_host_tensor()) {
        bool should_arg_be_const;
        TF_RETURN_IF_ERROR(ShouldArgumentBeConstant(&ctx->op_kernel(), i,
                                                    ctx->function_library(),
                                                    &should_arg_be_const));
        if (should_arg_be_const) {
          constant_arguments[i] = xla_tensor->host_tensor();
        }
      }
    }

    if (constant_arguments.count(i) == 0) {
      bool must_argument_be_const;
      TF_RETURN_IF_ERROR(MustArgumentBeConstant(&ctx->op_kernel(), i,
                                                ctx->function_library(),
                                                &must_argument_be_const));

      if (must_argument_be_const) {
        // Slow path; the argument is not available as a host constant so we
        // must fetch it synchronously.
        Tensor host_tensor;
        AllocatorAttributes attrs;
        attrs.set_on_host(true);
        TF_RETURN_IF_ERROR(ctx->allocate_temp(
            device_tensor.dtype(), device_tensor.shape(), &host_tensor, attrs));
        Notification n;
        Status status;
        ctx->op_device_context()->CopyDeviceTensorToCPU(
            &device_tensor, "ConstantArgument",
            reinterpret_cast<Device*>(ctx->device()), &host_tensor,
            [&](Status s) {
              status = s;
              n.Notify();
            });
        n.WaitForNotification();
        if (!status.ok()) {
          LOG(ERROR) << "Copying tensor of shape "
                     << device_tensor.shape().DebugString() << " from "
                     << ctx->device()->name() << "to CPU failed with "
                     << status.ToString();
          return status;
        }
        constant_arguments[i] = host_tensor;
      }
    }
  }

  // We store information about the JIT-compiled XLA computation
  // in the ResourceMgr.
  ResourceMgr* rm = ctx->resource_manager();
  CHECK(rm);

  XlaCompilationCache* cache;
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<XlaCompilationCache>(
      rm->default_container(), "xla_cache", &cache,
      [&](XlaCompilationCache** cache) {
        *cache = new XlaCompilationCache(metadata.client(),
                                         metadata.jit_device_type());
        return Status::OK();
      }));
  // Hold the reference to the JIT during evaluation. (We could probably
  // free it sooner because the ResourceMgr will retain a reference, but
  // this is more obviously correct.)
  core::ScopedUnref cache_ref(cache);

  XlaCompiler::Options options;
  options.device_type = metadata.jit_device_type();
  options.client = metadata.client();
  options.flib_def = ctx->function_library()->GetFunctionLibraryDefinition();
  options.shape_representation_fn = metadata.shape_representation_fn();

  XlaCompiler::CompileOptions compile_options;
  compile_options.is_entry_computation = true;
  // Optimization: where possible, have the computation return a naked array
  // rather than a one-element tuple.
  compile_options.always_return_tuple = false;

  std::map<int, OptionalTensor> variable_args = GetVariables(ctx);

  std::vector<XlaCompiler::Argument> args;

  TF_RETURN_IF_ERROR(XlaComputationLaunchContext::BuildXlaCompilerArguments(
      constant_arguments, variable_args, ctx, &args));

  return cache->CompileSingleOp(options, args, ctx, compile_options, result,
                                executable);
}

void XlaCompileOnDemandOp::Compute(OpKernelContext* ctx) {
  const XlaCompiler::CompilationResult* result;
  xla::LocalExecutable* executable;
  const XlaDevice::Metadata* metadata;
  OP_REQUIRES_OK(ctx, XlaDevice::GetMetadata(ctx, &metadata));
  OP_REQUIRES_OK(ctx, Compile(ctx, *metadata, &result, &executable));
  OP_REQUIRES_OK(ctx, Run(ctx, *metadata, result, executable));
}

}  // namespace tensorflow
