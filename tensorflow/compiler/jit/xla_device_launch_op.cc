/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/xla_device_launch_op.h"

#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/xla_compilation_cache.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_context.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

namespace {

Status BuildCompilationCache(ResourceMgr* rm, XlaCompilationCache** compiler) {
  XlaDevice::Metadata* metadata;
  Status s = rm->Lookup<XlaDevice::Metadata>(rm->default_container(),
                                             "xla_metadata", &metadata);
  if (!s.ok()) {
    return s;
  }
  core::ScopedUnref metadata_ref(metadata);
  XlaCompiler::Options options;
  options.device_type = metadata->jit_device_type();
  options.client = metadata->client();
  options.allow_cpu_custom_calls = false;
  options.local_executable_has_hybrid_result = false;
  *compiler = new XlaCompilationCache(options);
  return Status::OK();
}

}  // namespace

XlaDeviceLaunchOp::XlaDeviceLaunchOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  const NameAttrList* func;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("function", &func));
  function_ = *func;
  VLOG(1) << "XlaDeviceLaunch created function="
          << Canonicalize(function_.name(), function_.attr());
  DataTypeVector constant_types;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tconstants", &constant_types));
  num_constant_args_ = constant_types.size();
}

void XlaDeviceLaunchOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "XlaDeviceLaunch::Compute "
          << Canonicalize(function_.name(), function_.attr());
  // We store information about the JIT-compiled XLA computation
  // in the ResourceMgr.
  ResourceMgr* rm = ctx->resource_manager();
  OP_REQUIRES(ctx, rm, errors::Internal("No resource manager."));

  XlaCompilationCache* compiler;
  OP_REQUIRES_OK(ctx,
                 rm->LookupOrCreate<XlaCompilationCache>(
                     rm->default_container(), "xla_compiler", &compiler,
                     [rm](XlaCompilationCache** compiler) {
                       return BuildCompilationCache(rm, compiler);
                     }));
  // Hold the reference to the JIT during evaluation. (We could probably
  // free it sooner because the ResourceMgr will retain a reference, but
  // this is more obviously correct.)
  core::ScopedUnref compiler_ref(compiler);

  const XlaCompiler::CompilationResult* kernel;
  OP_REQUIRES_OK(
      ctx,
      compiler->Compile(function_, num_constant_args_, ctx, &kernel, nullptr));

  VLOG(1) << "Executing XLA Computation...";

  OP_REQUIRES(ctx, ctx->num_outputs() == kernel->outputs.size(),
              errors::Internal("Unexpected number of outputs"));

  // Run the computation, if any. There might not be a computation if all
  // outputs were compile-time constants.
  std::vector<std::unique_ptr<xla::GlobalData>> outputs;
  if (!kernel->computation.IsNull()) {
    auto opaque_shape = xla::ShapeUtil::MakeOpaqueShape();

    // Convert argument tensors to xla::GlobalData pointers.
    std::vector<std::shared_ptr<xla::GlobalData>> arg_handles(
        kernel->xla_input_shapes.size());
    std::vector<xla::GlobalData*> arg_ptrs(kernel->xla_input_shapes.size());
    for (int i = 0; i < kernel->xla_input_shapes.size(); ++i) {
      int input_num = kernel->xla_input_shapes[i].first;
      arg_handles[i] =
          XlaTransferManager::GetTensorGlobalData(ctx->input(input_num));
      arg_ptrs[i] = arg_handles[i].get();
    }

    // Execute the computation.
    xla::ExecutionProfile profile;
    xla::ExecutionOptions execution_options;
    *execution_options.mutable_shape_with_output_layout() =
        kernel->xla_output_shape;
    Env* env = Env::Default();
    auto start_time = env->NowMicros();
    auto result = compiler->client()->Execute(kernel->computation, arg_ptrs,
                                              &execution_options, &profile);
    auto elapsed = env->NowMicros() - start_time;
    OP_REQUIRES(ctx, result.ok(), result.status());

    VLOG(1) << "Elapsed time: " << elapsed << "us";
    VLOG(1) << "ExecutionProfile: " << profile.DebugString();

    if (xla::ShapeUtil::IsTuple(kernel->xla_output_shape)) {
      auto outputs_or_error =
          compiler->client()->DeconstructTuple(*result.ValueOrDie());
      OP_REQUIRES(ctx, outputs_or_error.ok(), outputs_or_error.status());
      outputs = outputs_or_error.ConsumeValueOrDie();
    } else {
      outputs.push_back(result.ConsumeValueOrDie());
    }
  }

  XlaDeviceContext* device_context = ctx->op_device_context<XlaDeviceContext>();

  // Copy XLA outputs to the operator's outputs.
  int output_num = 0;
  for (int i = 0; i < ctx->num_outputs(); ++i) {
    Tensor* output;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(i, kernel->outputs[i].shape, &output));
    if (kernel->outputs[i].is_constant) {
      // TODO(phawkins): mark constant _XlaLaunch outputs as HostMemory and
      // remove the copy from this code.
      Status status;
      device_context->CopyCPUTensorToDevice(
          &kernel->outputs[i].constant_value, nullptr, output,
          [&status](const Status& s) { status = s; });
      if (!status.ok()) {
        ctx->SetStatus(status);
        return;
      }
    } else {
      CHECK_LT(output_num, outputs.size());
      XlaTransferManager::SetTensorGlobalData(
          std::shared_ptr<xla::GlobalData>(std::move(outputs[output_num])),
          output);
      ++output_num;
    }
  }

  VLOG(1) << "Done";
}

XlaDeviceLaunchOp::~XlaDeviceLaunchOp() {
  VLOG(1) << "XlaDeviceLaunch destroyed";
}

}  // namespace tensorflow
