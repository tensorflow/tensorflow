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

#include "tensorflow/compiler/jit/kernels/xla_device_launch_op.h"

#include "tensorflow/compiler/jit/defs.h"
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
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

namespace {

Status BuildCompilationCache(ResourceMgr* rm, XlaCompilationCache** cache) {
  XlaDevice::Metadata* metadata;
  Status s = rm->Lookup<XlaDevice::Metadata>(rm->default_container(),
                                             "xla_metadata", &metadata);
  if (!s.ok()) {
    return s;
  }
  core::ScopedUnref metadata_ref(metadata);
  *cache =
      new XlaCompilationCache(metadata->client(), metadata->jit_device_type());
  return Status::OK();
}

}  // namespace

XlaDeviceLaunchOp::XlaDeviceLaunchOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  const NameAttrList* func;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("function", &func));
  function_ = *func;
  VLOG(1) << "XlaDeviceLaunch created function="
          << Canonicalize(function_.name(), AttrSlice(&function_.attr()));
  DataTypeVector constant_types;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tconstants", &constant_types));
  num_constant_args_ = constant_types.size();
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Nresources", &num_resource_args_));
}

std::vector<OptionalTensor> SnapshotResourceVariables(OpKernelContext* ctx,
                                                      int num_variables) {
  std::vector<OptionalTensor> snapshot(num_variables);
  int first_variable = ctx->num_inputs() - num_variables;
  for (int i = 0; i < num_variables; ++i) {
    Var* variable = nullptr;
    ResourceHandle handle = HandleFromInput(ctx, first_variable + i);
    if (LookupResource(ctx, handle, &variable).ok()) {
      mutex_lock lock(*variable->mu());
      snapshot[i].name = handle.name();
      snapshot[i].present = true;
      snapshot[i].value = *variable->tensor();
    }
  }
  return snapshot;
}

void XlaDeviceLaunchOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "XlaDeviceLaunch::Compute "
          << Canonicalize(function_.name(), AttrSlice(&function_.attr()));
  // We store information about the JIT-compiled XLA computation
  // in the ResourceMgr.
  ResourceMgr* rm = ctx->resource_manager();
  OP_REQUIRES(ctx, rm, errors::Internal("No resource manager."));

  XlaCompilationCache* cache;
  OP_REQUIRES_OK(ctx, rm->LookupOrCreate<XlaCompilationCache>(
                          rm->default_container(), "xla_compiler", &cache,
                          [rm](XlaCompilationCache** cache) {
                            return BuildCompilationCache(rm, cache);
                          }));
  // Holds the reference to the JIT during evaluation. (We could probably
  // free it sooner because the ResourceMgr will retain a reference, but
  // this is more obviously correct.)
  core::ScopedUnref cache_ref(cache);

  std::vector<OptionalTensor> variables =
      SnapshotResourceVariables(ctx, num_resource_args_);

  XlaCompiler::Options options;
  options.client = cache->client();
  options.device_type = &cache->device_type();
  options.flib_def = ctx->function_library()->GetFunctionLibraryDefinition();
  options.graph_def_version = ctx->function_library()->graph_def_version();
  options.allow_cpu_custom_calls = false;
  options.local_executable_has_hybrid_result = false;

  const XlaCompiler::CompilationResult* kernel;
  OP_REQUIRES_OK(ctx, cache->Compile(options, function_, num_constant_args_,
                                     variables, ctx, &kernel, nullptr));

  VLOG(1) << "XLA compilation complete...";

  OP_REQUIRES(ctx, ctx->num_outputs() == kernel->outputs.size(),
              errors::Internal("Unexpected number of outputs"));

  // Runs the computation, if any. There might not be a computation if all
  // outputs were compile-time constants.
  std::vector<std::unique_ptr<xla::GlobalData>> outputs;
  if (!kernel->computation->IsNull()) {
    auto opaque_shape = xla::ShapeUtil::MakeOpaqueShape();

    // Builds the inputs to the computation.
    std::vector<std::shared_ptr<xla::GlobalData>> arg_handles(
        kernel->input_mapping.size());
    std::vector<xla::GlobalData*> arg_ptrs(kernel->input_mapping.size());

    // Adds the argument tensors.
    const int first_variable_arg = ctx->num_inputs() - num_resource_args_;
    for (int i = 0; i < kernel->input_mapping.size(); ++i) {
      int op_input_num = kernel->input_mapping[i];

      if (op_input_num >= first_variable_arg) {
        arg_handles[i] = XlaTransferManager::GetTensorGlobalData(
            variables[op_input_num - first_variable_arg].value);
      } else {
        arg_handles[i] =
            XlaTransferManager::GetTensorGlobalData(ctx->input(op_input_num));
      }
      arg_ptrs[i] = arg_handles[i].get();
    }

    // Execute the computation.
    xla::ExecutionProfile profile;
    xla::ExecutionOptions execution_options;
    *execution_options.mutable_shape_with_output_layout() =
        kernel->xla_output_shape;
    Env* env = Env::Default();
    auto start_time = env->NowMicros();
    VLOG(1) << "Executing XLA Computation...";
    auto result = cache->client()->Execute(*kernel->computation, arg_ptrs,
                                           &execution_options, &profile);
    auto elapsed = env->NowMicros() - start_time;
    OP_REQUIRES(ctx, result.ok(), result.status());

    VLOG(1) << "Elapsed time: " << elapsed << "us";
    VLOG(1) << "ExecutionProfile: " << profile.DebugString();

    if (xla::ShapeUtil::IsTuple(kernel->xla_output_shape)) {
      auto outputs_or_error =
          cache->client()->DeconstructTuple(*result.ValueOrDie());
      OP_REQUIRES(ctx, outputs_or_error.ok(), outputs_or_error.status());
      outputs = outputs_or_error.ConsumeValueOrDie();
    } else {
      outputs.push_back(result.ConsumeValueOrDie());
    }
  }

  XlaDeviceContext* device_context = ctx->op_device_context<XlaDeviceContext>();

  // Copy XLA outputs to the operator's outputs.
  VLOG(2) << "Setting operator output";
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

  // Apply variable updates, if any.
  VLOG(2) << "Applying variable updates";
  for (int i = 0; i < kernel->variable_updates.size(); ++i) {
    const XlaCompiler::VariableUpdate& write = kernel->variable_updates[i];
    OP_REQUIRES(ctx,
                write.input_index >= 0 && write.input_index < ctx->num_inputs(),
                errors::Internal("Invalid input index for variable write."));
    // This code is very close to being a clone of AssignVariableOp, but the
    // key difference is that the contents of an XLA device tensor cannot be
    // copied safely; instead we must use
    // XlaTransferManager::SetTensorGlobalData.
    Var* variable = nullptr;
    // TODO(b/35625933): tensorflow::Var should contain a PersistentTensor, not
    // a Tensor.
    OP_REQUIRES_OK(ctx, LookupOrCreateResource<Var>(
                            ctx, HandleFromInput(ctx, write.input_index),
                            &variable, [this, ctx, &write](Var** ptr) {
                              *ptr = new Var(write.type);
                              PersistentTensor unused;
                              Tensor* tmp;
                              TF_RETURN_IF_ERROR(ctx->allocate_persistent(
                                  write.type, write.shape, &unused, &tmp));
                              *(*ptr)->tensor() = *tmp;
                              return Status::OK();
                            }));
    core::ScopedUnref s(variable);

    mutex_lock ml(*variable->mu());
    OP_REQUIRES(ctx, variable->tensor()->dtype() == write.type,
                errors::Internal("Mismatched type in variable write"));
    if (!variable->tensor()->shape().IsSameSize(write.shape)) {
      PersistentTensor unused;
      Tensor* tmp;
      OP_REQUIRES_OK(ctx, ctx->allocate_persistent(write.type, write.shape,
                                                   &unused, &tmp));
      *variable->tensor() = *tmp;
    }
    XlaTransferManager::SetTensorGlobalData(
        std::shared_ptr<xla::GlobalData>(std::move(outputs[output_num])),
        variable->tensor());
    ++output_num;
  }

  VLOG(1) << "Done";
}

XlaDeviceLaunchOp::~XlaDeviceLaunchOp() {
  VLOG(1) << "XlaDeviceLaunch destroyed";
}

}  // namespace tensorflow
