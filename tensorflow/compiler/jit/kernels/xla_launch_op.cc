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

#include "tensorflow/compiler/jit/kernels/xla_launch_op.h"

#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/util/stream_executor_util.h"

namespace tensorflow {

XlaLocalLaunchOp::XlaLocalLaunchOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), device_type_(ctx->device_type()) {
  const NameAttrList* func;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("function", &func));
  function_ = *func;
  DataTypeVector constant_types;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tconstants", &constant_types));
  num_constant_args_ = constant_types.size();
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Nresources", &num_resource_args_));
  if (device_type_ == DeviceType(DEVICE_CPU)) {
    platform_id_ = se::host::kHostPlatformId;
  } else if (device_type_ == DeviceType(DEVICE_GPU)) {
    platform_id_ = se::cuda::kCudaPlatformId;
  } else {
    platform_id_ = nullptr;
  }
}

Status XlaLocalLaunchOp::BuildCompilationCache(OpKernelContext* ctx,
                                               XlaCompilationCache** cache) {
  const XlaDevice::Metadata* metadata;
  Status s = XlaDevice::GetMetadata(ctx, &metadata);
  if (s.ok()) {
    *cache = new XlaCompilationCache(metadata->client(),
                                     metadata->jit_device_type());
    return Status::OK();
  }

  auto platform = se::MultiPlatformManager::PlatformWithId(platform_id_);
  if (!platform.ok()) {
    return platform.status();
  }
  xla::LocalClientOptions client_options;
  client_options.set_platform(platform.ValueOrDie());
  client_options.set_intra_op_parallelism_threads(
      ctx->device()->tensorflow_cpu_worker_threads()->num_threads);
  auto client = xla::ClientLibrary::GetOrCreateLocalClient(client_options);
  if (!client.ok()) {
    return client.status();
  }
  const XlaOpRegistry::DeviceRegistration* registration;
  if (!XlaOpRegistry::GetCompilationDevice(device_type_.type(),
                                           &registration)) {
    return errors::InvalidArgument("No JIT device registered for ",
                                   device_type_.type());
  }
  *cache = new XlaCompilationCache(
      client.ValueOrDie(), DeviceType(registration->compilation_device_name));
  return Status::OK();
}

void XlaLocalLaunchOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "XlaLocalLaunchOp::Compute "
          << Canonicalize(function_.name(), AttrSlice(&function_.attr()));
  // We store information about the JIT-compiled XLA computation
  // in the ResourceMgr.
  ResourceMgr* rm = ctx->resource_manager();
  OP_REQUIRES(ctx, rm, errors::Internal("No resource manager."));

  se::Stream* stream =
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

  XlaCompilationCache* cache;
  OP_REQUIRES_OK(ctx, rm->LookupOrCreate<XlaCompilationCache>(
                          rm->default_container(), "xla_cache", &cache,
                          [this, ctx](XlaCompilationCache** cache) {
                            return BuildCompilationCache(ctx, cache);
                          }));
  // Hold the reference to the JIT during evaluation. (We could probably
  // free it sooner because the ResourceMgr will retain a reference, but
  // this is more obviously correct.)
  core::ScopedUnref cache_ref(cache);

  const XlaDevice::Metadata* metadata;
  Status s = XlaDevice::GetMetadata(ctx, &metadata);
  bool allocate_xla_tensors = s.ok();

  // Get the platform_id_ for XLA_* devices.
  if (platform_id_ == nullptr) {
    if (s.ok()) {
      platform_id_ = metadata->platform()->id();
    }
  }

  std::map<int, OptionalTensor> variables =
      SnapshotResourceVariables(ctx, num_resource_args_);

  xla::LocalClient* client = static_cast<xla::LocalClient*>(cache->client());

  XlaAllocator local_xla_allocator(client->backend().platform(),
                                   ctx->device()->GetAllocator({}));
  xla::DeviceMemoryAllocator* xla_allocator;
  // If we are on an XlaDevice, use the underlying XLA platform's allocator
  // directly. We could use the StreamExecutor's allocator which may
  // theoretically be more correct, but XLA returns a nice OOM message in a
  // Status and StreamExecutor does not.
  //
  // Importantly we can't use ctx->device()->GetAllocator() as the allocator
  // (which local_xla_allocator above uses) as on an XlaDevice, this is a
  // dummy allocator that returns XlaTensor objects. The XlaCompiler needs a
  // real allocator to allocate real buffers.
  if (allocate_xla_tensors) {
    xla_allocator = client->backend().memory_allocator();
  } else {
    xla_allocator = &local_xla_allocator;
  }

  XlaCompiler::Options options;
  options.client = client;
  options.device_type = &cache->device_type();
  options.flib_def = ctx->function_library()->GetFunctionLibraryDefinition();
  options.graph_def_version = ctx->function_library()->graph_def_version();
  options.allow_cpu_custom_calls = (platform_id_ == se::host::kHostPlatformId);
  options.device_allocator = xla_allocator;
  // TODO(b/77671268): We don't set variable_representation_shape_fn here. This
  // is restricted to Variables, but we need something like this to apply to
  // normal Tensors too.

  const XlaCompiler::CompilationResult* kernel;
  xla::LocalExecutable* executable;

  std::map<int, Tensor> constant_args;
  for (int i = 0; i < num_constant_args_; ++i) {
    constant_args.insert({i, ctx->input(i)});
  }
  OP_REQUIRES_OK(ctx, cache->Compile(options, function_, constant_args,
                                     variables, ctx, &kernel, &executable,
                                     /*compile_options=*/nullptr));

  VLOG(1) << "Executing XLA Computation...";

  XlaComputationLaunchContext launch_context(
      num_resource_args_, client, xla_allocator, allocate_xla_tensors);
  launch_context.PopulateInputs(ctx, kernel, variables);

  // Execute the computation.
  VLOG(2) << "Executing computation.";
  xla::ExecutableRunOptions run_options;
  run_options.set_stream(stream);
  run_options.set_allocator(xla_allocator);
  run_options.set_intra_op_thread_pool(&ctx->eigen_cpu_device());
  run_options.set_rng_seed(ctx->step_id());
  Env* env = Env::Default();
  auto start_time = env->NowMicros();

  auto run_result = executable->Run(launch_context.arguments(), run_options);
  OP_REQUIRES(ctx, run_result.ok(), run_result.status());

  auto elapsed = env->NowMicros() - start_time;
  VLOG(2) << "Elapsed time: " << elapsed << "us";

  launch_context.PopulateOutputs(ctx, kernel, run_result.ConsumeValueOrDie());
  VLOG(1) << "Done";
}

XlaLocalLaunchOp::~XlaLocalLaunchOp() {
  VLOG(1) << "XlaLocalLaunchOp destroyed";
}

REGISTER_KERNEL_BUILDER(Name("_XlaLaunch").Device(DEVICE_CPU),
                        XlaLocalLaunchOp);

REGISTER_KERNEL_BUILDER(Name("_XlaLaunch")
                            .Device(DEVICE_GPU)
                            .HostMemory("constants")
                            .HostMemory("resources"),
                        XlaLocalLaunchOp);

}  // namespace tensorflow
