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

namespace gpu = perftools::gputools;

namespace tensorflow {

// Adapter class that wraps a Tensorflow allocator as an XLA allocator.
// Assumes that the Tensorflow allocator permits asynchronous deallocation:
// see comment on `AllowsAsynchronousDeallocation()`.
class XlaAllocator : public xla::DeviceMemoryAllocator {
 public:
  XlaAllocator(const gpu::Platform* platform, OpKernelContext* op_context);
  ~XlaAllocator() override;
  xla::StatusOr<gpu::DeviceMemoryBase> Allocate(int device_ordinal, uint64 size,
                                                bool retry_on_failure) override;
  Status Deallocate(int device_ordinal, gpu::DeviceMemoryBase* mem) override;

  // Register an Tensor (input or resource variable) with the allocator. If
  // the operation returns an alias to one of its inputs, then the allocator
  // needs to be able to handle it.
  Status RegisterArgument(const Tensor* t);

  // Makes 'tensor' a wrapper around the data buffer at 'ptr'. The buffer is
  // interpreted as having data type 'dtype' and shape 'shape'.
  Status MakeTensorFromBuffer(gpu::DeviceMemoryBase buffer, DataType dtype,
                              const TensorShape& shape, Tensor* tensor) const;

  // The Tensorflow BFC allocator used on GPU allows host-side deallocation
  // before GPU execution takes place. Tensorflow uses the ordering of the main
  // compute stream to enforce a happens-before relationship between a memory
  // allocation and code that reuses the same memory. If Tensorflow adds
  // support for multiple GPU streams or allocators with different ordering
  // requirements, this code may need to change.
  // (This attribute has no effect on CPU.)
  bool AllowsAsynchronousDeallocation() const override { return true; }

 private:
  OpKernelContext* const op_context_;

  // Map from pointer address to the owning Tensor; used by
  // MakeTensorFromBuffer. Also used to automatically release Tensors when the
  // allocator is freed.
  std::unordered_map<void*, Tensor> tensors_;
};

XlaAllocator::XlaAllocator(const gpu::Platform* platform,
                           OpKernelContext* op_context)
    : xla::DeviceMemoryAllocator(platform), op_context_(op_context) {}

XlaAllocator::~XlaAllocator() = default;

xla::StatusOr<gpu::DeviceMemoryBase> XlaAllocator::Allocate(
    int device_ordinal, uint64 size, bool retry_on_failure) {
  AllocatorAttributes allocator_attrs;
  allocator_attrs.set_on_host(false);

  AllocationAttributes allocation_attrs;
  allocation_attrs.no_retry_on_failure = !retry_on_failure;

  Tensor t;
  Status status = op_context_->allocate_temp(
      DT_UINT8, TensorShape({static_cast<int64>(size)}), &t, allocator_attrs,
      allocation_attrs);
  if (!status.ok()) {
    VLOG(2) << "Allocation failed " << size;
    return status;
  }
  void* data =
      reinterpret_cast<void*>(const_cast<char*>(t.tensor_data().data()));
  tensors_[data] = t;
  return gpu::DeviceMemoryBase(data, size);
}

Status XlaAllocator::RegisterArgument(const Tensor* t) {
  void* data =
      reinterpret_cast<void*>(const_cast<char*>(t->tensor_data().data()));
  tensors_[data] = *t;
  return Status::OK();
}

Status XlaAllocator::Deallocate(int device_ordinal,
                                gpu::DeviceMemoryBase* mem) {
  if (mem->opaque() != nullptr) {
    if (tensors_.erase(mem->opaque()) == 0) {
      return tensorflow::errors::InvalidArgument("Unknown tensor address");
    }
  }
  return Status::OK();
}

Status XlaAllocator::MakeTensorFromBuffer(gpu::DeviceMemoryBase buffer,
                                          DataType dtype,
                                          const TensorShape& shape,
                                          Tensor* out_tensor) const {
  void* ptr = const_cast<void*>(buffer.opaque());
  auto it = tensors_.find(ptr);
  if (it == tensors_.end()) {
    return errors::InvalidArgument("Unknown tensor address");
  }
  const Tensor& tensor = it->second;

  int64 output_size = DataTypeSize(dtype) * shape.num_elements();
  if (tensor.TotalBytes() == output_size) {
    out_tensor->UnsafeCopyFromInternal(tensor, dtype, shape);
  } else {
    Tensor slice = tensor.Slice(0, output_size);
    out_tensor->UnsafeCopyFromInternal(slice, dtype, shape);
  }
  return Status::OK();
}

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
    platform_id_ = gpu::host::kHostPlatformId;
  } else if (device_type_ == DeviceType(DEVICE_GPU)) {
    platform_id_ = gpu::cuda::kCudaPlatformId;
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

  auto platform = gpu::MultiPlatformManager::PlatformWithId(platform_id_);
  if (!platform.ok()) {
    return StreamExecutorUtil::ConvertStatus(platform.status());
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

std::vector<OptionalTensor> SnapshotResourceVariables(OpKernelContext* ctx,
                                                      int num_variables) {
  std::vector<OptionalTensor> snapshot(num_variables);
  int first_variable = ctx->num_inputs() - num_variables;
  for (int i = 0; i < num_variables; ++i) {
    Var* variable = nullptr;
    ResourceHandle handle = HandleFromInput(ctx, first_variable + i);
    if (LookupResource(ctx, handle, &variable).ok()) {
      tf_shared_lock lock(*variable->mu());
      snapshot[i].name = handle.name();
      snapshot[i].present = true;
      snapshot[i].value = *variable->tensor();
    }
  }
  return snapshot;
}

void XlaLocalLaunchOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "XlaLocalLaunchOp::Compute "
          << Canonicalize(function_.name(), AttrSlice(&function_.attr()));
  // We store information about the JIT-compiled XLA computation
  // in the ResourceMgr.
  ResourceMgr* rm = ctx->resource_manager();
  OP_REQUIRES(ctx, rm, errors::Internal("No resource manager."));

  gpu::Stream* stream =
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

  // Get the platform_id_ for XLA_* devices.
  if (platform_id_ == nullptr) {
    const XlaDevice::Metadata* metadata;
    Status s = XlaDevice::GetMetadata(ctx, &metadata);
    if (s.ok()) {
      platform_id_ = metadata->platform()->id();
    }
  }

  std::vector<OptionalTensor> variables =
      SnapshotResourceVariables(ctx, num_resource_args_);

  xla::LocalClient* client = static_cast<xla::LocalClient*>(cache->client());

  // Builds an XLA allocator for the device.
  XlaAllocator xla_allocator(client->platform(), ctx);

  XlaCompiler::Options options;
  options.client = client;
  options.device_type = &cache->device_type();
  options.flib_def = ctx->function_library()->GetFunctionLibraryDefinition();
  options.graph_def_version = ctx->function_library()->graph_def_version();
  options.allow_cpu_custom_calls = (platform_id_ == gpu::host::kHostPlatformId);
  options.device_allocator = &xla_allocator;

  const XlaCompiler::CompilationResult* kernel;
  xla::LocalExecutable* executable;

  OP_REQUIRES_OK(ctx, cache->Compile(options, function_, num_constant_args_,
                                     variables, ctx, &kernel, &executable,
                                     /*compile_options=*/nullptr));

  VLOG(1) << "Executing XLA Computation...";

  std::unique_ptr<xla::ShapedBuffer> output;
  // Build xla::ShapedBuffers that point directly to the Tensor buffers.
  std::vector<std::unique_ptr<xla::ShapedBuffer>> arg_buffers;
  arg_buffers.reserve(kernel->xla_input_shapes.size() + 1);
  arg_buffers.resize(kernel->xla_input_shapes.size());
  std::vector<xla::ShapedBuffer*> arg_ptrs(arg_buffers.size());

  const int first_variable_arg = ctx->num_inputs() - num_resource_args_;
  // Pass remaining parameters.
  const Tensor* t;
  for (int i = 0; i < kernel->xla_input_shapes.size(); ++i) {
    int arg_num = kernel->input_mapping[i];
    const xla::Shape& shape = kernel->xla_input_shapes[i];
    if (arg_num >= first_variable_arg) {
      t = &(variables[arg_num - first_variable_arg].value);
    } else {
      t = &(ctx->input(arg_num));
    }

    gpu::DeviceMemoryBase dmem = gpu::DeviceMemoryBase(
        const_cast<char*>(t->tensor_data().data()), t->tensor_data().size());

    const xla::Shape on_device_shape =
        client->backend().transfer_manager()->HostShapeToDeviceShape(shape);
    CHECK(xla::ShapeUtil::Equal(shape, on_device_shape))
        << "On-device shape "
        << xla::ShapeUtil::HumanStringWithLayout(on_device_shape)
        << " not the same as on-host shape "
        << xla::ShapeUtil::HumanStringWithLayout(shape);
    arg_buffers[i] = xla::MakeUnique<xla::ShapedBuffer>(
        /*on_host_shape=*/shape, /*on_device_shape=*/shape, client->platform(),
        client->default_device_ordinal());
    arg_buffers[i]->set_buffer(dmem, /*index=*/{});
    arg_ptrs[i] = arg_buffers[i].get();

    OP_REQUIRES_OK(ctx, xla_allocator.RegisterArgument(t));
  }

  // Execute the computation.
  VLOG(2) << "Executing computation.";
  xla::ExecutableRunOptions run_options;
  run_options.set_stream(stream);
  run_options.set_allocator(&xla_allocator);
  run_options.set_intra_op_thread_pool(&ctx->eigen_cpu_device());
  Env* env = Env::Default();
  auto start_time = env->NowMicros();
  auto run_result = executable->Run(arg_ptrs, run_options);
  OP_REQUIRES(ctx, run_result.ok(), run_result.status());

  output = run_result.ConsumeValueOrDie()->release();
  auto elapsed = env->NowMicros() - start_time;
  VLOG(2) << "Elapsed time: " << elapsed << "us";

  // Computation output should always be a tuple.
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Result tuple shape: " << output->on_host_shape().DebugString();
  }
  CHECK_EQ(ctx->num_outputs(), kernel->outputs.size());

  // Copy XLA results to the OpOutputList.
  int output_num = 0;
  for (int i = 0; i < ctx->num_outputs(); ++i) {
    if (kernel->outputs[i].is_constant) {
      // Output is a constant.
      const Tensor& const_tensor = kernel->outputs[i].constant_value;
      const size_t total_bytes = const_tensor.TotalBytes();
      if (stream && total_bytes > 0) {
        // Copy host -> device. (Empty tensors don't have backing buffers.)
        VLOG(1) << "Constant output tensor on device";
        Tensor* output_tensor;
        TF_CHECK_OK(
            ctx->allocate_output(i, const_tensor.shape(), &output_tensor));

        const void* src_ptr = DMAHelper::base(&const_tensor);
        void* dst_ptr = DMAHelper::base(output_tensor);
        gpu::DeviceMemoryBase gpu_dst_ptr(dst_ptr, total_bytes);
        stream->ThenMemcpy(&gpu_dst_ptr, src_ptr, total_bytes);
      } else {
        // No copy required.
        ctx->set_output(i, const_tensor);
      }
    } else {
      const TensorShape& shape = kernel->outputs[i].shape;
      VLOG(2) << "Retval " << i << " shape " << shape.DebugString();

      gpu::DeviceMemoryBase buffer = output->buffer({output_num});
      Tensor output_tensor;
      // Looks up the owning Tensor by buffer address.
      OP_REQUIRES_OK(ctx, xla_allocator.MakeTensorFromBuffer(
                              buffer, ctx->expected_output_dtype(i), shape,
                              &output_tensor));
      ctx->set_output(i, output_tensor);
      ++output_num;
    }

    if (VLOG_IS_ON(3)) {
      VLOG(3) << ctx->mutable_output(i)->DebugString();
    }
  }

  // Apply variable updates, if any.
  VLOG(2) << "Applying variable updates";
  for (int i = 0; i < kernel->resource_updates.size(); ++i) {
    const XlaCompiler::ResourceUpdate& write = kernel->resource_updates[i];
    OP_REQUIRES(ctx,
                write.input_index >= 0 && write.input_index < ctx->num_inputs(),
                errors::Internal("Invalid input index for variable write."));

    gpu::DeviceMemoryBase buffer = output->buffer({output_num});

    Var* variable = nullptr;
    // TODO(b/35625933): tensorflow::Var should contain a PersistentTensor, not
    // a Tensor.
    OP_REQUIRES_OK(ctx, LookupOrCreateResource<Var>(
                            ctx, HandleFromInput(ctx, write.input_index),
                            &variable, [this, ctx, &write](Var** ptr) {
                              *ptr = new Var(write.type);
                              return Status::OK();
                            }));

    core::ScopedUnref s(variable);

    mutex_lock ml(*variable->mu());
    OP_REQUIRES(ctx, variable->tensor()->dtype() == write.type,
                errors::Internal("Mismatched type in variable write"));

    // Looks up the owning Tensor by buffer address.
    OP_REQUIRES_OK(
        ctx, xla_allocator.MakeTensorFromBuffer(buffer, write.type, write.shape,
                                                variable->tensor()));
    ++output_num;
  }

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
