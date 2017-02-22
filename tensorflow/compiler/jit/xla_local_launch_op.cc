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

#include "tensorflow/compiler/jit/xla_local_launch_op.h"

#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_local_runtime_context.h"
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
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/util/stream_executor_util.h"

namespace gpu = perftools::gputools;

namespace tensorflow {

REGISTER_OP("_XlaLaunch")
    .Input("constants: Tconstants")
    .Attr("Tconstants: list(type) >= 0")
    .Input("args: Targs")
    .Attr("Targs: list(type) >= 0")
    .Input("resources: Nresources * resource")
    .Attr("Nresources: int >= 0")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Attr("function: func")
    // XLA random-number generation ops are stateful.
    // TODO(phawkins): create stateful and non-stateful variants of _XlaLaunch.
    .SetIsStateful()
    .Doc("XLA Launch Op. For use by the XLA JIT only.");

// Adapter class that wraps a Tensorflow allocator as an XLA allocator.
class XlaAllocator : public xla::DeviceMemoryAllocator {
 public:
  XlaAllocator(const perftools::gputools::Platform* platform,
               OpKernelContext* op_context);
  ~XlaAllocator() override;
  xla::StatusOr<perftools::gputools::DeviceMemoryBase> Allocate(
      int device_ordinal, uint64 size, bool retry_on_failure = true) override;
  Status Deallocate(int device_ordinal,
                    perftools::gputools::DeviceMemoryBase* mem) override;

  // Makes 'tensor' a wrapper around the data buffer at 'ptr'. The buffer is
  // interpreted as having data type 'dtype' and shape 'shape'.
  Status MakeTensorFromBuffer(gpu::DeviceMemoryBase buffer, DataType dtype,
                              const TensorShape& shape, Tensor* tensor) const;

 private:
  OpKernelContext* const op_context_;

  // Map from pointer address to the owning Tensor; used by
  // MakeTensorFromBuffer. Also used to automatically release Tensors when the
  // allocator is freed.
  std::unordered_map<void*, Tensor> tensors_;
};

XlaAllocator::XlaAllocator(const perftools::gputools::Platform* platform,
                           OpKernelContext* op_context)
    : xla::DeviceMemoryAllocator(platform), op_context_(op_context) {}

XlaAllocator::~XlaAllocator() = default;

xla::StatusOr<perftools::gputools::DeviceMemoryBase> XlaAllocator::Allocate(
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
  TF_RET_CHECK(data != nullptr);
  tensors_[data] = t;
  return perftools::gputools::DeviceMemoryBase(data, size);
}

Status XlaAllocator::Deallocate(int device_ordinal,
                                perftools::gputools::DeviceMemoryBase* mem) {
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

  int num_resource_args;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Nresources", &num_resource_args));
  OP_REQUIRES(ctx, num_resource_args == 0,
              errors::Unimplemented(
                  "XlaLocalLaunchOp does not support resource variables"));
}

Status XlaLocalLaunchOp::BuildCompilationCache(XlaCompilationCache** compiler) {
  gpu::Platform::Id platform_id;
  if (device_type_ == DeviceType(DEVICE_CPU)) {
    platform_id = gpu::host::kHostPlatformId;
  } else if (device_type_ == DeviceType(DEVICE_GPU)) {
    platform_id = gpu::cuda::kCudaPlatformId;
  } else {
    return errors::InvalidArgument("Unknown device type for local _XlaLaunch");
  }

  auto platform = gpu::MultiPlatformManager::PlatformWithId(platform_id);
  if (!platform.ok()) {
    return StreamExecutorUtil::ConvertStatus(platform.status());
  }
  auto client =
      xla::ClientLibrary::GetOrCreateLocalClient(platform.ValueOrDie());
  if (!client.ok()) {
    return client.status();
  }
  const XlaOpRegistry::DeviceRegistration* registration;
  if (!XlaOpRegistry::GetCompilationDevice(device_type_.type(),
                                           &registration)) {
    return errors::InvalidArgument("No JIT device registered for ",
                                   device_type_.type());
  }
  XlaCompiler::Options options;
  options.device_type = DeviceType(registration->compilation_device_name);
  options.client = client.ValueOrDie();
  options.allow_cpu_custom_calls = (platform_id == gpu::host::kHostPlatformId);
  options.local_executable_has_hybrid_result = true;
  *compiler = new XlaCompilationCache(options);
  return Status::OK();
}

void XlaLocalLaunchOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "XlaLocalLaunchOp::Compute "
          << Canonicalize(function_.name(), function_.attr());
  // We store information about the JIT-compiled XLA computation
  // in the ResourceMgr.
  ResourceMgr* rm = ctx->resource_manager();
  OP_REQUIRES(ctx, rm, errors::Internal("No resource manager."));

  gpu::Stream* stream =
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

  XlaCompilationCache* compiler;
  OP_REQUIRES_OK(ctx,
                 rm->LookupOrCreate<XlaCompilationCache>(
                     rm->default_container(), "xla_compiler", &compiler,
                     [this](XlaCompilationCache** compiler) {
                       return BuildCompilationCache(compiler);
                     }));
  // Hold the reference to the JIT during evaluation. (We could probably
  // free it sooner because the ResourceMgr will retain a reference, but
  // this is more obviously correct.)
  core::ScopedUnref compiler_ref(compiler);

  xla::LocalClient* client = static_cast<xla::LocalClient*>(compiler->client());

  const XlaCompiler::CompilationResult* kernel;
  xla::LocalExecutable* executable;
  OP_REQUIRES_OK(ctx, compiler->Compile(function_, num_constant_args_, {}, ctx,
                                        &kernel, &executable));

  VLOG(1) << "Executing XLA Computation...";

  // Builds an XLA allocator for the device.
  XlaAllocator xla_allocator(client->platform(), ctx);
  XlaLocalRuntimeContext local_runtime_context;

  std::unique_ptr<xla::ShapedBuffer> output;
  bool output_is_tuple;
  if (!kernel->computation.IsNull()) {
    // Build xla::ShapedBuffers that point directly to the Tensor buffers.
    std::vector<std::unique_ptr<xla::ShapedBuffer>> arg_buffers;
    arg_buffers.reserve(kernel->xla_input_shapes.size() + 1);
    arg_buffers.resize(kernel->xla_input_shapes.size());
    std::vector<xla::ShapedBuffer*> arg_ptrs(arg_buffers.size());

    // Pass remaining parameters.
    for (int i = 0; i < kernel->xla_input_shapes.size(); ++i) {
      int arg_num = kernel->xla_input_shapes[i].first;
      const xla::Shape& shape = kernel->xla_input_shapes[i].second;
      gpu::DeviceMemoryBase dmem(
          const_cast<char*>(ctx->input(arg_num).tensor_data().data()),
          ctx->input(arg_num).tensor_data().size());

      arg_buffers[i] =
          xla::ShapedBuffer::MakeArrayShapedBuffer(
              shape, client->platform(), client->default_device_ordinal(), dmem)
              .ConsumeValueOrDie();
      arg_ptrs[i] = arg_buffers[i].get();
    }

    // Make the final parameter point at local_runtime_context.
    if (kernel->requires_runtime_context) {
      gpu::DeviceMemoryBase local_runtime_context_dmem(
          &local_runtime_context, sizeof(local_runtime_context));
      arg_buffers.push_back(
          xla::ShapedBuffer::MakeArrayShapedBuffer(
              xla::ShapeUtil::MakeOpaqueShape(), client->platform(),
              client->default_device_ordinal(), local_runtime_context_dmem)
              .ConsumeValueOrDie());
      arg_ptrs.push_back(arg_buffers.back().get());
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

    if (local_runtime_context.error) {
      ctx->CtxFailure(errors::InvalidArgument(
          "Compiled kernel returned error: ", local_runtime_context.error_msg));
      return;
    }

    output = std::move(run_result.ValueOrDie());
    auto elapsed = env->NowMicros() - start_time;
    VLOG(2) << "Elapsed time: " << elapsed << "us";

    // Computation output should always be a tuple.
    if (VLOG_IS_ON(2)) {
      VLOG(2) << "Result tuple shape: " << output->shape().DebugString();
    }
    output_is_tuple = xla::ShapeUtil::IsTuple(output->shape());
  }
  CHECK_EQ(ctx->num_outputs(), kernel->outputs.size());

  // Copy XLA results to the OpOutputList.
  int output_num = 0;
  for (int i = 0; i < ctx->num_outputs(); ++i) {
    if (kernel->outputs[i].is_constant) {
      // Output is a constant
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

      gpu::DeviceMemoryBase buffer;
      if (output_is_tuple) {
        buffer = output->buffer({output_num});
      } else {
        CHECK_EQ(0, output_num);
        buffer = output->buffer({});
      }
      Tensor output_tensor;
      // Looks up the owning Tensor by buffer address.
      OP_REQUIRES_OK(
          ctx,
          xla_allocator.MakeTensorFromBuffer(
              buffer, ctx->expected_output_dtype(i), shape, &output_tensor));
      ctx->set_output(i, output_tensor);
      ++output_num;
    }

    if (VLOG_IS_ON(3)) {
      VLOG(3) << ctx->mutable_output(i)->DebugString();
    }
  }

  VLOG(1) << "Done";
}

XlaLocalLaunchOp::~XlaLocalLaunchOp() {
  VLOG(1) << "XlaLocalLaunchOp destroyed";
}

REGISTER_KERNEL_BUILDER(Name("_XlaLaunch").Device(DEVICE_CPU),
                        XlaLocalLaunchOp);

REGISTER_KERNEL_BUILDER(
    Name("_XlaLaunch").Device(DEVICE_GPU).HostMemory("constants"),
    XlaLocalLaunchOp);

}  // namespace tensorflow
