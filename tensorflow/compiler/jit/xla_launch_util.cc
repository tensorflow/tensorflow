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

#include "tensorflow/compiler/jit/xla_launch_util.h"

#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
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
#include "tensorflow/core/util/stream_executor_util.h"

namespace gpu = perftools::gputools;

namespace tensorflow {

std::map<int, OptionalTensor> SnapshotResourceVariables(OpKernelContext* ctx,
                                                        int num_variables) {
  std::map<int, OptionalTensor> snapshot;
  int first_variable = ctx->num_inputs() - num_variables;
  for (int i = 0; i < num_variables; ++i) {
    Var* variable = nullptr;
    ResourceHandle handle = HandleFromInput(ctx, first_variable + i);
    OptionalTensor& tensor = snapshot[first_variable + i];
    if (LookupResource(ctx, handle, &variable).ok()) {
      tf_shared_lock lock(*variable->mu());
      tensor.name = handle.name();
      tensor.present = true;
      tensor.value = *variable->tensor();
    }
  }
  return snapshot;
}

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

XlaComputationLaunchContext::XlaComputationLaunchContext(
    int64 num_resource_args, xla::LocalClient* client,
    XlaAllocator* xla_allocator)
    : num_resource_args_(num_resource_args),
      client_(client),
      xla_allocator_(xla_allocator) {}

void XlaComputationLaunchContext::PopulateInputs(
    OpKernelContext* ctx, const XlaCompiler::CompilationResult* kernel,
    const std::map<int, OptionalTensor>& variables) {
  // Build xla::ShapedBuffers that point directly to the Tensor buffers.
  arg_buffers_.reserve(kernel->xla_input_shapes.size() + 1);
  arg_buffers_.resize(kernel->xla_input_shapes.size());
  arg_ptrs_ = std::vector<xla::ShapedBuffer*>(arg_buffers_.size());

  // Pass remaining parameters.
  const Tensor* t;
  for (int i = 0; i < kernel->xla_input_shapes.size(); ++i) {
    int arg_num = kernel->input_mapping[i];
    const xla::Shape& shape = kernel->xla_input_shapes[i];
    if (variables.count(arg_num)) {
      t = &(variables.at(arg_num).value);
      CHECK(t);
    } else {
      t = &(ctx->input(arg_num));
    }

    gpu::DeviceMemoryBase dmem = gpu::DeviceMemoryBase(
        const_cast<char*>(t->tensor_data().data()), t->tensor_data().size());

    const xla::Shape on_device_shape =
        client_->backend().transfer_manager()->HostShapeToDeviceShape(shape);
    CHECK(xla::ShapeUtil::Equal(shape, on_device_shape))
        << "On-device shape "
        << xla::ShapeUtil::HumanStringWithLayout(on_device_shape)
        << " not the same as on-host shape "
        << xla::ShapeUtil::HumanStringWithLayout(shape);
    arg_buffers_[i] = xla::MakeUnique<xla::ShapedBuffer>(
        /*on_host_shape=*/shape, /*on_device_shape=*/shape, client_->platform(),
        client_->default_device_ordinal());
    arg_buffers_[i]->set_buffer(dmem, /*index=*/{});
    arg_ptrs_[i] = arg_buffers_[i].get();

    OP_REQUIRES_OK(ctx, xla_allocator_->RegisterArgument(t));
  }
}

void XlaComputationLaunchContext::PopulateOutputs(
    OpKernelContext* ctx, const XlaCompiler::CompilationResult* kernel,
    std::unique_ptr<xla::ShapedBuffer> output) {
  gpu::Stream* stream =
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

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
      OP_REQUIRES_OK(ctx, xla_allocator_->MakeTensorFromBuffer(
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
    // TODO(b/35625933): tensorflow::Var should contain a PersistentTensor,
    // not a Tensor.
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
    OP_REQUIRES_OK(ctx,
                   xla_allocator_->MakeTensorFromBuffer(
                       buffer, write.type, write.shape, variable->tensor()));
    ++output_num;
  }
}

}  // namespace tensorflow
