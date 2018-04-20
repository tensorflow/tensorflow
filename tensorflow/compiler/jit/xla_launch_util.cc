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
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/stream_executor_util.h"

namespace {
namespace gpu = perftools::gputools;
using xla::ScopedShapedBuffer;
using xla::ShapedBuffer;
}  // anonymous namespace

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

XlaAllocator::XlaAllocator(const gpu::Platform* platform, Allocator* wrapped)
    : xla::DeviceMemoryAllocator(platform), wrapped_(wrapped) {}

XlaAllocator::~XlaAllocator() {}

xla::StatusOr<gpu::DeviceMemoryBase> XlaAllocator::Allocate(
    int device_ordinal, uint64 size, bool retry_on_failure) {
  void* data = wrapped_->AllocateRaw(Allocator::kAllocatorAlignment, size);
  if (data == nullptr) {
    return errors::ResourceExhausted("Out of memory while trying to allocate ",
                                     size, " bytes.");
  } else {
    return gpu::DeviceMemoryBase(data, size);
  }
}

Status XlaAllocator::Deallocate(int device_ordinal,
                                gpu::DeviceMemoryBase* mem) {
  wrapped_->DeallocateRaw(mem->opaque());
  return Status::OK();
}

namespace {
// Return the 'index''th subtree of the given ShapedBuffer as a
// ScopedShapedBuffer. The returned ScopedShapedBuffer takes ownership of the
// subtree, and sets the input's buffer pointers to nullptr for the subtree.
ScopedShapedBuffer ExtractSubShapedBuffer(
    ShapedBuffer* shaped_buffer, int index,
    xla::DeviceMemoryAllocator* allocator) {
  xla::Shape on_host_shape = xla::ShapeUtil::GetTupleElementShape(
      shaped_buffer->on_host_shape(), index);
  xla::Shape on_device_shape = xla::ShapeUtil::GetTupleElementShape(
      shaped_buffer->on_device_shape(), index);

  ShapedBuffer sub_shaped_buffer(on_host_shape, on_device_shape,
                                 shaped_buffer->platform(),
                                 shaped_buffer->device_ordinal());

  auto& shape_tree = shaped_buffer->buffers();
  auto& sub_shape_tree = sub_shaped_buffer.buffers();
  sub_shape_tree.CopySubtreeFrom(shape_tree,
                                 /*source_base_index=*/{index},
                                 /*target_base_index=*/{});
  for (auto& index_to_buffer : shape_tree) {
    if (!index_to_buffer.first.empty() && index_to_buffer.first[0] == index) {
      index_to_buffer.second = gpu::DeviceMemoryBase(nullptr, 0);
    }
  }
  return ScopedShapedBuffer(std::move(sub_shaped_buffer), allocator);
}
}  // namespace

XlaComputationLaunchContext::XlaComputationLaunchContext(
    int64 num_resource_args, xla::LocalClient* client,
    xla::DeviceMemoryAllocator* xla_allocator, bool allocate_xla_tensors)
    : num_resource_args_(num_resource_args),
      client_(client),
      xla_allocator_(xla_allocator),
      allocate_xla_tensors_(allocate_xla_tensors) {}

void XlaComputationLaunchContext::PopulateInputs(
    OpKernelContext* ctx, const XlaCompiler::CompilationResult* kernel,
    const std::map<int, OptionalTensor>& variables) {
  // Build ShapedBuffers that point directly to the Tensor buffers.
  arg_buffers_.reserve(kernel->xla_input_shapes.size() + 1);
  arg_buffers_.resize(kernel->xla_input_shapes.size());
  arg_ptrs_ = std::vector<ShapedBuffer*>(arg_buffers_.size());

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

    const xla::Shape on_device_shape =
        client_->backend().transfer_manager()->HostShapeToDeviceShape(shape);
    if (xla::ShapeUtil::IsTuple(on_device_shape)) {
      const XlaTensor* xla_tensor = XlaTensor::FromTensor(t);
      CHECK(xla_tensor && xla_tensor->has_shaped_buffer());
      arg_ptrs_[i] = const_cast<ShapedBuffer*>(&xla_tensor->shaped_buffer());
    } else {
      CHECK(xla::ShapeUtil::Equal(shape, on_device_shape))
          << "On-device shape "
          << xla::ShapeUtil::HumanStringWithLayout(on_device_shape)
          << " not the same as on-host shape "
          << xla::ShapeUtil::HumanStringWithLayout(shape);
      gpu::DeviceMemoryBase dmem = XlaTensor::DeviceMemoryFromTensor(*t);
      arg_buffers_[i] = xla::MakeUnique<ShapedBuffer>(
          /*on_host_shape=*/shape, /*on_device_shape=*/shape,
          client_->platform(), client_->default_device_ordinal());
      arg_buffers_[i]->set_buffer(dmem, /*index=*/{});
      arg_ptrs_[i] = arg_buffers_[i].get();
    }
  }
}

void XlaComputationLaunchContext::PopulateOutputs(
    OpKernelContext* ctx, const XlaCompiler::CompilationResult* kernel,
    ScopedShapedBuffer output) {
  gpu::Stream* stream =
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

  // Computation output should always be a tuple.
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Result tuple shape: " << output.on_host_shape().DebugString();
    VLOG(2) << "Result tuple shape (on device): "
            << output.on_device_shape().DebugString();
  }
  CHECK_EQ(ctx->num_outputs(), kernel->outputs.size());

  // Copy XLA results to the OpOutputList.
  int output_num = 0;
  for (int i = 0; i < ctx->num_outputs(); ++i) {
    Allocator* allocator = ctx->device()->GetAllocator({});
    if (kernel->outputs[i].is_constant) {
      // Output is a constant.
      const Tensor& const_tensor = kernel->outputs[i].constant_value;
      Tensor* output_tensor;
      const size_t total_bytes = const_tensor.TotalBytes();
      if (stream && total_bytes > 0) {
        // Copy host -> device. (Empty tensors don't have backing buffers.)
        // Manually allocate memory using an XlaTensorBuffer so we can allocate
        // as much memory as the device requires (as given by
        // GetByteSizeRequirement). This avoids XlaTransferManager having to
        // reallocate the device buffer later.
        VLOG(1) << "Constant output tensor on device";

        OP_REQUIRES_OK(
            ctx, ctx->allocate_output(i, const_tensor.shape(), &output_tensor));
        if (XlaTensor* xla_tensor = XlaTensor::FromTensor(output_tensor)) {
          OP_REQUIRES_OK(ctx, xla_tensor->AllocateShapedBuffer(
                                  const_tensor.dtype(), const_tensor.shape(),
                                  client_, stream->parent()->device_ordinal()));
        }

        Device* device = dynamic_cast<Device*>(ctx->device());
        OP_REQUIRES(ctx, device != nullptr,
                    errors::Internal("DeviceBase was not a Device."));
        ctx->op_device_context()->CopyCPUTensorToDevice(
            &const_tensor, device, output_tensor,
            [&](Status status) { TF_CHECK_OK(status); });

        if (device->device_type() == DEVICE_GPU) {
          // The GPUDeviceContext enqueues the host->device transfer in a
          // separate stream from the main compute stream. We must ensure the
          // compute stream is synchronized with the host->device transfer
          // stream now otherwise we will create a race condition.
          auto* gpu_device_context =
              static_cast<GPUDeviceContext*>(ctx->op_device_context());
          gpu_device_context->stream()->ThenWaitFor(
              gpu_device_context->host_to_device_stream());
        }
      } else {
        // No copy required.
        ctx->set_output(i, const_tensor);
        output_tensor = ctx->mutable_output(i);
      }
      if (XlaTensor* xla_tensor = XlaTensor::FromTensor(output_tensor)) {
        xla_tensor->set_host_tensor(const_tensor);
      }
    } else {
      const TensorShape& shape = kernel->outputs[i].shape;
      VLOG(2) << "Retval " << i << " shape " << shape.DebugString();

      gpu::DeviceMemoryBase buffer = output.buffer({output_num});
      if (allocate_xla_tensors_) {
        Tensor* output_tensor;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(i, shape, &output_tensor));
        XlaTensor* xla_tensor = XlaTensor::FromTensor(output_tensor);
        CHECK(xla_tensor);
        xla_tensor->set_shaped_buffer(ScopedShapedBuffer(
            ExtractSubShapedBuffer(&output, output_num, xla_allocator_)));
      } else {
        Tensor output_tensor = XlaTensorBuffer::MakeTensor(
            ctx->expected_output_dtype(i), shape, buffer, allocator);
        output.set_buffer(gpu::DeviceMemoryBase(nullptr, 0), {output_num});
        ctx->set_output(i, output_tensor);
      }
      ++output_num;
    }

    if (VLOG_IS_ON(3)) {
      VLOG(3) << ctx->mutable_output(i)->DebugString();
    }
  }

  // Apply variable updates, if any.
  VLOG(2) << "Applying variable updates";
  for (int i = 0; i < kernel->resource_updates.size(); ++i) {
    Allocator* allocator = ctx->device()->GetAllocator({});
    const XlaCompiler::ResourceUpdate& write = kernel->resource_updates[i];
    OP_REQUIRES(ctx,
                write.input_index >= 0 && write.input_index < ctx->num_inputs(),
                errors::Internal("Invalid input index for variable write."));

    gpu::DeviceMemoryBase buffer = output.buffer({output_num});

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

    if (allocate_xla_tensors_) {
      Tensor output_tensor;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(write.type, write.shape, &output_tensor));
      XlaTensor* xla_tensor = XlaTensor::FromTensor(&output_tensor);
      CHECK(xla_tensor);
      xla_tensor->set_shaped_buffer(
          ExtractSubShapedBuffer(&output, output_num, xla_allocator_));
      *variable->tensor() = output_tensor;
    } else {
      Tensor output_tensor = XlaTensorBuffer::MakeTensor(
          write.type, write.shape, buffer, allocator);
      output.set_buffer(gpu::DeviceMemoryBase(nullptr, 0), {output_num});
      *variable->tensor() = output_tensor;
    }
    ++output_num;
  }
}

}  // namespace tensorflow
