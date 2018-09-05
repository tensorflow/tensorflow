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

#include <memory>

#include "absl/memory/memory.h"
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

namespace tensorflow {
namespace {
using xla::ScopedShapedBuffer;
using xla::ShapedBuffer;
}  // anonymous namespace

std::map<int, OptionalTensor> SnapshotResourceVariables(
    OpKernelContext* ctx, const std::vector<int>& variables) {
  std::map<int, OptionalTensor> snapshot;
  for (int i : variables) {
    Var* variable = nullptr;
    ResourceHandle handle = HandleFromInput(ctx, i);
    OptionalTensor& tensor = snapshot[i];
    if (LookupResource(ctx, handle, &variable).ok()) {
      tf_shared_lock lock(*variable->mu());
      tensor.name = handle.name();
      tensor.present = true;
      tensor.value = *variable->tensor();
    }
  }
  return snapshot;
}

XlaAllocator::XlaAllocator(const se::Platform* platform, Allocator* wrapped)
    : xla::DeviceMemoryAllocator(platform), wrapped_(wrapped) {}

XlaAllocator::~XlaAllocator() {}

xla::StatusOr<xla::OwningDeviceMemory> XlaAllocator::Allocate(
    int device_ordinal, uint64 size, bool retry_on_failure) {
  AllocationAttributes attrs;
  attrs.no_retry_on_failure = !retry_on_failure;
  void* data = nullptr;
  if (size != 0) {
    data = wrapped_->AllocateRaw(Allocator::kAllocatorAlignment, size, attrs);
    if (data == nullptr) {
      return errors::ResourceExhausted(
          "Out of memory while trying to allocate ", size, " bytes.");
    }
  }
  return xla::OwningDeviceMemory(se::DeviceMemoryBase(data, size),
                                 device_ordinal, this);
}

Status XlaAllocator::Deallocate(int device_ordinal, se::DeviceMemoryBase mem) {
  wrapped_->DeallocateRaw(mem.opaque());
  return Status::OK();
}

namespace internal {
// Return the 'index''th subtree of the given ShapedBuffer as a
// ScopedShapedBuffer. The returned ScopedShapedBuffer takes ownership of the
// subtree, and sets the input's buffer pointers to nullptr for the subtree.
ScopedShapedBuffer ExtractSubShapedBuffer(
    ShapedBuffer* shaped_buffer, int index,
    xla::DeviceMemoryAllocator* allocator) {
  const xla::Shape& on_host_shape = xla::ShapeUtil::GetTupleElementShape(
      shaped_buffer->on_host_shape(), index);
  const xla::Shape& on_device_shape = xla::ShapeUtil::GetTupleElementShape(
      shaped_buffer->on_device_shape(), index);

  ShapedBuffer sub_shaped_buffer(on_host_shape, on_device_shape,
                                 shaped_buffer->platform(),
                                 shaped_buffer->device_ordinal());

  auto& shape_tree = shaped_buffer->buffers();
  auto& sub_shape_tree = sub_shaped_buffer.buffers();
  sub_shape_tree.CopySubtreeFrom(shape_tree,
                                 /*source_base_index=*/{index},
                                 /*target_base_index=*/{});
  shape_tree.ForEachMutableElement(
      [index](const xla::ShapeIndex& shape_index,
              tensorflow::se::DeviceMemoryBase* data) {
        // shape_index is empty for the root node. Ignore that.
        if (!shape_index.empty() && shape_index[0] == index) {
          *data = tensorflow::se::DeviceMemoryBase(nullptr, 0);
        }
      });
  return ScopedShapedBuffer(std::move(sub_shaped_buffer), allocator);
}
}  // namespace internal
using internal::ExtractSubShapedBuffer;

XlaComputationLaunchContext::XlaComputationLaunchContext(
    xla::LocalClient* client, xla::DeviceMemoryAllocator* xla_allocator,
    bool allocate_xla_tensors, bool use_multiple_streams)
    : client_(client),
      xla_allocator_(xla_allocator),
      allocate_xla_tensors_(allocate_xla_tensors),
      use_multiple_streams_(use_multiple_streams) {
  if (use_multiple_streams_) {
    CHECK(allocate_xla_tensors_) << "To use multiple streams correctly we must "
                                    "be allocating XLA tensors!";
  }
}

void XlaComputationLaunchContext::PopulateInputs(
    OpKernelContext* ctx, const XlaCompiler::CompilationResult* kernel,
    const std::map<int, OptionalTensor>& variables) {
  se::Stream* stream =
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;
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

    if (use_multiple_streams_) {
      CHECK(stream) << "Must have a stream available when using XLA tensors!";
      XlaTensor* xla_tensor = XlaTensor::FromTensor(t);
      CHECK(xla_tensor);
      if (se::Event* event = xla_tensor->GetDefinitionEvent(stream)) {
        stream->ThenWaitFor(event);
        xla_tensor->SetDefinedOn(stream);
      }
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
      se::DeviceMemoryBase dmem = XlaTensor::DeviceMemoryFromTensor(*t);
      arg_buffers_[i] = absl::make_unique<ShapedBuffer>(
          /*on_host_shape=*/shape, /*on_device_shape=*/shape,
          client_->platform(), client_->default_device_ordinal());
      arg_buffers_[i]->set_buffer(dmem, /*index=*/{});
      arg_ptrs_[i] = arg_buffers_[i].get();
    }
  }
}

Status XlaComputationLaunchContext::PopulateOutputs(
    OpKernelContext* ctx, const XlaCompiler::CompilationResult* kernel,
    ScopedShapedBuffer output) {
  se::Stream* stream =
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

  // Computation output should always be a tuple.
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Result tuple shape: " << output.on_host_shape().DebugString();
    VLOG(2) << "Result tuple shape (on device): "
            << output.on_device_shape().DebugString();
  }
  CHECK_EQ(ctx->num_outputs(), kernel->outputs.size());

  // If the on-host-shape isn't a tuple, create a new single-element tuple
  // buffer with a nullptr root index table. This allows the code below to treat
  // output as a tuple unconditionally.
  if (!xla::ShapeUtil::IsTuple(output.on_host_shape())) {
    ShapedBuffer nontuple_buffer = output.release();
    ShapedBuffer buffer(
        xla::ShapeUtil::MakeTupleShape({nontuple_buffer.on_host_shape()}),
        xla::ShapeUtil::MakeTupleShape({nontuple_buffer.on_device_shape()}),
        output.platform(), output.device_ordinal());
    buffer.buffers().CopySubtreeFrom(nontuple_buffer.buffers(),
                                     /*source_base_index=*/{},
                                     /*target_base_index=*/{0});
    output = ScopedShapedBuffer(std::move(buffer), output.memory_allocator());
  }

  std::shared_ptr<se::Event> definition_event;
  if (use_multiple_streams_) {
    definition_event = std::make_shared<se::Event>(stream->parent());
    if (!definition_event->Init()) {
      return errors::Internal("Failed to initialize tensor definition event.");
    }
    stream->ThenRecordEvent(definition_event.get());
  }

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

        TF_RETURN_IF_ERROR(
            ctx->allocate_output(i, const_tensor.shape(), &output_tensor));

        Device* device = dynamic_cast<Device*>(ctx->device());
        if (device == nullptr) {
          return errors::Internal("DeviceBase was not a Device.");
        }
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
      const DataType& type = kernel->outputs[i].type;
      VLOG(2) << "Retval " << i << " shape " << shape.DebugString() << " type "
              << DataTypeString(type);
      if (type == DT_RESOURCE) {
        ctx->set_output(i, ctx->input(kernel->outputs[i].input_index));
      } else {
        se::DeviceMemoryBase buffer = output.buffer({output_num});
        if (allocate_xla_tensors_) {
          Tensor* output_tensor;
          TF_RETURN_IF_ERROR(ctx->allocate_output(i, shape, &output_tensor));
          XlaTensor* xla_tensor = XlaTensor::FromTensor(output_tensor);
          if (xla_tensor) {
            xla_tensor->set_shaped_buffer(ScopedShapedBuffer(
                ExtractSubShapedBuffer(&output, output_num, xla_allocator_)));
            if (use_multiple_streams_) {
              xla_tensor->SetDefinedOn(stream, definition_event);
            }
          } else {
            // xla_tensor wasn't valid, which must mean this is a zero-element
            // tensor.
            CHECK_EQ(output_tensor->TotalBytes(), 0);
          }
        } else {
          Tensor output_tensor = XlaTensorBuffer::MakeTensor(
              ctx->expected_output_dtype(i), shape, buffer, allocator);
          output.set_buffer(xla::OwningDeviceMemory(), {output_num});
          ctx->set_output(i, output_tensor);
        }
        ++output_num;
      }
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
    if (write.input_index < 0 || write.input_index >= ctx->num_inputs()) {
      return errors::Internal("Invalid input index for variable write.");
    }

    se::DeviceMemoryBase buffer = output.buffer({output_num});

    Var* variable = nullptr;
    // TODO(b/35625933): tensorflow::Var should contain a PersistentTensor,
    // not a Tensor.
    TF_RETURN_IF_ERROR(LookupOrCreateResource<Var>(
        ctx, HandleFromInput(ctx, write.input_index), &variable,
        [&write](Var** ptr) {
          *ptr = new Var(write.type);
          return Status::OK();
        }));

    core::ScopedUnref s(variable);

    mutex_lock ml(*variable->mu());
    if (variable->tensor()->dtype() != write.type) {
      return errors::Internal("Mismatched type in variable write");
    }

    if (allocate_xla_tensors_) {
      Tensor output_tensor;
      TF_RETURN_IF_ERROR(
          ctx->allocate_temp(write.type, write.shape, &output_tensor));
      XlaTensor* xla_tensor = XlaTensor::FromTensor(&output_tensor);
      CHECK(xla_tensor);
      xla_tensor->set_shaped_buffer(
          ExtractSubShapedBuffer(&output, output_num, xla_allocator_));
      if (use_multiple_streams_) {
        xla_tensor->SetDefinedOn(stream, definition_event);
      }
      *variable->tensor() = output_tensor;
    } else {
      Tensor output_tensor = XlaTensorBuffer::MakeTensor(
          write.type, write.shape, buffer, allocator);
      output.set_buffer(xla::OwningDeviceMemory(), {output_num});
      *variable->tensor() = output_tensor;
    }
    ++output_num;
  }
  return Status::OK();
}

}  // namespace tensorflow
