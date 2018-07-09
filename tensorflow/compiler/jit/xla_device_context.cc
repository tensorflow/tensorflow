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

#include "tensorflow/compiler/jit/xla_device_context.h"

#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {

// The allocator used for Tensors assigned to the XLA device.
XlaDeviceAllocator::XlaDeviceAllocator() {}
XlaDeviceAllocator::~XlaDeviceAllocator() = default;

string XlaDeviceAllocator::Name() { return "xla"; }

void* XlaDeviceAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  // We always return an empty XlaTensor object, encoded as an opaque tagged
  // pointer. We can return an empty object and ignore num_bytes here because we
  // have control over all of the uses of this device tensor, and can lazily
  // allocate memory when used. This allows us to also know the shape of the
  // allocated Tensor, which is useful if the device's tensor representation
  // differs from the host.
  return XlaTensor::ToOpaquePointer(new XlaTensor());
}

void XlaDeviceAllocator::DeallocateRaw(void* ptr) {
  delete XlaTensor::FromOpaquePointer(ptr);
}

void XlaDeviceAllocator::GetStats(AllocatorStats* stats) { stats->Clear(); }

XlaTransferManager::XlaTransferManager(
    se::Stream* stream, xla::LocalClient* client, bool transfer_as_literal,
    XlaCompiler::ShapeRepresentationFn shape_representation_fn)
    : stream_(stream),
      client_(client),
      transfer_manager_(client->backend().transfer_manager()),
      transfer_as_literal_(transfer_as_literal),
      shape_representation_fn_(std::move(shape_representation_fn)) {
  if (!shape_representation_fn_) {
    shape_representation_fn_ =
        [](const TensorShape& shape,
           DataType dtype) -> xla::StatusOr<TensorShape> { return shape; };
  }
}

Status XlaTransferManager::TransferLiteralToDevice(
    const Tensor& host_tensor, Tensor* device_tensor) const {
  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(TensorShapeToXLAShape(host_tensor.dtype(),
                                           host_tensor.shape(), &xla_shape));
  // Create a reference to hold onto host_tensor until after the literal has
  // been transferred. Also make sure the literal exists until the function
  // asynchronously completes, as it will be wrapped in an xla::LiteralSlice.
  TensorReference ref(host_tensor);
  auto literal = std::make_shared<xla::BorrowingLiteral>(
      static_cast<const char*>(DMAHelper::base(&host_tensor)), xla_shape);

  const xla::ShapedBuffer& shaped_buffer =
      XlaTensor::FromTensor(device_tensor)->shaped_buffer();
  VLOG(1) << "Transfer to device as literal: " << literal->ToString() << " "
          << shaped_buffer.ToString();
  TF_RETURN_IF_ERROR(transfer_manager_->TransferLiteralToDeviceAsync(
      stream_, *literal, shaped_buffer));
  // Unref the host tensor, and capture the literal shared_ptr too so it goes
  // out of scope when the lambda completes.
  stream_->ThenDoHostCallback([ref, literal]() { ref.Unref(); });
  return Status::OK();
}

void XlaTransferManager::TransferLiteralFromDevice(
    Tensor* host_tensor, const Tensor& device_tensor,
    const StatusCallback& done) const {
  const xla::ShapedBuffer& shaped_buffer =
      XlaTensor::FromTensor(&device_tensor)->shaped_buffer();

  TensorReference ref(device_tensor);
  transfer_manager_->TransferLiteralFromDevice(
      stream_, shaped_buffer,
      [=, &shaped_buffer](
          xla::StatusOr<std::unique_ptr<xla::Literal> > literal_or) {
        ref.Unref();
        done([&]() -> Status {
          TF_ASSIGN_OR_RETURN(auto literal, std::move(literal_or));
          VLOG(1) << "Transfer from device as literal: " << literal->ToString()
                  << " " << shaped_buffer.ToString();
          Tensor tensor;
          TF_RETURN_IF_ERROR(
              LiteralToHostTensor(*literal, host_tensor->dtype(), &tensor));
          // Reshape the tensor back to its declared shape.
          Status status;
          if (!host_tensor->CopyFrom(tensor, device_tensor.shape())) {
            status = errors::Internal(
                "Tensor::CopyFrom failed when copying from XLA device to CPU");
          }
          return status;
        }());
      });
}

void XlaTransferManager::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                               Device* device,
                                               Tensor* device_tensor,
                                               StatusCallback done) const {
  if (cpu_tensor->NumElements() > 0) {
    VLOG(2) << "CopyCPUTensorToDevice "
            << reinterpret_cast<const void*>(cpu_tensor->tensor_data().data())
            << " "
            << reinterpret_cast<const void*>(
                   device_tensor->tensor_data().data())
            << " " << cpu_tensor->NumElements() << " "
            << cpu_tensor->shape().DebugString() << " "
            << device_tensor->shape().DebugString();

    void* src_ptr = const_cast<void*>(DMAHelper::base(cpu_tensor));
    const int64 total_bytes = cpu_tensor->TotalBytes();

    XlaTensor* xla_tensor = XlaTensor::FromTensor(device_tensor);
    CHECK(xla_tensor);

    Status status;
    xla::StatusOr<TensorShape> shape_or_status = shape_representation_fn_(
        device_tensor->shape(), device_tensor->dtype());
    if (!shape_or_status.ok()) {
      done(shape_or_status.status());
      return;
    }
    TensorShape shape = shape_or_status.ValueOrDie();
    if (!xla_tensor->has_shaped_buffer()) {
      status = xla_tensor->AllocateShapedBuffer(
          device_tensor->dtype(), shape, client_,
          stream_->parent()->device_ordinal());
      if (!status.ok()) {
        return done(status);
      }
    }

    if (transfer_as_literal_) {
      Tensor reshaped_cpu_tensor;
      if (!reshaped_cpu_tensor.CopyFrom(*cpu_tensor, shape)) {
        done(errors::Internal(
            "Tensor::CopyFrom failed when copying from CPU to XLA device"));
        return;
      }
      status = TransferLiteralToDevice(reshaped_cpu_tensor, device_tensor);
    } else {
      se::DeviceMemoryBase dev_dst_ptr =
          XlaTensor::DeviceMemoryFromTensor(*device_tensor);
      stream_->ThenMemcpy(&dev_dst_ptr, src_ptr, total_bytes);
      // TODO(hpucha): Make this asynchronous.
      Status block_status = stream_->BlockHostUntilDone();
      if (!block_status.ok()) {
        status = xla::InternalError(
            "Failed to complete data transfer on stream %p: %s", stream_,
            block_status.error_message().c_str());
      }
    }
    xla_tensor->set_host_tensor(*cpu_tensor);

    done(status);
    return;
  }

  VLOG(2) << "CopyCPUTensorToDevice empty tensor";
  done(Status::OK());
}

void XlaTransferManager::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                               StringPiece tensor_name,
                                               Device* device,
                                               Tensor* cpu_tensor,
                                               StatusCallback done) {
  if (device_tensor->NumElements() > 0) {
    VLOG(2) << "CopyDeviceTensorToCPU "
            << reinterpret_cast<const void*>(
                   device_tensor->tensor_data().data())
            << " "
            << reinterpret_cast<const void*>(cpu_tensor->tensor_data().data())
            << " " << device_tensor->NumElements() << " "
            << cpu_tensor->shape().DebugString() << " "
            << device_tensor->shape().DebugString();

    const int64 total_bytes = cpu_tensor->TotalBytes();
    se::DeviceMemoryBase dev_src_ptr =
        XlaTensor::DeviceMemoryFromTensor(*device_tensor);
    void* dst_ptr = DMAHelper::base(cpu_tensor);

    Status status;
    if (transfer_as_literal_) {
      TransferLiteralFromDevice(cpu_tensor, *device_tensor, done);
      return;
    } else {
      stream_->ThenMemcpy(dst_ptr, dev_src_ptr, total_bytes);
      // TODO(hpucha): Make this asynchronous.
      Status block_status = stream_->BlockHostUntilDone();
      if (!block_status.ok()) {
        status = xla::InternalError(
            "Failed to complete data transfer on stream %p: %s", stream_,
            block_status.error_message().c_str());
      }
      done(status);
    }
    return;
  }

  VLOG(2) << "CopyDeviceTensorToCPU empty tensor";
  done(Status::OK());
}

void XlaTransferManager::CopyDeviceTensorToDevice(const Tensor& src_tensor,
                                                  Tensor* dst_tensor,
                                                  const StatusCallback& done) {
  // Perform memory allocation now, and enqueue the device-to-device transfer.
  Status status = [&]() -> Status {
    if (src_tensor.NumElements() == 0) {
      return Status::OK();
    }
    XlaTensor* xla_src = XlaTensor::FromTensor(&src_tensor);
    XlaTensor* xla_dst = XlaTensor::FromTensor(dst_tensor);
    CHECK(xla_src && xla_dst)
        << "Missing destination tensor for device-to-device copy";
    if (!xla_dst->has_shaped_buffer()) {
      TF_ASSIGN_OR_RETURN(
          TensorShape shape,
          shape_representation_fn_(src_tensor.shape(), src_tensor.dtype()));
      TF_RETURN_IF_ERROR(
          xla_dst->AllocateShapedBuffer(src_tensor.dtype(), shape, client_,
                                        stream_->parent()->device_ordinal()));
    }
    auto from_iter = xla_src->shaped_buffer().buffers().begin();
    auto to_iter = xla_dst->shaped_buffer().buffers().begin();
    for (auto end_iter = xla_src->shaped_buffer().buffers().end();
         from_iter != end_iter; ++from_iter, ++to_iter) {
      stream_->ThenMemcpyD2D(&to_iter->second, from_iter->second,
                             to_iter->second.size());
    }
    return Status::OK();
  }();
  if (!status.ok()) {
    return done(status);
  } else {
    stream_->ThenDoHostCallback([=]() { done(Status::OK()); });
  }
}

XlaDeviceContext::XlaDeviceContext(
    se::Stream* stream, xla::LocalClient* client, bool transfer_as_literal,
    XlaCompiler::ShapeRepresentationFn shape_representation_fn)
    : manager_(stream, client, transfer_as_literal,
               std::move(shape_representation_fn)) {}

void XlaDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                             Device* device,
                                             Tensor* device_tensor,
                                             StatusCallback done) const {
  manager_.CopyCPUTensorToDevice(cpu_tensor, device, device_tensor, done);
}

void XlaDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                             StringPiece tensor_name,
                                             Device* device, Tensor* cpu_tensor,
                                             StatusCallback done) {
  manager_.CopyDeviceTensorToCPU(device_tensor, tensor_name, device, cpu_tensor,
                                 done);
}

void XlaDeviceContext::CopyDeviceTensorToDevice(const Tensor& src_tensor,
                                                Tensor* dst_tensor,
                                                const StatusCallback& done) {
  manager_.CopyDeviceTensorToDevice(src_tensor, dst_tensor, done);
}

}  // namespace tensorflow
