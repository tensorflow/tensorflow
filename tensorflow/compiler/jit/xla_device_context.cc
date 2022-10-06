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

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/stream_executor/platform/port.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {

// The allocator used for Tensors assigned to the XLA device.
XlaDeviceAllocator::XlaDeviceAllocator(
    stream_executor::StreamExecutor* stream_executor)
    : stream_executor_(stream_executor) {}

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

std::optional<AllocatorStats> XlaDeviceAllocator::GetStats() {
  std::optional<stream_executor::AllocatorStats> se_stats =
      stream_executor_->GetAllocatorStats();
  if (!se_stats) {
    return std::nullopt;
  }

  tensorflow::AllocatorStats tf_stats;
  tf_stats.num_allocs = se_stats->num_allocs;
  tf_stats.bytes_in_use = se_stats->bytes_in_use;
  tf_stats.peak_bytes_in_use = se_stats->peak_bytes_in_use;
  tf_stats.largest_alloc_size = se_stats->largest_alloc_size;
  tf_stats.bytes_limit = se_stats->bytes_limit;
  tf_stats.bytes_reserved = se_stats->bytes_reserved;
  tf_stats.peak_bytes_reserved = se_stats->peak_bytes_reserved;
  tf_stats.bytes_reservable_limit = se_stats->bytes_reservable_limit;
  tf_stats.largest_free_block_bytes = se_stats->largest_free_block_bytes;
  return tf_stats;
}

bool XlaDeviceAllocator::ClearStats() {
  if (!stream_executor_->SynchronizeAllActivity()) {
    return false;
  }
  return stream_executor_->ClearAllocatorStats();
}

XlaDeviceContext::XlaDeviceContext(
    std::shared_ptr<se::Stream> compute_stream,
    std::shared_ptr<se::Stream> host_to_device_stream,
    std::shared_ptr<se::Stream> device_to_host_stream,
    std::vector<std::shared_ptr<se::Stream>> device_to_device_streams,
    xla::LocalClient* client,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    thread::ThreadPool* thread_pool)
    : stream_(std::move(compute_stream)),
      host_to_device_stream_(std::move(host_to_device_stream)),
      device_to_host_stream_(std::move(device_to_host_stream)),
      device_to_device_streams_(std::move(device_to_device_streams)),
      client_(client),
      transfer_manager_(client->backend().transfer_manager()),
      shape_determination_fns_(std::move(shape_determination_fns)),
      thread_pool_(thread_pool) {
  CHECK(host_to_device_stream_ != nullptr);
  CHECK(stream_ != nullptr);
}

void XlaDeviceContext::CopyTensorInSameDevice(const Tensor* input_tensor,
                                              Device* device,
                                              Tensor* output_tensor,
                                              StatusCallback done) const {
  done(errors::Unimplemented("XLA->XLA same-device copies not implemented."));
}

void XlaDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                             Device* device,
                                             Tensor* device_tensor,
                                             StatusCallback done,
                                             bool sync_dst_compute) const {
  if (cpu_tensor->NumElements() == 0) {
    VLOG(2) << "CopyCPUTensorToDevice empty tensor";
    done(OkStatus());
    return;
  }

  VLOG(2) << "CopyCPUTensorToDevice " << this << " "
          << reinterpret_cast<const void*>(cpu_tensor->tensor_data().data())
          << " "
          << reinterpret_cast<const void*>(device_tensor->tensor_data().data())
          << " " << cpu_tensor->NumElements() << " "
          << cpu_tensor->shape().DebugString() << " "
          << device_tensor->shape().DebugString();

  XlaTensor* xla_tensor = XlaTensor::FromTensor(device_tensor);
  CHECK(xla_tensor);

  XlaLayoutPreference layout_preference =
      shape_determination_fns_.layout_preference_fn(
          device_tensor->shape(), device_tensor->dtype(), std::nullopt);
  Status status = [&]() -> Status {
    TF_ASSIGN_OR_RETURN(xla::Shape shape,
                        shape_determination_fns_.shape_representation_fn(
                            device_tensor->shape(), device_tensor->dtype(),
                            /*fast_mem=*/false, layout_preference));

    // The device tensor should always be fresh.
    TF_RET_CHECK(!xla_tensor->has_shaped_buffer());

    TF_RETURN_IF_ERROR(
        xla_tensor->AllocateShapedBuffer(device_tensor->dtype(), shape, client_,
                                         stream_->parent()->device_ordinal()));

    // The cpu_tensor and literal that we created here hold the data of host
    // tensor in descending layout. The layout could be different from layout in
    // device_tensor (but the logical shape has to be the same). The
    // transfer_manager is responsible to do corresponding transposing when
    // transferring the data to device.
    xla::BorrowingLiteral literal(
        static_cast<const char*>(DMAHelper::base(cpu_tensor)),
        xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions()));

    VLOG(2) << "Transfer to device as literal: " << literal.ToString() << " "
            << xla_tensor->shaped_buffer().ToString();
    if (UseMultipleStreams() &&
        !transfer_manager_->CanShapedBufferBeAccessedNow(
            stream_->parent(), xla_tensor->shaped_buffer())) {
      // Initially wait for the compute stream so that memory allocations are
      // synchronized.
      host_to_device_stream_->ThenWaitFor(stream_.get());
    }

    TF_RETURN_IF_ERROR(transfer_manager_->TransferLiteralToDeviceAsync(
        host_to_device_stream_.get(), literal, xla_tensor->shaped_buffer()));

    if (UseMultipleStreams()) {
      auto event = std::make_shared<se::Event>(stream_->parent());
      TF_RET_CHECK(event->Init()) << "Event failed to initialize!";
      host_to_device_stream_->ThenRecordEvent(event.get());
      xla_tensor->ResetDefinitionEvent(std::move(event),
                                       host_to_device_stream_.get());
    }

    return OkStatus();
  }();
  if (!status.ok()) {
    done(status);
    return;
  }

  // Create a reference to hold onto cpu_tensor until after the literal has
  // been transferred
  TensorReference ref(*cpu_tensor);
  if (UseMultipleStreams()) {
    // Unref the host tensor when the transfer completes.
    // We don't defer the call to done() onto the stream here, and the reasons
    // why this is correct are subtle. We assume that:
    // a) all consumers of the device tensor will wait for its definition event.
    // b) if the tensor is destroyed, then the memory allocator will not hand
    //    out the same buffers until the transfer has completed.
    host_to_device_stream_->ThenDoHostCallback([ref]() { ref.Unref(); });
    done(status);
  } else {
    host_to_device_stream_->ThenDoHostCallback([ref, done]() {
      ref.Unref();
      done(OkStatus());
    });
  }
}

void XlaDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                             absl::string_view tensor_name,
                                             Device* device, Tensor* cpu_tensor,
                                             StatusCallback done) {
  if (device_tensor->NumElements() == 0) {
    VLOG(2) << "CopyDeviceTensorToCPU empty tensor";
    done(OkStatus());
    return;
  }
  VLOG(2) << "CopyDeviceTensorToCPU "
          << reinterpret_cast<const void*>(device_tensor->tensor_data().data())
          << " "
          << reinterpret_cast<const void*>(cpu_tensor->tensor_data().data())
          << " " << device_tensor->NumElements() << " "
          << cpu_tensor->shape().DebugString() << " "
          << device_tensor->shape().DebugString();

  std::shared_ptr<se::Stream> device_to_host_stream;
  if (device_to_host_stream_) {
    device_to_host_stream = device_to_host_stream_;
  } else {
    stream_executor::port::StatusOr<xla::StreamPool::Ptr> ptr_or_status =
        client_->mutable_backend()->BorrowStream(
            stream_->parent()->device_ordinal());
    if (!ptr_or_status.status().ok()) {
      done(ptr_or_status.status());
      return;
    }
    device_to_host_stream =
        std::shared_ptr<se::Stream>(std::move(ptr_or_status.value()));
  }

  XlaTensor* xla_tensor = XlaTensor::FromTensor(device_tensor);
  xla_tensor->WaitForDefinitionEventOnStream(device_to_host_stream.get());

  // Transfer manager requires the shape of the shaped buffer to be the same as
  // literal shape except for the layout.  Set the literal to use xla_tensor's
  // shape as it is derived from the cpu_tensor's shape using
  // shape_representation_fn_.
  xla::MutableBorrowingLiteral literal;
  TF_CHECK_OK(HostTensorToMutableBorrowingLiteral(
      xla::LayoutUtil::GetWithDefaultLayout(
          xla_tensor->shaped_buffer().on_host_shape()),
      cpu_tensor, &literal));

  TensorReference ref(*device_tensor);
  const bool device_allows_sync_on_completion =
      device->AllowsSyncOnCompletion();
  // Explicitly capture device_to_host_stream to make sure the stream is alive
  // before the transfer finishes.
  transfer_manager_->TransferLiteralFromDevice(
      device_to_host_stream.get(), xla_tensor->shaped_buffer(), literal,
      [this, ref, xla_tensor, done, device_to_host_stream,
       device_allows_sync_on_completion](xla::Status status) {
        Status done_status = status;
        VLOG(2) << "Transfer from device as literal: "
                << xla_tensor->shaped_buffer().ToString();
        // For devices don't allow sync on completion, the device execution is
        // deferred. We check the execution stream status here to avoid wrong
        // results from a failed stream being propagated to following
        // host-side ops.
        if (!device_allows_sync_on_completion) {
          done_status.Update(xla_tensor->RefreshStatusOfStreams());
        }
        done(done_status);
        ref.Unref();
        // If a stream is in a bad state, it gets deleted when it's returned to
        // the stream pool, i.e. when it leaves this scope. However, a stream
        // deleting itself in a host callback on itself can cause bad behaviors
        // on some platforms. Releasing it in another stream to avoid that.
        if (!device_allows_sync_on_completion &&
            !device_to_host_stream->RefreshStatus().ok()) {
          auto status_or_new_stream = client_->mutable_backend()->BorrowStream(
              stream_->parent()->device_ordinal());
          if (status_or_new_stream.ok()) {
            status_or_new_stream.value()->ThenDoHostCallback(
                [device_to_host_stream] {});
          }
        }
      });
}

se::Stream* XlaDeviceContext::GetDeviceToDeviceStream() {
  DCHECK_GT(device_to_device_streams_.size(), 0);
  absl::MutexLock lock(&mu_);
  int stream = next_stream_;
  next_stream_ = (next_stream_ + 1) % device_to_device_streams_.size();
  return device_to_device_stream(stream);
}

Status XlaDeviceContext::ThenExecute(Device* device,
                                     stream_executor::Stream* stream,
                                     std::function<void()> func) {
  VLOG(2) << "XlaDeviceContext::ThenExecute";
  stream->ThenDoHostCallback(std::move(func));
  return OkStatus();
}

}  // namespace tensorflow
