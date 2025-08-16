/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/se_raw_buffer.h"

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/profiler/lib/connected_traceme.h"

namespace xla {

PjRtFuture<> PjRtStreamExecutorDeviceEvent::GetReadyFuture() {
  PjRtFuture<>::Promise promise = PjRtFuture<>::CreatePromise();
  event_.AndThen([promise, event = event_]() mutable {
    if (auto* error = event.GetErrorIfPresent()) {
      promise.Set(*error);
    } else {
      promise.Set();
    }
  });

  return PjRtFuture<>(
      promise,
      /*on_block_start=*/
      [ready_event = FormRef(promise.async_value()),
       callee_method = callee_method_, callee_type = callee_type_]() {
        tsl::profiler::TraceMeProducer traceme(
            [&] { return absl::StrCat(callee_type, "::", callee_method); });
        return PjRtFutureHelpers::ProfilingKeys({traceme.GetContextId()});
      },
      /*on_block_end=*/
      [callee_method = callee_method_,
       callee_type = callee_type_](PjRtFutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme(
            [&] { return absl::StrCat(callee_type, "::", callee_method); },
            keys.traceme_context_id);
      });
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
PjRtStreamExecutorRawBuffer::CopyRawHostToDeviceAndReturnEvent(
    const void* src, int64_t offset, int64_t transfer_size) {
  return client_->CopyRawHostToDevice(local_device_, device_buffer_, src,
                                      offset, transfer_size);
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
PjRtStreamExecutorRawBuffer::CopyRawDeviceToHostAndReturnEvent(
    void* dst, int64_t offset, int64_t transfer_size) {
  return client_->CopyRawDeviceToHost(local_device_, device_buffer_, dst,
                                      offset, transfer_size);
}

ShapedBuffer PjRtStreamExecutorRawBuffer::AsShapedBuffer(
    const xla::Shape& shape) {
  auto* device = memory_space()->devices()[0];
  ShapedBuffer shaped_buffer(shape, device->local_device_id().value(),
                             device->local_hardware_id().value());
  ShapeTree<se::DeviceMemoryBase>::iterator iterator =
      shaped_buffer.buffers().begin();
  if (device_buffer_) {
    CHECK(iterator != shaped_buffer.buffers().end());
    iterator->second = device_buffer_->mem();
    ++iterator;
  }
  CHECK(iterator == shaped_buffer.buffers().end());
  return shaped_buffer;
}

void PjRtStreamExecutorRawBuffer::ReadDynamicShape(
    tsl::AsyncValueRef<xla::Shape> output_shape, xla::Shape shape) {
  auto* stream = local_device_->GetDeviceToHostStream();
  auto shaped_buffer = AsShapedBuffer(shape);
  TransferManager* transfer_manager =
      client_->client()->backend().transfer_manager();
  auto status = transfer_manager->ReadDynamicShapes(stream, &shaped_buffer,
                                                    &*output_shape);
  if (!status.ok()) {
    output_shape.SetError(status);
  } else {
    output_shape.SetStateConcrete();
  }
}

void PjRtStreamExecutorRawBuffer::CopyToLiteralAsync(
    PjRtFuture<>::Promise promise,
    tsl::RCReference<PjRtDeviceEventPromise> device_promise,
    MutableLiteralBase* literal, xla::Shape shape) {
  device_promise->SetError(
      absl::UnimplementedError("Cannot CopyToLiteralAsync."));
  promise.Set(absl::UnimplementedError("Cannot CopyToLiteralAsync."));
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
PjRtStreamExecutorRawBuffer::MakeAllocationReadyEvent() {
  return absl::UnimplementedError("Cannot make ready event");
}

void PjRtStreamExecutorRawBuffer::CopyTo(
    tsl::RCReference<CommonPjRtRawBuffer> dst_raw_buffer,
    tsl::RCReference<PjRtDeviceEventPromise> definition_event_promise,
    tsl::RCReference<PjRtDeviceEventPromise> src_usage_event_promise,
    ::tsl::AsyncValueRef<bool> allocation_event) {
  auto status = absl::UnimplementedError("CopyTo not implemented");
  src_usage_event_promise->SetError(status);
  if (allocation_event) {
    allocation_event.SetError(status);
  }
  definition_event_promise->SetError(status);
}

std::optional<absl::StatusOr<tsl::RCReference<PjRtRawBuffer>>>
CreateGPURawBuffer(PjRtBuffer* buffer) {
  if (auto* se_buffer = dynamic_cast<PjRtStreamExecutorBuffer*>(buffer)) {
    auto* se_client = dynamic_cast<PjRtStreamExecutorClient*>(buffer->client());
    if (se_client == nullptr) {
      return absl::InvalidArgumentError("invalid se-client");
    }
    PjRtStreamExecutorBuffer::ScopedHold hold(
        se_buffer->GetBufferWithUsageHold());
    if (!hold.ok()) {
      return hold.status();
    }
    if (!hold->device_memory()) {
      return absl::InvalidArgumentError(
          "Create raw buffer called on an invalid buffer");
    }
    return tsl::MakeRef<PjRtStreamExecutorRawBuffer>(
        se_client, se_buffer->memory_space(),
        se_buffer->device()->local_device_state(), hold->device_memory());
  }
  return std::nullopt;
}

REGISTER_PJRT_RAW_BUFFER_FACTORY(CreateGPURawBuffer);

}  // namespace xla
