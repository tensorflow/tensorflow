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

#include "xla/pjrt/cpu/raw_buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/cpu/abstract_cpu_buffer.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/pjrt/cpu/tracked_cpu_device_buffer.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/profiler/lib/connected_traceme.h"

namespace xla {

void CpuTrackedDeviceEventPromise::Set(
    tsl::RCReference<PjRtDeviceEvent> event) {
  auto tpu_event =
      tensorflow::down_cast<CpuTrackedDeviceEvent*>(event.get())->event();
  av_->ForwardTo(std::move(tpu_event));
}

PjRtFuture<> CpuTrackedDeviceEvent::GetReadyFuture() {
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

void CpuTrackedDeviceEvent::AndThen(absl::AnyInvocable<void() &&> cb) {
  event_.AndThen(std::move(cb));
}

/*static*/ absl::StatusOr<tsl::RCReference<CpuRawBuffer>>
CpuRawBuffer::Allocate(PjRtMemorySpace* memory_space, size_t size_bytes) {
  TF_ASSIGN_OR_RETURN(auto memory, CpuDeviceMemory::Allocate(size_bytes));
  return tsl::MakeRef<CpuRawBuffer>(memory_space, std::move(memory));
}

size_t CpuRawBuffer::GetOnDeviceSizeInBytes() const {
  return buffer_->size_bytes();
}

void* CpuRawBuffer::GetHostPointer() const { return buffer_->untyped_data(); }

absl::Status CpuRawBuffer::ValidateSlice(int64_t offset, int64_t slice_size) {
  size_t buffer_size = GetOnDeviceSizeInBytes();
  if (offset < 0 || offset > buffer_size || buffer_size - offset < slice_size) {
    return InvalidArgument(
        "Invalid slicing of buffer size %lld with "
        "invalid offset %lld, slice size %lld",
        buffer_size, offset, slice_size);
  }
  return absl::OkStatus();
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
CpuRawBuffer::CopyRawHostToDeviceAndReturnEvent(const void* src, int64_t offset,
                                                int64_t transfer_size) {
  TF_RETURN_IF_ERROR(ValidateSlice(offset, transfer_size));
  std::memcpy(static_cast<uint8_t*>(GetHostPointer()) + offset, src,
              transfer_size);
  return tsl::MakeRef<CpuTrackedDeviceEvent>(
      tsl::MakeAvailableAsyncValueRef<CpuEvent>());
}

PjRtFuture<> CpuRawBuffer::CopyRawDeviceToHost(void* dst, int64_t offset,
                                               int64_t transfer_size) {
  auto s = ValidateSlice(offset, transfer_size);
  if (!s.ok()) {
    return PjRtFuture<>(s);
  }
  std::memcpy(dst, static_cast<uint8_t*>(GetHostPointer()) + offset,
              transfer_size);
  return PjRtFuture<>(absl::OkStatus());
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> CpuRawBuffer::CopyFromLiteral(
    const LiteralSlice& literal, const xla::Layout& layout,
    AsyncWorkRunner* async_work_runner) {
  const xla::Shape& shape = literal.shape();
  if ((shape.has_layout() &&
       !xla::LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) ||
      !xla::LayoutUtil::IsMonotonicWithDim0Major(layout)) {
    return absl::UnimplementedError(
        "PjRt CPU's CopyFromLiteral does not support "
        "non-major-to-minor layout");
  }
  auto event = tsl::MakeConstructedAsyncValueRef<CpuEvent>();
  async_work_runner->Schedule([literal, event, buffer = buffer_]() {
    CHECK(buffer.IsConcrete());
    PackOrCopy(literal.shape().element_type(), literal, buffer->untyped_data(),
               buffer->size_bytes());
    event.SetStateConcrete();
  });
  return tsl::MakeRef<CpuTrackedDeviceEvent>(std::move(event), "CpuRawBuffer",
                                             "CopyFromLiteral");
}

absl::StatusOr<xla::Shape> MakeDefaultCpuBufferShape(
    xla::Shape shape, const xla::Layout* layout) {
  xla::LayoutUtil::SetToDefaultLayout(&shape);
  auto element_type = shape.element_type();
  if (primitive_util::IsSubByteNonPredType(element_type)) {
    shape.mutable_layout()->set_element_size_in_bits(
        primitive_util::BitWidth(element_type));
  }
  if (layout) {
    auto layout_copy = *layout;
    if (primitive_util::IsSubByteNonPredType(element_type)) {
      layout_copy.set_element_size_in_bits(
          primitive_util::BitWidth(element_type));
    }
    if (layout_copy != shape.layout()) {
      return absl::UnimplementedError(
          absl::StrCat("PjRt CPU buffers do not support non-default layout. ",
                       shape.ToString(), " vs ", layout_copy.ToString()));
    }
  }
  return shape;
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
CpuRawBuffer::MakeAllocationReadyEvent() {
  return tsl::MakeRef<CpuTrackedDeviceEvent>(
      tsl::MakeAvailableAsyncValueRef<CpuEvent>());
}

}  // namespace xla
