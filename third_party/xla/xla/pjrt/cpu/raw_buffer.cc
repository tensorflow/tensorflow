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
#include <memory>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/cpu_function_runtime.h"
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
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

constexpr size_t kSmallDataTransferByteSize = 102400;  // 100 KiB

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

/*static*/ absl::StatusOr<tsl::RCReference<CpuRawBuffer>>
CpuRawBuffer::ImportForeignMemory(
    void* data, absl::AnyInvocable<void() &&> on_delete_callback,
    size_t on_device_bytes_count, PjRtMemorySpace* memory_space) {
  if ((absl::bit_cast<std::uintptr_t>(data) &
       (cpu_function_runtime::MinAlign() - 1)) != 0) {
    return InvalidArgument(
        "Can't create a view of buffer with unaligned data, ptr: %#x is not "
        "aligned to %d bytes. ",
        reinterpret_cast<std::uintptr_t>(data),
        cpu_function_runtime::MinAlign());
  }
  return tsl::MakeRef<CpuRawBuffer>(
      memory_space,
      CpuDeviceMemory::CreateForeignMemory(data, on_device_bytes_count,
                                           std::move(on_delete_callback)));
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

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
CpuRawBuffer::CopyRawDeviceToHostAndReturnEvent(void* dst, int64_t offset,
                                                int64_t transfer_size) {
  TF_RETURN_IF_ERROR(ValidateSlice(offset, transfer_size));
  std::memcpy(dst, static_cast<uint8_t*>(GetHostPointer()) + offset,
              transfer_size);
  return tsl::MakeRef<CpuTrackedDeviceEvent>(
      tsl::MakeAvailableAsyncValueRef<CpuEvent>());
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> CpuRawBuffer::CopyFromLiteral(
    const LiteralSlice& literal, const xla::Layout& layout,
    AsyncWorkRunner* async_work_runner) {
  auto event = tsl::MakeConstructedAsyncValueRef<CpuEvent>();
  async_work_runner->Schedule([literal, layout, event, buffer = buffer_]() {
    CHECK(buffer.IsConcrete());
    const xla::Shape& shape = literal.shape();
    if ((!shape.has_layout() &&
         !xla::LayoutUtil::IsMonotonicWithDim0Major(layout)) ||
        (shape.layout() != layout)) {
      auto shape_copy = xla::ShapeUtil::MakeShape(
          literal.shape().element_type(), literal.shape().dimensions());
      shape_copy.mutable_layout()->mutable_minor_to_major()->assign(
          layout.minor_to_major().begin(), layout.minor_to_major().end());

      xla::Literal literal_copy(shape_copy);
      CHECK_OK(literal_copy.CopyFrom(literal));
      PackOrCopy(literal_copy.shape().element_type(), literal_copy,
                 buffer->untyped_data(), buffer->size_bytes());
    } else {
      PackOrCopy(literal.shape().element_type(), literal,
                 buffer->untyped_data(), buffer->size_bytes());
    }
    event.SetStateConcrete();
  });
  return tsl::MakeRef<CpuTrackedDeviceEvent>(std::move(event), "CpuRawBuffer",
                                             "CopyFromLiteral");
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
CpuRawBuffer::CopyFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    PjRtClient::HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer, const Shape& shape,
    AsyncWorkRunner* async_work_runner, absl::Mutex* transpose_mu,
    TransposePlanCache* transpose_cache) {
  tsl::AsyncValueRef<CpuDeviceMemory> device_buffer = buffer_;
  bool has_default_layout =
      !byte_strides || HasMajorToMinorLayout(type, dims, *byte_strides);
  const int bit_width = primitive_util::BitWidth(type);
  // Packed arrays are unpacked on host and packed on device.
  bool is_packed = primitive_util::IsSubByteNonPredType(type);

  size_t byte_size = ShapeUtil::ByteSizeOf(shape);
  size_t dst_byte_size = byte_size;
  if (is_packed) {
    byte_size *= 8 / bit_width;
  }
  auto dst_data_ptr = device_buffer->untyped_data();
  if (!has_default_layout || is_packed) {
    // If the input array does not have a major-to-minor layout, transpose it
    // into major-to-minor layout. Currently we choose to always do this
    // synchronously.
    // TODO(phawkins): consider performing the transpose asynchronously.
    // TODO(phawkins): parallelize the transpose.
    std::shared_ptr<TransposePlan> transpose;
    {
      absl::InlinedVector<int64_t, 4> permutation(dims.size());
      absl::c_iota(permutation, 0);
      TransposePlan::Options options;
      options.elem_size_in_bytes = primitive_util::ByteWidth(type);
      options.dims = dims;
      options.permutation = permutation;
      if (byte_strides) {
        options.input_layout = TransposePlan::Striding{*byte_strides};
      }
      absl::MutexLock lock(transpose_mu);
      TF_ASSIGN_OR_RETURN(transpose, transpose_cache->GetOrCreate(options));
    }
    if (!is_packed) {
      transpose->Execute(data, dst_data_ptr);
    } else {
      // First transpose the unpacked data into a new temporary buffer, then
      // pack the data.
      // TODO(reedwm): Fuse the transpose and packing by having TransposePlan
      // support packing.
      auto data_transposed = std::make_unique<char[]>(byte_size);
      transpose->Execute(data, data_transposed.get());
      absl::Span<const char> src_data_span(data_transposed.get(), byte_size);
      absl::Span<char> dst_data_span(static_cast<char*>(dst_data_ptr),
                                     dst_byte_size);
      PackIntN(bit_width, src_data_span, dst_data_span);
    }
    if (on_done_with_host_buffer) {
      std::move(on_done_with_host_buffer)();
      on_done_with_host_buffer = nullptr;
    }
  } else {
    bool should_sync_copy =
        host_buffer_semantics ==
            PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall ||
        (byte_size < kSmallDataTransferByteSize);
    if (should_sync_copy) {
      std::memcpy(dst_data_ptr, data, byte_size);
      if (on_done_with_host_buffer) {
        std::move(on_done_with_host_buffer)();
        on_done_with_host_buffer = nullptr;
      }
    } else {
      tsl::AsyncValueRef<CpuEvent> copy_event =
          tsl::MakeConstructedAsyncValueRef<CpuEvent>();
      auto result = tsl::MakeRef<CpuTrackedDeviceEvent>(
          copy_event.CopyRef(), "CpuRawBuffer", "CopyFromHostBuffer");
      async_work_runner->Schedule([device_buffer, dst_data_ptr, data, byte_size,
                                   copy_event = std::move(copy_event),
                                   on_done_with_host_buffer = std::move(
                                       on_done_with_host_buffer)]() mutable {
        tsl::profiler::TraceMe traceme("H2D Dispatch");
        std::memcpy(dst_data_ptr, data, byte_size);
        if (on_done_with_host_buffer) {
          std::move(on_done_with_host_buffer)();
          on_done_with_host_buffer = nullptr;
        }
        // Signal copy is complete.
        copy_event.SetStateConcrete();
      });
      return result;
    }
  }
  return tsl::MakeRef<CpuTrackedDeviceEvent>(
      tsl::MakeAvailableAsyncValueRef<CpuEvent>());
}

absl::StatusOr<xla::Shape> MakeDefaultCpuBufferShape(
    xla::Shape shape, const xla::Layout* layout) {
  if (layout) {
    shape.mutable_layout()->mutable_minor_to_major()->assign(
        layout->minor_to_major().begin(), layout->minor_to_major().end());
  } else {
    xla::LayoutUtil::SetToDefaultLayout(&shape);
  }
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

void CpuRawBuffer::ReadDynamicShape(tsl::AsyncValueRef<xla::Shape> output_shape,
                                    xla::Shape shape) {
  size_t offset = xla::ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  // Each dynamic dimension size is represented as a S32.
  int64_t metadata_size = sizeof(int32_t) * shape.dimensions().size();
  auto metadata_buffer = reinterpret_cast<const int32_t*>(
      reinterpret_cast<const uint8_t*>(buffer_->untyped_data()) + offset);
  if (buffer_->size_bytes() != metadata_size + offset) {
    output_shape.SetError(absl::InvalidArgumentError(absl::StrFormat(
        "Raw buffer size (%d) incompatible with original shape (%s)",
        buffer_->size_bytes(), shape.ToString(true))));
    return;
  }
  output_shape->clear_dynamic_dimensions();
  for (size_t i = 0; i < shape.dimensions().size(); ++i) {
    output_shape->set_dimensions(i, metadata_buffer[i]);
  }
  if (!ShapeUtil::DynamicShapeIsCompatible(output_shape.get(), shape)) {
    output_shape.SetError(absl::InvalidArgumentError(absl::StrFormat(
        "Output dynamic shape (%s) incompatible with original shape (%s)",
        output_shape->ToString(true), shape.ToString(true))));
    return;
  }
  output_shape.SetStateConcrete();
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
CpuRawBuffer::MakeAllocationReadyEvent() {
  return tsl::MakeRef<CpuTrackedDeviceEvent>(
      tsl::MakeAvailableAsyncValueRef<CpuEvent>());
}

void CpuRawBuffer::CopyToLiteralAsync(
    PjRtFuture<>::Promise promise,
    tsl::RCReference<PjRtDeviceEventPromise> device_promise,
    MutableLiteralBase* literal, xla::Shape shape) {
  absl::Span<const char> input_span{
      static_cast<const char*>(buffer_->untyped_data()), buffer_->size_bytes()};
  size_t output_size =
      static_cast<size_t>(ShapeUtil::ByteSizeOf(literal->shape()));
  absl::Span<char> output_span{static_cast<char*>(literal->untyped_data()),
                               output_size};
  if (primitive_util::IsSubByteNonPredType(shape.element_type())) {
    primitive_util::UnpackIntN(shape.element_type(), input_span, output_span);
  } else {
    std::memcpy(output_span.data(), input_span.data(), output_size);
  }
  device_promise->Set(tsl::MakeRef<CpuTrackedDeviceEvent>(
      tsl::MakeAvailableAsyncValueRef<CpuEvent>()));
  promise.Set(absl::OkStatus());
}

void CpuRawBuffer::CopyTo(
    tsl::RCReference<CommonPjRtRawBuffer> dst_raw_buffer,
    tsl::RCReference<PjRtDeviceEventPromise> definition_event_promise,
    tsl::RCReference<PjRtDeviceEventPromise> src_usage_event_promise,
    ::tsl::AsyncValueRef<bool> allocation_event) {
  if (allocation_event) {
    allocation_event.SetStateConcrete();
  }
  auto other_event = dst_raw_buffer->CopyRawHostToDeviceAndReturnEvent(
      GetHostPointer(), 0, GetOnDeviceSizeInBytes());
  if (!other_event.ok()) {
    definition_event_promise->SetError(other_event.status());
    src_usage_event_promise->SetError(other_event.status());
    return;
  }
  (*other_event)
      ->AndThen([src_usage_event_promise = std::move(src_usage_event_promise),
                 src_buffer = tsl::FormRef(this)]() {
        src_usage_event_promise->Set(tsl::MakeRef<CpuTrackedDeviceEvent>(
            tsl::MakeAvailableAsyncValueRef<CpuEvent>()));
      });
  definition_event_promise->Set(*std::move(other_event));
}

}  // namespace xla
