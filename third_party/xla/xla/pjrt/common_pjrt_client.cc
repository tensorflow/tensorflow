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

#include "xla/pjrt/common_pjrt_client.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/casts.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/Support/MathExtras.h"
#include "xla/error/error_codes.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/c/pjrt_c_api_device_event.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/device_event_utils.h"
#include "xla/pjrt/dynamic_shapes.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/staging_buffer.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/primitive_util.h"
#include "xla/runtime/device_id.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/context.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/lib/scoped_memory_debug_annotation.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

void CommonPjRtClient::TrackFuture(PjRtMemorySpace* memory_space,
                                   absl::string_view debug_info,
                                   const Future<>& future) {}

absl::Status CommonPjRtClient::WaitOnStream(PjRtMemorySpace* memory_space,
                                            PjRtDeviceEventRef event,
                                            std::intptr_t stream) {
  return absl::UnimplementedError(
      "WaitUntilBufferReadyOnStream is only implemented for GPU.");
}

tsl::AsyncValueRef<PjRtStagingBuffer>
CommonPjRtClient::AllocateForDelinearizationAsync(
    size_t size, PjRtMemorySpace* memory_space) {
  return tsl::MakeErrorAsyncValueRef(absl::UnimplementedError(
      "AllocateForDelinearizationAsync is not supported"));
}

void CommonPjRtClient::DelinearizeAsync(
    tsl::AsyncValueRef<PjRtStagingBuffer> staging_buffer,
    PjRtMemorySpace* memory_space, const Shape& shape,
    MutableLiteralBase* literal, tsl::Promise<void> promise) {
  tsl::Context context(tsl::ContextKind::kThread);
  staging_buffer.AndThen([this, staging_buffer, shape, literal,
                          context = std::move(context),
                          promise = std::move(promise)]() mutable {
    if (auto* error = staging_buffer.GetErrorIfPresent()) {
      promise.Set(*error);
      return;
    }
    auto run_delinearize = [this, staging_buffer, shape, literal,
                            context = std::move(context),
                            promise = std::move(promise)]() mutable {
      tsl::WithContext wc(context);
      absl::Span<const uint8_t> input_data = staging_buffer->const_data();
      absl::Status status = DelinearizeHostBuffer(input_data, shape, literal);
      staging_buffer.reset();
      promise.Set(status);
    };
    if (async_work_runner() != nullptr) {
      async_work_runner()->Execute(std::move(run_delinearize));
    } else {
      run_delinearize();
    }
  });
}

tsl::AsyncValueRef<PjRtStagingBuffer>
CommonPjRtClient::CreateStagingForZeroCopyLinearize(
    const void* data, const xla::Shape& device_shape,
    PjRtMemorySpace* memory_space,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer) {
  auto size_or = GetOnDeviceBytesCount(memory_space->kind_id(), device_shape);
  if (!size_or.ok()) {
    return tsl::MakeErrorAsyncValueRef(size_or.status());
  }
  absl::Span<uint8_t> span(
      const_cast<uint8_t*>(static_cast<const uint8_t*>(data)), *size_or);
  return PjRtStagingBuffer::Create(span, std::move(on_done_with_host_buffer));
}

absl::StatusOr<tsl::AsyncValueRef<PjRtStagingBuffer>>
CommonPjRtClient::AllocateLinearizeDest(bool sync,
                                        const xla::Shape& device_shape,
                                        absl::Span<const int64_t> byte_strides,
                                        PjRtRawBufferRef dest_buffer) {
  PjRtMemorySpace* memory_space = dest_buffer->memory_space();
  ASSIGN_OR_RETURN(size_t size, GetOnDeviceBytesCount(memory_space->kind_id(),
                                                      device_shape));
  if (dest_buffer->GetHostPointer() != nullptr) {
    absl::Span<uint8_t> span(
        static_cast<uint8_t*>(dest_buffer->GetHostPointer()), size);
    return PjRtStagingBuffer::Create(
        span, [dest_buffer = std::move(dest_buffer)]() {});
  }
  auto vec = std::make_unique<std::vector<uint8_t>>(size);
  absl::Span<uint8_t> span = absl::MakeSpan(*vec);
  return PjRtStagingBuffer::Create(span, [vec = std::move(vec)]() {});
}

absl::Status CommonPjRtClient::Linearize(
    absl::Span<uint8_t> dest, const void* data, PrimitiveType type,
    absl::Span<const int64_t> dims, absl::Span<const int64_t> byte_strides,
    const Layout& device_layout, absl::Span<const uint32_t> dynamic_sizes) {
  return absl::UnimplementedError("Linearize not supported");
}

absl::StatusOr<PjRtDeviceEventRef> CommonPjRtClient::LinearizeIntoImpl(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    const xla::Shape& device_shape, absl::Span<const uint32_t> dynamic_sizes,
    PjRtRawBufferRef raw_buffer) {
  auto* memory_space = raw_buffer->memory_space();
  tsl::profiler::TraceMeProducer producer("CommonPjRtClient::LinearizeIntoImpl",
                                          tsl::profiler::ContextType::kPjRt);

  tsl::AsyncValueRef<PjRtStagingBuffer> linearized;
  if (raw_buffer->GetHostPointer() == nullptr &&
      host_buffer_semantics != HostBufferSemantics::kImmutableOnlyDuringCall &&
      dynamic_sizes.empty() &&
      ShouldPerformZeroCopyLinearize(data, device_shape, type, dims,
                                     byte_strides)) {
    linearized = CreateStagingForZeroCopyLinearize(
        data, device_shape, memory_space, std::move(on_done_with_host_buffer));
  } else {
    ASSIGN_OR_RETURN(
        linearized,
        AllocateLinearizeDest(
            /*sync=*/host_buffer_semantics ==
                HostBufferSemantics::kImmutableOnlyDuringCall,
            device_shape, byte_strides.value_or(absl::Span<const int64_t>()),
            raw_buffer));
    if (host_buffer_semantics ==
        HostBufferSemantics::kImmutableOnlyDuringCall) {
      if (!linearized.IsAvailable()) {
        tsl::BlockUntilReady(linearized.GetAsyncValue());
      }
      if (auto* error = linearized.GetAsyncValue()->GetErrorIfPresent()) {
        return *error;
      }
      RETURN_IF_ERROR(
          Linearize(linearized->data(), data, type, dims,
                    byte_strides.value_or(absl::Span<const int64_t>()),
                    device_shape.layout(), dynamic_sizes));
      if (on_done_with_host_buffer) {
        std::move(on_done_with_host_buffer)();
        on_done_with_host_buffer = nullptr;
      }
    } else {
      PjRtDeviceEventPromiseRef copy_event_promise;
      PjRtDeviceEventRef copy_event;
      ASSIGN_OR_RETURN(
          std::tie(copy_event_promise, copy_event),
          CreateLinkedEventPromise(memory_space, "BufferFromHostBuffer"));

      async_work_runner()->ExecuteWhenReady(
          {linearized.CopyRCRef()},
          [this, linearized = std::move(linearized), copy_event_promise,
           raw_buffer = std::move(raw_buffer), data, type,
           dims = absl::InlinedVector<int64_t, 4>(dims.begin(), dims.end()),
           byte_strides = byte_strides.has_value()
                              ? absl::InlinedVector<int64_t, 4>(
                                    byte_strides->begin(), byte_strides->end())
                              : absl::InlinedVector<int64_t, 4>(),
           dynamic_sizes = absl::InlinedVector<uint32_t, 4>(
               dynamic_sizes.begin(), dynamic_sizes.end()),
           device_layout = device_shape.layout(),
           context_id{producer.GetContextId()},
           on_done_with_host_buffer =
               std::move(on_done_with_host_buffer)]() mutable {
            tsl::profiler::TraceMeConsumer consumer(
                "H2D Dispatch", tsl::profiler::ContextType::kPjRt, context_id);
            absl::Span<uint8_t> staging_data = linearized->data();
            auto status = [&]() -> absl::Status {
              if (auto* error =
                      linearized.GetAsyncValue()->GetErrorIfPresent()) {
                return *error;
              }
              return Linearize(staging_data, data, type, dims, byte_strides,
                               device_layout, dynamic_sizes);
            }();
            if (on_done_with_host_buffer) {
              std::move(on_done_with_host_buffer)();
              on_done_with_host_buffer = nullptr;
            }
            if (!status.ok()) {
              copy_event_promise.SetError(std::move(status));
              return;
            }
            PjRtDeviceEventRef copy_event;
            if (raw_buffer->GetHostPointer() == staging_data.data()) {
              linearized.reset();
              copy_event_promise.SetReady();
              return;
            }
            auto event = raw_buffer->CopyRawHostToDeviceAndReturnEvent(
                staging_data.data(), 0, staging_data.size());
            if (!event.ok()) {
              copy_event_promise.SetError(event.status());
              return;
            }
            event->ptr().DeleteWhenReady(
                tsl::RCReference<tsl::AsyncValue>(std::move(linearized)));
            copy_event_promise.Set(event.value());
          });
      return copy_event;
    }
  }

  if (!linearized.IsAvailable()) {
    tsl::BlockUntilReady(linearized.GetAsyncValue());
  }
  if (auto* error = linearized.GetAsyncValue()->GetErrorIfPresent()) {
    return *error;
  }
  auto staging_data = linearized->const_data();
  if (raw_buffer->GetHostPointer() == staging_data.data()) {
    PjRtDeviceEventPromiseRef copy_event_promise;
    PjRtDeviceEventRef copy_event;
    ASSIGN_OR_RETURN(
        std::tie(copy_event_promise, copy_event),
        CreateLinkedEventPromise(memory_space, "BufferFromHostBuffer"));
    copy_event_promise.SetReady();
    return copy_event;
  }
  auto event = raw_buffer->CopyRawHostToDeviceAndReturnEvent(
      staging_data.data(), 0, staging_data.size());
  if (!event.ok()) {
    PjRtDeviceEventPromiseRef copy_event_promise;
    PjRtDeviceEventRef copy_event;
    ASSIGN_OR_RETURN(
        std::tie(copy_event_promise, copy_event),
        CreateLinkedEventPromise(memory_space, "BufferFromHostBuffer"));
    copy_event_promise.SetError(event.status());
    return copy_event;
  }
  event->ptr().DeleteWhenReady(
      tsl::RCReference<tsl::AsyncValue>(std::move(linearized)));
  return event.value();
}

absl::Status CommonPjRtClient::DelinearizeHostBuffer(
    absl::Span<const uint8_t> input_data, const Shape& shape,
    MutableLiteralBase* literal) {
  xla::Layout literal_layout;
  bool need_transpose = false;
  if (shape.IsArray()) {
    if (literal->shape().has_layout()) {
      literal_layout = literal->shape().layout();
    } else {
      literal_layout =
          LayoutUtil::MakeDescendingLayout(shape.dimensions().size());
    }
    need_transpose = literal_layout != shape.layout();
  }

  absl::Span<const char> input_span{
      reinterpret_cast<const char*>(input_data.data()), input_data.size()};
  size_t output_size =
      static_cast<size_t>(ShapeUtil::ByteSizeOf(literal->shape()));
  absl::Span<char> output_span{static_cast<char*>(literal->untyped_data()),
                               output_size};

  if (need_transpose) {
    std::vector<char> staged_buffer;
    if (primitive_util::IsSubByteNonPredType(shape.element_type())) {
      staged_buffer.resize(output_size);
      primitive_util::UnpackIntN(shape.element_type(), input_span,
                                 absl::MakeSpan(staged_buffer));
      input_span = absl::MakeConstSpan(staged_buffer);
    }

    absl::InlinedVector<int64_t, 4> byte_strides(shape.dimensions().size());
    RETURN_IF_ERROR(
        ShapeUtil::UnpackedByteStrides(shape, absl::MakeSpan(byte_strides)));
    absl::Span<const int64_t> dims = shape.dimensions();
    absl::InlinedVector<int64_t, 4> permutation(dims.size());
    absl::c_reverse_copy(literal_layout.minor_to_major(), permutation.begin());
    TransposePlan::Options options;
    options.elem_size_in_bytes =
        primitive_util::ByteWidth(shape.element_type());
    options.dims = dims;
    options.permutation = permutation;
    options.input_striding = TransposePlan::Striding{byte_strides};
    ASSIGN_OR_RETURN(std::shared_ptr<TransposePlan> plan,
                     GetTransposePlan(options));
    plan->Execute(input_span.data(), output_span.data());
  } else {
    if (primitive_util::IsSubByteNonPredType(shape.element_type())) {
      primitive_util::UnpackIntN(shape.element_type(), input_span, output_span);
    } else {
      std::memcpy(output_span.data(), input_span.data(), output_size);
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> CommonPjRtClient::DefineBuffer(
    std::shared_ptr<const Shape> on_device_shape, PjRtMemorySpace* memory_space,
    PjRtRawBufferRef raw_buffer,
    absl::InlinedVector<PjRtDeviceEventRef, 2> definition_device_events) {
  if (on_device_shape->IsTuple()) {
    return absl::InvalidArgumentError(
        "DefineBuffer: Can't define a tuple buffer.");
  }
  if (raw_buffer && raw_buffer->memory_space() != memory_space) {
    return absl::InvalidArgumentError(
        absl::StrFormat("DefineBuffer: Mismatch in memory spaces: %s vs %s",
                        raw_buffer->memory_space()->DebugString(),
                        memory_space->DebugString()));
  }
  return std::make_unique<CommonPjRtBufferImpl>(
      std::move(on_device_shape),
      std::make_unique<AbstractTrackedDeviceBuffer>(
          std::move(raw_buffer), std::move(definition_device_events),
          use_stream_based_compaction()),
      memory_space);
}

Future<> CommonPjRtClient::CreateProfiledFuture(PjRtMemorySpace* memory_space,
                                                const char* callee_type,
                                                const char* callee_method,
                                                Future<> future) {
  return FutureHelpers::WithProfiling(
      std::move(future),
      /*on_block_start=*/
      [callee_type, callee_method] {
        tsl::profiler::TraceMeProducer traceme(
            [&] { return absl::StrCat(callee_type, "::", callee_method); });
        VLOG(1) << callee_type << "::" << callee_method;
        FutureHelpers::ProfilingKeys keys;
        keys.traceme_context_id = traceme.GetContextId();
        return keys;
      },
      /*on_block_end=*/
      [callee_type, callee_method](FutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme(
            [&] { return absl::StrCat(callee_type, "::", callee_method); },
            keys.traceme_context_id);
      });
}

std::pair<Promise<>, Future<>> CommonPjRtClient::CreateLinkedUserPromise(
    PjRtMemorySpace* memory_space, const char* callee_type,
    const char* callee_method, absl::string_view debug_info) {
  auto [promise, future] = MakePromise<>();
  auto profiled_future = CreateProfiledFuture(memory_space, callee_type,
                                              callee_method, std::move(future));
  TrackFuture(memory_space, debug_info, profiled_future);
  return std::make_pair(std::move(promise), std::move(profiled_future));
}

tsl::AsyncValueRef<bool> CommonPjRtClient::CreateAllocationEventForTransfers(
    PjRtMemorySpace* memory_space,
    const std::optional<std::string>& debug_info) {
  return tsl::AsyncValueRef<bool>();
}

absl::StatusOr<xla::Shape> CommonPjRtClient::GetCopyDestinationShape(
    const xla::Shape& shape, PjRtMemorySpace* src_memory_space,
    PjRtMemorySpace* dst_memory_space) {
  auto other_client =
      dynamic_cast<CommonPjRtClient*>(dst_memory_space->client());
  if (!other_client) {
    return absl::InternalError(absl::StrFormat(
        "GetCopyDestinationShape not supported %s -> %s",
        src_memory_space->ToString(), dst_memory_space->ToString()));
  }
  if (shape.IsToken()) {
    return shape;
  }
  return other_client->MakeDefaultShapeForMemorySpace(
      dst_memory_space,
      xla::ShapeUtil::MakeShapeWithDescendingLayout(shape.element_type(),
                                                    shape.dimensions()),
      /*layout=*/nullptr);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtClient::BufferFromHostLiteral(const LiteralSlice& literal,
                                        PjRtMemorySpace* memory_space,
                                        const Layout* device_layout) {
  const Shape& shape = literal.shape();

  if (shape.IsTuple()) {
    return InvalidArgument(
        "Tuples are not supported in CommonPjRtClient::BufferFromHostLiteral");
  }
  tsl::profiler::TraceMeProducer producer(
      "CommonPjRtClient::BufferFromHostLiteral",
      tsl::profiler::ContextType::kPjRt);
  ASSIGN_OR_RETURN(Shape device_shape, MakeDefaultShapeForMemorySpace(
                                           memory_space, shape, device_layout));
  ASSIGN_OR_RETURN(int64_t on_device_bytes_count,
                   GetOnDeviceBytesCount(memory_space, device_shape));
  ASSIGN_OR_RETURN(auto raw_buffer,
                   AllocateRawBuffer(memory_space, on_device_bytes_count,
                                     /*retry_on_oom=*/true,
                                     /*allocate_after=*/{}));
  ASSIGN_OR_RETURN(
      auto definition_event,
      LinearizeInto(literal, device_shape,
                    HostBufferSemantics::kImmutableUntilTransferCompletes,
                    raw_buffer));
  return DefineBuffer(std::move(device_shape), memory_space,
                      std::move(raw_buffer), {std::move(definition_event)});
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtClient::CreateUninitializedBuffer(const Shape& shape,
                                            PjRtMemorySpace* memory_space) {
  if (shape.IsTuple()) {
    return InvalidArgument(
        "Tuples are not supported in "
        "CommonPjRtClient::CreateUninitializedBuffer");
  }
  Shape device_shape;
  if (!primitive_util::IsArrayType(shape.element_type())) {
    device_shape = shape;
  } else {
    if (shape.has_layout()) {
      ASSIGN_OR_RETURN(device_shape, MakeDefaultShapeForMemorySpace(
                                         memory_space, shape, &shape.layout()));
    } else {
      ASSIGN_OR_RETURN(device_shape, MakeDefaultShapeForMemorySpace(
                                         memory_space, shape, nullptr));
    }
  }
  ASSIGN_OR_RETURN(int64_t on_device_bytes_count,
                   GetOnDeviceBytesCount(memory_space, device_shape));
  ASSIGN_OR_RETURN(auto raw_buffer,
                   AllocateRawBuffer(memory_space, on_device_bytes_count,
                                     /*retry_on_oom=*/true,
                                     /*allocate_after=*/{}));
  ASSIGN_OR_RETURN(auto definition_event,
                   raw_buffer->MakeAllocationReadyEvent());
  ASSIGN_OR_RETURN(auto output_buffer,
                   DefineBuffer(std::move(device_shape), memory_space,
                                raw_buffer, {std::move(definition_event)}));
  return output_buffer;
}

absl::StatusOr<
    std::pair<std::unique_ptr<PjRtBuffer>, PjRtFulfillAliasBufferCallback>>
CommonPjRtClient::CreateAliasBuffer(const Shape& shape,
                                    PjRtMemorySpace* memory_space) {
  PjRtRawBufferRef raw_buffer;
  PjRtFulfillAliasRawBufferCallback buffer_promise;

  auto shared_shape = std::make_shared<const Shape>(shape);
  ASSIGN_OR_RETURN(int64_t on_device_bytes_count,
                   GetOnDeviceBytesCount(memory_space, *shared_shape));

  ASSIGN_OR_RETURN(std::tie(raw_buffer, buffer_promise),
                   CreateRawBufferChannel(memory_space, on_device_bytes_count));

  xla::PjRtDeviceEventPromiseRef definition_event_promise;
  PjRtDeviceEventRef definition_event;
  ASSIGN_OR_RETURN(
      std::tie(definition_event_promise, definition_event),
      CreateLinkedEventPromise(memory_space, "CreateRawBufferChannel"));

  PjRtFulfillAliasBufferCallback fulfill_cb =
      [buffer_promise = std::move(buffer_promise),
       definition_event_promise = std::move(definition_event_promise),
       memory_space,
       shared_shape](absl::StatusOr<xla::PjRtBuffer*> buffer_or) mutable {
        if (!buffer_or.ok()) {
          definition_event_promise.SetError(buffer_or.status());
          std::move(buffer_promise)(buffer_or.status()).IgnoreError();
          return buffer_or.status();
        }
        xla::PjRtBuffer* buffer = buffer_or.value();
        if (buffer->on_device_shape() != *shared_shape) {
          auto status = absl::InvalidArgumentError(absl::StrFormat(
              "Shape mismatch in CreateRawBufferChannel fulfill: expected %s, "
              "got %s",
              shared_shape->ToString(), buffer->on_device_shape().ToString()));
          definition_event_promise.SetError(status);
          std::move(buffer_promise)(status).IgnoreError();
          return status;
        }
        xla::CommonPjRtBuffer* common_buffer =
            dynamic_cast<xla::CommonPjRtBuffer*>(buffer);
        if (common_buffer == nullptr) {
          auto status =
              absl::InternalError("Failed to cast to CommonPjRtBuffer");
          definition_event_promise.SetError(status);
          std::move(buffer_promise)(status).IgnoreError();
          return status;
        }
        xla::CommonPjRtBuffer::ScopedHold hold =
            common_buffer->GetBufferWithHold(
                xla::CommonPjRtBuffer::ScopedHold::kDonation);
        auto device_event_or = hold.buffer()->GetDefinitionEvent(memory_space);
        if (!device_event_or.ok()) {
          auto status = device_event_or.status();
          definition_event_promise.SetError(status);
          std::move(buffer_promise)(status).IgnoreError();
          return status;
        }
        auto status = std::move(buffer_promise)(hold.buffer()->raw_buffer());
        if (!status.ok()) {
          definition_event_promise.SetError(status);
          return status;
        }

        definition_event_promise.Set(std::move(*device_event_or));
        hold.ConfirmDonation();
        return absl::OkStatus();
      };

  ASSIGN_OR_RETURN(
      auto result_buffer,
      DefineBuffer(shared_shape, memory_space, std::move(raw_buffer),
                   {std::move(definition_event)}));

  return std::make_pair(std::move(result_buffer), std::move(fulfill_cb));
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtMemorySpace* memory_space, const Layout* device_layout) {
  Shape shape;
  if (type == TOKEN) {
    shape = ShapeUtil::MakeTokenShape();
  } else {
    ASSIGN_OR_RETURN(shape, ShapeUtil::MakeValidatedShape(type, dims));
  }
  ASSIGN_OR_RETURN(Shape device_shape, MakeDefaultShapeForMemorySpace(
                                           memory_space, shape, device_layout));
  auto shared_device_shape =
      std::make_shared<const Shape>(std::move(device_shape));
  if (host_buffer_semantics ==
          PjRtClient::HostBufferSemantics::kImmutableZeroCopy ||
      host_buffer_semantics ==
          PjRtClient::HostBufferSemantics::kMutableZeroCopy) {
    if (BufferFromHostBufferSupportsZeroCopy(data, type, dims, byte_strides,
                                             *shared_device_shape, memory_space,
                                             device_layout)) {
      ASSIGN_OR_RETURN(
          int64_t on_device_bytes_count,
          GetOnDeviceBytesCount(memory_space, *shared_device_shape));
      ASSIGN_OR_RETURN(
          auto raw_buffer,
          ImportForeignMemory(
              const_cast<void*>(data),  // CONST_CAST_OK=flag controlled.
              std::move(on_done_with_host_buffer), on_device_bytes_count,
              memory_space,
              host_buffer_semantics ==
                  PjRtClient::HostBufferSemantics::kMutableZeroCopy));
      ASSIGN_OR_RETURN(
          auto output_buffer,
          DefineBuffer(shared_device_shape, memory_space, raw_buffer,
                       absl::InlinedVector<PjRtDeviceEventRef, 2>{}));
      return output_buffer;
    }
  }

  ASSIGN_OR_RETURN(int64_t on_device_bytes_count,
                   GetOnDeviceBytesCount(memory_space, *shared_device_shape));
  ASSIGN_OR_RETURN(auto raw_buffer,
                   AllocateRawBuffer(memory_space, on_device_bytes_count,
                                     /*retry_on_oom=*/true,
                                     /*allocate_after=*/{}));
  ASSIGN_OR_RETURN(auto definition_event,
                   LinearizeHostBufferInto(data, type, dims, byte_strides,
                                           host_buffer_semantics,
                                           std::move(on_done_with_host_buffer),
                                           *shared_device_shape, raw_buffer));
  ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> output_buffer,
                   DefineBuffer(shared_device_shape, memory_space, raw_buffer,
                                {std::move(definition_event)}));
  return output_buffer;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtBuffer* donated_dst, const Layout* device_layout) {
  auto* common_donated_dst = dynamic_cast<CommonPjRtBuffer*>(donated_dst);
  PjRtMemorySpace* memory_space = donated_dst->memory_space();
  if (memory_space->client() != this) {
    return InvalidArgument("Invalid buffer passed to BufferFromHostBuffer: %s",
                           memory_space->DebugString());
  }
  ASSIGN_OR_RETURN(const Shape shape,
                   ShapeUtil::MakeValidatedShape(type, dims));
  ASSIGN_OR_RETURN(Shape device_shape, MakeDefaultShapeForMemorySpace(
                                           memory_space, shape, device_layout));
  auto shared_device_shape =
      std::make_shared<const Shape>(std::move(device_shape));
  if (host_buffer_semantics ==
          PjRtClient::HostBufferSemantics::kImmutableZeroCopy ||
      host_buffer_semantics ==
          PjRtClient::HostBufferSemantics::kMutableZeroCopy) {
    if (BufferFromHostBufferSupportsZeroCopy(data, type, dims, byte_strides,
                                             *shared_device_shape, memory_space,
                                             device_layout)) {
      return BufferFromHostBuffer(data, type, dims, byte_strides,
                                  host_buffer_semantics,
                                  std::move(on_done_with_host_buffer),
                                  donated_dst->memory_space(), device_layout);
    }
  }

  ASSIGN_OR_RETURN(int64_t on_device_bytes_count,
                   GetOnDeviceBytesCount(memory_space, *shared_device_shape));
  auto hold = common_donated_dst->GetBufferWithHold(
      CommonPjRtBuffer::ScopedHold::kDonation);
  if (!hold.ok()) {
    return InvalidArgument("Invalid buffer passed to BufferFromHostBuffer: %s",
                           hold.status().ToString());
  }
  auto raw_buffer = hold.buffer()->raw_buffer();
  if (raw_buffer->GetOnDeviceSizeInBytes() != on_device_bytes_count) {
    return InvalidArgument(
        "Invalid buffer passed to BufferFromHostBuffer: %d vs %d",
        raw_buffer->GetOnDeviceSizeInBytes(), on_device_bytes_count);
  }

  PjRtDeviceEventPromiseRef definition_event_promise;
  PjRtDeviceEventRef definition_event;
  ASSIGN_OR_RETURN(
      std::tie(definition_event_promise, definition_event),
      CreateLinkedEventPromise(memory_space, "BufferFromHostBuffer"));
  auto events = hold.buffer()->GetAsyncValueDefinitionAndUsageDeviceEvents();
  hold.ConfirmDonation();

  xla::ExecuteWhenReady(
      events, async_work_runner(),
      [this, definition_event_promise = std::move(definition_event_promise),
       data, type, dims, byte_strides, host_buffer_semantics,
       on_done_with_host_buffer = std::move(on_done_with_host_buffer),
       raw_buffer, shared_device_shape]() mutable {
        auto definition_event = LinearizeHostBufferInto(
            data, type, dims, byte_strides, host_buffer_semantics,
            std::move(on_done_with_host_buffer), *shared_device_shape,
            raw_buffer);
        if (definition_event.ok()) {
          definition_event_promise.Set(*std::move(definition_event));
        } else {
          definition_event_promise.SetError(definition_event.status());
        }
      });
  ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> output_buffer,
                   DefineBuffer(shared_device_shape, memory_space, raw_buffer,
                                {std::move(definition_event)}));
  return output_buffer;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtClient::CreateViewOfDeviceBuffer(
    void* device_ptr, const Shape& shape, PjRtMemorySpace* memory_space,
    std::function<void()> on_delete_callback,
    std::optional<std::intptr_t> stream) {
  if (stream) {
    return Unimplemented(
        "CommonPjRtClient::CreateViewOfDeviceBuffer does not support `stream` "
        "argument.");
  }
  ASSIGN_OR_RETURN(
      Shape device_shape,
      MakeDefaultShapeForMemorySpace(
          memory_space, shape, shape.has_layout() ? &shape.layout() : nullptr));
  ASSIGN_OR_RETURN(int64_t on_device_bytes_count,
                   GetOnDeviceBytesCount(memory_space, device_shape));
  ASSIGN_OR_RETURN(
      auto raw_buffer,
      ImportForeignMemory(device_ptr, std::move(on_delete_callback),
                          on_device_bytes_count, memory_space,
                          /*is_mutable=*/false));
  ASSIGN_OR_RETURN(
      auto output_buffer,
      DefineBuffer(std::move(device_shape), memory_space, raw_buffer,
                   absl::InlinedVector<PjRtDeviceEventRef, 2>{}));
  return output_buffer;
}

absl::StatusOr<xla::Shape> CommonPjRtClient::MakeDefaultShapeForMemorySpace(
    PjRtMemorySpace* memory_space, xla::Shape shape,
    const xla::Layout* layout) const {
  if (!shape.IsToken()) {
    if (layout) {
      *shape.mutable_layout() = *layout;
      if (primitive_util::IsSubByteNonPredType(shape.element_type())) {
        ASSIGN_OR_RETURN(
            xla::Layout default_layout,
            (*GetTopologyDescription())
                ->GetDefaultLayout(shape.element_type(), shape.dimensions()));
        if (default_layout.element_size_in_bits() !=
            shape.layout().element_size_in_bits()) {
          return InvalidArgument(
              "Device buffers require %d bits per element for an element type "
              "%s, but got layout %s for shape %s",
              default_layout.element_size_in_bits(),
              PrimitiveType_Name(shape.element_type()), layout->ToString(),
              shape.ToString());
        }
      }
    } else {
      ASSIGN_OR_RETURN(
          *shape.mutable_layout(),
          (*GetTopologyDescription())
              ->GetDefaultLayout(shape.element_type(), shape.dimensions()));
    }
  }
  return shape;
}

tsl::Future<> CommonPjRtClient::MakeTrackedReadyFuture(
    PjRtDeviceEventPtr device_event, PjRtMemorySpace* memory_space,
    const char* callee_type, const char* callee_method) {
  auto [promise, result] = CreateLinkedUserPromise(
      memory_space, callee_type, callee_method, callee_method);
  device_event.AndThen([promise = std::move(promise), device_event]() mutable {
    if (auto error = device_event.GetErrorIfPresent()) {
      promise.Set(*error);
    } else {
      promise.Set();
    }
  });
  return result;
}

absl::StatusOr<std::shared_ptr<TransposePlan>>
CommonPjRtClient::GetTransposePlan(const TransposePlan::Options& options) {
  absl::MutexLock lock(transpose_mu_);
  return transpose_cache_.GetOrCreate(options);
}

Future<> CommonPjRtRawBufferImpl::CopyRawHostToDevice(const void* src,
                                                      int64_t offset,
                                                      int64_t transfer_size) {
  auto event =
      CopyRawHostToDeviceAndReturnEvent(src, offset, transfer_size, {});
  if (!event.ok()) {
    return Future<>(event.status());
  }
  return absl::down_cast<CommonPjRtClient*>(memory_space()->client())
      ->MakeTrackedReadyFuture(event->ptr(), memory_space(),
                               "CommonPjRtRawBuffer", "CopyRawHostToDevice");
}

Future<> CommonPjRtRawBufferImpl::CopyRawDeviceToHost(void* dst, int64_t offset,
                                                      int64_t transfer_size) {
  auto event =
      CopyRawDeviceToHostAndReturnEvent(dst, offset, transfer_size, {});
  if (!event.ok()) {
    return Future<>(event.status());
  }
  return absl::down_cast<CommonPjRtClient*>(memory_space()->client())
      ->MakeTrackedReadyFuture(event->ptr(), memory_space(),
                               "CommonPjRtRawBuffer", "CopyRawDeviceToHost");
}

void CommonPjRtRawBufferImpl::ScheduleCopyTo(
    PjRtDeviceEventRefVector transfer_dependency_events,
    PjRtRawBufferRef dst_raw_buffer,
    PjRtDeviceEventPromiseRef definition_event_promise,
    PjRtDeviceEventPromiseRef src_usage_event_promise,
    absl::AnyInvocable<void(absl::Status) &&> allocation_event) {
  PjRtDeviceEventSpan events_span(transfer_dependency_events);
  CommonPjRtClient* client =
      absl::down_cast<CommonPjRtClient*>(memory_space()->client());
  AsyncWorkRunner* async_work_runner = client->async_work_runner();

  xla::ExecuteWhenReady(
      events_span, async_work_runner,
      [src_raw_buffer = tsl::FormRef(this),
       dst_raw_buffer = std::move(dst_raw_buffer),
       definition_event_promise = std::move(definition_event_promise),
       src_usage_event_promise = std::move(src_usage_event_promise),
       allocation_event = std::move(allocation_event),
       transfer_dependency_events =
           std::move(transfer_dependency_events)]() mutable {
        absl::Status status = xla::GetErrors(transfer_dependency_events);
        if (!status.ok()) {
          if (allocation_event) {
            std::move(allocation_event)(status);
          }
          definition_event_promise.SetError(status);
          src_usage_event_promise.SetError(status);
          return;
        }

        src_raw_buffer->CopyTo(
            std::move(dst_raw_buffer), std::move(definition_event_promise),
            std::move(src_usage_event_promise), std::move(allocation_event));
      });
}

void CommonPjRtBufferImpl::CopyToRemoteDevice(
    Future<std::string> serialized_descriptor, RemoteSendCallback on_done) {
  auto* common_client = absl::down_cast<CommonPjRtClient*>(client());
  PjRtDeviceEventRefVector definition_events;
  PjRtDeviceEventPromiseRef usage_event_promise;
  PjRtRawBufferRef raw_buffer;
  auto hold_status = AcquireScopedRawBuffer(
      [&](PjRtRawBufferRef buf_raw_buffer,
          PjRtDeviceEventRefVector buf_definition_events)
          -> absl::StatusOr<PjRtDeviceEventRef> {
        raw_buffer = std::move(buf_raw_buffer);
        definition_events = std::move(buf_definition_events);
        PjRtDeviceEventRef usage_event;
        if (common_client->event_tracking_enabled()) {
          // Dependencies are added later either to the src_buffer_ptr's
          // definition events if they are not yet available, and to a boxcar's
          // ready event once the send is added to a boxcar.
          const auto& current_anno =
              tsl::profiler::ScopedMemoryDebugAnnotation::CurrentAnnotation();
          std::string op_name =
              !current_anno.pending_op_name.empty()
                  ? absl::StrCat(" Op:", current_anno.pending_op_name)
                  : "";
          ASSIGN_OR_RETURN(
              std::tie(usage_event_promise, usage_event),
              common_client->CreateLinkedEventPromise(
                  memory_space(), absl::StrCat("RemoteSend", op_name)));
        } else {
          ASSIGN_OR_RETURN(std::tie(usage_event_promise, usage_event),
                           common_client->CreateLinkedEventPromise(
                               memory_space(), "CopyToRemoteDevice"));
        }
        return usage_event;
      },
      "CopyToRemoteDevice()");
  if (!hold_status.ok()) {
    on_done(hold_status, /*sends_were_enqueued=*/false);
    return;
  }

  common_client->ScheduleRemoteSend(
      memory_space(), std::move(raw_buffer), std::move(definition_events),
      std::move(usage_event_promise), std::move(serialized_descriptor),
      std::move(on_done));
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> CommonPjRtBufferImpl::Bitcast(
    xla::PrimitiveType element_type, absl::Span<const int64_t> dims,
    const Layout* device_layout) {
  if (!primitive_util::IsArrayType(on_device_shape_->element_type()) ||
      !primitive_util::IsArrayType(element_type)) {
    return InvalidArgument("Bitcast can only be used on array types.");
  }
  ASSIGN_OR_RETURN(const Shape shape,
                   ShapeUtil::MakeValidatedShape(element_type, dims));
  ASSIGN_OR_RETURN(Shape new_on_device_shape,
                   client()->MakeDefaultShapeForMemorySpace(
                       memory_space(), shape, device_layout));
  if (ShapeUtil::ArraySize(*on_device_shape_) !=
      ShapeUtil::ArraySize(new_on_device_shape)) {
    return InvalidArgument(
        "Bitcast requires a new on-device shape to have the same size of %d "
        "bytes, but got %d bytes.",
        ShapeUtil::ArraySize(*on_device_shape_),
        ShapeUtil::ArraySize(new_on_device_shape));
  }

  std::unique_ptr<AbstractTrackedDeviceBuffer> device_buffer = ReleaseBuffer();
  if (device_buffer == nullptr) {
    return InvalidArgument("Bitcast was called on deleted or donated buffer.");
  }
  return std::make_unique<CommonPjRtBufferImpl>(
      std::make_shared<const Shape>(std::move(new_on_device_shape)),
      std::move(device_buffer), memory_space());
}

void CommonPjRtClient::ScheduleRemoteSend(
    PjRtMemorySpace* memory_space, PjRtRawBufferRef raw_buffer,
    PjRtDeviceEventRefVector definition_events,
    PjRtDeviceEventPromiseRef usage_event_promise,
    Future<std::string> serialized_descriptor,
    PjRtBuffer::RemoteSendCallback on_done) {
  auto error = absl::UnimplementedError(
      absl::StrCat("ScheduleRemoteSend is not implemented for %s",
                   memory_space->DebugString()));
  on_done(error, /*sends_were_enqueued=*/false);
  usage_event_promise.SetError(error);
}

absl::Status CommonPjRtClient::PrepareArguments(
    const ExecuteOptions& options,
    absl::Span<PjRtBuffer* const> argument_handles,
    absl::Span<int const> donated_params, PjRtDeviceEventRefVector& extra_deps,
    PjRtDeviceEventRefVector& control_deps,
    absl::InlinedVector<PjRtRawBufferRef, 4>& input_buffers,
    absl::InlinedVector<CommonPjRtBuffer::ScopedHold, 4>& device_buffers,
    PjRtDevice* device, int replica, int partition,
    absl::Span<const Shape> parameter_device_shapes, bool& is_error,
    bool allow_fallback_for_donation) {
  absl::flat_hash_set<void*> extra_deps_seen;
  absl::flat_hash_set<void*> control_deps_seen;
  if (argument_handles.size() != parameter_device_shapes.size()) {
    return InvalidArgument(
        "Execution supplied %d arguments but compiled program expected %d",
        argument_handles.size(), parameter_device_shapes.size());
  }
  input_buffers.reserve(argument_handles.size());
  device_buffers.reserve(argument_handles.size());
  auto donate_it = donated_params.begin();
  {
    tsl::profiler::TraceMe t2("Handle inputs");
    // State for `TestBufferDonationClashes`.
    absl::flat_hash_map<const void*, std::pair<bool, int>> donation_clashes;
    donation_clashes.reserve(argument_handles.size());
    // The first element is the argument index of the donated buffer, and the
    // second element is the size in bytes of the donated buffer.
    std::vector<std::pair<int, size_t>> donated_buffer_stats;
    for (int i = 0; i < argument_handles.size(); ++i) {
      PjRtBuffer* handle = argument_handles[i];
      auto* tfrt_buffer = absl::down_cast<CommonPjRtBufferImpl*>(handle);
      if (tfrt_buffer->device() != device) {
        return InvalidArgument(
            "Buffer passed to Execute() as argument %d to replica %d is on "
            "device %s, but replica is assigned to device %s.",
            i, replica, tfrt_buffer->device()->DebugString(),
            device->DebugString());
      }

      const Shape& expected_shape = parameter_device_shapes[i];
      const Shape& on_device_shape = tfrt_buffer->on_device_shape();

      if (options.strict_shape_checking &&
          // Skip shape check for non-array shapes (e.g. tuples).
          expected_shape.IsArray() && on_device_shape.IsArray() &&
          // Dynamic shapes cannot be compared directly.
          expected_shape.is_static() && on_device_shape.is_static() &&
          !xla::Shape::Equal().IgnoreMemorySpaceInLayout()(expected_shape,
                                                           on_device_shape)) {
        return InvalidArgument(
            "Buffer passed to Execute() as argument %d to replica %d has "
            "unexpected shape: %s (expected %s).",
            i, replica, on_device_shape.ToString(/*print_layout=*/true),
            expected_shape.ToString(/*print_layout=*/true));
      }

      const bool donated_param =
          donate_it != donated_params.end() && *donate_it == i;
      if (donated_param) {
        ++donate_it;
      }
      const bool donation_denied_at_runtime =
          options.non_donatable_input_indices.contains(i);
      if (donated_param && donation_denied_at_runtime &&
          on_device_shape.has_layout() &&
          on_device_shape.layout().memory_space() == Layout::kHostMemorySpace) {
        return absl::UnimplementedError(
            "pinned_host buffers do not support donation denial at runtime via "
            "`ExecuteOptions::non_donatable_input_indices`");
      }
      bool must_donate = donated_param && !donation_denied_at_runtime;
      RETURN_IF_ERROR(TestBufferDonationClashes(
          tfrt_buffer, donation_clashes, must_donate, i, replica, partition));
      if (allow_fallback_for_donation && must_donate) {
        // On CPU, we allow donation to succeed by introducing a copy. This was
        // added when enabling buffer donation on CPU since it turned out that a
        // number of users were holding external references to buffers that were
        // supposed to be donated. We may wish to tighten those semantics in the
        // future.
        device_buffers.emplace_back([&]() -> CommonPjRtBuffer::ScopedHold {
          auto result = tfrt_buffer->GetBufferWithHold(
              CommonPjRtBuffer::ScopedHold::kDonation);
          if (!result.ok()) {
            return tfrt_buffer->GetBufferWithHold(
                CommonPjRtBuffer::ScopedHold::kUsage);
          }
          return result;
        }());
      } else {
        device_buffers.emplace_back(tfrt_buffer->GetBufferWithHold(
            must_donate ? CommonPjRtBuffer::ScopedHold::kDonation
                        : CommonPjRtBuffer::ScopedHold::kUsage));
      }
      CommonPjRtBuffer::ScopedHold& hold = device_buffers.back();
      if (!hold.ok()) {
        return InvalidArgument(
            "Invalid buffer passed to Execute() as argument %d to replica %d: "
            "%s",
            i, replica, hold.status().ToString());
      }
      auto* device_buffer = hold.buffer();

      if (device_buffer->raw_buffer()) {
        PjRtRawBufferRef actual_buffer = device_buffer->raw_buffer();
        if (on_device_shape.is_dynamic() && !expected_shape.is_dynamic()) {
          ASSIGN_OR_RETURN(auto handle_logical_device_shape,
                           handle->logical_on_device_shape());
          auto* client = absl::down_cast<CommonPjRtClient*>(
              actual_buffer->memory_space()->client());
          auto ds_kind = client->GetDynamicShapeKind(
              actual_buffer->memory_space()->kind_id());
          auto status_or_buffer = xla::RemoveDynamicShapeMetadataIfPresent(
              actual_buffer, on_device_shape, handle_logical_device_shape,
              ds_kind);

          if (!status_or_buffer.ok()) {
            absl::Status status = status_or_buffer.status();
            tsl::errors::AppendToMessage(
                &status, absl::StrCat("; Error when preparing the input buffer "
                                      "to Execute() as argument ",
                                      i, " to replica ", replica));
            return status;
          }
          actual_buffer = std::move(status_or_buffer).value();
        }
        input_buffers.push_back(std::move(actual_buffer));
      } else {
        is_error = true;
      }

      // Definition events are never modified after buffer construction.
      for (const auto& ev : device_buffer->definition_events()) {
        if (ev) {
          switch (ev.ptr().state()) {
            case PJRT_DeviceEvent_State_Error:
              is_error = true;
              ABSL_FALLTHROUGH_INTENDED;
            case PJRT_DeviceEvent_State_Unavailable:
              if (extra_deps_seen.insert(ev.ptr().ToC().device_event).second) {
                extra_deps.push_back(ev);
              }
              break;
            case PJRT_DeviceEvent_State_Ready:
              break;
          }
        }
      }
      // If we are trying to donate this buffer, we must wait on its usage
      // events as well as its definition events to ensure that all reads on
      // this buffer (e.g., d2h transfer) have been completed before it can be
      // mutated. Usage holds on this buffer are excluded during a donation hold
      // so we know that its usage events won't be modified while we are
      // enqueueing, but we ignore any errors from usage events.
      if (hold.type() == CommonPjRtBuffer::ScopedHold::kDonation) {
        if (VLOG_IS_ON(1)) {
          ASSIGN_OR_RETURN(size_t on_device_size,
                           tfrt_buffer->GetOnDeviceSizeInBytes());
          donated_buffer_stats.emplace_back(std::make_pair(i, on_device_size));
        }
        for (const auto& ev : device_buffer->usage_events()) {
          if (ev.ptr().state() == PJRT_DeviceEvent_State_Ready) {
            continue;
          }
          if (control_deps_seen.insert(ev.ptr().ToC().device_event).second) {
            control_deps.push_back(ev);
          }
        }
      }
    }
    // Debug logging of buffer donation and input buffer shapes and size.
    if (VLOG_IS_ON(1)) {
      // Buffer donation information.
      if (!argument_handles.empty()) {
        LOG(INFO) << donated_buffer_stats.size() << " arguments out of total "
                  << argument_handles.size() << " arguments will be donated.";
        for (auto [index, buffer_size] : donated_buffer_stats) {
          LOG(INFO) << "Argument " << index << " with size " << buffer_size
                    << " will be donated.";
        }
      }
      // Input buffers shape and size.
      for (int i = 0; i < input_buffers.size(); ++i) {
        size_t buffer_size = input_buffers[i]->GetOnDeviceSizeInBytes();
        ASSIGN_OR_RETURN(Shape actual_input_shape,
                         argument_handles[i]->logical_on_device_shape());
        VLOG(2) << "input buffer with index " << i
                << " has shape: " << actual_input_shape.ToString()
                << " and size: " << buffer_size;
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<absl::InlinedVector<PjRtRawBufferRef, 4>>
CommonPjRtClient::AllocateOutputBuffersWithInputReuse(
    const Shape& output_device_shape,
    absl::Span<const CommonPjRtBuffer::ScopedHold> input_device_buffer_holds,
    const HloInputOutputAliasConfig& alias_config, PjRtDevice* device,
    absl::Span<const int> output_memory_space_kind_ids,
    const ExecuteOptions& options) {
  tsl::profiler::TraceMe traceme("AllocateOutputBuffersWithInputReuse");
  VLOG(1) << "Creating an output buffer, which may be partially donated, with "
             "shape "
          << output_device_shape.ToString();
  absl::InlinedVector<PjRtRawBufferRef, 4> buffers;
  if (output_device_shape.IsTuple() &&
      output_device_shape.tuple_shapes().empty()) {
    return buffers;
  }
  int num_input_pjrt_buffers = input_device_buffer_holds.size();
  absl::Span<const Shape> output_leaf_shapes =
      output_device_shape.IsTuple()
          ? absl::MakeSpan(output_device_shape.tuple_shapes())
          : absl::MakeSpan(&output_device_shape, 1);
  auto get_alias = [&](int i) {
    return output_device_shape.IsTuple() ? alias_config.GetAliasedParameter({i})
                                         : alias_config.GetAliasedParameter({});
  };
  buffers.resize(output_leaf_shapes.size());

  auto should_allocate_new_buffer =
      [&](std::optional<HloInputOutputAliasConfig::Alias> alias) -> bool {
    if (!alias.has_value()) {
      return true;
    }
    int parameter_index = alias->parameter_number;
    // Handle "Case 3." input
    // donation below. ^ denotes donation pair. i0,  i1^  ->   r0^ where
    // parameter_is_tupled_arguments=true
    //
    // e.g. For alias: {0, {1}, may-alias}
    // We should check the donation eligibility of the second buffer in the
    // input list.
    if (num_input_pjrt_buffers > 1 && alias->parameter_index.size() == 1) {
      parameter_index = alias->parameter_index[0];
    }
    if (input_device_buffer_holds[parameter_index].type() !=
        CommonPjRtBuffer::ScopedHold::kDonation) {
      return true;
    }
    auto& buffer =
        input_device_buffer_holds[parameter_index].buffer()->raw_buffer();
    return buffer && !buffer->is_mutable();
  };

  struct PendingAllocation {
    int output_index;
    const Shape* shape;
    int64_t size_bytes;
  };

  absl::flat_hash_map<PjRtMemorySpace*, std::vector<PendingAllocation>>
      pending_allocations;
  int64_t total_allocated_bytes = 0;

  for (int i = 0; i < output_leaf_shapes.size(); ++i) {
    std::optional<HloInputOutputAliasConfig::Alias> alias = get_alias(i);
    if (should_allocate_new_buffer(alias)) {
      const Shape& leaf_shape = output_leaf_shapes[i];
      const auto& current_anno =
          tsl::profiler::ScopedMemoryDebugAnnotation::CurrentAnnotation();
      tsl::profiler::ScopedMemoryDebugAnnotation anno(
          "dummy", current_anno.pending_region_type, 0, [&leaf_shape]() {
            return ShapeUtil::HumanStringWithLayout(leaf_shape);
          });
      int kind_id = output_memory_space_kind_ids[i];
      PjRtMemorySpace* memory_space = nullptr;
      for (PjRtMemorySpace* ms : device->memory_spaces()) {
        if (kind_id == ms->kind_id()) {
          memory_space = ms;
          break;
        }
      }
      if (memory_space == nullptr) {
        return absl::InternalError(
            absl::StrCat("No memory space found (kind_id: ", kind_id, ")"));
      }
      ASSIGN_OR_RETURN(int64_t on_device_bytes,
                       GetOnDeviceBytesCount(memory_space, leaf_shape));

      total_allocated_bytes += on_device_bytes;

      if (!options.use_output_arena) {
        ASSIGN_OR_RETURN(auto raw_buffer, AllocateRawBufferForExecute(
                                              memory_space, on_device_bytes,
                                              /*retry_on_oom=*/false));
        buffers[i] = std::move(raw_buffer);
      } else {
        // If arena allocation is requested, we defer the allocation.
        pending_allocations[memory_space].push_back(
            {i, &leaf_shape, on_device_bytes});
      }
    } else {
      // a tuple output element alias to input. There are 3 supported cases.
      // Case 1: alias a non-tuple input.
      // Case 2: alias a tuple input leaf while a single tuple PjRtBuffer is
      // passed to PjRtLoadExecutable::Execute.
      // Case 3: alias a tuple input leaf while individual input PjRtBuffer
      // leaves are passed to PjRtLoadExecutable::Execute.
      const ShapeIndex& shape_index = alias->parameter_index;
      size_t parameter_number;
      if (shape_index.empty()) {
        // Case 1: (o, i, {}) alias non-tuple input i
        CHECK_LT(alias->parameter_number, num_input_pjrt_buffers);
        parameter_number = alias->parameter_number;
      } else if (num_input_pjrt_buffers == 1 && shape_index.size() == 1 &&
                 shape_index[0] != 0) {
        // Case 2: (o, 0, {i}) alias a single tuple input's i-th element
        //  where i > 0.
        return Unimplemented("Alias %s not supported: found %d inputs.",
                             alias->ToString(), num_input_pjrt_buffers);
      } else if (shape_index.size() == 1) {
        // Case 3: (o, 0, {i}) alias a single tuple input's i-th element but
        // the input PjRtBuffers have not been tuplized yet
        parameter_number = shape_index[0];
      } else {
        return Unimplemented("Alias %s not supported: found %d inputs.",
                             alias->ToString(), num_input_pjrt_buffers);
      }
      const CommonPjRtBuffer::ScopedHold& input_hold =
          input_device_buffer_holds[parameter_number];
      buffers[i] = input_hold.buffer()->raw_buffer();
    }
  }

  // If pending_allocations is not empty, these allocations should be done in
  // an arena.
  for (const auto& [memory_space, allocs] : pending_allocations) {
    tsl::profiler::TraceMe trace_arena("Arena Allocation");
    ASSIGN_OR_RETURN(size_t alignment, GetDeviceAddressAlignment());
    std::vector<PjRtRawBufferInterface::SliceInfo> slices(allocs.size());
    int64_t total_arena_size_in_bytes = 0;
    for (int i = 0; i < allocs.size(); ++i) {
      PjRtRawBufferInterface::SliceInfo slice;
      slice.offset = total_arena_size_in_bytes;
      slice.size = allocs[i].size_bytes;
      slices[i] = std::move(slice);
      total_arena_size_in_bytes += allocs[i].size_bytes;

      // Make each sub buffer aligned.
      total_arena_size_in_bytes =
          llvm::alignTo(total_arena_size_in_bytes, alignment);
    }

    ASSIGN_OR_RETURN(
        PjRtRawBufferRef arena_buffer,
        AllocateRawBufferForExecute(memory_space, total_arena_size_in_bytes,
                                    /*retry_on_oom=*/false));

    tsl::profiler::TraceMe trace_arena_slicing("Arena Slicing");
    ASSIGN_OR_RETURN(std::vector<PjRtRawBufferRef> sliced_buffers,
                     arena_buffer->MultiSlice(slices));

    for (int i = 0; i < allocs.size(); ++i) {
      buffers[allocs[i].output_index] = std::move(sliced_buffers[i]);
    }
  }

  if (VLOG_IS_ON(1)) {
    LOG(INFO)
        << "Total size of new output buffers allocated in this execution: "
        << total_allocated_bytes
        << (options.use_output_arena ? " (using arena)" : "");
  }
  return std::move(buffers);
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
CommonPjRtClient::MakeCrossHostReceiveBuffers(
    absl::Span<const Shape> shapes, PjRtDevice* absl_nonnull pjrt_device,
    PjRtCrossHostRecvNotifier notifier) {
  VLOG(2) << "Making " << shapes.size() << " cross host receive buffers";
  if (shapes.empty()) {
    return InvalidArgument(
        "shapes parameter empty in MakeCrossHostReceiveBuffers");
  }

  ASSIGN_OR_RETURN(auto memory_space, pjrt_device->default_memory_space());

  std::vector<PjRtRawBufferRef> raw_buffers;
  PjRtDeviceEventRefVector transfer_dependency_events;
  std::vector<xla::Shape> dst_shapes;
  raw_buffers.reserve(shapes.size());
  // Reserve one extra for internal use.
  transfer_dependency_events.reserve(shapes.size() + 1);
  dst_shapes.reserve(shapes.size());
  for (const Shape& shape : shapes) {
    if (shape.IsTuple()) {
      return InvalidArgument(
          "Tuple shape %s not supported in MakeCrossHostReceiveBuffers",
          ShapeUtil::HumanString(shape));
    }
    ASSIGN_OR_RETURN(xla::Shape dst_shape,
                     MakeDefaultShapeForMemorySpace(
                         memory_space, shape,
                         shape.has_layout() ? &shape.layout() : nullptr));
    ASSIGN_OR_RETURN(int64_t on_device_bytes_count,
                     GetOnDeviceBytesCount(memory_space, dst_shape));
    dst_shapes.push_back(std::move(dst_shape));
    ASSIGN_OR_RETURN(PjRtRawBufferRef raw_buffer,
                     AllocateRawBuffer(memory_space, on_device_bytes_count,
                                       /*retry_on_oom=*/true,
                                       /*allocate_after=*/{}));

    PjRtDeviceEventPtr buffer_av = raw_buffer->GetRawBufferAsyncValue();
    transfer_dependency_events.push_back(buffer_av.CopyRef());
    raw_buffers.push_back(std::move(raw_buffer));
  }

  ASSIGN_OR_RETURN(
      PjRtDeviceEventRefVector definition_events,
      CrossHostReceiveBuffersInto(raw_buffers, std::move(notifier),
                                  std::move(transfer_dependency_events)));

  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  buffers.reserve(shapes.size());
  for (int i = 0; i < raw_buffers.size(); ++i) {
    ASSIGN_OR_RETURN(
        std::unique_ptr<PjRtBuffer> output_buffer,
        DefineBuffer(std::move(dst_shapes[i]), memory_space, raw_buffers[i],
                     {PjRtDeviceEventPtr(definition_events[i]).CopyRef()}));
    buffers.push_back(std::move(output_buffer));
  }
  return buffers;
}

// Send functionality for second cross-host transfers API; wraps
// CrossHostTransferBuffers.
absl::StatusOr<std::vector<Future<>>> CommonPjRtClient::CrossHostSendBuffers(
    absl::Span<PjRtBuffer* const> buffers,
    absl::Span<const GlobalDeviceId> dst_global_device_ids,
    std::vector<CrossHostTransferKey> transfer_keys) {
  // Validate arguments.
  if (dst_global_device_ids.size() != buffers.size() ||
      transfer_keys.size() != buffers.size()) {
    return InvalidArgument(
        "CrossHostSendBuffers: buffers, dst_global_device_ids, and "
        "transfer_keys must have the same length, but got %d, %d, and %d.",
        buffers.size(), dst_global_device_ids.size(), transfer_keys.size());
  }
  for (int i = 0; i < buffers.size(); ++i) {
    // Each transfer must be between an addressable and a non-addressable
    // device. If both devices are addressable, then both a data transfer and a
    // 'normal' XLA SPMD executable may try to acquire the same GPU clique,
    // causing issues.
    PjRtDevice* src_device = buffers[i]->device();
    if (!src_device->IsAddressable()) {
      return InvalidArgument(
          "CrossHostSendBuffers: buffer %d is on non-addressable device with "
          "global device id %d.",
          i, src_device->global_device_id().value());
    }
    ASSIGN_OR_RETURN(PjRtDevice * dst_device,
                     LookupDevice(dst_global_device_ids[i]));
    if (dst_device->IsAddressable()) {
      return InvalidArgument(
          "CrossHostSendBuffers: destination device for buffer %d is "
          "addressable (global device id %d), but cross-host transfers must "
          "be between an addressable and a non-addressable device.",
          i, dst_global_device_ids[i].value());
    }
  }

  // Create futures and promises.
  std::vector<Future<>> futures;
  std::vector<std::shared_ptr<Promise<>>> promises;
  futures.reserve(buffers.size());
  promises.reserve(buffers.size());
  for (int i = 0; i < buffers.size(); ++i) {
    auto [promise, future] = MakePromise<>();
    futures.push_back(std::move(future));
    promises.push_back(std::move(promise).ToShared());
  }

  // Extract the raw buffers and definition events for each of the input send
  // buffers.
  std::vector<PjRtRawBufferRef> raw_buffers;
  raw_buffers.reserve(buffers.size());

  PjRtDeviceEventRefVector transfer_dependencies;
  std::vector<PjRtDeviceEventPromiseRef> usage_event_promises;
  usage_event_promises.reserve(buffers.size());

  for (int i = 0; i < buffers.size(); ++i) {
    PjRtDeviceEventPromiseRef usage_event_promise;
    PjRtDeviceEventRef usage_event;
    ASSIGN_OR_RETURN(std::tie(usage_event_promise, usage_event),
                     CreateLinkedEventPromise(
                         buffers[i]->memory_space(),
                         absl::StrFormat("CrossHostSendBuffers buffer %i", i)));
    usage_event_promises.push_back(std::move(usage_event_promise));
    usage_event.AndThen([promise = std::move(promises[i]), usage_event]() {
      CHECK(usage_event.async_value()->IsAvailable());
      if (usage_event.async_value()->IsError()) {
        promise->Set(usage_event.async_value()->GetError());
      } else {
        promise->Set(absl::OkStatus());
      }
    });
    RETURN_IF_ERROR(
        absl::down_cast<CommonPjRtBufferImpl*>(buffers[i])
            ->AcquireScopedRawBuffer(
                [&](PjRtRawBufferRef buf_raw_buffer,
                    PjRtDeviceEventRefVector buf_definition_events) mutable
                    -> absl::StatusOr<PjRtDeviceEventRef> {
                  // Note: CrossHostTransferBuffers ensures that
                  // a reference to buf_raw_buffer is retained
                  // for the duration of the transfer.
                  raw_buffers.push_back(std::move(buf_raw_buffer));
                  ConsumeEvents(
                      std::move(buf_definition_events),
                      [&](PjRtDeviceEventRef&& ev) {
                        transfer_dependencies.push_back(std::move(ev));
                      });
                  return PjRtDeviceEventRef(usage_event);
                },
                "CrossHostSendBuffers"));
  }

  // Build the CrossHostTransferSpec for each buffer.
  std::vector<CrossHostTransferSpec> transfer_specs;
  transfer_specs.reserve(buffers.size());
  for (int i = 0; i < buffers.size(); ++i) {
    transfer_specs.push_back(CrossHostTransferSpec{
        /*src_global_device_id=*/buffers[i]->device()->global_device_id(),
        dst_global_device_ids[i], std::move(raw_buffers[i])});
  }

  // Schedule sends.
  absl::StatusOr<PjRtDeviceEventRefVector> usage_events_or =
      CrossHostTransferBuffers(std::move(transfer_dependencies),
                               std::move(transfer_specs));
  if (!usage_events_or.ok()) {
    for (auto& promise : usage_event_promises) {
      promise.SetError(usage_events_or.status());
    }
    return usage_events_or.status();
  }
  PjRtDeviceEventRefVector usage_events = std::move(usage_events_or).value();

  int usage_event_index = 0;
  ConsumeEvents(std::move(usage_events), [&](PjRtDeviceEventRef&& ev) {
    usage_event_promises[usage_event_index++].Set(std::move(ev));
  });

  return futures;
}

// Receive functionality for second cross-host transfers API; wraps
// CrossHostTransferBuffers.
absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
CommonPjRtClient::CrossHostReceiveBuffers(
    xla::PjRtDevice* device, absl::Span<const xla::Shape> shapes,
    absl::Span<const GlobalDeviceId> src_global_device_ids,
    std::vector<CrossHostTransferKey> transfer_keys) {
  // Validate arguments.
  if (shapes.empty()) {
    return InvalidArgument("shapes parameter empty in CrossHostReceiveBuffers");
  }
  if (src_global_device_ids.size() != shapes.size() ||
      transfer_keys.size() != shapes.size()) {
    return InvalidArgument(
        "CrossHostReceiveBuffers: shapes, src_global_device_ids, and "
        "transfer_keys must have the same length, but got %d, %d, and %d.",
        shapes.size(), src_global_device_ids.size(), transfer_keys.size());
  }
  // Each transfer must be between an addressable and a non-addressable
  // device. If both devices are addressable, then both a data transfer and a
  // 'normal' XLA SPMD executable may try to acquire the same GPU clique,
  // causing issues.
  if (!device->IsAddressable()) {
    return InvalidArgument(
        "CrossHostReceiveBuffers: destination device is non-addressable "
        "(global device id %d), but cross-host transfers must be between an  "
        "addressable and a non-addressable device.",
        device->global_device_id().value());
  }
  for (int i = 0; i < src_global_device_ids.size(); ++i) {
    ASSIGN_OR_RETURN(PjRtDevice * src_device,
                     LookupDevice(src_global_device_ids[i]));
    if (src_device->IsAddressable()) {
      return InvalidArgument(
          "CrossHostReceiveBuffers: source device for buffer %d is addressable "
          "(global device id %d), but cross-host transfers must be between an "
          "addressable and a non-addressable device.",
          i, src_global_device_ids[i].value());
    }
  }

  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode(
        absl::StrFormat("[%v] CommonPjRtClient::CrossHostReceiveBuffers",
                        device->local_device_id()),
        {{"num_shapes", shapes.size()}});
  });

  // Create a single definition event for all receive buffers.
  ASSIGN_OR_RETURN(PjRtMemorySpace * memory_space,
                   device->default_memory_space());
  PjRtDeviceEventPromiseRef definition_event_promise;
  PjRtDeviceEventRef definition_event;
  ASSIGN_OR_RETURN(
      std::tie(definition_event_promise, definition_event),
      CreateLinkedEventPromise(memory_space,
                               absl::StrFormat("CrossHostReceiveBuffers")));

  // Build output receive buffers, collect their allocation events, and form
  // their transfer specs.
  PjRtDeviceEventRefVector allocation_events;
  std::vector<CrossHostTransferSpec> transfer_specs;
  transfer_specs.reserve(shapes.size());
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  buffers.reserve(shapes.size());

  for (int i = 0; i < shapes.size(); ++i) {
    // Allocate the raw buffer and define its owning PjRtBuffer.
    ASSIGN_OR_RETURN(
        Shape on_device_shape,
        MakeDefaultShapeForMemorySpace(
            memory_space, shapes[i],
            shapes[i].has_layout() ? &shapes[i].layout() : nullptr));
    ASSIGN_OR_RETURN(size_t on_device_bytes_count,
                     GetOnDeviceBytesCount(memory_space, on_device_shape));
    ASSIGN_OR_RETURN(PjRtRawBufferRef raw_buffer,
                     AllocateRawBuffer(memory_space, on_device_bytes_count,
                                       /*retry_on_oom=*/true,
                                       /*allocate_after=*/{}));
    ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> buffer,
                     DefineBuffer(std::move(on_device_shape), memory_space,
                                  raw_buffer, {definition_event}));

    // Store a ref to the allocation event as a transfer dependency so that
    // the receive waits for the buffer allocation to complete.
    ASSIGN_OR_RETURN(PjRtDeviceEventRef allocation_event,
                     raw_buffer->MakeAllocationReadyEvent());
    allocation_events.push_back(std::move(allocation_event));
    transfer_specs.push_back(CrossHostTransferSpec{src_global_device_ids[i],
                                                   device->global_device_id(),
                                                   std::move(raw_buffer)});
    buffers.push_back(std::move(buffer));
  }

  // Schedule receives.
  absl::StatusOr<PjRtDeviceEventRefVector> definition_events_or =
      CrossHostTransferBuffers(std::move(allocation_events),
                               std::move(transfer_specs));
  if (!definition_events_or.ok()) {
    definition_event_promise.SetError(definition_events_or.status());
    return definition_events_or.status();
  }
  PjRtDeviceEventRefVector definition_events =
      std::move(definition_events_or).value();

  // Populate definition event. We use definition_events[0] because all
  // transfers scheduled by CrossHostReceiveBuffers receive data into the same
  // device (the 'device' given as input to this function), and because
  // CrossHostTransferBuffers will only assign different definition events for
  // transfers into different devices.
  bool definition_event_set = false;
  ConsumeEvents(std::move(definition_events), [&](PjRtDeviceEventRef&& ev) {
    if (!definition_event_set) {
      definition_event_promise.Set(std::move(ev));
      definition_event_set = true;
    }
  });

  return buffers;
}

static std::unique_ptr<PjRtBuffer> CreateOutputLeafBuffer(
    std::shared_ptr<const Shape> output_leaf_shape,
    PjRtDeviceEventRef definition_event, bool is_predetermined_error,
    CommonPjRtClient* client, PjRtDevice* device, PjRtRawBufferRef leaf_buffer,
    int kind_id) {
  PjRtMemorySpace* memory_space = nullptr;
  if (leaf_buffer) {
    memory_space = leaf_buffer->memory_space();
  } else {
    for (PjRtMemorySpace* ms : device->memory_spaces()) {
      if (kind_id == ms->kind_id()) {
        memory_space = ms;
        break;
      }
    }
    CHECK(memory_space) << "No memory space found for device: "
                        << device->DebugString() << " kind: " << kind_id;
  }
  auto buffer_or = client->DefineBuffer(std::move(output_leaf_shape),
                                        memory_space, std::move(leaf_buffer),
                                        {std::move(definition_event)});
  CHECK_OK(buffer_or);
  return *std::move(buffer_or);
}

std::vector<std::unique_ptr<PjRtBuffer>> CommonPjRtClient::CreateOutputs(
    const std::shared_ptr<const Shape>& output_device_shape,
    PjRtDeviceEventRef definition_event, PjRtDevice* device,
    absl::Span<const int> output_memory_space_kind_ids,
    absl::InlinedVector<PjRtRawBufferRef, 4> output_leaf_buffers,
    bool is_predetermined_error) {
  tsl::profiler::TraceMe t1("CommonPjRtClient::CreateOutputs");
  std::vector<std::unique_ptr<PjRtBuffer>> res;
  if (output_device_shape->IsTuple()) {
    const auto& tuple_shapes = output_device_shape->tuple_shapes();
    res.reserve(tuple_shapes.size());
    auto get_buffer = [&](int i) {
      return i < output_leaf_buffers.size() ? std::move(output_leaf_buffers[i])
                                            : PjRtRawBufferRef();
    };
    for (int i = 0; i < tuple_shapes.size(); ++i) {
      // Share sub-shape from the parent shape's tuple_shapes.
      auto leaf_shape =
          std::shared_ptr<const Shape>(output_device_shape, &tuple_shapes[i]);
      res.push_back(CreateOutputLeafBuffer(
          std::move(leaf_shape), definition_event, is_predetermined_error, this,
          device, get_buffer(i), output_memory_space_kind_ids[i]));
    }
  } else if (!output_device_shape->IsTuple() &&
             output_leaf_buffers.size() == 1) {
    // Share the shape directly from the executable.
    res.push_back(CreateOutputLeafBuffer(
        output_device_shape, std::move(definition_event),
        is_predetermined_error, this, device, std::move(output_leaf_buffers[0]),
        output_memory_space_kind_ids[0]));
  } else {
    CHECK(is_predetermined_error)
        << "Nontuple results must have a single result buffer.";
    res.push_back(CreateOutputLeafBuffer(output_device_shape,
                                         std::move(definition_event),
                                         is_predetermined_error, this, device,
                                         {}, output_memory_space_kind_ids[0]));
  }
  return res;
}

absl::StatusOr<CommonPjRtLoadedExecutable::DeviceAndAssignment>
CommonPjRtLoadedExecutable::LookupDeviceAndAssignment(
    const ExecuteOptions& options, int replica, int partition,
    PjRtDevice* device) const {
  DeviceAndAssignment result;
  if (device == nullptr) {
    if (device_assignment_ == nullptr) {
      return InvalidArgument(
          "device_assignment_ must be set if device is not provided.");
    }
    const int64_t device_id = (*device_assignment_)(replica, partition);
    GlobalDeviceId global_device_id(device_id);
    ASSIGN_OR_RETURN(device, client()->LookupDevice(global_device_id));
    result.device_assignment = device_assignment_;
  } else {
    if (device_assignment_ != nullptr) {
      return InvalidArgument(
          "device_assignment_ must not be set if device is provided.");
    }
    CHECK_EQ(replica, 0);
    CHECK_EQ(partition, 0);
    CHECK(addressable_devices_.empty());
    result.device_assignment = std::make_shared<DeviceAssignment>(1, 1);
    (*result.device_assignment)(0, 0) = device->id();
  }
  CHECK_EQ(device->process_index(), client()->process_index());
  result.device = device;
  result.replica = replica;
  result.partition = partition;
  return result;
}

absl::Status CommonPjRtLoadedExecutable::ExecutePrepare(
    ExecuteLaunchArgs& launch_args,
    absl::Span<PjRtBuffer* const> argument_handles, xla::RunId run_id,
    int replica, int partition, const ExecuteOptions& options,
    size_t host_callback_idx, PjRtDevice* device, int attempt) const {
  tsl::profiler::TraceMe traceme("CommonPjRtLoadedExecutable::ExecutePrepare");
  ASSIGN_OR_RETURN(
      auto device_and_assign,
      LookupDeviceAndAssignment(options, replica, partition, device));
  // Fill in device to launch_args so it will be present even if ExecutePrepare
  // fails with OOM.
  device = device_and_assign.device;
  launch_args.device = device;

  // Execute takes `extra_deps` and waits for those to be
  // fulfilled before executing the program and returning an available
  // `execute_event` signaling that the program execution is complete. To avoid
  // clobbering inputs, we must ensure that
  //   `extra_deps` = inputs' definition events + donated inputs' usage events.
  // This also ensures that the returned `execute_event` dominates all inputs'
  // events, and thus output buffer only need to contain `execute_event` as the
  // single definition event.
  launch_args.extra_deps.reserve(argument_handles.size());
  launch_args.control_deps.reserve(argument_handles.size());

  bool is_error = false;
  RETURN_IF_ERROR(CommonPjRtClient::PrepareArguments(
      options, argument_handles, ParametersThatMustBeDonated(),
      launch_args.extra_deps, launch_args.control_deps,
      launch_args.input_buffers, launch_args.device_buffers, device, replica,
      partition, parameter_device_shapes_, is_error,
      client()->allow_fallback_for_donation()));

  absl::InlinedVector<PjRtRawBufferRef, 4> output_leaf_buffers;
  if (!is_error || !client()->supports_predetermined_error()) {
    // Allocate output with input reuse. Any allocation errors are returned
    // immediately. Derived classes may use custom logic for allocation.
    ASSIGN_OR_RETURN(output_leaf_buffers,
                     client()->AllocateOutputBuffersWithInputReuse(
                         *output_device_shape_, launch_args.device_buffers,
                         input_output_alias_config(), device,
                         output_memory_space_kind_ids_, options));
    VLOG(3) << "Created output buffer: " << output_device_shape_->ToString();

    RETURN_IF_ERROR(CheckBufferCompatibilities(
        options, launch_args.input_buffers, argument_handles));
  }

  ASSIGN_OR_RETURN(launch_args.executable,
                   LoadRawExecutable(options, host_callback_idx, run_id,
                                     std::move(device_and_assign), attempt));
  launch_args.options = &options;
  launch_args.is_predetermined_error = is_error;
  launch_args.output_leaf_buffers = std::move(output_leaf_buffers);
  return absl::OkStatus();
}

absl::Span<int const> CommonPjRtLoadedExecutable::ParametersThatMustBeDonated()
    const {
  return parameters_that_must_be_donated_;
}

absl::Status CommonPjRtLoadedExecutable::CheckBufferCompatibilities(
    const ExecuteOptions& options,
    absl::Span<const PjRtRawBufferRef> input_buffers,
    absl::Span<PjRtBuffer* const> argument_handles) const {
  if (input_buffers.size() != input_buffer_sizes_in_bytes_.size()) {
    return InvalidArgument(
        "Execution supplied %lld buffers but compiled program expected %lld "
        "buffers",
        input_buffers.size(), input_buffer_sizes_in_bytes_.size());
  }
  const CommonPjRtClient* common_client = this->client();

  for (int i = 0; i < input_buffers.size(); ++i) {
    const auto& expected_shape = parameter_device_shapes_[i];
    const auto& actual_shape = argument_handles[i]->on_device_shape();

    PjRtDynamicShapeKind ds_kind = common_client->GetDynamicShapeKind(
        argument_handles[i]->memory_space()->kind_id());

    bool both_are_dynamic =
        !expected_shape.is_static() && !actual_shape.is_static();

    if (both_are_dynamic && ds_kind == PjRtDynamicShapeKind::kPrefix) {
      // Both shapes dynamic of kPrefix kind.
      // Element type check
      if (expected_shape.element_type() != actual_shape.element_type()) {
        return error::RuntimeProgramInputMismatch(
            "Executable(%s) expected parameter %d to have element type %s but "
            "got buffer with element type %s",
            name(), i, PrimitiveType_Name(expected_shape.element_type()),
            PrimitiveType_Name(actual_shape.element_type()));
      }
      // Rank check
      if (expected_shape.dimensions().size() !=
          actual_shape.dimensions().size()) {
        return error::RuntimeProgramInputMismatch(
            "Executable(%s) expected parameter %d to have rank %d but "
            "got buffer with rank %d",
            name(), i, expected_shape.dimensions().size(),
            actual_shape.dimensions().size());
      }
      // Layout check
      if (!xla::LayoutUtil::LayoutsInShapesEqual(expected_shape,
                                                 actual_shape)) {
        return error::RuntimeProgramInputMismatch(
            "Executable(%s) expected parameter %d to have layout %s but "
            "got buffer with layout %s",
            name(), i, expected_shape.layout().ToString(),
            actual_shape.layout().ToString());
      }
      // Bounds check
      ASSIGN_OR_RETURN(Shape actual_logical_shape,
                       argument_handles[i]->logical_on_device_shape());
      for (int d = 0; d < expected_shape.dimensions().size(); ++d) {
        if (actual_logical_shape.dimensions(d) > expected_shape.dimensions(d)) {
          return error::RuntimeProgramInputMismatch(
              "Executable(%s) expected parameter %d dimension %d runtime size "
              "<= %lld, but got buffer with size %lld",
              name(), i, d, expected_shape.dimensions(d),
              actual_logical_shape.dimensions(d));
        }
      }
    } else {
      size_t buffer_size = input_buffers[i]->GetOnDeviceSizeInBytes();
      if (input_buffer_sizes_in_bytes_[i] != buffer_size) {
        return error::RuntimeProgramInputMismatch(
            "Executable(%s) expected parameter %d of size %lld (%s) but got "
            "buffer with incompatible size %lld (%s)",
            name(), i, input_buffer_sizes_in_bytes_[i],
            expected_shape.ToString(true), buffer_size,
            actual_shape.ToString(true));
      }
    }

    if (!parameter_memory_space_kind_ids_.empty()) {
      if (argument_handles[i]->memory_space()->kind_id() !=
          parameter_memory_space_kind_ids_[i]) {
        return error::RuntimeProgramInputMismatch(
            "Executable(%s) got a parameter buffer for parameter %d in an "
            "unexpected memory space '%s'",
            name(), i, argument_handles[i]->memory_space()->kind());
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<PjRtLoadedExecutable::Result>
CommonPjRtLoadedExecutable::ExecuteLaunch(ExecuteLaunchArgs& launch_args,
                                          bool fill_future) const {
  auto results = std::move(*launch_args.executable)
                     .Execute(*launch_args.options, launch_args.input_buffers,
                              launch_args.output_leaf_buffers,
                              std::move(launch_args.extra_deps),
                              std::move(launch_args.control_deps),
                              launch_args.is_predetermined_error, fill_future);
  {
    tsl::profiler::TraceMe t3("Handle input event recording");
    // Handle input event recording.
    for (CommonPjRtBuffer::ScopedHold& b : launch_args.device_buffers) {
      if (b.type() == CommonPjRtBuffer::ScopedHold::kUsage) {
        b.ConvertUsageHold(results.primary_execute_event);
      } else {
        CHECK(b.type() == CommonPjRtBuffer::ScopedHold::kDonation);
        b.ConfirmDonation();
      }
    }
  }

  RETURN_IF_ERROR(results.inline_status);

  return PjRtLoadedExecutable::Result(
      {/*future=*/std::move(results.future),
       /*buffers=*/client()->CreateOutputs(
           output_device_shape_, std::move(results.primary_execute_event),
           launch_args.device, output_memory_space_kind_ids_,
           std::move(launch_args.output_leaf_buffers),
           launch_args.is_predetermined_error)});
}

absl::Status CommonPjRtLoadedExecutable::ExecutePrepareWithOomRetries(
    std::optional<ExecuteLaunchArgs>& launch_args,
    absl::Span<PjRtBuffer* const> argument_handles, xla::RunId run_id,
    int replica, int partition, const ExecuteOptions& options,
    size_t host_callback_idx, PjRtDevice* device) const {
  absl::Status prepare_status;
  int attempts = 0;
  while (true) {
    launch_args.emplace();
    prepare_status =
        ExecutePrepare(*launch_args, argument_handles, run_id, replica,
                       partition, options, host_callback_idx, device, attempts);
    ++attempts;
    if (!absl::IsResourceExhausted(prepare_status)) {
      break;
    }
    if (!client()->ShouldRetryOnOom(attempts, launch_args->device, this,
                                    prepare_status)) {
      break;
    }
  }
  if (!prepare_status.ok()) {
    LOG(ERROR) << "ExecutePrepareWithOomRetries failed: " << prepare_status;
  }
  return prepare_status;
}

static absl::Status ValidateHostTransferCallbacks(
    absl::Span<const std::vector<SendCallback>> send_callbacks,
    absl::Span<const std::vector<RecvCallback>> recv_callbacks,
    size_t num_devices) {
  if (!send_callbacks.empty() && send_callbacks.size() != num_devices) {
    return InvalidArgument(
        "The number of send callback vectors does not match the number of "
        "devices");
  }
  if (!recv_callbacks.empty() && recv_callbacks.size() != num_devices) {
    return InvalidArgument(
        "The number of recv callback vectors does not match the number of "
        "devices");
  }
  return absl::OkStatus();
}

absl::StatusOr<PjRtLoadedExecutable::Result>
CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice(
    absl::Span<PjRtBuffer* const> argument_handles, xla::RunId run_id,
    int replica, int partition, const ExecuteOptions& options, bool fill_future,
    PjRtDevice* device) const {
  tsl::profiler::TraceMe traceme(
      "CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice");
  std::optional<ExecuteLaunchArgs> launch_args;
  RETURN_IF_ERROR(ExecutePrepareWithOomRetries(
      launch_args, argument_handles, run_id, replica, partition, options,
      /*host_callback_idx=*/0, device));
  return ExecuteLaunch(*launch_args, fill_future);
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
CommonPjRtLoadedExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<tsl::Future<void>>& returned_future, bool fill_future) const {
  RunId run_id = options.launch_id != 0 ? RunId(options.launch_id)
                                        : RunId::CreateUniqueId();
  tsl::profiler::TraceMe traceme([&]() {
    return tsl::profiler::TraceMeEncode(
        absl::StrFormat("CommonPjRtLoadedExecutable::ExecuteSharded (%s)",
                        name()),
        {{"name", name()},
         {"num_replicas", num_replicas()},
         {"num_partitions", num_partitions()},
         {"num_addressable_devices", addressable_devices_.size()}});
  });
  for (int i = 0; i < addressable_devices_.size(); ++i) {
    if (addressable_devices_[i] == device) {
      RETURN_IF_ERROR(ValidateHostTransferCallbacks(
          options.send_callbacks, options.recv_callbacks, /*num_devices=*/1));
      ASSIGN_OR_RETURN(auto result,
                       ExecuteHelperOnSingleDevice(
                           argument_handles, run_id,
                           addressable_device_logical_ids_[i].replica,
                           addressable_device_logical_ids_[i].partition,
                           options, fill_future));
      returned_future = std::move(result.future);
      return std::move(result.buffers);
    }
  }
  return InvalidArgument(
      "ExecuteShard attempted to execute on device id %d which is not "
      "addressable by this client",
      device->global_device_id().value());
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
CommonPjRtLoadedExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<tsl::Future<void>>& returned_future, bool fill_future) const {
  tsl::profiler::TraceMe traceme([&]() {
    return tsl::profiler::TraceMeEncode(
        absl::StrFormat("CommonPjRtLoadedExecutable::ExecutePortable (%s)",
                        name()),
        {{"name", name()},
         {"num_replicas", num_replicas()},
         {"num_partitions", num_partitions()},
         {"num_addressable_devices", addressable_devices_.size()}});
  });
  if (num_replicas() != 1 || num_partitions() != 1) {
    return InvalidArgument(
        "ExecutePortable expects a single-core executable but gets "
        "one with %d replica %d partition",
        num_replicas(), num_partitions());
  }
  if (device == nullptr) {
    return InvalidArgument("ExecutePortable expects a device to be specified");
  }
  if (!device->IsAddressable()) {
    return InvalidArgument(
        "ExecutePortable attempted to execute on device id %d which is not "
        "addressable by this client",
        device->global_device_id().value());
  }

  RETURN_IF_ERROR(ValidateHostTransferCallbacks(
      options.send_callbacks, options.recv_callbacks, /*num_devices=*/1));
  VLOG(1) << "ExecutePortable executes single-core portable executable "
          << name();
  RunId run_id = options.launch_id != 0 ? RunId(options.launch_id)
                                        : RunId::CreateUniqueId();
  ASSIGN_OR_RETURN(auto result,
                   ExecuteHelperOnSingleDevice(argument_handles, run_id,
                                               /*replica=*/0,
                                               /*partition=*/0, options,
                                               fill_future, device));
  returned_future = std::move(result.future);
  return std::move(result.buffers);
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
CommonPjRtLoadedExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<tsl::Future<void>>>& returned_futures) const {
  RunId run_id = options.launch_id != 0 ? RunId(options.launch_id)
                                        : RunId::CreateUniqueId();
  int num_addressable_devices = addressable_devices_.size();

  VLOG(1) << absl::StreamFormat(
      "CommonPjRtLoadedExecutable::Execute: run_id=%d, execution_mode=%v",
      run_id.ToInt(), options.execution_mode);

  if (!client()->allows_execute_recursion() &&
      ThisThreadIsInsideHostCallback()) {
    // Because TPU is single threaded, and the host callback currently blocking
    // the TPU, we should not initiate any outstanding computations because that
    // risks deadlocking the TPU.
    return InvalidArgument("Execute() called from inside host callback.");
  }

  tsl::profiler::TraceMeProducer producer(
      [&] {
        return tsl::profiler::TraceMeEncode(
            absl::StrFormat("CommonPjRtLoadedExecutable::Execute (%s)", name()),
            {{"run_id", run_id.ToInt()},
             {"execution_mode", absl::StrCat(options.execution_mode)},
             {"name", name()},
             {"num_replicas", num_replicas()},
             {"num_partitions", num_partitions()},
             {"num_addressable_devices", num_addressable_devices}});
      },
      tsl::profiler::ContextType::kPjRt, run_id.ToInt());

  if (argument_handles.size() != num_addressable_devices) {
    return InvalidArgument(
        "Attempted to execute with %d argument lists when local device "
        "count is %d (total replica count: %d, partition count: %d)",
        argument_handles.size(), num_addressable_devices, num_replicas(),
        num_partitions());
  }

  VLOG(1) << "Executing computation " << name()
          << "; num_replicas=" << num_replicas()
          << " num_partitions=" << num_partitions()
          << " num_addressable_devices=" << num_addressable_devices;

  RETURN_IF_ERROR(ValidateHostTransferCallbacks(options.send_callbacks,
                                                options.recv_callbacks,
                                                addressable_devices_.size()));

  std::vector<absl::StatusOr<Result>> results(num_addressable_devices);
  if (num_addressable_devices == 1) {
    // Fast-path if there is only one device — run the computation on the
    // current thread.
    const int replica = addressable_device_logical_ids_[0].replica;
    const int partition = addressable_device_logical_ids_[0].partition;
    results[0] = ExecuteHelperOnSingleDevice(argument_handles[0], run_id,
                                             replica, partition, options,
                                             returned_futures.has_value());
  } else {
    absl::Mutex mu;
    int preparing = num_addressable_devices;
    int launching = num_addressable_devices;
    int failed = 0;
    absl::Status first_failure_status;

    {
      // The gang_schedule mutex ensures that all calls to Schedule() happen
      // atomically and cannot interleave with calls to Execute on other
      // threads. If calls to Schedule are not atomic, then the threads can get
      // stuck waiting for done_preparing to become true.
      absl::MutexLock gang_schedule(client()->gang_scheduler());
      auto context_id = producer.GetContextId();
      for (int i = 0; i < num_addressable_devices; ++i) {
        const int replica = addressable_device_logical_ids_[i].replica;
        const int partition = addressable_device_logical_ids_[i].partition;
        PjRtDevice* device = addressable_devices_[i];
        client()->LaunchOnDevice(
            device, [&, context_id, i, replica, partition, device] {
              tsl::profiler::TraceMeConsumer consumer(
                  [&] {
                    return tsl::profiler::TraceMeEncode(
                        absl::StrFormat(
                            "[%d] CommonPjRtLoadedExecutable::Execute (%s)", i,
                            name()),
                        {{"name", name()},
                         {"replica", replica},
                         {"partition", partition},
                         {"global_device_id", device->global_device_id()}});
                  },
                  tsl::profiler::ContextType::kPjRt, context_id);

              // Two phase launch. Phase 1: Prepare on all cores. Abort
              // launch on prepare failure.
              std::optional<ExecuteLaunchArgs> launch_args;
              absl::Status launch_status = ExecutePrepareWithOomRetries(
                  launch_args, argument_handles[i], run_id, replica, partition,
                  options,
                  /*host_callback_idx=*/i);
              // Wait for prepare to finish on all cores.
              if (client()->supports_two_phase_launch()) {
                absl::MutexLock lock(mu);
                preparing--;
                auto done_preparing = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) {
                  return preparing == 0;
                };
                mu.Await(absl::Condition(&done_preparing));
                if (!launch_status.ok()) {
                  if (failed == 0) {
                    first_failure_status = launch_status;
                  }
                  failed++;
                }
                if (failed > 0) {
                  // Poison results for all cores.
                  results[i] = first_failure_status;
                  // Abort phase 2 if Prepare fails for any core.
                  --launching;
                  return;
                }
              }

              // Phase 2: Launch. It cannot fail.
              results[i] =
                  ExecuteLaunch(*launch_args, returned_futures.has_value());

              absl::MutexLock lock(mu);
              --launching;
            });
      }
    }

    // Wait until we either fail Phase 1 or completes two phases.
    tsl::profiler::TraceMe trace_wait("Wait for LaunchOnDevice completion");
    auto done = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) {
      return launching == 0;
    };
    absl::MutexLock lock(mu);
    mu.Await(absl::Condition(&done));
  }
  VLOG(3) << "Replicated execution complete.";

  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> wrapped_results(
      num_addressable_devices);
  if (returned_futures.has_value()) {
    returned_futures->reserve(num_addressable_devices);
  }
  for (int i = 0; i < num_addressable_devices; ++i) {
    const int replica = addressable_device_logical_ids_[i].replica;
    const int partition = addressable_device_logical_ids_[i].partition;
    auto& statusor = results[i];
    if (!statusor.ok()) {
      if (absl::IsResourceExhausted(statusor.status())) {
        client()->CallOomHandlers();
      }
      if (returned_futures.has_value()) {
        returned_futures->clear();
      }
      if (num_addressable_devices == 1) {
        return statusor.status();
      }
      return AppendStatus(
          statusor.status(),
          absl::StrFormat("while running replica %d and partition %d of a "
                          "replicated computation (other "
                          "replicas may have failed as well).",
                          replica, partition));
    }
    wrapped_results[i] = std::move(statusor->buffers);
    if (returned_futures.has_value()) {
      returned_futures->push_back(*std::move(statusor->future));
    }
  }
  return wrapped_results;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtBufferImpl::CopyToCpuMemorySpace(xla::Shape dst_shape,
                                           PjRtMemorySpace* dst_memory_space) {
  auto* dst_client =
      dynamic_cast<CommonPjRtClient*>(dst_memory_space->client());
  if (!dst_client) {
    return absl::InvalidArgumentError(
        "CopyToCpuMemorySpace only supported across CommonPjRtClient "
        "subclassed clients");
  }
  auto shared_dst_shape =
      std::make_shared<const xla::Shape>(std::move(dst_shape));
  ASSIGN_OR_RETURN(
      int64_t on_device_bytes_count,
      dst_client->GetOnDeviceBytesCount(dst_memory_space, *shared_dst_shape));
  ASSIGN_OR_RETURN(
      auto dst_raw_buffer,
      dst_client->AllocateRawBuffer(dst_memory_space, on_device_bytes_count,
                                    /*retry_on_oom=*/true, {}));
  PjRtDeviceEventPromiseRef definition_event_promise;
  PjRtDeviceEventRef definition_event;
  ASSIGN_OR_RETURN(std::tie(definition_event_promise, definition_event),
                   dst_client->CreateLinkedEventPromise(dst_memory_space, ""));
  ASSIGN_OR_RETURN(
      auto buffer,
      dst_client->DefineBuffer(shared_dst_shape, dst_memory_space,
                               dst_raw_buffer, {std::move(definition_event)}));
  auto* base_ptr = dst_raw_buffer->GetHostPointer();
  std::unique_ptr<MutableLiteralBase> literal;
  bool needs_second_copy = false;
  if (!primitive_util::IsSubByteNonPredType(shared_dst_shape->element_type()) &&
      base_ptr) {
    literal = std::make_unique<MutableBorrowingLiteral>(
        reinterpret_cast<char*>(base_ptr), *shared_dst_shape);
  } else {
    literal = std::make_unique<Literal>(*shared_dst_shape);
    needs_second_copy = true;
  }

  auto copied = ToLiteral(literal.get());
  copied.OnReady([literal = std::move(literal), dst_client, needs_second_copy,
                  dst_raw_buffer = std::move(dst_raw_buffer),
                  shared_dst_shape = std::move(shared_dst_shape),
                  definition_event_promise = std::move(
                      definition_event_promise)](absl::Status status) mutable {
    if (!status.ok()) {
      definition_event_promise.SetError(status);
    } else {
      absl::StatusOr<PjRtDeviceEventRef> status_or_h2d_transfer_event;
      if (needs_second_copy) {
        status_or_h2d_transfer_event = dst_client->LinearizeInto(
            *literal, *shared_dst_shape,
            PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
            dst_raw_buffer);
        if (!status_or_h2d_transfer_event.ok()) {
          definition_event_promise.SetError(status);
        } else {
          status_or_h2d_transfer_event.value().AndThen(
              [literal = std::move(literal)] {});
          definition_event_promise.Set(
              *std::move(status_or_h2d_transfer_event));
        }
      } else {
        definition_event_promise.SetReady();
      }
    }
  });

  return buffer;
}

static absl::AnyInvocable<void(absl::Status) &&> ToAllocationCallback(
    ::tsl::AsyncValueRef<bool> allocation_event) {
  if (!allocation_event) {
    return {};
  }
  return [allocation_event =
              std::move(allocation_event)](absl::Status status) mutable {
    if (status.ok()) {
      allocation_event.SetStateConcrete();
    } else {
      allocation_event.SetError(status);
    }
  };
}

static absl::Status CommonCopyToMemorySpace(
    CommonPjRtBuffer* src_buffer, PjRtMemorySpace* dst_memory_space,
    std::shared_ptr<const xla::Shape> dst_shape,
    PjRtDeviceEventPromiseRef& definition_event_promise,
    PjRtDeviceEventPromiseRef& src_usage_event_promise,
    PjRtRawBufferRef& src_raw_buffer, PjRtRawBufferRef& dst_raw_buffer,
    std::unique_ptr<PjRtBuffer>& dst_buffer,
    PjRtDeviceEventRefVector& definition_events,
    ::tsl::AsyncValueRef<bool>& allocation_event) {
  auto* src_memory_space = src_buffer->memory_space();
  CommonPjRtClient* const src_client =
      absl::down_cast<CommonPjRtClient*>(src_buffer->client());
  CommonPjRtClient* const dst_client =
      dynamic_cast<CommonPjRtClient*>(dst_memory_space->client());
  if (!dst_client) {
    return absl::InvalidArgumentError(
        "CommonCopyToMemorySpace only supported across CommonPjRtClient "
        "subclassed clients");
  }
  ASSIGN_OR_RETURN(
      const int64_t on_device_bytes_count,
      dst_client->GetOnDeviceBytesCount(dst_memory_space, *dst_shape));

  std::optional<std::string> debug_info = std::nullopt;
  if (dst_client->event_tracking_enabled()) {
    const auto& current_anno =
        tsl::profiler::ScopedMemoryDebugAnnotation::CurrentAnnotation();
    if (!current_anno.pending_op_name.empty() &&
        !current_anno.pending_region_type.empty()) {
      debug_info = std::make_optional<std::string>(absl::StrCat(
          current_anno.pending_op_name, " ", current_anno.pending_region_type));
    }
  }

  static std::atomic<uint64_t> start_transfer_id = []() {
    absl::BitGen bits;
    return absl::Uniform<uint64_t>(bits);
  }();
  uint64_t transfer_id = start_transfer_id.fetch_add(1);

  allocation_event = dst_client->CreateAllocationEventForTransfers(
      dst_memory_space, debug_info);
  PjRtDeviceEventRef definition_event;
  if (dst_client->event_tracking_enabled()) {
    ASSIGN_OR_RETURN(
        std::tie(definition_event_promise, definition_event),
        dst_client->CreateLinkedEventPromise(
            dst_memory_space,
            absl::StrCat("CopyToMemorySpace CrossDeviceSink: ", transfer_id,
                         " Op:", debug_info.value_or(""))));
  } else {
    ASSIGN_OR_RETURN(
        std::tie(definition_event_promise, definition_event),
        dst_client->CreateLinkedEventPromise(dst_memory_space, ""));
  }

  auto status = [&]() -> absl::Status {
    if (dst_raw_buffer) {
      if (dst_raw_buffer->GetOnDeviceSizeInBytes() != on_device_bytes_count) {
        return InvalidArgument(
            "Invalid buffer passed to CommonCopyToMemorySpace: %d vs %d",
            dst_raw_buffer->GetOnDeviceSizeInBytes(), on_device_bytes_count);
      }
    } else {
      ASSIGN_OR_RETURN(dst_raw_buffer,
                       dst_client->AllocateRawBuffer(
                           dst_memory_space, on_device_bytes_count,
                           /*retry_on_oom=*/true, allocation_event));
    }
    ASSIGN_OR_RETURN(
        dst_buffer,
        dst_client->DefineBuffer(dst_shape, dst_memory_space, dst_raw_buffer,
                                 {std::move(definition_event)}));
    RETURN_IF_ERROR(src_buffer->AcquireScopedRawBuffer(
        [&](PjRtRawBufferRef buf_raw_buffer,
            PjRtDeviceEventRefVector buf_definition_events)
            -> absl::StatusOr<PjRtDeviceEventRef> {
          src_raw_buffer = std::move(buf_raw_buffer);
          PjRtDeviceEventRef usage_event;
          if (definition_events.empty()) {
            definition_events = std::move(buf_definition_events);
          } else {
            ConsumeEvents(std::move(buf_definition_events),
                          [&](PjRtDeviceEventRef&& ev) {
                            definition_events.push_back(std::move(ev));
                          });
          }
          if (src_client->event_tracking_enabled()) {
            ASSIGN_OR_RETURN(
                std::tie(src_usage_event_promise, usage_event),
                dst_client->CreateLinkedEventPromise(
                    src_memory_space,
                    absl::StrCat(
                        "CopyToMemorySpace CrossDeviceSrc: ", transfer_id,
                        " Op:", debug_info.value_or(""))));
          } else {
            ASSIGN_OR_RETURN(
                std::tie(src_usage_event_promise, usage_event),
                dst_client->CreateLinkedEventPromise(src_memory_space, ""));
          }
          return usage_event;
        }));
    return absl::OkStatus();
  }();
  if (!status.ok()) {
    if (allocation_event) {
      allocation_event.SetError(status);
    }
    definition_event_promise.SetError(status);
    return status;
  }

  if (!src_raw_buffer) {
    PjRtDeviceEventSpan deps_span(definition_events);
    xla::ExecuteWhenReady(
        deps_span, src_client->async_work_runner(),
        [dst_raw_buffer = std::move(dst_raw_buffer),
         definition_events = std::move(definition_events),
         definition_event_promise = std::move(definition_event_promise),
         src_usage_event_promise = std::move(src_usage_event_promise),
         allocation_event = std::move(allocation_event)]() mutable {
          auto set_error = [&](absl::Status status) {
            if (allocation_event) {
              allocation_event.SetError(status);
            }
            definition_event_promise.SetError(status);
            src_usage_event_promise.SetError(status);
          };
          absl::Status status = xla::GetErrors(definition_events);
          if (!status.ok()) {
            set_error(status);
            return;
          }
          set_error(
              absl::InternalError("src_raw_buffer is nullptr for copy but no "
                                  "definition events were errors."));
        });
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtBufferImpl::CopyFromCpuToMemorySpace(
    xla::Shape dst_shape, PjRtMemorySpace* dst_memory_space) {
  tsl::profiler::TraceMe traceme("CopyToMemorySpace");
  CommonPjRtClient* const src_client =
      absl::down_cast<CommonPjRtClient*>(client());
  auto* dst_client =
      dynamic_cast<CommonPjRtClient*>(dst_memory_space->client());
  if (!dst_client) {
    return absl::InvalidArgumentError(
        "DirectCopyToMemorySpace only supported across CommonPjRtClient "
        "subclassed clients");
  }
  PjRtDeviceEventPromiseRef definition_event_promise;
  PjRtDeviceEventPromiseRef src_usage_event_promise;
  PjRtRawBufferRef src_raw_buffer;
  PjRtRawBufferRef dst_raw_buffer;
  std::unique_ptr<PjRtBuffer> dst_buffer;
  PjRtDeviceEventRefVector definition_events;
  ::tsl::AsyncValueRef<bool> allocation_event;
  auto shared_dst_shape =
      std::make_shared<const xla::Shape>(std::move(dst_shape));
  RETURN_IF_ERROR(CommonCopyToMemorySpace(
      this, dst_memory_space, shared_dst_shape, definition_event_promise,
      src_usage_event_promise, src_raw_buffer, dst_raw_buffer, dst_buffer,
      definition_events, allocation_event));
  if (src_raw_buffer) {
    PjRtDeviceEventSpan deps_span(definition_events);
    xla::ExecuteWhenReady(
        deps_span, src_client->async_work_runner(),
        [dst_raw_buffer = std::move(dst_raw_buffer),
         src_raw_buffer = std::move(src_raw_buffer), dst_client = dst_client,
         src_shape = on_device_shape(),
         device_shape = std::move(shared_dst_shape),
         definition_events = std::move(definition_events),
         definition_event_promise = std::move(definition_event_promise),
         src_usage_event_promise = std::move(src_usage_event_promise),
         allocation_event = std::move(allocation_event)]() mutable {
          auto set_error = [&](absl::Status status) {
            if (allocation_event) {
              allocation_event.SetError(status);
            }
            definition_event_promise.SetError(status);
            src_usage_event_promise.SetError(status);
          };
          absl::Status status = xla::GetErrors(definition_events);
          if (!status.ok()) {
            set_error(status);
            return;
          }
          auto* base_ptr = src_raw_buffer->GetHostPointer();
          if (!base_ptr) {
            set_error(absl::InternalError(
                "CopyFromCpuToMemorySpace expects that "
                "src_raw_buffer->GetHostPointer() is nonnull"));
            return;
          }
          if (allocation_event) {
            allocation_event.SetStateConcrete();
          }
          std::unique_ptr<MutableLiteralBase> literal =
              std::make_unique<MutableBorrowingLiteral>(
                  reinterpret_cast<char*>(base_ptr), src_shape);
          auto* dst_memory_space = dst_raw_buffer->memory_space();
          auto status_or_h2d_transfer_event = dst_client->LinearizeInto(
              *literal, *device_shape,
              PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
              std::move(dst_raw_buffer));
          CHECK_OK(status_or_h2d_transfer_event);
          auto h2d_transfer_event = *std::move(status_or_h2d_transfer_event);
          h2d_transfer_event.AndThen(
              [src_raw_buffer = std::move(src_raw_buffer),
               literal = std::move(literal),
               src_usage_event_promise = std::move(src_usage_event_promise)]() {
                src_usage_event_promise.SetReady();
              });
          if (dst_client->event_tracking_enabled()) {
            dst_client->AppendDescriptionToEvent(
                dst_memory_space, h2d_transfer_event.ptr(),
                " TransferToDevice ", {definition_event_promise.event()});
          }
          definition_event_promise.Set(std::move(h2d_transfer_event));
        });
  }
  return dst_buffer;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtBufferImpl::CopyToMemorySpace(PjRtMemorySpace* dst_memory_space) {
  // Copying across PjRtClients involves a copy through the host.
  if (dst_memory_space->client() == client()) {
    ASSIGN_OR_RETURN(auto dest_shape,
                     client()->GetCopyDestinationShape(
                         on_device_shape(), memory_space(), dst_memory_space));
    if (xla::Shape::Equal().IgnoreMemorySpaceInLayout()(dest_shape,
                                                        on_device_shape())) {
      return DirectCopyToMemorySpace(dst_memory_space);
    }
    if (!primitive_util::IsSubByteNonPredType(dest_shape.element_type())) {
      if (client()->IsOnCpu(dst_memory_space) &&
          xla::LayoutUtil::IsMonotonicWithDim0Major(dest_shape.layout()) &&
          dest_shape.layout().tiles().empty()) {
        return CopyToCpuMemorySpace(std::move(dest_shape), dst_memory_space);
      }
      if (client()->IsOnCpu(memory_space()) &&
          xla::LayoutUtil::IsMonotonicWithDim0Major(
              on_device_shape().layout()) &&
          on_device_shape().layout().tiles().empty()) {
        return CopyFromCpuToMemorySpace(std::move(dest_shape),
                                        dst_memory_space);
      }
    }
  }
  if (auto* other_client =
          dynamic_cast<CommonPjRtClient*>(dst_memory_space->client())) {
    return CopyToMemorySpaceFallbackThroughLiteral(dst_memory_space);
  } else {
    return CopyToMemorySpaceSyncThroughLiteral(dst_memory_space);
  }
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtBufferImpl::CopyToMemorySpace(PjRtBuffer* donated_dst) {
  PjRtMemorySpace* dst_memory_space = donated_dst->memory_space();
  // Copying across PjRtClients involves a copy through the host.
  if (dst_memory_space->client() == client()) {
    ASSIGN_OR_RETURN(auto dest_shape,
                     client()->GetCopyDestinationShape(
                         on_device_shape(), memory_space(), dst_memory_space));
    if (xla::Shape::Equal().IgnoreMemorySpaceInLayout()(dest_shape,
                                                        on_device_shape())) {
      return DirectCopyToMemorySpace(donated_dst);
    }
    if (!primitive_util::IsSubByteNonPredType(dest_shape.element_type())) {
      if (client()->IsOnCpu(dst_memory_space) &&
          xla::LayoutUtil::IsMonotonicWithDim0Major(dest_shape.layout()) &&
          dest_shape.layout().tiles().empty()) {
        return CopyToCpuMemorySpace(std::move(dest_shape), dst_memory_space);
      }
      if (client()->IsOnCpu(memory_space()) &&
          xla::LayoutUtil::IsMonotonicWithDim0Major(
              on_device_shape().layout()) &&
          on_device_shape().layout().tiles().empty()) {
        return CopyFromCpuToMemorySpace(std::move(dest_shape),
                                        dst_memory_space);
      }
    }
  }
  if (auto* other_client =
          dynamic_cast<CommonPjRtClient*>(dst_memory_space->client())) {
    return CopyToMemorySpaceFallbackThroughLiteral(dst_memory_space);
  } else {
    return CopyToMemorySpaceSyncThroughLiteral(dst_memory_space);
  }
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtBufferImpl::CopyToMemorySpaceSyncThroughLiteral(
    PjRtMemorySpace* dst_memory_space) {
  // Copy across PjRtClients by copying through host
  ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal,
                   PjRtBuffer::ToLiteral().Await());
  absl::InlinedVector<int64_t, 4> byte_strides(
      literal->shape().dimensions().size());
  RETURN_IF_ERROR(ShapeUtil::UnpackedByteStrides(literal->shape(),
                                                 absl::MakeSpan(byte_strides)));
  // Avoid use-after-free on `literal` due to unsequenced move and use.
  Literal* literal_pointer = literal.get();
  return dst_memory_space->client()->BufferFromHostBuffer(
      literal_pointer->untyped_data(), literal_pointer->shape().element_type(),
      literal_pointer->shape().dimensions(), byte_strides,
      PjRtClient::HostBufferSemantics::kImmutableZeroCopy,
      [literal{std::move(literal)}]() { /* frees literal */ }, dst_memory_space,
      /*device_layout=*/nullptr);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtBufferImpl::CopyToMemorySpaceFallbackThroughLiteral(
    PjRtMemorySpace* dst_memory_space) {
  Shape shape = on_device_shape().IsToken()
                    ? on_device_shape()
                    : ShapeUtil::MakeShapeWithDescendingLayout(
                          on_device_shape().element_type(),
                          on_device_shape().dimensions());
  ASSIGN_OR_RETURN(
      auto manager,
      dst_memory_space->client()->CreateBuffersForAsyncHostToDevice(
          {shape}, dst_memory_space));
  std::unique_ptr<PjRtBuffer> dst_buffer = manager->RetrieveBuffer(0);

  auto literal = std::make_unique<Literal>();
  Future<> d2h_future = LazyToLiteral(
      [raw_literal = literal.get(),
       shape = std::move(shape)]() -> Future<MutableLiteralBase*> {
        *raw_literal = Literal(shape);
        return Future<MutableLiteralBase*>(raw_literal);
      });
  d2h_future.OnReady(
      [manager = std::move(manager),
       literal = std::move(literal)](absl::Status status) mutable {
        if (!status.ok()) {
          manager->SetBufferError(0, status);
          return;
        }
        auto* raw_manager = manager.get();
        auto* raw_literal = literal.get();
        CHECK_OK(raw_manager->TransferLiteralToBuffer(
            0, *raw_literal,
            [literal = std::move(literal), manager = std::move(manager)]() {
              // Keep `literal` and `manager` alive until the H2D transfer is
              // complete.
            }));
      });

  return dst_buffer;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtBufferImpl::DirectCopyToMemorySpace(
    PjRtMemorySpace* dst_memory_space) {
  tsl::profiler::TraceMe traceme("CopyToMemorySpace");
  if (!dynamic_cast<CommonPjRtClient*>(dst_memory_space->client())) {
    return absl::InvalidArgumentError(
        "DirectCopyToMemorySpace only supported across CommonPjRtClient "
        "subclassed clients");
  }
  PjRtDeviceEventPromiseRef definition_event_promise;
  PjRtDeviceEventPromiseRef src_usage_event_promise;
  PjRtRawBufferRef src_raw_buffer;
  PjRtRawBufferRef dst_raw_buffer;
  std::unique_ptr<PjRtBuffer> dst_buffer;
  PjRtDeviceEventRefVector definition_events;
  ::tsl::AsyncValueRef<bool> allocation_event;
  RETURN_IF_ERROR(CommonCopyToMemorySpace(
      this, dst_memory_space, on_device_shape_, definition_event_promise,
      src_usage_event_promise, src_raw_buffer, dst_raw_buffer, dst_buffer,
      definition_events, allocation_event));
  if (src_raw_buffer) {
    src_raw_buffer->ScheduleCopyTo(
        std::move(definition_events), std::move(dst_raw_buffer),
        std::move(definition_event_promise), std::move(src_usage_event_promise),
        ToAllocationCallback(std::move(allocation_event)));
  }
  return dst_buffer;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtBufferImpl::DirectCopyToMemorySpace(PjRtBuffer* donated_dst) {
  PjRtMemorySpace* dst_memory_space = donated_dst->memory_space();
  tsl::profiler::TraceMe traceme("CopyToMemorySpace");
  if (!dynamic_cast<CommonPjRtClient*>(dst_memory_space->client())) {
    return absl::InvalidArgumentError(
        "DirectCopyToMemorySpace only supported across CommonPjRtClient "
        "subclassed clients");
  }
  PjRtDeviceEventPromiseRef definition_event_promise;
  PjRtDeviceEventPromiseRef src_usage_event_promise;
  PjRtRawBufferRef src_raw_buffer;
  PjRtRawBufferRef dst_raw_buffer;
  std::unique_ptr<PjRtBuffer> dst_buffer;
  PjRtDeviceEventRefVector definition_events;
  auto* common_donated_dst = dynamic_cast<CommonPjRtBuffer*>(donated_dst);
  auto hold = common_donated_dst->GetBufferWithHold(
      CommonPjRtBuffer::ScopedHold::kDonation);
  if (!hold.ok()) {
    return InvalidArgument("Invalid buffer passed to BufferFromHostBuffer: %s",
                           hold.status().ToString());
  }
  dst_raw_buffer = hold.buffer()->raw_buffer();
  definition_events =
      hold.buffer()->GetAsyncValueDefinitionAndUsageDeviceEvents();
  hold.ConfirmDonation();

  ::tsl::AsyncValueRef<bool> allocation_event;
  RETURN_IF_ERROR(CommonCopyToMemorySpace(
      this, dst_memory_space, on_device_shape_, definition_event_promise,
      src_usage_event_promise, src_raw_buffer, dst_raw_buffer, dst_buffer,
      definition_events, allocation_event));
  if (src_raw_buffer) {
    src_raw_buffer->ScheduleCopyTo(
        std::move(definition_events), std::move(dst_raw_buffer),
        std::move(definition_event_promise), std::move(src_usage_event_promise),
        ToAllocationCallback(std::move(allocation_event)));
  }
  return dst_buffer;
}

Future<> CommonPjRtBufferImpl::LazyToLiteral(
    absl::AnyInvocable<Future<MutableLiteralBase*>() &&> generator) {
  return ToLiteralImpl(nullptr, std::move(generator));
}

Future<> CommonPjRtBufferImpl::ToLiteral(MutableLiteralBase* literal) {
  return ToLiteralImpl(literal, [] {
    return Future<MutableLiteralBase*>(
        FailedPrecondition("ToLiteral generator should never be called"));
  });
}

Future<> CommonPjRtBufferImpl::ToLiteralImpl(
    MutableLiteralBase* literal,
    absl::AnyInvocable<Future<MutableLiteralBase*>() &&> generator) {
  tsl::profiler::TraceMeProducer producer("CommonPjRtBuffer::ToLiteral",
                                          tsl::profiler::ContextType::kPjRt);
  VLOG(1) << "CommonPjRtBuffer::ToLiteral";
  auto common_client = absl::down_cast<CommonPjRtClient*>(client());
  if (!common_client->allows_recursion() && ThisThreadIsInsideHostCallback()) {
    // Because TPU is single threaded, and the host callback currently blocking
    // the TPU, we should not block on any outstanding computations because that
    // risks deadlocking the TPU.
    return Future<>(
        InvalidArgument("ToLiteral() called from inside host callback."));
  }
  auto device_shape = on_device_shape();
  absl::StatusOr<Shape> logical_shape = logical_on_device_shape();
  if (!logical_shape.ok()) {
    return Future<>(logical_shape.status());
  }

  // TODO(zhangqiaorjc): Fast path if zero device_buffer wait events.
  // Make two copies because EnqueueWorkWhenReady below needs two different
  // lifetimes.
  PjRtDeviceEventRefVector src_definition_events;

  PjRtDeviceEventPromiseRef device_promise;
  PjRtRawBufferRef raw_buffer;
  auto hold_status = AcquireScopedRawBuffer(
      [&](PjRtRawBufferRef buf_raw_buffer,
          PjRtDeviceEventRefVector definition_events)
          -> absl::StatusOr<PjRtDeviceEventRef> {
        src_definition_events = std::move(definition_events);
        if (buf_raw_buffer) {
          raw_buffer = std::move(buf_raw_buffer);
          PjRtDeviceEventRef device_event;
          ASSIGN_OR_RETURN(std::tie(device_promise, device_event),
                           common_client->CreateLinkedEventPromise(
                               memory_space_, "ToLiteral Leaf: 0"));
          return device_event;
        }
        return PjRtDeviceEventRef();
      },
      "ToLiteral()");
  if (!hold_status.ok()) {
    return Future<>(std::move(hold_status));
  }

  auto [promise, result] = common_client->CreateLinkedUserPromise(
      memory_space(), "CommonPjRtBuffer", "ToLiteral", "ToLiteralEvent");
  if (device_promise) {
    if (common_client->event_tracking_enabled()) {
      common_client->AddEventDependencies(
          memory_space(), device_promise.event(), src_definition_events);
    }
  }

  // Wait for buffer definition events to finish before d2h dispatch.
  // D2H dispatch should be in parallel, e.g. one Execute event finish may
  // trigger multiple outputs' D2H, they should happen in different threads in
  // parallel.
  PjRtDeviceEventSpan deps_span(src_definition_events);
  xla::ExecuteWhenReady(
      deps_span, common_client->async_work_runner(),
      [common_client, shape = *std::move(logical_shape),
       device_shape = std::move(device_shape),
       src_definition_events = std::move(src_definition_events),
       raw_buffer = std::move(raw_buffer),
       device_promise = std::move(device_promise), literal,
       generator = std::move(generator), promise = std::move(promise),
       context_id = producer.GetContextId()]() mutable {
        auto copy_literal_async =
            [shape = std::move(shape), device_shape = std::move(device_shape),
             src_definition_events = std::move(src_definition_events),
             raw_buffer = std::move(raw_buffer),
             device_promise = std::move(device_promise),
             promise = std::move(promise), context_id = context_id,
             common_client](
                const absl::StatusOr<MutableLiteralBase*>& value) mutable {
              tsl::profiler::TraceMeConsumer traceme(
                  [&] {
                    return tsl::profiler::TraceMeEncode(
                        "D2H Dispatch",
                        {{"shape", shape.ToString(/*print_layout=*/true)}});
                  },
                  tsl::profiler::ContextType::kPjRt, context_id);

              // Notify all pending events with `status`.
              auto notify_all = [&](absl::Status status) {
                promise.Set(status);
                if (device_promise) {
                  device_promise.SetError(status);
                }
              };

              if (!value.ok()) {
                notify_all(value.status());
                return;
              }
              MutableLiteralBase* literal = *std::move(value);

              if (!ShapeUtil::Compatible(shape, literal->shape())) {
                notify_all(absl::InternalError(absl::StrFormat(
                    "Shape mismatch during ToLiteral conversion %s vs %s",
                    shape.ToString(), literal->shape().ToString())));
                return;
              }
              // Errors in src buffer are surfaced to user.
              for (size_t i = 0; i < src_definition_events.size(); ++i) {
                const auto& ev = src_definition_events[i];
                if (auto error = ev.GetErrorIfPresent()) {
                  notify_all(std::move(*error));
                  return;
                }
              }
              // Fast path for token shape, no need to copy in this case.
              // Already checked that the shape is compatible with the literal.
              if (shape.element_type() == TOKEN) {
                // A sanity check to ensure token buffers have no data.
                if (raw_buffer->GetOnDeviceSizeInBytes() != 0) {
                  notify_all(absl::InternalError(absl::StrFormat(
                      "Token buffer should have zero bytes, but has size %d.",
                      raw_buffer->GetOnDeviceSizeInBytes())));
                  return;
                }
                if (device_promise) {
                  device_promise.SetReady();
                }
                promise.Set();
                return;
              }
              auto ds_kind = common_client->GetDynamicShapeKind(
                  raw_buffer->memory_space()->kind_id());
              auto status_or_buffer = xla::RemoveDynamicShapeMetadataIfPresent(
                  raw_buffer, device_shape, shape, ds_kind);
              if (!status_or_buffer.ok()) {
                notify_all(status_or_buffer.status());
                return;
              }
              raw_buffer = *status_or_buffer;
              if (common_client->ShouldDoDirectTransfer(
                      *literal, shape, raw_buffer->memory_space())) {
                tsl::profiler::TraceMe traceme([&] {
                  return tsl::profiler::TraceMeEncode(
                      "PjRtBuffer::ToLiteralZeroCopyPath",
                      {{"shape", shape.ToString(/*print_layout=*/true)}});
                });
                auto d2h_event = raw_buffer->CopyRawDeviceToHostAndReturnEvent(
                    literal->untyped_data(), 0, literal->size_bytes());
                if (!d2h_event.ok()) {
                  notify_all(d2h_event.status());
                  return;
                }
                device_promise.Set(*std::move(d2h_event));
                ScopedLauncher launcher(
                    [device_promise, promise = std::move(promise)]() mutable {
                      if (auto error =
                              device_promise.event().GetErrorIfPresent()) {
                        promise.Set(*error);
                      } else {
                        promise.Set();
                      }
                    },
                    common_client->async_work_runner());
                launcher.AddDependency(device_promise.event());
                return;
              }
              PjRtMemorySpace* memory_space = raw_buffer->memory_space();
              auto staging_buffer = ToStagingBuffer(
                  std::move(raw_buffer), device_promise,
                  [common_client](size_t size, PjRtMemorySpace* memory_space) {
                    return common_client->AllocateForDelinearizationAsync(
                        size, memory_space);
                  });
              common_client->DelinearizeAsync(std::move(staging_buffer),
                                              memory_space, shape, literal,
                                              std::move(promise));
            };

        if (literal != nullptr) {
          copy_literal_async(literal);
        } else {
          Future<MutableLiteralBase*> generated = std::move(generator)();
          if (generated.IsKnownReady()) {
            copy_literal_async(generated.Await());
          } else {
            generated.OnReady(
                *common_client->async_work_runner(),
                [copy_literal_async = std::move(copy_literal_async)](
                    const absl::StatusOr<MutableLiteralBase*>& value) mutable {
                  copy_literal_async(value);
                });
          }
        }
      });
  return result;
}

absl::StatusOr<PjRtRawBufferRef>
CommonPjRtBufferImpl::CreateRawAliasOfBuffer() {
  PjRtRawBufferRef raw_buffer;
  RETURN_IF_ERROR(AcquireScopedRawBuffer(
      [&](PjRtRawBufferRef buf_raw_buffer,
          PjRtDeviceEventRefVector definition_events)
          -> absl::StatusOr<PjRtDeviceEventRef> {
        raw_buffer = std::move(buf_raw_buffer);
        return PjRtDeviceEventRef();
      },
      "CreateRawAliasOfBuffer()"));
  return raw_buffer;
}

static std::optional<absl::StatusOr<PjRtRawBufferRef>>
CommonPjRtBufferImpl_CreateRawAliasOfBuffer(PjRtBuffer* buffer) {
  if (auto* common_buffer = dynamic_cast<CommonPjRtBufferImpl*>(buffer)) {
    return common_buffer->CreateRawAliasOfBuffer();
  }
  return std::nullopt;
}

REGISTER_PJRT_RAW_BUFFER_FACTORY(CommonPjRtBufferImpl_CreateRawAliasOfBuffer);

absl::StatusOr<std::unique_ptr<CommonPjRtBufferImpl::ExternalReference>>
CommonPjRtBufferImpl::AcquireExternalReference() {
  ScopedHold hold = GetBufferWithHold(ScopedHold::kExternalReference);
  RETURN_IF_ERROR(hold.status());

  class ScopedHoldAsExternalReference : public ExternalReference {
   public:
    explicit ScopedHoldAsExternalReference(ScopedHold hold,
                                           PjRtRawBufferRef raw_buffer)
        : external_reference_(std::move(hold)),
          raw_buffer_(std::move(raw_buffer)) {
      CHECK(external_reference_.type() == ScopedHold::kExternalReference);
      if (!raw_buffer_) {
        data_ptr_ = nullptr;
      } else {
        data_ptr_ = raw_buffer_->OpaqueDeviceMemoryDataPointer();
      }
    }

    absl::Status WaitUntilBufferReadyOnStream(std::intptr_t stream) override {
      return external_reference_.buffer()->WaitUntilBufferReadyOnStream(
          raw_buffer_->memory_space(), stream);
    }

    ~ScopedHoldAsExternalReference() override = default;

   private:
    ScopedHold external_reference_;
    PjRtRawBufferRef raw_buffer_;
  };

  auto raw_buffer = hold.buffer()->raw_buffer();
  return std::unique_ptr<ExternalReference>(
      std::make_unique<ScopedHoldAsExternalReference>(std::move(hold),
                                                      std::move(raw_buffer)));
}

Future<> CommonPjRtBufferImpl::CopyRawToHost(void* dst, int64_t offset,
                                             int64_t transfer_size) {
  return CopyRawToHostFuture(Future<void*>(dst), offset, transfer_size);
}

Future<> CommonPjRtBufferImpl::CopyRawToHostFuture(Future<void*> dst,
                                                   int64_t offset,
                                                   int64_t transfer_size) {
  auto buf_client = absl::down_cast<CommonPjRtClient*>(client());
  PjRtDeviceEventRefVector definition_events;
  PjRtRawBufferRef raw_buffer;
  // tsl::RCReference<tsl::IndirectAsyncValue> indirect_usage_event;
  PjRtDeviceEventPromiseRef usage_event_promise;
  PjRtDeviceEventRef usage_event;
  auto hold_status = AcquireScopedRawBuffer(
      [&](PjRtRawBufferRef buf_raw_buffer,
          PjRtDeviceEventRefVector buf_definition_events)
          -> absl::StatusOr<PjRtDeviceEventRef> {
        definition_events = std::move(buf_definition_events);
        if (buf_raw_buffer) {
          auto on_device_size = buf_raw_buffer->GetOnDeviceSizeInBytes();
          if (offset < 0 || offset > on_device_size ||
              on_device_size - offset < transfer_size) {
            return InvalidArgument(
                "Copy raw buffer called on buffer size %lld with "
                "invalid offset %lld, transfer size %lld",
                on_device_size, offset, transfer_size);
          }
          raw_buffer = std::move(buf_raw_buffer);
        }
        ASSIGN_OR_RETURN(
            std::tie(usage_event_promise, usage_event),
            buf_client->CreateLinkedEventPromise(memory_space(), [&]() {
              const auto& current_anno = tsl::profiler::
                  ScopedMemoryDebugAnnotation::CurrentAnnotation();
              std::string op_name =
                  !current_anno.pending_op_name.empty()
                      ? absl::StrCat(" Op:", current_anno.pending_op_name)
                      : "";
              return absl::StrCat("CopyRawSubBufferToHost offset:", offset,
                                  " size:", transfer_size, op_name);
            }));
        return usage_event;
      },
      "CopyRawSubBufferToHost()");
  if (!hold_status.ok()) {
    return Future<>(std::move(hold_status));
  }

  if (buf_client->event_tracking_enabled()) {
    if (!dst.IsReady()) {
      buf_client->RegisterClientThreadWait(
          memory_space(), usage_event_promise.event(), "CopyRawToHostFuture");
    }
    buf_client->AddEventDependencies(
        memory_space(), usage_event_promise.event(), definition_events);
  }

  dst.OnReady([buf_client, transfer_size, offset,
               raw_buffer = std::move(raw_buffer),
               definition_events = std::move(definition_events),
               usage_event_promise = std::move(usage_event_promise)](
                  absl::StatusOr<void*> dst) mutable {
    if (!dst.ok()) {
      usage_event_promise.SetError(dst.status());
      return;
    }

    // We do this before the call to EnqueueWorkWhenReady because we are going
    // to std::move(definition_events) and indirect_usage_event.
    PjRtDeviceEventSpan definition_events_ref = definition_events;
    xla::ExecuteWhenReady(
        definition_events_ref, buf_client->async_work_runner(),
        [dst = *dst, transfer_size, offset, raw_buffer = std::move(raw_buffer),
         definition_events = std::move(definition_events),
         usage_event_promise = std::move(usage_event_promise)]() mutable {
          // Errors in src buffer are surfaced to user.
          for (size_t i = 0; i < definition_events.size(); ++i) {
            const auto& ev = definition_events[i];
            if (auto error = ev.GetErrorIfPresent()) {
              // Signal the usage event to unblock consumers of buffer.
              usage_event_promise.SetError(std::move(*error));
              return;
            }
          }
          auto d2h_event = raw_buffer->CopyRawDeviceToHostAndReturnEvent(
              dst, offset, transfer_size);
          if (!d2h_event.ok()) {
            usage_event_promise.SetError(d2h_event.status());
          } else {
            usage_event_promise.Set(*d2h_event);
          }
        });
  });
  return absl::down_cast<CommonPjRtClient*>(memory_space()->client())
      ->MakeTrackedReadyFuture(usage_event.ptr(), memory_space(),
                               "CommonPjRtBuffer", "CopyRawToHostFuture");
}

absl::StatusOr<Shape> CommonPjRtBufferImpl::logical_on_device_shape() {
  Shape device_shape = on_device_shape();
  if (device_shape.is_static()) {
    StripMetadataForLogicalShape(device_shape);
    return device_shape;
  }
  auto buf_client = absl::down_cast<CommonPjRtClient*>(client());
  auto output_shape = tsl::MakeConstructedAsyncValueRef<Shape>(device_shape);
  RETURN_IF_ERROR(AcquireScopedRawBuffer(
      [&](PjRtRawBufferRef raw_buffer,
          PjRtDeviceEventRefVector definition_events)
          -> absl::StatusOr<PjRtDeviceEventRef> {
        auto ds_kind = client()->GetDynamicShapeKind(memory_space()->kind_id());
        PjRtDeviceEventSpan deps_span(definition_events);
        xla::ExecuteWhenReady(
            deps_span, buf_client->async_work_runner(),
            [definition_events = std::move(definition_events),
             raw_buffer = raw_buffer, output_shape = output_shape,
             device_shape = std::move(device_shape), ds_kind]() mutable {
              tsl::profiler::TraceMe traceme("D2H Read Shape Metadata");
              absl::Status status = xla::GetErrors(definition_events);
              if (!status.ok()) {
                output_shape.SetError(absl::InternalError(
                    absl::StrCat("Cannot read dynamic shape due to error in "
                                 "device buffer: ",
                                 status.message())));
                return;
              }
              xla::ReadDynamicShape(raw_buffer, output_shape, device_shape,
                                    ds_kind);
            });
        tsl::BlockUntilReady(output_shape.CopyRCRef().get());
        if (auto* error = output_shape.GetErrorIfPresent()) {
          return Internal("logical_on_device_shape failed: %s",
                          error->message());
        }

        return PjRtDeviceEventRef();
      },
      "logical_on_device_shape()"));

  return client()->UpdateLayoutForDynamicShapes(memory_space()->kind_id(),
                                                output_shape.get());
}

void CommonPjRtBufferImpl::Delete() {
  VLOG(2) << "CommonPjRtBuffer::Delete (" << this << ") with shape "
          << on_device_shape_->ToString(true);
  if (auto device_buffer = ReleaseBuffer()) {
    device_buffer.release()->Delete(memory_space_);
  }
}

bool CommonPjRtBufferImpl::IsOnCpu() const {
  return absl::down_cast<CommonPjRtClient*>(client())->IsOnCpu(memory_space());
}

CommonPjRtBufferImpl::CommonPjRtBufferImpl(
    std::shared_ptr<const Shape> on_device_shape,
    std::unique_ptr<AbstractTrackedDeviceBuffer> tracked_device_buffer,
    PjRtMemorySpace* memory_space)
    : CommonPjRtBuffer(std::move(tracked_device_buffer), memory_space),
      on_device_shape_(std::move(on_device_shape)) {}

CommonPjRtBufferImpl::~CommonPjRtBufferImpl() { Delete(); }

PjRtDevice* CommonPjRtBufferImpl::device() const {
  CHECK_EQ(memory_space_->devices().size(), 1);
  return absl::down_cast<PjRtDevice*>(memory_space_->devices()[0]);
}

CommonPjRtClient* CommonPjRtBufferImpl::client() const {
  return absl::down_cast<CommonPjRtClient*>(memory_space()->client());
}

absl::StatusOr<size_t> CommonPjRtBufferImpl::GetOnDeviceSizeInBytes() const {
  return client()->GetOnDeviceBytesCount(memory_space(), *on_device_shape_);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
CommonPjRtBufferImpl::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  if (on_device_shape_->IsTuple()) {
    return InvalidArgument(
        "ReleaseDeviceMemoryOwnership allowed only for non-tuple");
  }
  auto device_buffer = ReleaseBuffer();
  if (device_buffer == nullptr) {
    return {nullptr};
  }

  if (wait_for_operations_to_complete) {
    RETURN_IF_ERROR(device_buffer->BlockForOperationsToComplete(memory_space_));
  }

  class RawBufferAsExternalReference : public PjRtBuffer::ExternalReference {
   public:
    explicit RawBufferAsExternalReference(PjRtRawBufferRef raw_buffer)
        : raw_buffer_(std::move(raw_buffer)) {
      if (!raw_buffer_) {
        data_ptr_ = nullptr;
      } else {
        data_ptr_ = raw_buffer_->OpaqueDeviceMemoryDataPointer();
      }
    }

    ~RawBufferAsExternalReference() override = default;

   private:
    PjRtRawBufferRef raw_buffer_;
  };

  std::unique_ptr<PjRtBuffer::ExternalReference> ref;
  if (device_buffer) {
    ref = std::make_unique<RawBufferAsExternalReference>(
        device_buffer->raw_buffer());
  }
  return ref;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtBufferImpl::DonateWithControlDependency(Future<> dependency) {
  ASSIGN_OR_RETURN(auto extra_definition_event,
                   client()->CreateDeviceEvent(memory_space(), dependency));
  absl::StatusOr<std::unique_ptr<AbstractTrackedDeviceBuffer>> tracked_buffer =
      DonateTrackedBuffer();
  if (!tracked_buffer.ok()) {
    return InvalidArgument(
        "Invalid buffer passed to DonateWithControlDependency: %s",
        tracked_buffer.status().ToString());
  }
  (*tracked_buffer)
      ->UnsafePrependDefinitionEvent(std::move(extra_definition_event));

  return std::make_unique<CommonPjRtBufferImpl>(
      on_device_shape_, std::move(*tracked_buffer), memory_space());
}

Future<> CommonPjRtBufferImpl::GetReadyFuture() {
  absl::MutexLock lock(mu_);
  if (!device_buffer()) {
    return Future<>(InvalidArgument(
        "GetReadyFuture() called on deleted or donated buffer"));
  }
  if (!definition_future_) {
    auto future = device_buffer()->GetReadyFuture(memory_space());
    definition_future_ = client()->CreateProfiledFuture(
        memory_space(), "CommonPjRtBuffer", "Await", std::move(future));
  }
  return definition_future_;
}

}  // namespace xla
