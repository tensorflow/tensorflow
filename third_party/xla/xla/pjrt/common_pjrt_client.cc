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
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/lib/scoped_memory_debug_annotation.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

std::pair<PjRtFuture<>::Promise, PjRtFuture<>>
CommonPjRtClient::CreateLinkedUserPromise(PjRtMemorySpace* memory_space,
                                          const char* callee_type,
                                          const char* callee_method,
                                          absl::string_view debug_info) {
  PjRtFuture<>::Promise promise = PjRtFuture<>::CreatePromise();
  auto result = PjRtFuture<>(
      promise,
      /*on_block_start=*/
      [ready_event = FormRef(promise.async_value()), callee_type,
       callee_method]() {
        tsl::profiler::TraceMeProducer traceme(
            [&] { return absl::StrCat(callee_type, "::", callee_method); });
        VLOG(1) << callee_type << "::" << callee_method;
        PjRtFutureHelpers::ProfilingKeys keys;
        keys.traceme_context_id = traceme.GetContextId();
        return keys;
      },
      /*on_block_end=*/
      [callee_type, callee_method](PjRtFutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme(
            [&] { return absl::StrCat(callee_type, "::", callee_method); },
            keys.traceme_context_id);
      });
  return std::make_pair(std::move(promise), std::move(result));
}

tsl::AsyncValueRef<bool> CommonPjRtClient::CreateAllocationEventForTransfers(
    PjRtMemorySpace* memory_space,
    const std::optional<std::string>& debug_info) {
  return tsl::AsyncValueRef<bool>();
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
  TF_ASSIGN_OR_RETURN(
      Shape device_shape,
      MakeDefaultShapeForMemorySpace(memory_space, shape, device_layout));
  TF_ASSIGN_OR_RETURN(
      auto promise_and_event,
      CreateLinkedEventPromise(memory_space, "BufferFromHostLiteral"));
  tsl::RCReference<CommonPjRtRawBuffer> raw_buffer;
  std::unique_ptr<PjRtBuffer> output_buffer;
  absl::Status s = [&]() {
    TF_ASSIGN_OR_RETURN(int64_t on_device_bytes_count,
                        GetOnDeviceBytesCount(memory_space, device_shape));
    TF_ASSIGN_OR_RETURN(raw_buffer,
                        AllocateRawBuffer(memory_space, on_device_bytes_count,
                                          /*retry_on_oom=*/true,
                                          /*allocate_after=*/{}));
    TF_ASSIGN_OR_RETURN(output_buffer,
                        DefineBuffer(device_shape, raw_buffer,
                                     {std::move(promise_and_event.second)},
                                     /*raw_buffer_is_mutable=*/true));
    return absl::OkStatus();
  }();
  if (!s.ok()) {
    promise_and_event.first->SetError(s);
    return s;
  }

  async_work_runner()->Schedule(
      [this, shape, literal, raw_buffer = std::move(raw_buffer),
       definition_event = std::move(promise_and_event.first),
       device_layout = device_shape.layout(),
       context_id = producer.GetContextId()]() mutable {
        tsl::profiler::TraceMeConsumer consumer(
            "BufferFromHostLiteral H2D Dispatch",
            tsl::profiler::ContextType::kPjRt, context_id);
        auto status_or_h2d_transfer_event =
            LinearizeInto(literal, device_layout, std::move(raw_buffer));
        CHECK_OK(status_or_h2d_transfer_event);
        auto h2d_transfer_event = *std::move(status_or_h2d_transfer_event);
        if (event_tracking_enabled()) {
          h2d_transfer_event->AppendDescriptionToEvent(
              " TransferToDevice ", {definition_event.get()});
        }
        definition_event->Set(std::move(h2d_transfer_event));
      });
  return output_buffer;
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
      TF_ASSIGN_OR_RETURN(
          device_shape,
          MakeDefaultShapeForMemorySpace(memory_space, shape, &shape.layout()));
    } else {
      TF_ASSIGN_OR_RETURN(device_shape, MakeDefaultShapeForMemorySpace(
                                            memory_space, shape, nullptr));
    }
  }
  TF_ASSIGN_OR_RETURN(int64_t on_device_bytes_count,
                      GetOnDeviceBytesCount(memory_space, device_shape));
  TF_ASSIGN_OR_RETURN(auto raw_buffer,
                      AllocateRawBuffer(memory_space, on_device_bytes_count,
                                        /*retry_on_oom=*/true,
                                        /*allocate_after=*/{}));
  TF_ASSIGN_OR_RETURN(auto definition_event,
                      raw_buffer->MakeAllocationReadyEvent());
  TF_ASSIGN_OR_RETURN(
      auto output_buffer,
      DefineBuffer(device_shape, raw_buffer, {std::move(definition_event)},
                   /*raw_buffer_is_mutable=*/true));
  return output_buffer;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtMemorySpace* memory_space, const Layout* device_layout) {
  TF_ASSIGN_OR_RETURN(const Shape shape,
                      ShapeUtil::MakeValidatedShape(type, dims));
  TF_ASSIGN_OR_RETURN(
      Shape device_shape,
      MakeDefaultShapeForMemorySpace(memory_space, shape, device_layout));
  if (host_buffer_semantics ==
          PjRtClient::HostBufferSemantics::kImmutableZeroCopy ||
      host_buffer_semantics ==
          PjRtClient::HostBufferSemantics::kMutableZeroCopy) {
    if (BufferFromHostBufferSupportsZeroCopy(data, type, dims, byte_strides,
                                             device_shape, memory_space,
                                             device_layout)) {
      TF_ASSIGN_OR_RETURN(int64_t on_device_bytes_count,
                          GetOnDeviceBytesCount(memory_space, device_shape));
      TF_ASSIGN_OR_RETURN(
          auto raw_buffer,
          ImportForeignMemory(
              const_cast<void*>(data),  // CONST_CAST_OK=flag controlled.
              std::move(on_done_with_host_buffer), on_device_bytes_count,
              memory_space));
      TF_ASSIGN_OR_RETURN(
          auto output_buffer,
          DefineBuffer(
              device_shape, raw_buffer,
              absl::InlinedVector<tsl::RCReference<PjRtDeviceEvent>, 4>{},
              /*raw_buffer_is_mutable=*/host_buffer_semantics ==
                  PjRtClient::HostBufferSemantics::kMutableZeroCopy));
      return output_buffer;
    }
  }

  TF_ASSIGN_OR_RETURN(int64_t on_device_bytes_count,
                      GetOnDeviceBytesCount(memory_space, device_shape));
  TF_ASSIGN_OR_RETURN(auto raw_buffer,
                      AllocateRawBuffer(memory_space, on_device_bytes_count,
                                        /*retry_on_oom=*/true,
                                        /*allocate_after=*/{}));
  TF_ASSIGN_OR_RETURN(
      auto definition_event,
      LinearizeHostBufferInto(
          data, type, dims, byte_strides, host_buffer_semantics,
          std::move(on_done_with_host_buffer), device_shape, raw_buffer));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtBuffer> output_buffer,
      DefineBuffer(device_shape, raw_buffer, {std::move(definition_event)},
                   /*raw_buffer_is_mutable=*/true));
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
  TF_ASSIGN_OR_RETURN(Shape device_shape, MakeDefaultShapeForMemorySpace(
                                              memory_space, shape, nullptr));
  TF_ASSIGN_OR_RETURN(int64_t on_device_bytes_count,
                      GetOnDeviceBytesCount(memory_space, device_shape));
  TF_ASSIGN_OR_RETURN(
      auto raw_buffer,
      ImportForeignMemory(device_ptr, std::move(on_delete_callback),
                          on_device_bytes_count, memory_space));
  TF_ASSIGN_OR_RETURN(
      auto output_buffer,
      DefineBuffer(device_shape, raw_buffer,
                   absl::InlinedVector<tsl::RCReference<PjRtDeviceEvent>, 4>{},
                   /*raw_buffer_is_mutable=*/false));
  return output_buffer;
}

absl::StatusOr<xla::Shape> CommonPjRtClient::MakeDefaultShapeForMemorySpace(
    PjRtMemorySpace* memory_space, xla::Shape shape,
    const xla::Layout* layout) const {
  if (layout) {
    *shape.mutable_layout() = *layout;
  } else {
    TF_ASSIGN_OR_RETURN(
        *shape.mutable_layout(),
        (*GetTopologyDescription())
            ->GetDefaultLayout(shape.element_type(), shape.dimensions()));
  }
  return shape;
}

void CommonPjRtBufferImpl::CopyToRemoteDevice(
    PjRtFuture<std::string> serialized_descriptor, RemoteSendCallback on_done) {
  auto* common_client = tensorflow::down_cast<CommonPjRtClient*>(client());
  std::vector<tsl::RCReference<tsl::AsyncValue>> definition_events;
  tsl::RCReference<PjRtDeviceEventPromise> usage_event_promise;
  tsl::RCReference<CommonPjRtRawBuffer> raw_buffer;
  auto hold_status = AcquireScopedRawBuffer(
      [&](tsl::RCReference<CommonPjRtRawBuffer> buf_raw_buffer,
          std::vector<tsl::RCReference<tsl::AsyncValue>> buf_definition_events)
          -> absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> {
        raw_buffer = std::move(buf_raw_buffer);
        definition_events = std::move(buf_definition_events);
        tsl::RCReference<PjRtDeviceEvent> usage_event;
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
          TF_ASSIGN_OR_RETURN(
              std::tie(usage_event_promise, usage_event),
              common_client->CreateLinkedEventPromise(
                  memory_space(), absl::StrCat("RemoteSend", op_name)));
        } else {
          TF_ASSIGN_OR_RETURN(std::tie(usage_event_promise, usage_event),
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

void CommonPjRtClient::ScheduleRemoteSend(
    PjRtMemorySpace* memory_space,
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
    std::vector<tsl::RCReference<tsl::AsyncValue>> definition_events,
    tsl::RCReference<PjRtDeviceEventPromise> usage_event_promise,
    PjRtFuture<std::string> serialized_descriptor,
    PjRtBuffer::RemoteSendCallback on_done) {
  auto error = absl::UnimplementedError(
      absl::StrCat("ScheduleRemoteSend is not implemented for %s",
                   memory_space->DebugString()));
  on_done(error, /*sends_were_enqueued=*/false);
  usage_event_promise->SetError(error);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtBufferImpl::DirectCopyToMemorySpace(
    PjRtMemorySpace* dst_memory_space) {
  tsl::profiler::TraceMe traceme("CopyToMemorySpace");
  auto* src_memory_space = memory_space();
  CommonPjRtClient* const src_client =
      tensorflow::down_cast<CommonPjRtClient*>(client());
  CommonPjRtClient* const dst_client =
      dynamic_cast<CommonPjRtClient*>(dst_memory_space->client());
  if (!dst_client) {
    return absl::InvalidArgumentError(
        "DirectCopyToMemorySpace only supported across CommonPjRtClient "
        "subclassed clients");
  }
  TF_ASSIGN_OR_RETURN(const int64_t on_device_bytes_count,
                      GetOnDeviceSizeInBytes());

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

  auto allocation_event = dst_client->CreateAllocationEventForTransfers(
      dst_memory_space, debug_info);
  tsl::RCReference<PjRtDeviceEventPromise> definition_event_promise;
  tsl::RCReference<PjRtDeviceEvent> definition_event;
  if (dst_client->event_tracking_enabled()) {
    TF_ASSIGN_OR_RETURN(
        std::tie(definition_event_promise, definition_event),
        dst_client->CreateLinkedEventPromise(
            dst_memory_space,
            absl::StrCat("CopyToMemorySpace CrossDeviceSink: ", transfer_id,
                         " Op:", debug_info.value_or(""))));
  } else {
    TF_ASSIGN_OR_RETURN(
        std::tie(definition_event_promise, definition_event),
        dst_client->CreateLinkedEventPromise(dst_memory_space, ""));
  }

  tsl::RCReference<PjRtDeviceEventPromise> src_usage_event_promise;
  tsl::RCReference<CommonPjRtRawBuffer> src_raw_buffer;
  tsl::RCReference<CommonPjRtRawBuffer> dst_raw_buffer;
  std::unique_ptr<PjRtBuffer> dst_buffer;
  std::vector<tsl::RCReference<tsl::AsyncValue>> definition_events;
  auto status = [&]() -> absl::Status {
    TF_ASSIGN_OR_RETURN(
        dst_raw_buffer,
        dst_client->AllocateRawBuffer(dst_memory_space, on_device_bytes_count,
                                      /*retry_on_oom=*/true, allocation_event));
    TF_ASSIGN_OR_RETURN(
        dst_buffer, dst_client->DefineBuffer(on_device_shape(), dst_raw_buffer,
                                             {std::move(definition_event)},
                                             /*raw_buffer_is_mutable=*/true));
    TF_RETURN_IF_ERROR(AcquireScopedRawBuffer(
        [&](tsl::RCReference<CommonPjRtRawBuffer> buf_raw_buffer,
            std::vector<tsl::RCReference<tsl::AsyncValue>>
                buf_definition_events)
            -> absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> {
          src_raw_buffer = std::move(buf_raw_buffer);
          tsl::RCReference<PjRtDeviceEvent> usage_event;
          definition_events = std::move(buf_definition_events);
          if (src_client->event_tracking_enabled()) {
            TF_ASSIGN_OR_RETURN(
                std::tie(src_usage_event_promise, usage_event),
                dst_client->CreateLinkedEventPromise(
                    src_memory_space,
                    absl::StrCat(
                        "CopyToMemorySpace CrossDeviceSrc: ", transfer_id,
                        " Op:", debug_info.value_or(""))));
          } else {
            TF_ASSIGN_OR_RETURN(
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
    definition_event_promise->SetError(status);
    return status;
  }

  if (src_raw_buffer) {
    src_raw_buffer->ScheduleCopyTo(
        src_client->async_work_runner(), std::move(definition_events),
        std::move(dst_raw_buffer), std::move(definition_event_promise),
        std::move(src_usage_event_promise), std::move(allocation_event));
  } else {
    absl::Span<const tsl::RCReference<tsl::AsyncValue>> definition_events_span =
        definition_events;
    src_client->async_work_runner()->ScheduleWhenReady(
        definition_events_span,
        [dst_raw_buffer = std::move(dst_raw_buffer),
         definition_events = std::move(definition_events),
         definition_event_promise = std::move(definition_event_promise),
         src_usage_event_promise = std::move(src_usage_event_promise),
         allocation_event = std::move(allocation_event)]() {
          auto set_error = [&](absl::Status status) {
            if (allocation_event) {
              allocation_event.SetError(status);
            }
            definition_event_promise->SetError(status);
            src_usage_event_promise->SetError(status);
          };
          for (const auto& av : definition_events) {
            if (auto* error = av->GetErrorIfPresent()) {
              set_error(*error);
              return;
            }
          }
          set_error(
              absl::InternalError("src_raw_buffer is nullptr for copy but no "
                                  "definition events were errors."));
        });
  }

  return dst_buffer;
}

PjRtFuture<> CommonPjRtBufferImpl::LazyToLiteral(
    absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator) {
  return ToLiteralImpl(nullptr, std::move(generator));
}

PjRtFuture<> CommonPjRtBufferImpl::ToLiteral(MutableLiteralBase* literal) {
  return ToLiteralImpl(literal, [] {
    return FailedPrecondition("ToLiteral generator should never be called");
  });
}

PjRtFuture<> CommonPjRtBufferImpl::ToLiteralImpl(
    MutableLiteralBase* literal,
    absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator) {
  tsl::profiler::TraceMe traceme("CommonPjRtBuffer::ToLiteral");
  VLOG(1) << "CommonPjRtBuffer::ToLiteral";
  auto common_client = tensorflow::down_cast<CommonPjRtClient*>(client());
  if (!common_client->allows_recursion() && ThisThreadIsInsideHostCallback()) {
    // Because TPU is single threaded, and the host callback currently blocking
    // the TPU, we should not block on any outstanding computations because that
    // risks deadlocking the TPU.
    return PjRtFuture<>(
        InvalidArgument("ToLiteral() called from inside host callback."));
  }
  absl::StatusOr<Shape> device_shape = logical_on_device_shape();
  if (!device_shape.ok()) {
    return PjRtFuture<>(device_shape.status());
  }

  // TODO(zhangqiaorjc): Fast path if zero device_buffer wait events.
  // Make two copies because EnqueueWorkWhenReady below needs two different
  // lifetimes.
  std::vector<tsl::RCReference<tsl::AsyncValue>> src_definition_events_avs;

  tsl::RCReference<PjRtDeviceEventPromise> device_promise;
  tsl::RCReference<CommonPjRtRawBuffer> raw_buffer;
  auto hold_status = AcquireScopedRawBuffer(
      [&](tsl::RCReference<CommonPjRtRawBuffer> buf_raw_buffer,
          std::vector<tsl::RCReference<tsl::AsyncValue>> definition_events)
          -> absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> {
        src_definition_events_avs = std::move(definition_events);
        if (buf_raw_buffer) {
          raw_buffer = std::move(buf_raw_buffer);
          tsl::RCReference<PjRtDeviceEvent> device_event;
          TF_ASSIGN_OR_RETURN(std::tie(device_promise, device_event),
                              common_client->CreateLinkedEventPromise(
                                  memory_space_, "ToLiteral Leaf: 0"));
          return device_event;
        }
        return tsl::RCReference<PjRtDeviceEvent>();
      },
      "ToLiteral()");
  if (!hold_status.ok()) {
    return PjRtFuture<>(std::move(hold_status));
  }

  auto [promise, result] = common_client->CreateLinkedUserPromise(
      memory_space(), "CommonPjRtBuffer", "ToLiteral", "ToLiteralEvent");
  if (device_promise) {
    device_promise->AddEventDependencies(src_definition_events_avs);
  }

  // Wait for buffer definition events to finish before d2h dispatch.
  // D2H dispatch should be in parallel, e.g. one Execute event finish may
  // trigger multiple outputs' D2H, they should happen in different threads in
  // parallel.
  absl::Span<const tsl::RCReference<tsl::AsyncValue>>
      src_definition_events_avs_copy = src_definition_events_avs;
  common_client->async_work_runner()->ScheduleWhenReady(
      src_definition_events_avs_copy,
      [shape = *std::move(device_shape),
       src_definition_events_avs = std::move(src_definition_events_avs),
       raw_buffer = std::move(raw_buffer),
       device_promise = std::move(device_promise), literal,
       generator = std::move(generator),
       promise = std::move(promise)]() mutable {
        // Notify all pending events with `status`.
        auto notify_all = [&](absl::Status status) {
          promise.Set(status);
          if (device_promise) {
            device_promise->SetError(status);
          }
        };
        tsl::profiler::TraceMe traceme([&] {
          return tsl::profiler::TraceMeEncode(
              "D2H Dispatch",
              {{"shape", shape.ToString(/*print_layout=*/true)}});
        });
        if (literal == nullptr) {
          absl::StatusOr<MutableLiteralBase*> generated =
              std::move(generator)();
          if (!generated.ok()) {
            notify_all(generated.status());
            return;
          }
          literal = *generated;
        }
        DCHECK(ShapeUtil::Compatible(shape, literal->shape()));
        // Errors in src buffer are surfaced to user.
        for (const auto& av : src_definition_events_avs) {
          if (auto* error = av->GetErrorIfPresent()) {
            notify_all(*error);
            return;
          }
        }

        raw_buffer->CopyToLiteralAsync(promise, device_promise, literal,
                                       std::move(shape));
      });
  return result;
}

absl::StatusOr<tsl::RCReference<PjRtRawBuffer>>
CommonPjRtBufferImpl::CreateRawAliasOfBuffer() {
  tsl::RCReference<CommonPjRtRawBuffer> raw_buffer;
  TF_RETURN_IF_ERROR(AcquireScopedRawBuffer(
      [&](tsl::RCReference<CommonPjRtRawBuffer> buf_raw_buffer,
          std::vector<tsl::RCReference<tsl::AsyncValue>> definition_events)
          -> absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> {
        raw_buffer = std::move(buf_raw_buffer);
        return tsl::RCReference<PjRtDeviceEvent>();
      },
      "CreateRawAliasOfBuffer()"));
  return raw_buffer;
}

static std::optional<absl::StatusOr<tsl::RCReference<PjRtRawBuffer>>>
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
  TF_RETURN_IF_ERROR(hold.status());

  class ScopedHoldAsExternalReference : public ExternalReference {
   public:
    explicit ScopedHoldAsExternalReference(
        ScopedHold hold, tsl::RCReference<CommonPjRtRawBuffer> raw_buffer)
        : external_reference_(std::move(hold)),
          raw_buffer_(std::move(raw_buffer)) {
      CHECK(external_reference_.type() == ScopedHold::kExternalReference);
      if (!raw_buffer_) {
        data_ptr_ = nullptr;
      } else {
        data_ptr_ = raw_buffer_->OpaqueDeviceMemoryDataPointer();
      }
    }

    ~ScopedHoldAsExternalReference() override = default;

   private:
    ScopedHold external_reference_;
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer_;
  };

  auto raw_buffer = hold.buffer()->GetRawBuffer(memory_space_);
  return std::unique_ptr<ExternalReference>(
      std::make_unique<ScopedHoldAsExternalReference>(std::move(hold),
                                                      std::move(raw_buffer)));
}

PjRtFuture<> CommonPjRtBufferImpl::CopyRawToHost(void* dst, int64_t offset,
                                                 int64_t transfer_size) {
  return CopyRawToHostFuture(PjRtFuture<void*>(dst), offset, transfer_size);
}

PjRtFuture<> CommonPjRtBufferImpl::CopyRawToHostFuture(PjRtFuture<void*> dst,
                                                       int64_t offset,
                                                       int64_t transfer_size) {
  auto buf_client = tensorflow::down_cast<CommonPjRtClient*>(client());
  std::vector<tsl::RCReference<tsl::AsyncValue>> definition_events;
  tsl::RCReference<CommonPjRtRawBuffer> raw_buffer;
  // tsl::RCReference<tsl::IndirectAsyncValue> indirect_usage_event;
  tsl::RCReference<PjRtDeviceEventPromise> usage_event_promise;
  tsl::RCReference<PjRtDeviceEvent> usage_event;
  auto hold_status = AcquireScopedRawBuffer(
      [&](tsl::RCReference<CommonPjRtRawBuffer> buf_raw_buffer,
          std::vector<tsl::RCReference<tsl::AsyncValue>> buf_definition_events)
          -> absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> {
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
        TF_ASSIGN_OR_RETURN(
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
    return PjRtFuture<>(std::move(hold_status));
  }

  if (buf_client->event_tracking_enabled()) {
    if (!dst.IsReady()) {
      usage_event_promise->RegisterClientThreadWait("CopyRawToHostFuture");
    }
    usage_event_promise->AddEventDependencies(definition_events);
  }

  dst.OnReady([buf_client, transfer_size, offset,
               raw_buffer = std::move(raw_buffer),
               definition_events = std::move(definition_events),
               usage_event_promise = std::move(usage_event_promise)](
                  absl::StatusOr<void*> dst) mutable {
    if (!dst.ok()) {
      usage_event_promise->SetError(dst.status());
      return;
    }

    // We do this before the call to EnqueueWorkWhenReady because we are going
    // to std::move(definition_events) and indirect_usage_event.
    absl::Span<const tsl::RCReference<tsl::AsyncValue>> definition_events_ref =
        definition_events;
    buf_client->async_work_runner()->ScheduleWhenReady(
        definition_events_ref,
        [dst = *dst, transfer_size, offset, raw_buffer = std::move(raw_buffer),
         definition_events = std::move(definition_events),
         usage_event_promise = std::move(usage_event_promise)]() mutable {
          // Errors in src buffer are surfaced to user.
          for (const auto& av : definition_events) {
            if (auto* error = av->GetErrorIfPresent()) {
              // Signal the usage event to unblock consumers of buffer.
              usage_event_promise->SetError(*error);
              return;
            }
          }
          auto d2h_event = raw_buffer->CopyRawDeviceToHostAndReturnEvent(
              dst, offset, transfer_size);
          if (!d2h_event.ok()) {
            usage_event_promise->SetError(d2h_event.status());
          } else {
            usage_event_promise->Set(*d2h_event);
          }
        });
  });
  return usage_event->GetReadyFuture();
}

absl::StatusOr<Shape> CommonPjRtBufferImpl::logical_on_device_shape() {
  Shape device_shape = on_device_shape();
  if (device_shape.is_static()) {
    return device_shape;
  }
  auto buf_client = tensorflow::down_cast<CommonPjRtClient*>(client());
  auto output_shape = tsl::MakeConstructedAsyncValueRef<Shape>(device_shape);
  TF_RETURN_IF_ERROR(AcquireScopedRawBuffer(
      [&](tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
          std::vector<tsl::RCReference<tsl::AsyncValue>> definition_events)
          -> absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> {
        absl::Span<const tsl::RCReference<tsl::AsyncValue>>
            definition_events_ref = definition_events;
        buf_client->async_work_runner()->ScheduleWhenReady(
            definition_events_ref,
            [definition_events = std::move(definition_events),
             raw_buffer = raw_buffer, output_shape = output_shape,
             device_shape = std::move(device_shape)]() mutable {
              tsl::profiler::TraceMe traceme("D2H Read Shape Metadata");
              // Errors in src buffer are surfaced to user.
              for (const auto& av : definition_events) {
                if (auto* error = av->GetErrorIfPresent()) {
                  output_shape.SetError(absl::InternalError(
                      absl::StrCat("Cannot read dynamic shape due to error in "
                                   "device buffer: ",
                                   error->message())));
                  return;
                }
              }
              raw_buffer->ReadDynamicShape(output_shape,
                                           std::move(device_shape));
            });
        tsl::BlockUntilReady(output_shape.CopyRCRef().get());
        if (auto* error = output_shape.GetErrorIfPresent()) {
          return Internal("logical_on_device_shape failed: %s",
                          error->message());
        }

        return tsl::RCReference<PjRtDeviceEvent>();
      },
      "logical_on_device_shape()"));

  return output_shape.get();
}

void CommonPjRtBufferImpl::Delete() {
  VLOG(2) << "CommonPjRtBuffer::Delete (" << this << ") with shape "
          << on_device_shape().ToString(true) << " and size "
          << GetOnDeviceSizeInBytes().value_or(0);
  if (auto device_buffer = ReleaseBuffer()) {
    device_buffer.release()->Delete(memory_space_);
  }
}

}  // namespace xla
