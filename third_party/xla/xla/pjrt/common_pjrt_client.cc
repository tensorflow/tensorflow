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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
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
  TF_ASSIGN_OR_RETURN(int64_t on_device_bytes_count,
                      GetOnDeviceBytesCount(memory_space, device_shape));
  TF_ASSIGN_OR_RETURN(auto raw_buffer,
                      AllocateRawBuffer(memory_space, on_device_bytes_count,
                                        /*retry_on_oom=*/true,
                                        /*allocate_after=*/{}));
  TF_ASSIGN_OR_RETURN(auto output_buffer,
                      DefineBuffer(device_shape, raw_buffer,
                                   {std::move(promise_and_event.second)},
                                   /*raw_buffer_is_mutable=*/true));

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
  TF_ASSIGN_OR_RETURN(
      Shape device_shape,
      MakeDefaultShapeForMemorySpace(
          memory_space, ShapeUtil::MakeShape(type, dims), device_layout));
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

  auto allocation_event = dst_client->CreateAllocationEventForTransfers(
      dst_memory_space, debug_info);
  tsl::RCReference<PjRtDeviceEventPromise> definition_event_promise;
  tsl::RCReference<PjRtDeviceEvent> definition_event;
  if (dst_client->event_tracking_enabled()) {
    TF_ASSIGN_OR_RETURN(
        std::tie(definition_event_promise, definition_event),
        dst_client->CreateLinkedEventPromise(
            dst_memory_space, absl::StrCat("CopyToMemorySpace_H2D Op:",
                                           debug_info.value_or(""))));
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
                    src_memory_space, absl::StrCat("CopyToMemorySpace_D2H Op:",
                                                   debug_info.value_or(""))));
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
    definition_event_promise->SetError(status);
    return status;
  }

  absl::Span<const tsl::RCReference<tsl::AsyncValue>> definition_events_span =
      definition_events;
  src_client->async_work_runner()->ScheduleWhenReady(
      definition_events_span,
      [src_raw_buffer = std::move(src_raw_buffer),
       dst_raw_buffer = std::move(dst_raw_buffer),
       definition_events = std::move(definition_events),
       definition_event_promise = std::move(definition_event_promise),
       src_usage_event_promise = std::move(src_usage_event_promise),
       allocation_event = std::move(allocation_event)]() {
        for (const auto& av : definition_events) {
          if (auto* error = av->GetErrorIfPresent()) {
            auto status = *error;
            if (allocation_event) {
              allocation_event.SetError(status);
            }
            definition_event_promise->SetError(status);
            src_usage_event_promise->SetError(status);
            return;
          }
        }

        src_raw_buffer->CopyTo(
            std::move(dst_raw_buffer), std::move(definition_event_promise),
            std::move(src_usage_event_promise), std::move(allocation_event));
      });

  return dst_buffer;
}

}  // namespace xla
