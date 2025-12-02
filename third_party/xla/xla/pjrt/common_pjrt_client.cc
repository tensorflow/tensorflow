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
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/pjrt_client.h"
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

void CommonPjRtClient::TrackFuture(PjRtMemorySpace* memory_space,
                                   absl::string_view debug_info,
                                   const Future<>& future) {}

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
  auto [promise, future] = Future<>::MakePromise();
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
  TF_ASSIGN_OR_RETURN(
      Shape device_shape,
      MakeDefaultShapeForMemorySpace(memory_space, shape, device_layout));
  TF_ASSIGN_OR_RETURN(int64_t on_device_bytes_count,
                      GetOnDeviceBytesCount(memory_space, device_shape));
  TF_ASSIGN_OR_RETURN(auto raw_buffer,
                      AllocateRawBuffer(memory_space, on_device_bytes_count,
                                        /*retry_on_oom=*/true,
                                        /*allocate_after=*/{}));
  TF_ASSIGN_OR_RETURN(
      auto definition_event,
      LinearizeInto(literal, device_shape,
                    HostBufferSemantics::kImmutableUntilTransferCompletes,
                    raw_buffer));
  return DefineBuffer(device_shape, memory_space, std::move(raw_buffer),
                      {std::move(definition_event)},
                      /*raw_buffer_is_mutable=*/true);
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
  TF_ASSIGN_OR_RETURN(auto output_buffer,
                      DefineBuffer(device_shape, memory_space, raw_buffer,
                                   {std::move(definition_event)},
                                   /*raw_buffer_is_mutable=*/true));
  return output_buffer;
}

absl::StatusOr<
    std::pair<std::unique_ptr<PjRtBuffer>, PjRtFulfillAliasBufferCallback>>
CommonPjRtClient::CreateAliasBuffer(const Shape& shape,
                                    PjRtMemorySpace* memory_space) {
  tsl::RCReference<CommonPjRtRawBuffer> raw_buffer;
  PjRtFulfillAliasRawBufferCallback buffer_promise;

  TF_ASSIGN_OR_RETURN(std::tie(raw_buffer, buffer_promise),
                      CreateRawBufferChannel(memory_space));

  tsl::RCReference<xla::PjRtDeviceEventPromise> definition_event_promise;
  tsl::RCReference<xla::PjRtDeviceEvent> definition_event;
  TF_ASSIGN_OR_RETURN(
      std::tie(definition_event_promise, definition_event),
      CreateLinkedEventPromise(memory_space, "CreateRawBufferChannel"));

  PjRtFulfillAliasBufferCallback fulfill_cb =
      [buffer_promise = std::move(buffer_promise),
       definition_event_promise = std::move(definition_event_promise),
       memory_space,
       shape](absl::StatusOr<xla::PjRtBuffer*> buffer_or) mutable {
        if (!buffer_or.ok()) {
          definition_event_promise->SetError(buffer_or.status());
          std::move(buffer_promise)(buffer_or.status()).IgnoreError();
          return buffer_or.status();
        }
        xla::PjRtBuffer* buffer = buffer_or.value();
        if (buffer->on_device_shape() != shape) {
          auto status = absl::InvalidArgumentError(absl::StrFormat(
              "Shape mismatch in CreateRawBufferChannel fulfill: expected %s, "
              "got %s",
              shape.ToString(), buffer->on_device_shape().ToString()));
          definition_event_promise->SetError(status);
          std::move(buffer_promise)(status).IgnoreError();
          return status;
        }
        xla::CommonPjRtBuffer* common_buffer =
            dynamic_cast<xla::CommonPjRtBuffer*>(buffer);
        if (common_buffer == nullptr) {
          auto status =
              absl::InternalError("Failed to cast to CommonPjRtBuffer");
          definition_event_promise->SetError(status);
          std::move(buffer_promise)(status).IgnoreError();
          return status;
        }
        xla::CommonPjRtBuffer::ScopedHold hold =
            common_buffer->GetBufferWithHold(
                xla::CommonPjRtBuffer::ScopedHold::kDonation);
        auto device_event_or = hold.buffer()->GetDefinitionEvent(memory_space);
        if (!device_event_or.ok()) {
          auto status = device_event_or.status();
          definition_event_promise->SetError(status);
          std::move(buffer_promise)(status).IgnoreError();
          return status;
        }
        auto status = std::move(buffer_promise)(hold.buffer()->raw_buffer());
        if (!status.ok()) {
          definition_event_promise->SetError(status);
          return status;
        }

        definition_event_promise->Set(std::move(*device_event_or));
        hold.ConfirmDonation();
        return absl::OkStatus();
      };

  TF_ASSIGN_OR_RETURN(auto result_buffer,
                      DefineBuffer(shape, memory_space, std::move(raw_buffer),
                                   {std::move(definition_event)},
                                   /*raw_buffer_is_mutable=*/true));

  return std::make_pair(std::move(result_buffer), std::move(fulfill_cb));
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
              device_shape, memory_space, raw_buffer,
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
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> output_buffer,
                      DefineBuffer(device_shape, memory_space, raw_buffer,
                                   {std::move(definition_event)},
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
  TF_ASSIGN_OR_RETURN(
      Shape device_shape,
      MakeDefaultShapeForMemorySpace(
          memory_space, shape, shape.has_layout() ? &shape.layout() : nullptr));
  TF_ASSIGN_OR_RETURN(int64_t on_device_bytes_count,
                      GetOnDeviceBytesCount(memory_space, device_shape));
  TF_ASSIGN_OR_RETURN(
      auto raw_buffer,
      ImportForeignMemory(device_ptr, std::move(on_delete_callback),
                          on_device_bytes_count, memory_space));
  TF_ASSIGN_OR_RETURN(
      auto output_buffer,
      DefineBuffer(device_shape, memory_space, raw_buffer,
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
    Future<std::string> serialized_descriptor, RemoteSendCallback on_done) {
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
    Future<std::string> serialized_descriptor,
    PjRtBuffer::RemoteSendCallback on_done) {
  auto error = absl::UnimplementedError(
      absl::StrCat("ScheduleRemoteSend is not implemented for %s",
                   memory_space->DebugString()));
  on_done(error, /*sends_were_enqueued=*/false);
  usage_event_promise->SetError(error);
}

absl::StatusOr<absl::InlinedVector<tsl::RCReference<CommonPjRtRawBuffer>, 4>>
CommonPjRtClient::AllocateOutputBuffersWithInputReuse(
    const Shape& output_device_shape,
    absl::Span<const CommonPjRtBuffer::ScopedHold> input_device_buffer_holds,
    const HloInputOutputAliasConfig& alias_config, PjRtDevice* device,
    absl::Span<const int> output_memory_space_kind_ids) {
  tsl::profiler::TraceMe traceme("AllocateOutputBuffersWithInputReuse");
  VLOG(1) << "Creating an output buffer, which may be partially donated, with "
             "shape "
          << output_device_shape.ToString();
  absl::InlinedVector<tsl::RCReference<CommonPjRtRawBuffer>, 4> buffers;
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
  buffers.reserve(output_leaf_shapes.size());

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
    return input_device_buffer_holds[parameter_index].type() !=
           CommonPjRtBuffer::ScopedHold::kDonation;
  };
  std::vector<size_t> output_buffer_sizes;
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
      TF_ASSIGN_OR_RETURN(int64_t on_device_bytes,
                          GetOnDeviceBytesCount(memory_space, leaf_shape));
      TF_ASSIGN_OR_RETURN(auto raw_buffer,
                          AllocateRawBuffer(memory_space, on_device_bytes,
                                            /*retry_on_oom=*/false,
                                            /*allocate_after=*/{}));
      buffers.push_back(std::move(raw_buffer));
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
      buffers.push_back(input_hold.buffer()->raw_buffer());
    }
  }

  if (VLOG_IS_ON(1)) {
    int64_t total_size = 0;
    for (const auto size : output_buffer_sizes) {
      total_size += size;
    }
    LOG(INFO)
        << "Total size of new output buffers allocated in this execution: "
        << total_size;
  }
  return std::move(buffers);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtBufferImpl::CopyToCpuMemorySpace(const xla::Shape& dst_shape,
                                           PjRtMemorySpace* dst_memory_space) {
  auto* dst_client =
      dynamic_cast<CommonPjRtClient*>(dst_memory_space->client());
  if (!dst_client) {
    return absl::InvalidArgumentError(
        "CopyToCpuMemorySpace only supported across CommonPjRtClient "
        "subclassed clients");
  }
  TF_ASSIGN_OR_RETURN(
      int64_t on_device_bytes_count,
      dst_client->GetOnDeviceBytesCount(dst_memory_space, dst_shape));
  TF_ASSIGN_OR_RETURN(
      auto dst_raw_buffer,
      dst_client->AllocateRawBuffer(dst_memory_space, on_device_bytes_count,
                                    /*retry_on_oom=*/true, {}));
  tsl::RCReference<PjRtDeviceEventPromise> definition_event_promise;
  tsl::RCReference<PjRtDeviceEvent> definition_event;
  TF_ASSIGN_OR_RETURN(
      std::tie(definition_event_promise, definition_event),
      dst_client->CreateLinkedEventPromise(dst_memory_space, ""));
  TF_ASSIGN_OR_RETURN(
      auto buffer,
      dst_client->DefineBuffer(dst_shape, dst_memory_space, dst_raw_buffer,
                               {std::move(definition_event)},
                               /*raw_buffer_is_mutable=*/true));
  auto* base_ptr = dst_raw_buffer->GetHostPointer();
  std::unique_ptr<MutableLiteralBase> literal;
  bool needs_second_copy = false;
  if (!primitive_util::IsSubByteNonPredType(dst_shape.element_type()) &&
      base_ptr) {
    literal = std::make_unique<MutableBorrowingLiteral>(
        reinterpret_cast<char*>(base_ptr), dst_shape);
  } else {
    literal = std::make_unique<Literal>(dst_shape);
    needs_second_copy = true;
  }

  auto copied = ToLiteral(literal.get());
  copied.OnReady([literal = std::move(literal), dst_client, needs_second_copy,
                  dst_raw_buffer = std::move(dst_raw_buffer), dst_shape,
                  definition_event_promise = std::move(
                      definition_event_promise)](absl::Status status) mutable {
    if (!status.ok()) {
      definition_event_promise->SetError(status);
    } else {
      absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
          status_or_h2d_transfer_event;
      if (needs_second_copy) {
        status_or_h2d_transfer_event = dst_client->LinearizeInto(
            *literal, dst_shape,
            PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
            dst_raw_buffer);
        if (!status_or_h2d_transfer_event.ok()) {
          definition_event_promise->SetError(status);
        } else {
          status_or_h2d_transfer_event.value()->AndThen(
              [literal = std::move(literal)] {});
          definition_event_promise->Set(
              *std::move(status_or_h2d_transfer_event));
        }
      } else {
        definition_event_promise->SetReady();
      }
    }
  });

  return buffer;
}

static absl::Status CommonCopyToMemorySpace(
    CommonPjRtBuffer* src_buffer, PjRtMemorySpace* dst_memory_space,
    const xla::Shape& dst_shape,
    tsl::RCReference<PjRtDeviceEventPromise>& definition_event_promise,
    tsl::RCReference<PjRtDeviceEventPromise>& src_usage_event_promise,
    tsl::RCReference<CommonPjRtRawBuffer>& src_raw_buffer,
    tsl::RCReference<CommonPjRtRawBuffer>& dst_raw_buffer,
    std::unique_ptr<PjRtBuffer>& dst_buffer,
    std::vector<tsl::RCReference<tsl::AsyncValue>>& definition_events,
    ::tsl::AsyncValueRef<bool>& allocation_event) {
  auto* src_memory_space = src_buffer->memory_space();
  CommonPjRtClient* const src_client =
      tensorflow::down_cast<CommonPjRtClient*>(src_buffer->client());
  CommonPjRtClient* const dst_client =
      dynamic_cast<CommonPjRtClient*>(dst_memory_space->client());
  if (!dst_client) {
    return absl::InvalidArgumentError(
        "CommonCopyToMemorySpace only supported across CommonPjRtClient "
        "subclassed clients");
  }
  TF_ASSIGN_OR_RETURN(
      const int64_t on_device_bytes_count,
      dst_client->GetOnDeviceBytesCount(dst_memory_space, dst_shape));

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

  auto status = [&]() -> absl::Status {
    TF_ASSIGN_OR_RETURN(
        dst_raw_buffer,
        dst_client->AllocateRawBuffer(dst_memory_space, on_device_bytes_count,
                                      /*retry_on_oom=*/true, allocation_event));
    TF_ASSIGN_OR_RETURN(
        dst_buffer,
        dst_client->DefineBuffer(dst_shape, dst_memory_space, dst_raw_buffer,
                                 {std::move(definition_event)},
                                 /*raw_buffer_is_mutable=*/true));
    TF_RETURN_IF_ERROR(src_buffer->AcquireScopedRawBuffer(
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

  if (!src_raw_buffer) {
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

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtBufferImpl::CopyFromCpuToMemorySpace(
    const xla::Shape& dst_shape, PjRtMemorySpace* dst_memory_space) {
  tsl::profiler::TraceMe traceme("CopyToMemorySpace");
  CommonPjRtClient* const src_client =
      tensorflow::down_cast<CommonPjRtClient*>(client());
  auto* dst_client =
      dynamic_cast<CommonPjRtClient*>(dst_memory_space->client());
  if (!dst_client) {
    return absl::InvalidArgumentError(
        "DirectCopyToMemorySpace only supported across CommonPjRtClient "
        "subclassed clients");
  }
  tsl::RCReference<PjRtDeviceEventPromise> definition_event_promise;
  tsl::RCReference<PjRtDeviceEventPromise> src_usage_event_promise;
  tsl::RCReference<CommonPjRtRawBuffer> src_raw_buffer;
  tsl::RCReference<CommonPjRtRawBuffer> dst_raw_buffer;
  std::unique_ptr<PjRtBuffer> dst_buffer;
  std::vector<tsl::RCReference<tsl::AsyncValue>> definition_events;
  ::tsl::AsyncValueRef<bool> allocation_event;
  TF_RETURN_IF_ERROR(CommonCopyToMemorySpace(
      this, dst_memory_space, dst_shape, definition_event_promise,
      src_usage_event_promise, src_raw_buffer, dst_raw_buffer, dst_buffer,
      definition_events, allocation_event));
  if (src_raw_buffer) {
    absl::Span<const tsl::RCReference<tsl::AsyncValue>> definition_events_span =
        definition_events;
    src_client->async_work_runner()->ScheduleWhenReady(
        definition_events_span,
        [dst_raw_buffer = std::move(dst_raw_buffer),
         src_raw_buffer = std::move(src_raw_buffer), dst_client = dst_client,
         src_shape = on_device_shape(), device_shape = dst_shape,
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
          auto status_or_h2d_transfer_event = dst_client->LinearizeInto(
              *literal, device_shape,
              PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
              std::move(dst_raw_buffer));
          CHECK_OK(status_or_h2d_transfer_event);
          auto h2d_transfer_event = *std::move(status_or_h2d_transfer_event);
          h2d_transfer_event->AndThen(
              [src_raw_buffer = std::move(src_raw_buffer),
               literal = std::move(literal),
               src_usage_event_promise = std::move(src_usage_event_promise)]() {
                src_usage_event_promise->SetReady();
              });
          if (dst_client->event_tracking_enabled()) {
            h2d_transfer_event->AppendDescriptionToEvent(
                " TransferToDevice ", {definition_event_promise.get()});
          }
          definition_event_promise->Set(std::move(h2d_transfer_event));
        });
  }
  return dst_buffer;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
CommonPjRtBufferImpl::CopyToMemorySpace(PjRtMemorySpace* dst_memory_space) {
  // Copying across PjRtClients involves a copy through the host.
  if (dst_memory_space->client() == client()) {
    TF_ASSIGN_OR_RETURN(auto dest_shape, client()->GetCopyDestinationShape(
                                             on_device_shape(), memory_space(),
                                             dst_memory_space));
    if (xla::Shape::Equal().IgnoreMemorySpaceInLayout()(dest_shape,
                                                        on_device_shape())) {
      return DirectCopyToMemorySpace(dst_memory_space);
    }
    if (!primitive_util::IsSubByteNonPredType(dest_shape.element_type())) {
      if (client()->IsOnCpu(dst_memory_space) &&
          xla::LayoutUtil::IsMonotonicWithDim0Major(dest_shape.layout()) &&
          dest_shape.layout().tiles().empty()) {
        return CopyToCpuMemorySpace(dest_shape, dst_memory_space);
      }
      if (client()->IsOnCpu(memory_space()) &&
          xla::LayoutUtil::IsMonotonicWithDim0Major(
              on_device_shape().layout()) &&
          on_device_shape().layout().tiles().empty()) {
        return CopyFromCpuToMemorySpace(dest_shape, dst_memory_space);
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
  TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal, ToLiteralSync());
  absl::InlinedVector<int64_t, 4> byte_strides(
      literal->shape().dimensions().size());
  TF_RETURN_IF_ERROR(ShapeUtil::UnpackedByteStrides(
      literal->shape(), absl::MakeSpan(byte_strides)));
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
  Shape shape = ShapeUtil::MakeShapeWithDescendingLayout(
      on_device_shape().element_type(), on_device_shape().dimensions());
  TF_ASSIGN_OR_RETURN(
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
  CommonPjRtClient* const src_client =
      tensorflow::down_cast<CommonPjRtClient*>(client());
  if (!dynamic_cast<CommonPjRtClient*>(dst_memory_space->client())) {
    return absl::InvalidArgumentError(
        "DirectCopyToMemorySpace only supported across CommonPjRtClient "
        "subclassed clients");
  }
  tsl::RCReference<PjRtDeviceEventPromise> definition_event_promise;
  tsl::RCReference<PjRtDeviceEventPromise> src_usage_event_promise;
  tsl::RCReference<CommonPjRtRawBuffer> src_raw_buffer;
  tsl::RCReference<CommonPjRtRawBuffer> dst_raw_buffer;
  std::unique_ptr<PjRtBuffer> dst_buffer;
  std::vector<tsl::RCReference<tsl::AsyncValue>> definition_events;
  ::tsl::AsyncValueRef<bool> allocation_event;
  TF_RETURN_IF_ERROR(CommonCopyToMemorySpace(
      this, dst_memory_space, on_device_shape(), definition_event_promise,
      src_usage_event_promise, src_raw_buffer, dst_raw_buffer, dst_buffer,
      definition_events, allocation_event));
  if (src_raw_buffer) {
    src_raw_buffer->ScheduleCopyTo(
        src_client->async_work_runner(), std::move(definition_events),
        std::move(dst_raw_buffer), std::move(definition_event_promise),
        std::move(src_usage_event_promise), std::move(allocation_event));
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
  tsl::profiler::TraceMe traceme("CommonPjRtBuffer::ToLiteral");
  VLOG(1) << "CommonPjRtBuffer::ToLiteral";
  auto common_client = tensorflow::down_cast<CommonPjRtClient*>(client());
  if (!common_client->allows_recursion() && ThisThreadIsInsideHostCallback()) {
    // Because TPU is single threaded, and the host callback currently blocking
    // the TPU, we should not block on any outstanding computations because that
    // risks deadlocking the TPU.
    return Future<>(
        InvalidArgument("ToLiteral() called from inside host callback."));
  }
  absl::StatusOr<Shape> device_shape = logical_on_device_shape();
  if (!device_shape.ok()) {
    return Future<>(device_shape.status());
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
    return Future<>(std::move(hold_status));
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
        auto copy_literal_async =
            [shape = std::move(shape),
             src_definition_events_avs = std::move(src_definition_events_avs),
             raw_buffer = std::move(raw_buffer),
             device_promise = std::move(device_promise),
             promise = std::move(promise)](
                const absl::StatusOr<MutableLiteralBase*>& value) mutable {
              tsl::profiler::TraceMe traceme([&] {
                return tsl::profiler::TraceMeEncode(
                    "D2H Dispatch",
                    {{"shape", shape.ToString(/*print_layout=*/true)}});
              });

              // Notify all pending events with `status`.
              auto notify_all = [&](absl::Status status) {
                promise.Set(status);
                if (device_promise) {
                  device_promise->SetError(status);
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
              for (const auto& av : src_definition_events_avs) {
                if (auto* error = av->GetErrorIfPresent()) {
                  notify_all(*error);
                  return;
                }
              }

              raw_buffer->CopyToLiteralAsync(std::move(promise), device_promise,
                                             literal, std::move(shape));
            };

        if (literal != nullptr) {
          copy_literal_async(literal);
        } else {
          Future<MutableLiteralBase*> generated = std::move(generator)();
          generated.OnReady(
              [copy_literal_async = std::move(copy_literal_async)](
                  const absl::StatusOr<MutableLiteralBase*>& value) mutable {
                copy_literal_async(value);
              });
        }
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

    absl::Status WaitUntilBufferReadyOnStream(std::intptr_t stream) override {
      return external_reference_.buffer()->WaitUntilBufferReadyOnStream(stream);
    }

    ~ScopedHoldAsExternalReference() override = default;

   private:
    ScopedHold external_reference_;
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer_;
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
    return Future<>(std::move(hold_status));
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

bool CommonPjRtBufferImpl::IsOnCpu() const {
  return tensorflow::down_cast<CommonPjRtClient*>(client())->IsOnCpu(
      memory_space());
}

CommonPjRtBufferImpl::CommonPjRtBufferImpl(
    const Shape& on_device_shape,
    std::unique_ptr<AbstractTrackedDeviceBuffer> tracked_device_buffer,
    PjRtMemorySpace* memory_space)
    : CommonPjRtBuffer(std::move(tracked_device_buffer), memory_space),
      on_device_shape_(on_device_shape) {}

CommonPjRtBufferImpl::~CommonPjRtBufferImpl() { Delete(); }

PjRtDevice* CommonPjRtBufferImpl::device() const {
  CHECK_EQ(memory_space_->devices().size(), 1);
  return tensorflow::down_cast<PjRtDevice*>(memory_space_->devices()[0]);
}

CommonPjRtClient* CommonPjRtBufferImpl::client() const {
  return tensorflow::down_cast<CommonPjRtClient*>(memory_space()->client());
}

absl::StatusOr<size_t> CommonPjRtBufferImpl::GetOnDeviceSizeInBytes() const {
  return client()->GetOnDeviceBytesCount(memory_space(), on_device_shape_);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
CommonPjRtBufferImpl::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  if (on_device_shape_.IsTuple()) {
    return InvalidArgument(
        "ReleaseDeviceMemoryOwnership allowed only for non-tuple");
  }
  auto device_buffer = ReleaseBuffer();
  if (device_buffer == nullptr) {
    return {nullptr};
  }

  if (wait_for_operations_to_complete) {
    TF_RETURN_IF_ERROR(
        device_buffer->BlockForOperationsToComplete(memory_space_));
  }

  class RawBufferAsExternalReference : public PjRtBuffer::ExternalReference {
   public:
    explicit RawBufferAsExternalReference(
        tsl::RCReference<CommonPjRtRawBuffer> raw_buffer)
        : raw_buffer_(std::move(raw_buffer)) {
      if (!raw_buffer_) {
        data_ptr_ = nullptr;
      } else {
        data_ptr_ = raw_buffer_->OpaqueDeviceMemoryDataPointer();
      }
    }

    ~RawBufferAsExternalReference() override = default;

   private:
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer_;
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
  auto hold = GetBufferWithHold(CommonPjRtBuffer::ScopedHold::kDonation);
  if (!hold.ok()) {
    return InvalidArgument(
        "Invalid buffer passed to DonateWithControlDependency: %s",
        hold.status().ToString());
  }
  // Make the new buffer which is identical to the old, except for the new
  // definition event.
  TF_ASSIGN_OR_RETURN(auto new_tracked_buffer,
                      hold.buffer()->CloneWithControlDependency(
                          memory_space(), std::move(dependency)));
  hold.ConfirmDonation();

  return std::make_unique<CommonPjRtBufferImpl>(
      on_device_shape(),
      std::unique_ptr<AbstractTrackedDeviceBuffer>(
          tensorflow::down_cast<AbstractTrackedDeviceBuffer*>(
              new_tracked_buffer.release())),
      memory_space());
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
