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

#include "xla/pjrt/host_to_device_transfer_manager.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/lib/scoped_memory_debug_annotation.h"

namespace xla {

class CommonAsyncHostToDeviceTransferManager
    : public PjRtClient::AsyncHostToDeviceTransferManager {
 public:
  static absl::StatusOr<
      std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
  Create(absl::Span<const PjRtClient::ShapeSpec> shape_specs,
         std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
         PjRtMemorySpace* memory_space) {
    if (device_layouts.has_value() &&
        device_layouts->size() != shape_specs.size()) {
      return InvalidArgument(
          "Number of layouts %d does not match the number of shapes %d",
          device_layouts->size(), shape_specs.size());
    }

    auto* client =
        tensorflow::down_cast<CommonPjRtClient*>(memory_space->client());
    std::optional<std::string> debug_info = std::nullopt;
    const auto& current_anno =
        tsl::profiler::ScopedMemoryDebugAnnotation::CurrentAnnotation();
    if (current_anno.pending_op_name && current_anno.pending_region_type) {
      debug_info = std::make_optional<std::string>(absl::StrCat(
          current_anno.pending_op_name, " ", current_anno.pending_region_type));
    }

    absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers;
    // Each buffer gets an allocation event, which is set when the first chunk
    // of data arrives, and triggers actual HBM allocation immediately before
    // the data is copied to the device. This lazy allocation design avoids
    // holding on to empty, unusable HBM while waiting for data, for example
    // from a remote server,
    absl::InlinedVector<std::unique_ptr<ScopedEvent>, 4> allocation_events;
    absl::InlinedVector<tsl::RCReference<PjRtDeviceEventPromise>, 4>
        definition_events;
    absl::InlinedVector<Shape, 4> device_shapes;
    absl::InlinedVector<tsl::RCReference<CommonPjRtRawBuffer>, 4>
        undispatched_buffer_refs;
    absl::InlinedVector<size_t, 4> buffer_sizes;
    undispatched_buffer_refs.reserve(shape_specs.size());
    buffer_sizes.reserve(shape_specs.size());
    buffers.reserve(shape_specs.size());
    allocation_events.reserve(shape_specs.size());
    definition_events.reserve(shape_specs.size());
    device_shapes.reserve(shape_specs.size());
    for (int i = 0; i < shape_specs.size(); ++i) {
      const PjRtClient::ShapeSpec& shape_spec = shape_specs[i];
      if (shape_spec.element_type == TUPLE) {
        return Unimplemented(
            "Async buffer transfer of tuples not implemented.");
      }

      // We make an event that will become available when the final transfer
      // is complete.
      tsl::RCReference<PjRtDeviceEventPromise> definition_event_promise;
      tsl::RCReference<PjRtDeviceEvent> definition_event;
      if (client->event_tracking_enabled()) {
        TF_ASSIGN_OR_RETURN(
            std::tie(definition_event_promise, definition_event),
            client->CreateLinkedEventPromise(
                memory_space,
                absl::StrCat("AsyncHostToDeviceTransferManager Op:",
                             debug_info.value_or(""))));
      } else {
        TF_ASSIGN_OR_RETURN(
            std::tie(definition_event_promise, definition_event),
            client->CreateLinkedEventPromise(memory_space, ""));
      }
      definition_events.push_back(std::move(definition_event_promise));

      auto allocation_event =
          client->CreateAllocationEventForTransfers(memory_space, debug_info);
      if (allocation_event) {
        allocation_events.push_back(
            std::make_unique<ScopedEvent>(allocation_event));
      } else {
        allocation_events.push_back({});
      }

      Shape& device_shape = device_shapes.emplace_back(
          ShapeUtil::MakeShape(shape_spec.element_type, shape_spec.dims));
      if (device_layouts.has_value() && (*device_layouts)[i].has_value()) {
        *device_shape.mutable_layout() = *(*device_layouts)[i];
      } else {
        TF_ASSIGN_OR_RETURN(
            *device_shape.mutable_layout(),
            (*client->GetTopologyDescription())
                ->GetDefaultLayout(shape_spec.element_type, shape_spec.dims));
      }
      TF_ASSIGN_OR_RETURN(
          int64_t on_device_bytes_count,
          client->GetOnDeviceBytesCount(memory_space, device_shape));
      TF_ASSIGN_OR_RETURN(
          auto raw_buffer,
          client->AllocateRawBuffer(memory_space, on_device_bytes_count,
                                    allocation_event));
      TF_ASSIGN_OR_RETURN(auto buffer,
                          client->DefineBuffer(device_shape, raw_buffer,
                                               {std::move(definition_event)}));
      buffers.push_back(std::move(buffer));
      undispatched_buffer_refs.push_back(raw_buffer);
      buffer_sizes.push_back(on_device_bytes_count);
    }

    return std::unique_ptr<CommonAsyncHostToDeviceTransferManager>(
        new CommonAsyncHostToDeviceTransferManager(
            std::move(buffers), std::move(undispatched_buffer_refs),
            std::move(buffer_sizes), std::move(allocation_events),
            std::move(definition_events), std::move(device_shapes),
            client->async_work_runner(), client, memory_space,
            std::move(debug_info)));
  }

  ~CommonAsyncHostToDeviceTransferManager() override {
    auto transfers_finished = [this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return transfers_in_flight_ == 0;
    };
    {
      absl::MutexLock l(&mu_);
      // Make sure we don't leave dangling pointers in cleanup routines even
      // if the client lets the object go out of scope.
      mu_.Await(absl::Condition(&transfers_finished));
      for (const auto in_flight : buffer_transfers_in_flight_) {
        CHECK_EQ(in_flight, 0);
      }
      // Since there are no transfers in flight, we can't race on the
      // definition_events_ here. Make sure it has been notified, to avoid
      // blocking the definition event forever. If the transfers completed
      // successfully the event will be set after the call to on_done, which
      // might trigger this destructor, so we reset each entry in
      // definition_events_ in the case of successful completion which is why
      // definition_events_[x].GetAsyncValue might return nullptr.
      for (auto& event : definition_events_) {
        if (event) {
          event->SetError(absl::InternalError(
              "Async transfer object was deleted before transfers completed."));
        }
      }
    }
  }

  size_t buffer_count() const override { return buffers_.size(); };

  size_t buffer_size(int buffer_index) const override {
    DCHECK_LT(buffer_index, buffer_sizes_.size());
    return buffer_sizes_[buffer_index];
  }

  PjRtDevice* device() const override { return memory_space_->devices()[0]; }

  std::unique_ptr<PjRtBuffer> RetrieveBuffer(int buffer_index) override {
    DCHECK_LT(buffer_index, buffers_.size());
    return std::move(buffers_[buffer_index]);
  };

  absl::Status TransferLiteralToBuffer(
      int buffer_index, const LiteralSlice& literal,
      absl::AnyInvocable<void() &&> on_done) override {
    absl::ReleasableMutexLock l(&mu_);

    DCHECK_LT(buffer_index, undispatched_buffer_refs_.size());
    tsl::RCReference<CommonPjRtRawBuffer>& undispatched_buffer_ref =
        undispatched_buffer_refs_[buffer_index];
    if (!undispatched_buffer_ref) {
      return InvalidArgument(
          "TransferLiteralToBuffer requested for buffer index %d which has "
          "already been fully transferred",
          buffer_index);
    }
    // Unblock allocating the underlying memory.
    allocation_events_[buffer_index].reset();

    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer;
    tsl::RCReference<PjRtDeviceEventPromise> definition_event;
    using std::swap;
    swap(raw_buffer, undispatched_buffer_ref);
    CHECK(raw_buffer);
    swap(definition_event, definition_events_[buffer_index]);
    CHECK(definition_event);

    ++transfers_in_flight_;
    CHECK_EQ(buffer_transfers_in_flight_[buffer_index], 0);
    ++buffer_transfers_in_flight_[buffer_index];

    // We release the lock here because EnqueueWork might sometimes run the
    // closure in this thread!
    l.Release();

    tsl::profiler::TraceMeProducer producer("TransferLiteralToBuffer",
                                            tsl::profiler::ContextType::kPjRt);

    // The host to device transfer is performed on a thread pool, mostly because
    // it includes linearization that may be slow.
    // TODO(misard) assess if it would be preferable to introduce a heuristic to
    // put the transfer into the calling thread for small literals.
    async_work_runner_->Schedule(
        [this, buffer_index, literal, raw_buffer = std::move(raw_buffer),
         definition_event = std::move(definition_event),
         on_done = std::move(on_done),
         context_id = producer.GetContextId()]() mutable {
          tsl::profiler::TraceMeConsumer consumer(
              "TransferLiteralToBuffer H2D Dispatch",
              tsl::profiler::ContextType::kPjRt, context_id);
          auto status_or_h2d_transfer_event = client_->LinearizeInto(
              literal, device_shapes_[buffer_index].layout(), raw_buffer);
          CHECK_OK(status_or_h2d_transfer_event);
          auto h2d_transfer_event = *std::move(status_or_h2d_transfer_event);
          if (client_->event_tracking_enabled()) {
            h2d_transfer_event->AppendDescriptionToEvent(
                " TransferToDevice TransferLiteralToBuffer",
                {definition_event.get()});
          }

          auto cleanup = [this, buffer_index,
                          transfer_event = h2d_transfer_event,
                          definition_event = std::move(definition_event),
                          on_done = std::move(on_done)]() mutable {
            {
              absl::MutexLock l(&mu_);

              CHECK_GT(transfers_in_flight_, 0);
              --transfers_in_flight_;
              CHECK_EQ(buffer_transfers_in_flight_[buffer_index], 1);
              --buffer_transfers_in_flight_[buffer_index];
              CHECK_GT(remaining_buffer_count_, 0);
              --remaining_buffer_count_;
            }

            // Call on_done after finishing all housekeeping and releasing the
            // lock.
            //
            // NOTE: on_done may call ~AsyncHostToDeviceTransferManager(), so we
            // don't touch any class members after this point.
            std::move(on_done)();

            // Unblock the definition event after calling on_done, just in case
            // the caller wanted some serialization between finding out about
            // the buffers becoming available and them being released.
            CHECK(definition_event);
            // Dependency of event on transfer_event was recorded above in
            // AppendDescriptionToEvent.
            definition_event->Set(std::move(transfer_event));
          };
          h2d_transfer_event->AndThen(std::move(cleanup));
        });

    return absl::OkStatus();
  }

  absl::Status TransferRawDataToBuffer(
      int buffer_index, absl::string_view data,
      absl::AnyInvocable<void() &&> on_done) override {
    return TransferRawDataToSubBuffer(buffer_index, data.data(),
                                      /*offset=*/0, data.size(),
                                      /*is_last_transfer=*/true,
                                      std::move(on_done));
  }

  absl::Status TransferRawDataToSubBuffer(
      int buffer_index, const void* data, int64_t offset, int64_t transfer_size,
      bool is_last_transfer, absl::AnyInvocable<void() &&> on_done) override {
    absl::ReleasableMutexLock l(&mu_);
    DCHECK_LT(buffer_index, undispatched_buffer_refs_.size());
    tsl::RCReference<CommonPjRtRawBuffer> undispatched_buffer_ref;
    // Drop reference to the buffer if this is the last transfer.
    if (is_last_transfer) {
      std::swap(undispatched_buffer_ref,
                undispatched_buffer_refs_[buffer_index]);
    } else {
      undispatched_buffer_ref = undispatched_buffer_refs_[buffer_index];
    }
    if (!undispatched_buffer_ref) {
      return InvalidArgument(
          "TransferRawData requested for buffer index %d which has "
          "already been fully transferred",
          buffer_index);
    }
    CHECK(definition_events_[buffer_index]);
    std::string op_name = "TransferRawDataToSubBuffer";
    std::string region_type = "";
    if (debug_info_.has_value()) {
      std::vector<std::string> debug_info =
          absl::StrSplit(debug_info_.value(), ';');
      op_name = debug_info.empty() ? "" : debug_info.front();
      region_type = debug_info.size() > 1 ? debug_info.back() : "";
    }
    tsl::profiler::ScopedMemoryDebugAnnotation anno(
        op_name.c_str(), region_type.c_str(), 0, []() { return ""; });
    // Unblock allocating the underlying memory.
    allocation_events_[buffer_index].reset();

    ++transfers_in_flight_;
    ++buffer_transfers_in_flight_[buffer_index];

    // Release the lock for two reasons:
    //   (1) Asynchronous calls to this function spend most of their time in
    //       `::tpu::System::TransferToDevice`, so this reduces lock contention.
    //   (2) Cleanup of this class may be called within the `on_done` of
    //        `h2d_transfer_event.AndThen`, which would cause deadlock.
    l.Release();
    TF_ASSIGN_OR_RETURN(
        auto h2d_transfer_event,
        undispatched_buffer_ref->CopyRawHostToDeviceAndReturnEvent(
            data, offset, transfer_size));
    if (client_->event_tracking_enabled()) {
      // Acquire when logging, for the sake of definition_events_.
      absl::MutexLock l(&mu_);
      std::string op_name = debug_info_.has_value()
                                ? absl::StrCat(" Op:", debug_info_.value())
                                : "";
      h2d_transfer_event->AppendDescriptionToEvent(
          absl::StrCat(" TransferToDevice TransferRawData offset:", offset,
                       " size:", transfer_size,
                       " last_transfer:", is_last_transfer, op_name),
          {definition_events_[buffer_index].get()});
    }

    h2d_transfer_event->AndThen([this, buffer_index,
                                 transfer_event = h2d_transfer_event,
                                 on_done = std::move(on_done)]() mutable {
      tsl::RCReference<PjRtDeviceEventPromise> definition_event;
      {
        absl::MutexLock l(&mu_);

        CHECK_GT(transfers_in_flight_, 0);
        --transfers_in_flight_;
        CHECK_GT(buffer_transfers_in_flight_[buffer_index], 0);
        --buffer_transfers_in_flight_[buffer_index];
        auto& definition_event_ref = definition_events_[buffer_index];
        if (buffer_transfers_in_flight_[buffer_index] == 0 &&
            !undispatched_buffer_refs_[buffer_index]) {
          CHECK_GT(remaining_buffer_count_, 0);
          --remaining_buffer_count_;
          using std::swap;
          swap(definition_event, definition_event_ref);
        }
        if (definition_event_ref) {
          // If this is not the last completed transfer, then we need to set the
          // error while holding the lock to avoid a race.
          auto state = transfer_event->state();
          if (state == PjRtDeviceEvent::State::kError) {
            definition_event_ref->SetError(transfer_event->status());
            definition_event_ref = tsl::RCReference<PjRtDeviceEventPromise>();
          } else {
            CHECK(state == PjRtDeviceEvent::State::kReady);
          }
        }
      }

      // Call on_done after finishing all housekeeping and releasing the
      // lock.
      //
      // NOTE: on_done may call ~AsyncHostToDeviceTransferManager(), so we
      // don't touch any class members after this point.
      std::move(on_done)();

      // Unblock the definition event after calling on_done, just in case
      // the caller wanted some serialization between finding out about the
      // buffers becoming available and them being released.
      if (definition_event) {
        // Dependency of event on transfer_event was recorded above in
        // AppendDescriptionToEvent.
        definition_event->Set(std::move(transfer_event));
      }
    });
    return absl::OkStatus();
  }

  void SetBufferError(int buffer_index, absl::Status error) override {
    absl::MutexLock l(&mu_);
    // For a given buffer_index, SetBufferError can't be called twice, or
    // called after the last transfer has been enqueued.
    auto definition_event = std::move(definition_events_[buffer_index]);
    CHECK(definition_event);
    definition_event->SetError(error);
    if (allocation_events_[buffer_index]) {
      allocation_events_[buffer_index]->SetError(error);
    }
  }

  void AddTransferMetadata(const TransferMetadata& meta) override {
    if (client_->event_tracking_enabled()) {
      absl::MutexLock l(&mu_);
      std::string annotation =
          absl::StrCat(" ", absl::StrJoin(meta, " ", absl::PairFormatter(":")));
      for (int i = 0; i < definition_events_.size(); ++i) {
        const auto& event = definition_events_[i];
        if (definition_events_.size() > 1) {
          absl::StrAppend(&annotation, " buf_idx:", i);
        }
        event->AppendDescriptionToEvent(annotation, {});
      }
    }
  }

 private:
  // Helper class that holds an event and makes the event available when the
  // class goes out of scope. Used for the events that unblock TpuBuffer
  // allocation to ensure that the allocations are unblocked in all error cases.
  class ScopedEvent {
   public:
    explicit ScopedEvent(::tsl::AsyncValueRef<bool> event)
        : event_(std::move(event)) {}
    ~ScopedEvent() {
      if (event_) {
        event_.SetStateConcrete();
      }
    }

    void SetError(const absl::Status& error) {
      event_.SetError(error);
      event_.reset();
    }

   private:
    ::tsl::AsyncValueRef<bool> event_;
  };

  CommonAsyncHostToDeviceTransferManager(
      absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers,
      absl::InlinedVector<tsl::RCReference<CommonPjRtRawBuffer>, 4> raw_buffers,
      absl::InlinedVector<size_t, 4> buffer_sizes,
      absl::InlinedVector<std::unique_ptr<ScopedEvent>, 4> allocation_events,
      absl::InlinedVector<tsl::RCReference<PjRtDeviceEventPromise>, 4>
          definition_events,
      absl::InlinedVector<Shape, 4> device_shapes,
      AsyncWorkRunner* async_work_runner, CommonPjRtClient* client,
      PjRtMemorySpace* memory_space, std::optional<std::string> debug_info)
      : debug_info_(std::move(debug_info)),
        buffers_(std::move(buffers)),
        allocation_events_(std::move(allocation_events)),
        buffer_sizes_(std::move(buffer_sizes)),
        undispatched_buffer_refs_(std::move(raw_buffers)),
        definition_events_(std::move(definition_events)),
        device_shapes_(std::move(device_shapes)),
        remaining_buffer_count_(buffers_.size()),
        transfers_in_flight_(0),
        async_work_runner_(async_work_runner),
        client_(client),
        memory_space_(memory_space) {
    DCHECK_EQ(memory_space_->devices().size(), 1);
    buffer_transfers_in_flight_.resize(undispatched_buffer_refs_.size(), 0);
  }

  std::optional<std::string> debug_info_;

  absl::Mutex mu_;
  // The newly created buffers, which will be returned to the caller via
  // Retrieve.
  absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers_;
  // Holders for events that block allocation of the underlying memory for the
  // buffers. When data is first available for a buffer the corresponding
  // allocation ScopedEvent is destroyed, which triggers allocation of the
  // memory.
  absl::InlinedVector<std::unique_ptr<ScopedEvent>, 4> allocation_events_
      ABSL_GUARDED_BY(mu_);
  // Cached versions of the sizes of all the buffers, so we can return them
  // without acquiring mu_.
  absl::InlinedVector<size_t, 4> buffer_sizes_;
  // References to the underlying storage for all the buffers, which ensures
  // that the buffers can't be freed before all transfers are dispatched. The
  // reference to each buffer is dropped immediately after the last transfer
  // for that buffer has been dispatched.
  absl::InlinedVector<tsl::RCReference<CommonPjRtRawBuffer>, 4>
      undispatched_buffer_refs_ ABSL_GUARDED_BY(mu_);
  // Number of transfers in flight for each buffer. Used to determine when the
  // last transfer has completed, in case the completions arrive out of order.
  absl::InlinedVector<int, 4> buffer_transfers_in_flight_ ABSL_GUARDED_BY(mu_);
  // Per buffer definition event. It is made available once the buffer is ready
  // (either because the transfer for that buffer completed, or because an error
  // was recorded for that buffer).
  absl::InlinedVector<tsl::RCReference<PjRtDeviceEventPromise>, 4>
      definition_events_ ABSL_GUARDED_BY(mu_);
  // Device shapes for all buffers with either compact or custom layout.
  const absl::InlinedVector<Shape, 4> device_shapes_;
  // Count of buffers that have not yet been fully transferred.
  size_t remaining_buffer_count_ ABSL_GUARDED_BY(mu_);
  // Count of transfers that have been started but have not yet called cleanup.
  // Used to block in the destructor to avoid dangling pointers in cleanup.
  int transfers_in_flight_ ABSL_GUARDED_BY(mu_);

  AsyncWorkRunner* async_work_runner_;  // not owned.
  CommonPjRtClient* client_;            // not owned.
  PjRtMemorySpace* memory_space_;       // not owned.
};

absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
CreateAsyncHostToDeviceTransferManager(
    absl::Span<const PjRtClient::ShapeSpec> shape_specs,
    std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
    PjRtMemorySpace* memory_space) {
  return CommonAsyncHostToDeviceTransferManager::Create(
      shape_specs, device_layouts, memory_space);
}

}  // namespace xla
