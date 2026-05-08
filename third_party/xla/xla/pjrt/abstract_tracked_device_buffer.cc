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

#include "xla/pjrt/abstract_tracked_device_buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

Future<> AbstractTrackedDeviceBuffer::GetReadyFuture(
    PjRtMemorySpace* memory_space) {
  auto* client = absl::down_cast<CommonPjRtClient*>(memory_space->client());

  auto [definition_promise, definition_future] = tsl::MakePromise<void>();
  client->TrackFuture(memory_space, "BufferDefinitionEvent", definition_future);

  CHECK(usage_events_);
  std::vector<tsl::RCReference<tsl::AsyncValue>> dependencies;
  dependencies.reserve(definition_events().size() + 1);
  bool first_event_is_buffer_alloc = false;
  if (raw_buffer() && client->include_raw_buffer_in_ready_event()) {
    auto* alloc_event = raw_buffer()->GetRawBufferAsyncValue();
    if (alloc_event && !alloc_event->IsConcrete()) {
      first_event_is_buffer_alloc = true;
      dependencies.push_back(tsl::FormRef(alloc_event));
    }
  }
  for (const auto& ev : definition_events()) {
    if (!ev.async_value()->IsConcrete()) {
      dependencies.push_back(tsl::FormRef(ev.async_value()));
    }
  }
  if (client->event_tracking_enabled()) {
    client->AddEventDependencies(
        memory_space,
        PjRtDeviceEventPtr::FromAsyncValue(definition_future.async_value()),
        dependencies);
  }
  auto deps = absl::Span<const tsl::RCReference<tsl::AsyncValue>>(dependencies);
  tsl::RunWhenReady(deps, [definition_event = std::move(definition_promise),
                           first_event_is_buffer_alloc,
                           dependencies = std::move(dependencies)]() mutable {
    absl::Status status;
    for (size_t i = 0; i < dependencies.size(); ++i) {
      const auto& e = dependencies[i];
      if (auto* error = e->GetErrorIfPresent()) {
        if (i == 0 && first_event_is_buffer_alloc) {
          status.Update(absl::Status(
              absl::StatusCode::kFailedPrecondition,
              absl::StrCat("Error in buffer allocation: ", error->message())));
        } else {
          status.Update(*error);
        }
      }
    }
    definition_event.Set(std::move(status));
  });

  return definition_future;
}

absl::Status AbstractTrackedDeviceBuffer::BlockForOperationsToComplete(
    PjRtMemorySpace* memory_space) {
  std::vector<tsl::RCReference<tsl::AsyncValue>> avs;
  usage_events_->AppendTo(avs);
  for (const auto& av : avs) {
    tsl::BlockUntilReady(av.get());
  }

  for (const auto& ev : definition_events()) {
    if (ev) {
      tsl::BlockUntilReady(ev.async_value());
      if (auto* error = ev.async_value()->GetErrorIfPresent()) {
        return absl::InternalError(
            absl::StrFormat("Error Execute: %s", error->message()));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status AbstractTrackedDeviceBuffer::WaitUntilBufferReadyOnStream(
    PjRtMemorySpace* memory_space, std::intptr_t stream) {
  auto* client = absl::down_cast<CommonPjRtClient*>(memory_space->client());
  for (const auto& event : definition_events()) {
    TF_RETURN_IF_ERROR(client->WaitOnStream(memory_space, event, stream));
  }
  return absl::OkStatus();
}

AbstractTrackedDeviceBuffer::AbstractTrackedDeviceBuffer(
    PjRtRawBufferRef raw_buffer,
    absl::InlinedVector<PjRtDeviceEventRef, 2> definition_events,
    bool use_stream_based_compaction)
    : raw_buffer_(std::move(raw_buffer)),
      definition_events_(std::move(definition_events)),
      usage_events_(
          std::make_unique<DefaultUsageEventSet>(use_stream_based_compaction)) {
}

void AbstractTrackedDeviceBuffer::Delete(PjRtMemorySpace* memory_space) {
  std::unique_ptr<AbstractTrackedDeviceBuffer> device_buffer(this);
  std::vector<PjRtDeviceEventRef> events;
  for (const auto& ev : device_buffer->definition_events()) {
    events.push_back(ev);
  }
  device_buffer->usage_events_->AppendTo(events);

  device_buffer->LockUseAndTransferUsageEvents();
  auto raw_buffer = device_buffer->raw_buffer();
  device_buffer.reset();
  if (raw_buffer) {
    raw_buffer.release()->DecrefAfter(std::move(events));
  }
}

void AbstractTrackedDeviceBuffer::UnsafePrependDefinitionEvent(
    PjRtDeviceEventRef extra_definition_event) {
  definition_events_.insert(definition_events_.begin(),
                            std::move(extra_definition_event));
}

CommonPjRtBuffer::CommonPjRtBuffer(
    std::unique_ptr<AbstractTrackedDeviceBuffer> device_buffer,
    PjRtMemorySpace* memory_space)
    : memory_space_(memory_space), device_buffer_(std::move(device_buffer)) {
  for (int i = 0; i < ScopedHold::Type::kMaxValue; ++i) {
    holds_[i] = 0;
  }
}

CommonPjRtBuffer::~CommonPjRtBuffer() {
  for (int i = 0; i < ScopedHold::Type::kMaxValue; ++i) {
    CHECK_EQ(holds_[i], 0) << "Non-zero type " << i << " hold on destruction.";
  }
}

void CommonPjRtBuffer::WaitForOutstandingUsageHolds() {
  auto not_in_usage_hold = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return holds_[ScopedHold::kUsage] == 0;
  };
  mu_.Await(absl::Condition(&not_in_usage_hold));
}

void CommonPjRtBuffer::WaitForOutstandingDonationHold() {
  auto not_in_donation_hold = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return holds_[ScopedHold::kDonation] == 0;
  };
  mu_.Await(absl::Condition(&not_in_donation_hold));
}

absl::StatusOr<AbstractTrackedDeviceBuffer*>
CommonPjRtBuffer::GetBufferForUsageOrExternalHoldLocked(ScopedHold::Type type) {
  // All callers should have called WaitForOutstandingDonationHold().
  CHECK_EQ(holds_[ScopedHold::kDonation], 0);
  if (device_buffer_ == nullptr) {
    return absl::InvalidArgumentError("Buffer has been deleted or donated.");
  } else {
    ++holds_[type];
  }
  return device_buffer_.get();
}

absl::StatusOr<std::unique_ptr<AbstractTrackedDeviceBuffer>>
CommonPjRtBuffer::GetBufferForDonationHoldLocked() {
  // All callers should have called WaitForOutstandingDonationHold().
  CHECK_EQ(holds_[ScopedHold::kDonation], 0);
  if (device_buffer_ == nullptr) {
    return absl::InvalidArgumentError("Donation requested for invalid buffer");
  }
  if (holds_[ScopedHold::kExternalReference] > 0) {
    return absl::InvalidArgumentError(
        "Donation requested for buffer with external reference");
  }
  // First add the donation hold.
  ++holds_[ScopedHold::kDonation];
  // Then wait for any usage holds to be dropped or converted. No new usage
  // holds can be added until we drop the donation hold so this wait will
  // complete eventually.
  WaitForOutstandingUsageHolds();
  // Because we added a donation hold, nobody could release the buffer while
  // we were waiting.
  CHECK(device_buffer_ != nullptr);
  return std::move(device_buffer_);
}

absl::StatusOr<std::unique_ptr<AbstractTrackedDeviceBuffer>>
CommonPjRtBuffer::DonateTrackedBuffer() {
  absl::MutexLock lock(&mu_);
  WaitForOutstandingDonationHold();

  auto buffer_or = GetBufferForDonationHoldLocked();
  if (!buffer_or.ok()) {
    return buffer_or.status();
  }

  auto buffer = std::move(buffer_or).value();
  CHECK_EQ(holds_[ScopedHold::kDonation], 1);
  holds_[ScopedHold::kDonation] = 0;
  return buffer;
}

void CommonPjRtBuffer::AcquireHoldLocked(ScopedHold* hold) {
  if (hold->type() == ScopedHold::kDonation) {
    hold->AcquireDonation(GetBufferForDonationHoldLocked());
    return;
  }

  hold->AcquireUsageOrExternalReference(
      GetBufferForUsageOrExternalHoldLocked(hold->type()));
}

void CommonPjRtBuffer::DropUsageOrExternalHold(
    ScopedHold::Type type, AbstractTrackedDeviceBuffer* buffer) {
  absl::MutexLock lock(mu_);
  CHECK(device_buffer_.get() == buffer || device_buffer_ == nullptr);
  CHECK_GT(holds_[type], 0);
  --holds_[type];
}

void CommonPjRtBuffer::DropDonationHold(
    std::unique_ptr<AbstractTrackedDeviceBuffer> buffer) {
  absl::MutexLock lock(mu_);
  CHECK_EQ(device_buffer_.get(), nullptr);
  device_buffer_ = std::move(buffer);
  CHECK_GT(holds_[ScopedHold::kDonation], 0);
  --holds_[ScopedHold::kDonation];
  CHECK_EQ(holds_[ScopedHold::kDonation], 0);
  CHECK_EQ(holds_[ScopedHold::kUsage], 0);
  CHECK_EQ(holds_[ScopedHold::kExternalReference], 0);
}

absl::Status CommonPjRtBuffer::ScopedHold::status() const {
  // Lazily create absl::Status values only when they are requested.
  switch (state_) {
    case kUninitialized:
      return absl::InvalidArgumentError("Buffer has not been initialized");
    case kValid:
      return absl::OkStatus();
    case kMoved:
      return absl::InvalidArgumentError("Buffer has been moved.");
    case kConverted:
      return absl::InvalidArgumentError("Buffer has been converted");
    case kReleased:
      return absl::InvalidArgumentError("Buffer has been released");
    case kDonated:
      return absl::InvalidArgumentError("Buffer has been donated");
    case kError:
      return status_;
    default:
      CHECK(false) << "Unexpected state value " << state_;
  }
}

void CommonPjRtBuffer::ScopedHold::DropHold() {
  if (ok()) {
    if (type_ == kDonation) {
      parent_->DropDonationHold(std::move(buffer_));
    } else {
      parent_->DropUsageOrExternalHold(type_, buffer_ptr_);
    }
  }
}

CommonPjRtBuffer::ScopedHold::~ScopedHold() { DropHold(); }

CommonPjRtBuffer::ScopedHold::ScopedHold(ScopedHold&& other)
    : parent_(other.parent_),
      type_(other.type_),
      state_(other.state_),
      status_(std::move(other.status_)),
      buffer_ptr_(other.buffer_ptr_),
      buffer_(std::move(other.buffer_)) {
  // Preserve the invariant that status is invalid if buffer == nullptr.
  other.SetState(kMoved);
}

void CommonPjRtBuffer::ScopedHold::AcquireDonation(
    absl::StatusOr<std::unique_ptr<AbstractTrackedDeviceBuffer>> buffer_or) {
  CHECK(!ok());
  if (buffer_or.ok()) {
    buffer_ = std::move(buffer_or).value();
    buffer_ptr_ = buffer_.get();
    SetState(kValid);
  } else {
    status_ = std::move(buffer_or).status();
    buffer_ = nullptr;
    buffer_ptr_ = nullptr;
    SetState(kError);
  }
  // Check the invariant holds.
  CHECK(!ok() || buffer_ptr_ != nullptr);
}

void CommonPjRtBuffer::ScopedHold::AcquireUsageOrExternalReference(
    absl::StatusOr<AbstractTrackedDeviceBuffer*> buffer_or) {
  CHECK(!ok());
  if (buffer_or.ok()) {
    buffer_.reset();
    buffer_ptr_ = buffer_or.value();
    SetState(kValid);
  } else {
    status_ = std::move(buffer_or).status();
    buffer_.reset();
    buffer_ = nullptr;
    SetState(kError);
  }
  // Check the invariant holds.
  CHECK(!ok() || buffer_ptr_ != nullptr);
}

void CommonPjRtBuffer::ScopedHold::ConfirmDonation() {
  CHECK(ok());
  CHECK_EQ(type(), kDonation);
  parent()->ConfirmDonation(buffer());
  SetState(kDonated);
}

void CommonPjRtBuffer::ConfirmDonation(
    AbstractTrackedDeviceBuffer* device_buffer) {
  absl::MutexLock lock(mu_);
  CHECK_EQ(holds_[ScopedHold::kUsage], 0);
  CHECK_EQ(holds_[ScopedHold::kExternalReference], 0);
  CHECK_EQ(holds_[ScopedHold::kDonation], 1);
  holds_[ScopedHold::kDonation] = 0;
  device_buffer->ConfirmDonation();
}

void CommonPjRtBuffer::ScopedHold::ConvertUsageHold(PjRtDeviceEventRef event) {
  CHECK(ok());
  CHECK_EQ(type(), kUsage);
  {
    absl::MutexLock lock(parent()->mu_);
    CHECK(parent()->device_buffer() == buffer() ||
          parent()->device_buffer() == nullptr);
    buffer()->AddUsageEvent(std::move(event));
    parent()->DecrementUsage();
  }
  SetState(kConverted);
}

std::unique_ptr<AbstractTrackedDeviceBuffer> CommonPjRtBuffer::ReleaseBuffer() {
  absl::MutexLock lock(mu_);
  {
    tsl::profiler::TraceMe t1("Wait for donation holds");
    // We first wait for a donation hold to complete if there is one in
    // progress. If the donation succeeds via ConfirmDonation() then it will
    // set device_buffer_ to nullptr before returning to this thread.
    WaitForOutstandingDonationHold();
  }
  if (device_buffer_ == nullptr) {
    // Buffer has been deleted.
    return nullptr;
  }
  // Return device_buffer_ by move which also sets it to nullptr, so
  // that no other thread can add a hold while we are in
  // WaitForOutstandingUsageHolds() below.
  auto buffer = std::move(device_buffer_);

  tsl::profiler::TraceMe t2("Wait for usage holds");
  WaitForOutstandingUsageHolds();
  return buffer;
}

bool CommonPjRtBuffer::IsDeleted() const {
  absl::MutexLock lock(mu_);
  return device_buffer_ == nullptr;
}

absl::Status CommonPjRtBuffer::AcquireScopedRawBuffer(
    absl::AnyInvocable<absl::StatusOr<PjRtDeviceEventRef>(
        PjRtRawBufferRef raw_buffer,
        std::vector<tsl::RCReference<tsl::AsyncValue>> definition_events) &&>
        scoped_acquire,
    const char* caller_name) {
  ScopedHold device_buffer(this, ScopedHold::kUsage);
  {
    absl::MutexLock lock(mu_);
    // Ensure that at most one donation hold can be in progress at a time.
    WaitForOutstandingDonationHold();
    AcquireHoldLocked(&device_buffer);
  }
  if (!device_buffer.ok()) {
    return InvalidArgument("%s called on deleted or donated buffer: %s",
                           caller_name, device_buffer.status().ToString());
  }
  TF_ASSIGN_OR_RETURN(
      auto device_event,
      std::move(scoped_acquire)(
          device_buffer.buffer()->raw_buffer(),
          device_buffer.buffer()->GetAsyncValueDefinitionEvents()));
  device_buffer.ConvertUsageHold(std::move(device_event));
  return absl::OkStatus();
}
absl::Status CommonPjRtBuffer::AcquireScopedRawBuffer(
    absl::AnyInvocable<absl::StatusOr<PjRtDeviceEventRef>(
        PjRtRawBufferRef raw_buffer,
        std::vector<PjRtDeviceEventRef> definition_events) &&>
        scoped_acquire,
    const char* caller_name) {
  ScopedHold device_buffer(this, ScopedHold::kUsage);
  {
    absl::MutexLock lock(mu_);
    // Ensure that at most one donation hold can be in progress at a time.
    WaitForOutstandingDonationHold();
    AcquireHoldLocked(&device_buffer);
  }
  if (!device_buffer.ok()) {
    return InvalidArgument("%s called on deleted or donated buffer: %s",
                           caller_name, device_buffer.status().ToString());
  }

  auto definition_events_span = device_buffer.buffer()->definition_events();
  std::vector<PjRtDeviceEventRef> definition_events(
      definition_events_span.begin(), definition_events_span.end());

  TF_ASSIGN_OR_RETURN(
      auto device_event,
      std::move(scoped_acquire)(device_buffer.buffer()->raw_buffer(),
                                std::move(definition_events)));
  device_buffer.ConvertUsageHold(std::move(device_event));
  return absl::OkStatus();
}

CommonPjRtBuffer::ScopedHold CommonPjRtBuffer::GetBufferWithHold(
    ScopedHold::Type type) {
  absl::MutexLock lock(mu_);
  // Ensure that at most one donation hold can be in progress at a time.
  WaitForOutstandingDonationHold();
  ScopedHold hold(this, type);
  AcquireHoldLocked(&hold);
  return hold;
}

void DefaultUsageEventSet::AddEvent(PjRtDeviceEventRef event) {
  if (!event) {
    return;
  }

  if (stream_based_compaction_) {
    if (event.state() == PJRT_DeviceEvent_State_Error) return;

    auto def_info = event.ptr().GetDefinitionStream();
    if (def_info.has_value()) {
      for (auto& existing : usage_events_) {
        auto existing_def_info = existing.ptr().GetDefinitionStream();
        if (!existing_def_info.has_value()) continue;

        if (existing_def_info->stream == def_info->stream) {
          if (existing_def_info->sequence_id < def_info->sequence_id) {
            existing = event;
          }
          return;
        }
      }
    }
    usage_events_.push_back(event);
    return;
  }

  // Trim ready events in order to avoid dynamic allocation as much as possible.
  // The cost will be amortized because a typical vector implementation grows
  // the capacity superlinearly.
  if (usage_events_.size() + 1 > usage_events_.capacity()) {
    int i = 0;
    while (i < usage_events_.size()) {
      auto& ev = usage_events_.at(i);
      if (ev.async_value()->IsAvailable()) {
        using std::swap;
        swap(ev, usage_events_.back());
        usage_events_.pop_back();
        continue;
      }
      ++i;
    }
  }
  usage_events_.push_back(std::move(event));
}

void DefaultUsageEventSet::AppendTo(
    std::vector<tsl::RCReference<tsl::AsyncValue>>& events) {
  events.reserve(events.size() + usage_events_.size());
  for (const auto& ev : usage_events_) {
    events.push_back(tsl::FormRef(ev.async_value()));
  }
}

void DefaultUsageEventSet::AppendTo(std::vector<PjRtDeviceEventRef>& events) {
  events.reserve(events.size() + usage_events_.size());
  for (const auto& ev : usage_events_) {
    events.push_back(ev);
  }
}

void DefaultUsageEventSet::AppendTo(PjRtDeviceEventSet& events) {
  for (const auto& ev : usage_events_) {
    events.AddEvent(ev);
  }
}

}  // namespace xla
