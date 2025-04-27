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

#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

CommonPjRtBuffer::CommonPjRtBuffer(
    std::unique_ptr<AbstractTrackedDeviceBuffer> device_buffer)
    : device_buffer_(std::move(device_buffer)) {
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
  absl::MutexLock lock(&mu_);
  CHECK(device_buffer_.get() == buffer || device_buffer_ == nullptr);
  CHECK_GT(holds_[type], 0);
  --holds_[type];
}

void CommonPjRtBuffer::DropDonationHold(
    std::unique_ptr<AbstractTrackedDeviceBuffer> buffer) {
  absl::MutexLock lock(&mu_);
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
  absl::MutexLock lock(&mu_);
  CHECK_EQ(holds_[ScopedHold::kUsage], 0);
  CHECK_EQ(holds_[ScopedHold::kExternalReference], 0);
  CHECK_EQ(holds_[ScopedHold::kDonation], 1);
  holds_[ScopedHold::kDonation] = 0;
  device_buffer->ConfirmDonation();
}

std::unique_ptr<AbstractTrackedDeviceBuffer> CommonPjRtBuffer::ReleaseBuffer() {
  absl::MutexLock lock(&mu_);
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

bool CommonPjRtBuffer::IsDeleted() {
  absl::MutexLock lock(&mu_);
  return device_buffer_ == nullptr;
}

}  // namespace xla
