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

#ifndef XLA_PJRT_ABSTRACT_TRACKED_DEVICE_BUFFER_H_
#define XLA_PJRT_ABSTRACT_TRACKED_DEVICE_BUFFER_H_

#include <array>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/raw_buffer.h"

namespace xla {

class AbstractTrackedDeviceBuffer {
 public:
  virtual ~AbstractTrackedDeviceBuffer() = default;

  // Construct (or return) a vector of tsl::AsyncValue events which
  // will become ready when this buffer is ready.
  virtual std::vector<tsl::RCReference<tsl::AsyncValue>>
  GetAsyncValueDefinitionEvents() = 0;

  // Construct (or return) a raw buffer which aliases the same
  // underlying memory as this AbstractTrackedDeviceBuffer.
  virtual tsl::RCReference<CommonPjRtRawBuffer> GetRawBuffer(
      PjRtMemorySpace* memory_space) = 0;

  // Only to be called via the result of
  // CommonPjRtBuffer::ScopedHold::ConvertUsageHold with an optional device
  // event to add to the usage events.
  virtual void AddUsageEvent(tsl::RCReference<PjRtDeviceEvent> event) = 0;

  // Only to be called by ScopedHold to mark a successful donation.
  virtual void ConfirmDonation() = 0;

  // Asynchronously frees all memory.
  virtual void Delete(PjRtMemorySpace* memory_space) = 0;

  // Clones an abstract buffer with an additional control dependency.
  virtual absl::StatusOr<std::unique_ptr<AbstractTrackedDeviceBuffer>>
  CloneWithControlDependency(PjRtMemorySpace* memory_space,
                             PjRtFuture<> dependency) {
    return absl::UnimplementedError(
        "DonateWithControlDependency is not supported.");
  }

  // Populates a future::promise when all the definition events are complete.
  virtual PjRtFuture<>::Promise GetReadyFuturePromise(
      PjRtMemorySpace* memory_space) {
    auto promise = PjRtFuture<>::CreatePromise();
    promise.Set(absl::UnimplementedError(
        absl::StrCat("GetReadyFuturePromise not supported for ",
                     memory_space->DebugString())));
    return promise;
  }

  // Waits for all usage and definition events to complete synchronously
  // and returns the status.
  virtual absl::Status BlockForOperationsToComplete(
      PjRtMemorySpace* memory_space) {
    return absl::UnimplementedError(
        absl::StrCat("BlockForOperationsToComplete not supported for ",
                     memory_space->DebugString()));
  }
};

class CommonPjRtBuffer : public PjRtBuffer {
 public:
  // Helper class to retain a "hold" on a CommonPjRtBuffer. A ScopedHold
  // may not outlive its parent CommonPjRtBuffer.
  //
  // There are three types of hold, as follows:
  //
  // 1) Usage hold: a transient hold while an operation using the buffer is
  //    being enqueued to the runtime.
  // A client acquires a usage hold by calling
  // CommonPjRtBuffer::GetBufferWithHold(kUsage) or the convenience
  // wrapper GetBufferWithUsageHold(). If the enqueue completes successfully the
  // hold should be released using a call to ConvertUsageHold. If the ScopedHold
  // is deleted without ConvertUsageHold being called, e.g., on error, the hold
  // is dropped. It is legal to drop a usage hold instead of calling
  // ConvertUsageHold, even if the buffer was successfully enqueued, as long as
  // the client ensures that all necessary synchronization has been done.
  //
  // 2) External hold: a potentially long-lived hold while the buffer is being
  //    shared by an external framework, e.g., NumPy.
  // A client acquires an external hold by calling
  // CommonPjRtBuffer::GetBufferWithHold(kExternal) or the convenience
  // wrapper GetBufferWithExternalReference and releases it by deleting the
  // ScopedHold. The external framework should not modify the underlying buffer
  // unless it is confident via its own synchronization that modifications do
  // not race with reads from the CommonPjRtBuffer.
  //
  // 3) Donation hold: a transient hold while an execution that donates the
  //    buffer is being enqueued to the runtime.
  // A client acquires a donation hold by calling
  // CommonPjRtBuffer::GetBufferWithHold(kDonation). If the enqueue
  // completes successfully the hold should be released using a call to
  // ConfirmDonation after which the buffer is invalid. If the ScopedHold is
  // deleted without ConfirmDonation being called, e.g., on error, the hold is
  // dropped and the buffer remains valid. If the buffer is successfully
  // enqueued the client *must* call ConfirmDonation.
  //
  // Donation holds behave like exclusive write locks: when a donation hold
  // has been acquired, any attempt to acquire another hold of any type will
  // block until the donation hold is dropped or confirmed. Acquiring a donation
  // hold will fail with an error if there is any outstanding external hold, and
  // will block if there are any outstanding usage holds until those holds are
  // dropped or converted.
  //
  // Calls to CommonPjRtBuffer::Release (and transitively to
  // CommonPjRtBuffer::Delete() and ~CommonPjRtBuffer()) will
  // block until all usage and donation holds are either deleted or
  // converted/confirmed.
  class ScopedHold {
   public:
    enum Type { kUsage = 0, kExternalReference, kDonation, kMaxValue };
    // Use a State enum instead of encoding the state in an error absl::Status
    // to avoid creating absl::Status values in non-error cases. Creating a
    // absl::Status entails several allocations and can add O(us) to every use
    // of a hold.
    enum State {
      kUninitialized = 0,
      kValid,
      kMoved,
      kConverted,
      kReleased,
      kDonated,
      kError
    };

    ~ScopedHold();
    ScopedHold(ScopedHold&& other);
    ScopedHold(const ScopedHold&) = delete;
    ScopedHold& operator=(const ScopedHold&) = delete;

    Type type() const { return type_; }

    absl::Status status() const;
    bool ok() const { return state_ == kValid; }

    // Access to the underlying device buffer storage. Requires this->ok().
    AbstractTrackedDeviceBuffer* buffer() const {
      CHECK_EQ(state_, kValid);
      CHECK_NE(buffer_ptr_, nullptr);
      return buffer_ptr_;
    }
    CommonPjRtBuffer* parent() const { return parent_; }

    // Confirms that the buffer was successfully donated to an execution.
    // Only valid for holds of type kDonation. Causes the buffer to become
    // invalid.
    void ConfirmDonation();

    // Converts the hold into a usage event. Only valid for holds of type
    // kUsage.
    void ConvertUsageHold(tsl::RCReference<PjRtDeviceEvent> event);

   protected:
    ScopedHold(CommonPjRtBuffer* parent, Type type)
        : parent_(parent), type_(type), state_(kUninitialized) {}

    // Sets buffer state.
    void SetState(State state) { state_ = state; }

   private:
    friend class CommonPjRtBuffer;

    // Acquires the unique ownership of the buffer. Called by parent_ to
    // initialize the donation hold.
    void AcquireDonation(
        absl::StatusOr<std::unique_ptr<AbstractTrackedDeviceBuffer>> buffer_or);

    // Acquires a non-owning reference of the buffer. Called by parent_ to
    // initialize the usage or external reference hold.
    void AcquireUsageOrExternalReference(
        absl::StatusOr<AbstractTrackedDeviceBuffer*> buffer_or);

    // Drops this hold. It resets `holds_` counters. If it is a donation hold
    // and an error occurs, it returns the device buffer to the
    // CommonPjRtBuffer.
    void DropHold();

    CommonPjRtBuffer* const parent_;
    const Type type_;

    // There is an invariant that if ok() then buffer_.value() != nullptr.
    State state_;
    absl::Status status_;
    // The non-owning pointer to the underlying buffer. It is not nullptr for
    // all types of holds.
    AbstractTrackedDeviceBuffer* buffer_ptr_ = nullptr;
    // If it is a donation hold, `buffer_` will not be nullptr. Otherwise, it is
    // a nullptr.
    std::unique_ptr<AbstractTrackedDeviceBuffer> buffer_;
  };

  bool IsDeleted() const override;

  absl::Status AcquireScopedRawBuffer(
      absl::AnyInvocable<absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>(
                             tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
                             std::vector<tsl::RCReference<tsl::AsyncValue>>
                                 definition_events) &&>
          scoped_acquire,
      const char* caller_name = "AcquireScopedRawBuffer");

  ScopedHold GetBufferWithHold(ScopedHold::Type type);

 protected:
  CommonPjRtBuffer(std::unique_ptr<AbstractTrackedDeviceBuffer> device_buffer,
                   PjRtMemorySpace* memory_space);
  ~CommonPjRtBuffer() override;

  // Blocks in mu_.Await until there are no more usage holds.
  void WaitForOutstandingUsageHolds() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Blocks in mu_.Await until there is no donation hold.
  void WaitForOutstandingDonationHold() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Adds a donation hold and returns device_buffer_. Returns an error if
  // device_buffer_ is null, or if a donation hold was requested when there is
  // an outstanding external hold.
  // Requires holds_[kDonation] == 0 (i.e., WaitForOutstandingDonationHolds()
  // must be called first.)
  absl::StatusOr<std::unique_ptr<AbstractTrackedDeviceBuffer>>
  GetBufferForDonationHoldLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Adds a hold of usage or external reference and returns non-owning
  // device_buffer_. Returns an error if device_buffer_ is null.
  // Requires holds_[kDonation] == 0 (i.e., WaitForOutstandingDonationHolds()
  // must be called first.)
  absl::StatusOr<AbstractTrackedDeviceBuffer*>
  GetBufferForUsageOrExternalHoldLocked(ScopedHold::Type type)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Adds a hold of hold->type() and initializes `hold` with device_buffer_.
  // Initializes hold with an error if device_buffer_ is null, or if a donation
  // hold was requested when there is an outstanding external hold.
  // Requires holds_[kDonation] == 0 (i.e., WaitForOutstandingDonationHolds()
  // must be called first.)
  void AcquireHoldLocked(ScopedHold* hold) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Drops a hold without taking any other action. Does a sanity check that
  // buffer==device_buffer_ or device_buffer_==nullptr.
  void DropUsageOrExternalHold(ScopedHold::Type type,
                               AbstractTrackedDeviceBuffer* buffer);

  // Drops a hold without taking any other action. Does a sanity check that
  // buffer==device_buffer_ or device_buffer_==nullptr.
  void DropDonationHold(std::unique_ptr<AbstractTrackedDeviceBuffer> buffer);

  // Drops a donation hold and makes *this invalid for further use. Does a
  // sanity check that buffer==device_buffer_. Called after device_buffer_ was
  // successfully donated to an execution.
  void ConfirmDonation(AbstractTrackedDeviceBuffer* device_buffer);

  void DecrementUsage() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    CHECK_GT(holds_[ScopedHold::kUsage], 0);
    --holds_[ScopedHold::kUsage];
  }

  std::unique_ptr<AbstractTrackedDeviceBuffer> ReleaseBuffer()
      ABSL_LOCKS_EXCLUDED(mu_);

  AbstractTrackedDeviceBuffer* device_buffer() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return device_buffer_.get();
  }

  mutable absl::Mutex mu_;
  PjRtFuture<>::Promise definition_promise_ ABSL_GUARDED_BY(mu_);
  PjRtMemorySpace* const memory_space_;

 private:
  std::unique_ptr<AbstractTrackedDeviceBuffer> device_buffer_
      ABSL_GUARDED_BY(mu_);
  // Count of holds on the buffer.
  std::array<int, ScopedHold::Type::kMaxValue> holds_ ABSL_GUARDED_BY(mu_);
};

}  // namespace xla

#endif  // XLA_PJRT_ABSTRACT_TRACKED_DEVICE_BUFFER_H_
