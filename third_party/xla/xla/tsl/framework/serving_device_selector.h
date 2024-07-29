/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XLA_TSL_FRAMEWORK_SERVING_DEVICE_SELECTOR_H_
#define XLA_TSL_FRAMEWORK_SERVING_DEVICE_SELECTOR_H_

#include <cstdint>
#include <deque>
#include <optional>

#include "absl/container/fixed_array.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tsl/platform/logging.h"

namespace tsl {

class ServingDeviceSelector;

// A RAII type for device reservation.
class DeviceReservation {
 public:
  DeviceReservation(int device_index, ServingDeviceSelector* selector);
  ~DeviceReservation();

  DeviceReservation(const DeviceReservation&) = delete;
  DeviceReservation& operator=(const DeviceReservation&) = delete;

  DeviceReservation(DeviceReservation&& r);
  DeviceReservation& operator=(DeviceReservation&& r);

  int device_index() const { return device_index_; }

  void reset();

 private:
  int device_index_;
  ServingDeviceSelector* device_selector_;
};

// Interface for runtime device selection for serving.
// NOTE: This interface is experimental and subject to change.
class ServingDeviceSelector {
 public:
  // Tracks the running average of certain program execution time.
  class RunningAverage {
   public:
    void Add(int64_t value) {
      DCHECK_GE(value, 0);
      sum_ += value;
      ++count_;
      latency_ = sum_ / count_;
    }

    int64_t Get() const { return latency_; }

   private:
    int64_t sum_ = 0;
    int64_t count_ = 0;
    int64_t latency_ = 0;
  };

  // Tracks the program execution information, including execution time.
  class ExecutionInfo {
   public:
    explicit ExecutionInfo(int64_t num_prefetch_result = 1)
        : running_average_(num_prefetch_result) {}

    virtual ~ExecutionInfo() = default;

    void AddTime(int64_t value, int result) {
      DCHECK_GE(value, 0);
      DCHECK_LT(result, running_average_.size());
      running_average_.at(result).Add(value);
    }

    int64_t GetTime(int result) const {
      DCHECK_LT(result, running_average_.size());
      return running_average_.at(result).Get();
    }

    // To be conservative when one of the path is missing.
    virtual int64_t MaybeGetValidTime(int result) const {
      return GetTime(result);
    }

   private:
    // Records program average execution time, one for each prefetch result.
    absl::FixedArray<RunningAverage> running_average_;
  };

  struct DeviceState {
    explicit DeviceState(int64_t priority_count = 1)
        : enqueued_programs(priority_count),
          scheduled_programs(priority_count) {}
    // TODO(b/295352859): Add more stats to track that are useful for the Policy
    // to use when selecting a device.
    struct ProgramInfo {
      std::string fingerprint;
      int32_t priority;
      int64_t req_id = -1;
      const ExecutionInfo* execution_info;
      int prefetch_results;
    };
    // A queue of enqueued programs, one for each priority level
    absl::FixedArray<std::deque<ProgramInfo>> enqueued_programs;
    // A queue of scheduled yet enqueued programs, one for each priority level.
    // May or may not have fingerprint.
    absl::FixedArray<std::deque<ProgramInfo>> scheduled_programs;
    // Timestamp in nanoseconds of last started program.
    int64_t last_started_ns = 0;
    // Fingerprint of last enqueued high priority program.
    std::string last_fingerprint;
    // The number of scheduled not yet enqueued programs with unknown
    // fingerprints.
    int32_t unknown_fingerprint_requests;
    // Whether execution timer was reset, true iff a program is enqueued while
    // all queues (for all priorities) were empty.
    bool timer_reset = true;
  };

  // Struct of all tracked device states, which will be passed to Policy.
  struct DeviceStates {
    absl::Span<const DeviceState> states;
  };

  // Policy used to select a device.
  class Policy {
   public:
    virtual ~Policy() = default;
    // Selects a device based on the tracked states of all devices.
    virtual int SelectDevice(absl::string_view program_fingerprint,
                             const DeviceStates& device_states) = 0;
  };

  virtual ~ServingDeviceSelector() = default;

  // Reserves a device according to a given selection policy. The reserved
  // device will be freed when the lifetime of the returned `DeviceReservation`
  // object ends.
  virtual DeviceReservation ReserveDevice(
      absl::string_view program_fingerprint) = 0;

  // Enqueues a program on the given device. Used only for load tracking
  // purposes when the device selection feature is unused.
  virtual void Enqueue(int32_t device_index, absl::string_view fingerprint) = 0;

  // Marks the completion of a program on the given device. Used only for load
  // tracking purposes when the device selection feature is unused.
  virtual void Completed(int32_t device_index, bool had_error) = 0;

 protected:
  // A helper function for Enqueue. The EnqueueHelper does the following things.
  //  1. If there are programs in the scheduled_programs queue of the given
  //     priority, move the program to the corresponding enqueued_programs
  //     queue. Update the fingerprint if it is unknown. This is a typical TF1
  //     use case.
  //  2. If there are no programs in the scheduled_programs queue of the given
  //     priority, create the program of the fingerprint and place it in the
  //     corresponding enqueued_programs queue.
  //     This can happen in two cases: (1) TFRT that doesn't need
  //     scheduled_programs queue. (2) In TF1, Schedule() was not called prior
  //     to Enqueue().
  // This helper also updates last_started_ns and timer_reset.
  static void EnqueueHelper(DeviceState& device_state, int32_t device_index,
                            ExecutionInfo& execution_info,
                            absl::string_view fingerprint, int32_t priority,
                            int64_t req_id, size_t priority_queue_count,
                            int prefetch_results, int64_t now_ns);
  // A helper function tells a program has completed on the given device.
  static void CompletedHelper(DeviceState& device_state, int32_t device_index,
                              int32_t priority,
                              std::optional<int64_t>& min_exec_time,
                              bool had_error, int64_t now_ns);
  // Helper to estimate the time until the core becomes idle in nanoseconds.
  // Only considers queues with priority at least as high as 'priority'.
  static int64_t EstimateTimeTillIdleNs(const DeviceState& device_state,
                                        int32_t priority, int64_t min_exec_time,
                                        int64_t now_ns);

 private:
  friend DeviceReservation;

  // Frees the given device reservation.
  virtual void FreeDeviceReservation(const DeviceReservation& reservation) = 0;
};

}  // namespace tsl

#endif  // XLA_TSL_FRAMEWORK_SERVING_DEVICE_SELECTOR_H_
