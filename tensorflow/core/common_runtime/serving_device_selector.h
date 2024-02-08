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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SERVING_DEVICE_SELECTOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SERVING_DEVICE_SELECTOR_H_

#include <cstdint>
#include <deque>

#include "absl/container/fixed_array.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

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
  // Enumerates the cases of prefetch hit and miss.
  enum class PrefetchResults { kPrefetchHit = 0, kPrefetchMiss };

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

    void AddTime(int64_t value,
                 PrefetchResults result = PrefetchResults::kPrefetchHit) {
      DCHECK_GE(value, 0);
      const int index = static_cast<int>(result);
      DCHECK_LT(index, running_average_.size());
      running_average_.at(index).Add(value);
    }

    int64_t GetTime(
        PrefetchResults result = PrefetchResults::kPrefetchHit) const {
      const int index = static_cast<int>(result);
      DCHECK_LT(index, running_average_.size());
      return running_average_.at(index).Get();
    }

    // To be conservative when one of the path is missing.
    int64_t MaybeGetValidTime(PrefetchResults result) const {
      const auto miss_time = GetTime(PrefetchResults::kPrefetchMiss);
      const auto hit_time = GetTime(PrefetchResults::kPrefetchHit);

      if (miss_time != 0 && hit_time != 0) {
        return result == PrefetchResults::kPrefetchMiss ? miss_time : hit_time;
      } else {
        auto dummy = std::max(miss_time, hit_time);
        return dummy;
      }
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
      absl::string_view fingerprint;
      int32_t priority;
      int64_t req_id = -1;
      ExecutionInfo* execution_info;
      PrefetchResults prefetch_results;
    };
    // A queue of enqueued programs, one for each priority level
    absl::FixedArray<std::deque<ProgramInfo>> enqueued_programs;
    // A queue of scheduled yet enqueued programs, one for each priority level.
    // May or may not have fingerprint.
    absl::FixedArray<std::deque<ProgramInfo>> scheduled_programs;
    // Timestamp in nanoseconds of last started program.
    int64_t last_started_ns = 0;
    // Fingerprint of last enqueued high priority program.
    absl::string_view last_fingerprint;
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

 private:
  friend DeviceReservation;

  // Frees the given device reservation.
  virtual void FreeDeviceReservation(const DeviceReservation& reservation) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SERVING_DEVICE_SELECTOR_H_
