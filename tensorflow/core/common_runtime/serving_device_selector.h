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

#include <deque>
#include <memory>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

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
  // The state for a single device.
  struct DeviceState {
    // TODO(b/295352859): Add more stats to track that are useful for the Policy
    // to use when selecting a device.
    struct ProgramInfo {
      absl::string_view fingerprint;
      int64_t req_id = -1;
    };
    std::deque<ProgramInfo> scheduled_programs;
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
