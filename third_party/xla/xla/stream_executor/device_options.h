/* Copyright 2015 The OpenXLA Authors.

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

// Contains device-level options that can be specified at a platform level.
// Example usage:
//    auto device_options = DeviceOptions::Default();

#ifndef XLA_STREAM_EXECUTOR_DEVICE_OPTIONS_H_
#define XLA_STREAM_EXECUTOR_DEVICE_OPTIONS_H_

#include <map>
#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "tsl/platform/logging.h"

namespace stream_executor {

// Indicates a set of options for a device's usage, which generally must be
// provided at StreamExecutor device-initialization time.
//
// These are intended to be useful-but-not-mandatorily-supported options for
// using devices on the underlying platform. Presently, if the option requested
// is not available on the target platform, a warning will be emitted.
struct DeviceOptions {
 public:
  // When it is observed that more memory has to be allocated for thread stacks,
  // this flag prevents it from ever being deallocated. Potentially saves
  // thrashing the thread stack memory allocation, but at the potential cost of
  // some memory space.
  static constexpr unsigned kDoNotReclaimStackAllocation = 0x1;

  // The following options refer to synchronization options when
  // using SynchronizeStream or SynchronizeContext.

  // Synchronize with spinlocks.
  static constexpr unsigned kScheduleSpin = 0x02;
  // Synchronize with spinlocks that also call CPU yield instructions.
  static constexpr unsigned kScheduleYield = 0x04;
  // Synchronize with a "synchronization primitive" (e.g. mutex).
  static constexpr unsigned kScheduleBlockingSync = 0x08;

  static constexpr unsigned kMask = 0xf;  // Mask of all available flags.

  // Constructs an or-d together set of device options.
  explicit DeviceOptions(unsigned flags) : flags_(flags) {
    CHECK((flags & kMask) == flags);
  }

  // Factory for the default set of device options.
  static DeviceOptions Default() { return DeviceOptions(0); }

  unsigned flags() const { return flags_; }

  bool operator==(const DeviceOptions& other) const {
    return flags_ == other.flags_ &&
           non_portable_tags == other.non_portable_tags;
  }

  bool operator!=(const DeviceOptions& other) const {
    return !(*this == other);
  }

  std::string ToString() const {
    std::vector<std::string> flags_on;
    if (flags_ & kDoNotReclaimStackAllocation) {
      flags_on.push_back("kDoNotReclaimStackAllocation");
    }
    if (flags_ & kScheduleSpin) {
      flags_on.push_back("kScheduleSpin");
    }
    if (flags_ & kScheduleYield) {
      flags_on.push_back("kScheduleYield");
    }
    if (flags_ & kScheduleBlockingSync) {
      flags_on.push_back("kScheduleBlockingSync");
    }
    return flags_on.empty() ? "none" : absl::StrJoin(flags_on, "|");
  }

  // Platform-specific device options. Expressed as key-value pairs to avoid
  // DeviceOptions subclass proliferation.
  std::map<std::string, std::string> non_portable_tags;

 private:
  unsigned flags_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_DEVICE_OPTIONS_H_
