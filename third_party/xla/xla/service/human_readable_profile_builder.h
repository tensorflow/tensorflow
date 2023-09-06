/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_HUMAN_READABLE_PROFILE_BUILDER_H_
#define XLA_SERVICE_HUMAN_READABLE_PROFILE_BUILDER_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/types.h"
#include "tsl/platform/logging.h"

namespace xla {

// HumanReadableProfileBuilder helps you create a textual profile of a
// computation, suitable for consumption by humans.
class HumanReadableProfileBuilder {
 public:
  explicit HumanReadableProfileBuilder(absl::string_view computation_name,
                                       bool is_entry_computation,
                                       int64_t total_cycles,
                                       double clock_rate_ghz)
      : computation_name_(computation_name),
        is_entry_computation_(is_entry_computation),
        total_cycles_(total_cycles),
        clock_rate_ghz_(clock_rate_ghz) {
    CHECK_GE(clock_rate_ghz, 1e-9);
  }

  int64_t total_cycles() const { return total_cycles_; }

  // Adds an operation to the profile.  If you don't know the number of
  // floating-point ops or bytes touched by the op, or if you don't know how
  // fast it would run optimally, pass -1 for that param.
  void AddOp(absl::string_view op_name, absl::string_view short_name,
             absl::string_view category, int64_t cycles, int64_t flop_count,
             int64_t transcendental_count, int64_t bytes_accessed,
             float optimal_seconds) {
    op_infos_.push_back({std::string(op_name), std::string(short_name),
                         std::string(category), cycles, flop_count,
                         transcendental_count, bytes_accessed,
                         optimal_seconds});
  }

  // Gets the human-readable profile.
  std::string ToString() const;

 private:
  struct OpInfo {
    std::string name;
    std::string short_name;
    std::string category;
    int64_t cycles;
    int64_t flop_count;  // -1 if unknown
    int64_t transcendental_count;
    int64_t bytes_accessed;  // -1 if unknown
    float optimal_seconds;   // -1 if unknown
  };

  double CyclesToSeconds(int64_t cycles) const {
    return cycles / clock_rate_ghz_ / 1e9;
  }
  double CyclesToMicroseconds(int64_t cycles) const {
    return cycles / clock_rate_ghz_ / 1000.0;
  }

  std::string computation_name_;
  bool is_entry_computation_;
  int64_t total_cycles_;
  double clock_rate_ghz_;
  std::vector<OpInfo> op_infos_;
};

}  // namespace xla

#endif  // XLA_SERVICE_HUMAN_READABLE_PROFILE_BUILDER_H_
