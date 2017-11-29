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

#ifndef THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PROFILE_PRINTER_H_
#define THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PROFILE_PRINTER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/types.h"

namespace xla {
// Instances of this class can pretty-print profile counters gathered from
// running an XLA computation without having access to the backing module.
class HloProfilePrinter {
 public:
  // Holds meta information about an HloInstruction.
  //
  // The pointer-typed fields can be owning or non-owning -- this decision is
  // manifested as the deleter_ function in the containing HloProfilePrinter.
  struct HloInstructionInfo {
    // Textual information for pretty printing.
    const char* long_name;
    const char* short_name;
    const char* category;

    // Metrics computed by HloCostAnalysis.
    float flop_count;
    float transcendental_count;
    float bytes_accessed;
    float optimal_seconds;

    // The index into the profile counters array for the HloInstruction
    // corresponding to this HloInstructionInfo.
    int64 profile_index;
  };

  // Holds meta information about an HloComputation.
  //
  // The pointer-typed fields can be owning or non-owning -- this decision is
  // manifested as the deleter_ function in the containing HloProfilePrinter.
  struct HloComputationInfo {
    const char* name;

    // The index into the profile counters array for the HloInstruction
    // corresponding to this HloComputationInfo.
    int64 profile_index;

    HloInstructionInfo* instructions;
    int64 instructions_size;
  };

  HloProfilePrinter(
      HloComputationInfo* computation_infos, int64 computation_infos_size,
      std::function<void(HloComputationInfo*, int64)> deleter = nullptr)
      : computation_infos_(computation_infos),
        computation_infos_size_(computation_infos_size),
        deleter_(std::move(deleter)) {}

  HloProfilePrinter(HloProfilePrinter&& other) {
    std::swap(other.computation_infos_, computation_infos_);
    std::swap(other.computation_infos_size_, computation_infos_size_);
    std::swap(other.deleter_, deleter_);
  }

  HloProfilePrinter(const HloProfilePrinter&) = delete;
  HloProfilePrinter& operator=(const HloProfilePrinter&) = delete;

  // Convert the profile counter sequence `counters` to a human readable string
  // representation.
  string ToString(const int64* counters, double clock_rate_ghz) const;

  ~HloProfilePrinter();

 private:
  // The `computation_infos_` field can be owning or non-owning -- this decision
  // is manifested as the deleter_ function.
  HloComputationInfo* computation_infos_ = nullptr;
  int64 computation_infos_size_ = 0;
  std::function<void(HloComputationInfo*, int64)> deleter_;
};
}  // namespace xla

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PROFILE_PRINTER_H_
