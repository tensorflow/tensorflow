/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_HLO_COST_ANALYSIS_WRAPPER_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_HLO_COST_ANALYSIS_WRAPPER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"

namespace tensorflow {
namespace profiler {

// Returns the input bitwidths of the given HLO instruction.
std::vector<uint32_t> GetInputBitwidths(const xla::HloInstruction& hlo);

// Wrapper class for mixing in unified logic across different HloCostAnalysis
// implementations.
class HloCostAnalysisWrapper {
 public:
  virtual ~HloCostAnalysisWrapper() = default;

  // Factory function type for creating a HloCostAnalysisWrapper.
  using Factory = absl::AnyInvocable<std::unique_ptr<HloCostAnalysisWrapper>()>;
  // Function type for the cost adjustment function associated with a given HLO.
  // The input is the cost associated with the HLO instruction and the output is
  // the adjusted cost.
  using CostPerCoreFn = absl::AnyInvocable<int64_t(int64_t)>;
  // Type for mapping from Xprof memory space to XLA memory space. Used when
  // constructing the memory accessed breakdown of an HLO instruction.
  using MemorySpaceMap =
      absl::flat_hash_map<PerformanceInfo::MemoryAccessed::MemorySpace,
                          int64_t>;

  // Generates a PerformanceInfo proto for the given HLO instruction. Should be
  // used in conjunction with PerformanceInfoWrapper.
  std::unique_ptr<PerformanceInfo> GeneratePerformanceInfo(
      const xla::HloInstruction& hlo) const;

  // Returns the number of operations required to compute the HLO.
  int64_t GetModelFlops(const xla::HloInstruction& hlo) const;

  // Returns the normalized model flops against the peak bandwidth.
  int64_t GetDeviceFlops(const xla::HloInstruction& hlo) const;

  // Returns the underlying xla::HloCostAnalysis object. This wrapper still
  // maintains ownership of the object.
  virtual xla::HloCostAnalysis* GetXlaCostAnalysis() const = 0;

 protected:
  // Returns a function that can be used to adjust the associated costs for the
  // given HLO instruction.
  virtual CostPerCoreFn GetCostPerCoreFunction(
      const xla::HloInstruction& hlo) const {
    return tsl::profiler::ValidHloCost;
  };

  // Returns a mapping from Xprof memory space to XLA memory space.
  virtual MemorySpaceMap GetMemorySpaceMapping() const = 0;

  // Returns the flops adjustment for the given HLO instruction. (e.g. when
  // quantization is used)
  virtual int64_t GetDeviceFlopsAdjustment(
      const xla::HloInstruction& hlo) const = 0;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_HLO_COST_ANALYSIS_WRAPPER_H_
