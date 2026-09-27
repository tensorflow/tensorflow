/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_SEQUENCER_H_
#define XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_SEQUENCER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"

namespace xla::numerics::comparison {

// Reorders a stream of RecoveredTensorSummaryProto messages based on the data
// dependencies defined in an HLO module.
//
// The recovery process for tensor summaries from runtime logs can result in
// summaries that are slightly out of order locally due to race conditions. This
// class corrects the ordering by performing a topological sort that respects
// the HLO computation graphs and their call hierarchy.
//
// The process is designed to handle large sets of summaries that may not fit
// into memory. It operates in three main stages:
// 1. Read all tensor keys and their file offsets from the input file into
//    memory.
// 2. Sort these keys based on a detailed comparison function that understands
//    HLO data dependencies and call graph semantics.
// 3. Read the full summaries from the input file using the sorted offsets and
//    write them in the correct order to the output file.
class OriginalTensorSummarySequencer {
 public:
  // Creates a sequencer for the given HLO module.
  // This method pre-processes the module to build data structures (like
  // topological orderings of computations) to enable efficient sorting.
  static absl::StatusOr<std::unique_ptr<OriginalTensorSummarySequencer>> Create(
      const HloModule* original_module);

  explicit OriginalTensorSummarySequencer(
      absl::flat_hash_map<std::string, int64_t>&& topo_ranks)
      : topo_ranks_(std::move(topo_ranks)) {}

  // Reads RecoveredTensorSummaryProtos from `input_path`, sorts them according
  // to the HLO module's structure, and writes the sorted protos to
  // `output_path`. Both paths are expected to point to riegeli files
  // containing RecoveredTensorSummaryProto messages.
  absl::StatusOr<std::unique_ptr<IsOriginalTensorAlreadyRecoveredCallback>>
  Sequence(absl::string_view input_path, absl::string_view output_path) const;

 private:
  int64_t GetRank(absl::string_view instruction_name) const;
  // Comparison function to establish a total order on tensor keys based on
  // HLO execution semantics.
  bool CompareKeys(const AbsoluteScopedTensorKey& a,
                   const AbsoluteScopedTensorKey& b) const;

  // A map from each computation to a map of its instruction names to their
  // topological rank.
  absl::flat_hash_map<std::string, int64_t> topo_ranks_;
};

}  // namespace xla::numerics::comparison

#endif  // XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_SEQUENCER_H_
