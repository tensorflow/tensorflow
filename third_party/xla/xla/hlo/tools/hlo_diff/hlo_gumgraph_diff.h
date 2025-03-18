/*
 * Copyright 2025 The OpenXLA Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_HLO_TOOLS_HLO_DIFF_HLO_GUMGRAPH_DIFF_H_
#define XLA_HLO_TOOLS_HLO_DIFF_HLO_GUMGRAPH_DIFF_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/tools/hlo_diff/graph/hlo_gumgraph.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_eval.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_summary.h"

namespace xla {
namespace hlo_diff {

// Options for computing the diff between two HLO modules.
struct DiffOptions {
  HloGumgraphFingerprintOptions fingerprint_options;
};

struct HloGumgraphDiffResults {
  std::unique_ptr<const DiffResult> diff_result;
  std::unique_ptr<const DiffSummary> diff_summary;
  std::unique_ptr<const DiffEval> diff_eval;
};

// Compares two HLO modules, computes and returns differences.
absl::StatusOr<HloGumgraphDiffResults> ComputeDiff(
    const HloModule& left, const HloModule& right,
    const DiffOptions& options = {}, bool run_eval = false);

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_HLO_GUMGRAPH_DIFF_H_
