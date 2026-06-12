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

#ifndef XLA_TOOLS_COMPARE_LITERALS_COMPARE_LITERALS_IMPL_H_
#define XLA_TOOLS_COMPARE_LITERALS_COMPARE_LITERALS_IMPL_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "xla/error_spec.h"
#include "xla/tools/run_hlo_module.pb.h"

namespace xla {

// Diff of the result and argument literals between two iterations of a
// RunHloModuleLiterals.
struct IterationDiff {
  // Index of the iteration in the RunHloModuleLiterals.
  int iteration_index = -1;
  // Whether there is a diff between the result or argument literals of this
  // iteration.
  bool has_diff = false;
  // Diff between the result literals of this iteration.
  std::string diff_str;
  // Diff between the argument literals of this iteration.
  // This is a concatenation of all the argument diffs.
  std::string argument_diffs;
};

// Diff between two RunHloModuleLiterals.
struct DiffResult {
  bool has_diff = false;
  std::vector<IterationDiff> iteration_diffs;
};

// Compares two RunHloModuleLiterals and populates diff_result. If diff_result
// already contains data, new iteration diffs will be appended to the end of its
// iteration_diffs vector. Returns kDataLoss if any literals differ,
// InvalidArgumentError if the number of iterations doesn't match, or OK if they
// are identical.
absl::Status CompareResults(const RunHloModuleLiterals& result1,
                            const RunHloModuleLiterals& result2,
                            ErrorSpec error_spec, DiffResult* diff_result);

// Compares two LiteralProto protos and returns an error if they are not
// approximately equal.
absl::Status CompareLiterals(const LiteralProto& lit1, const LiteralProto& lit2,
                             ErrorSpec error_spec);
}  // namespace xla

#endif  // XLA_TOOLS_COMPARE_LITERALS_COMPARE_LITERALS_IMPL_H_
