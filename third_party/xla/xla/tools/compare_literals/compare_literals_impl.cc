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

#include "xla/tools/compare_literals/compare_literals_impl.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "xla/error_spec.h"
#include "xla/literal.h"
#include "xla/literal_comparison.h"
#include "xla/tools/run_hlo_module.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Compares two LiteralProto protos and returns an error if they are not
// approximately equal.
absl::Status CompareLiterals(const LiteralProto& lit1, const LiteralProto& lit2,
                             ErrorSpec error_spec) {
  ASSIGN_OR_RETURN(Literal literal1, Literal::CreateFromProto(lit1));
  ASSIGN_OR_RETURN(Literal literal2, Literal::CreateFromProto(lit2));

  absl::Status status = literal_comparison::Near(
      literal1, literal2, error_spec,
      /*detailed_message=*/true, /*miscompare_callback=*/nullptr);
  if (!status.ok()) {
    return absl::Status(absl::StatusCode::kDataLoss, status.message());
  }
  return absl::OkStatus();
}

namespace {
// Compares two RunHloModuleIterationLiterals protos and fills in the
// iteration_diff.
absl::Status CompareIterations(const RunHloModuleIterationLiterals& iter1,
                               const RunHloModuleIterationLiterals& iter2,
                               ErrorSpec error_spec,
                               IterationDiff* iteration_diff) {
  absl::Status status = absl::OkStatus();
  // Compare arguments of this iteration.
  for (int i = 0; i < iter1.arguments_size(); ++i) {
    absl::Status s_arg =
        CompareLiterals(iter1.arguments(i), iter2.arguments(i), error_spec);
    status.Update(s_arg);
    if (!s_arg.ok()) {
      iteration_diff->has_diff = true;
      absl::StrAppend(&iteration_diff->argument_diffs, "Argument ", i, ": ",
                      s_arg.message(), "\n");
    }
  }
  // Compare result literal of this iteration.
  absl::Status s_literal =
      CompareLiterals(iter1.result(), iter2.result(), error_spec);
  status.Update(s_literal);
  if (!s_literal.ok()) {
    iteration_diff->has_diff = true;
    absl::StrAppend(&iteration_diff->diff_str, s_literal.message(), "\n");
  }
  return status;
}
}  // namespace

// Compares two RunHloModuleLiterals protos and fills in the diff_result.
absl::Status CompareResults(const RunHloModuleLiterals& result1,
                            const RunHloModuleLiterals& result2,
                            ErrorSpec error_spec, DiffResult* diff_result) {
  if (result1.iterations_size() != result2.iterations_size()) {
    return absl::InvalidArgumentError(
        "Number of iterations in proto files do not match.");
  }

  if (result1.iterations_size() == 0) {
    return absl::InvalidArgumentError("No iterations found in proto files.");
  }

  absl::Status status = absl::OkStatus();
  for (int i = 0; i < result1.iterations_size(); ++i) {
    IterationDiff iteration_diff;
    iteration_diff.iteration_index = i;
    absl::Status s =
        CompareIterations(result1.iterations(i), result2.iterations(i),
                          error_spec, &iteration_diff);
    diff_result->iteration_diffs.push_back(iteration_diff);
    if (!s.ok()) {
      // Update() only updates status if s is not OK and status is OK. This
      // ensures that status is OK iff all CompareIterations calls return OK and
      // that status contains the first error message (kDataLoss) if there are
      // multiple.
      status.Update(s);
      diff_result->has_diff = true;
      continue;
    }
  }
  return status;
}
}  // namespace xla
