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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_TEST_MATCHERS_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_TEST_MATCHERS_H_

#include <gmock/gmock.h>
#include "absl/status/statusor.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v1/compile_mlir_util.h"
#include "xla/tsl/platform/statusor.h"

template <typename T>
bool WasGraphAnalysisFailure(const absl::StatusOr<T>& status) {
  return (status.status() ==
          tensorflow::CompileToHloGraphAnalysisFailedError());
}

/* The third party version of the Graph Analysis always returns disabled so
 * these matchers short circuit on that error. */
MATCHER(IsOkOrFiltered,
        "Status was OK or equal to the Graph Analysis failure") {
  bool is_ok = arg.ok();
  auto graph_analysis_failure = WasGraphAnalysisFailure(arg);
  return testing::ExplainMatchResult(
      testing::IsTrue(), is_ok || graph_analysis_failure, result_listener);
}

MATCHER_P2(IncrementedOrFiltered, metric, value,
           "Metric was incremented by value or Status equal to the Graph "
           "Analysis failure") {
  auto graph_analysis_failure = WasGraphAnalysisFailure(arg);
  if (graph_analysis_failure) {
    return testing::ExplainMatchResult(testing::IsTrue(),
                                       graph_analysis_failure, result_listener);
  }
  return testing::ExplainMatchResult(testing::Eq(metric), value,
                                     result_listener);
}

MATCHER_P(ComputationProtoContains, regex,
          "If not a Graph Analysis failure then matches the computation result "
          "with the regex") {
  auto graph_analysis_failure = WasGraphAnalysisFailure(arg);
  if (graph_analysis_failure) {
    return testing::ExplainMatchResult(testing::IsTrue(),
                                       graph_analysis_failure, result_listener);
  }
  auto proto = arg.value().computation->proto().DebugString();
  return testing::ExplainMatchResult(testing::ContainsRegex(regex), proto,
                                     result_listener);
}

MATCHER_P(XlaComputationProtoContains, regex,
          "If not a Graph Analysis failure then matches the computation result "
          "with the regex") {
  auto graph_analysis_failure = WasGraphAnalysisFailure(arg);
  if (graph_analysis_failure) {
    return testing::ExplainMatchResult(testing::IsTrue(),
                                       graph_analysis_failure, result_listener);
  }
  auto proto = arg.value().proto().DebugString();
  return testing::ExplainMatchResult(testing::ContainsRegex(regex), proto,
                                     result_listener);
}

MATCHER_P(
    HasMlirModuleWith, expected,
    "If not a Graph Analysis failure then matches the mlir module result") {
  auto graph_analysis_failure = WasGraphAnalysisFailure(arg);
  if (graph_analysis_failure) {
    return testing::ExplainMatchResult(testing::IsTrue(),
                                       graph_analysis_failure, result_listener);
  }
  auto actual = arg.value();
  return testing::ExplainMatchResult(testing::ContainsRegex(expected), actual,
                                     result_listener);
}

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_TEST_MATCHERS_H_
