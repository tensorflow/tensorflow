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

#include "tensorflow/core/framework/metrics.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "tensorflow/core/lib/monitoring/cell_reader.h"

namespace {
using ::tensorflow::metrics::IncrementPhase2XlaCompilerCounter;
using ::tensorflow::metrics::Phase2XlaCompilerMetric;
using ::tensorflow::monitoring::testing::CellReader;

constexpr char kPhase2CompilationStatusStreamzName[] =
    "/tensorflow/core/tf2xla/api/v2/phase2_compilation_status";
constexpr char kMlirWithFallbackModeSuccess[] = "kMlirWithFallbackModeSuccess";
constexpr char kPhase2XlaCompilerStreamzName[] =
    "/tensorflow/compiler/tf2xla/xla_compiler/compilation_status";
constexpr char kCompileSingleOpXlaBuilderSuccess[] =
    "kCompileSingleOpMlirSuccess";
constexpr char kCompileSingleOpXlaBuilderFailure[] =
    "kCompileSingleOpXlaBuilderFailure";
constexpr char kCompileSingleOpMlirSuccess[] = "kCompileSingleOpMlirSuccess";
constexpr char kCompileSingleOpMlirFailure[] = "kCompileSingleOpMlirFailure";
constexpr char kCompileFunctionXlaBuilderSuccess[] =
    "kCompileFunctionMlirSuccess";
constexpr char kCompileFunctionXlaBuilderFailure[] =
    "kCompileFunctionXlaBuilderFailure";
constexpr char kCompileFunctionMlirSuccess[] = "kCompileFunctionMlirSuccess";
constexpr char kCompileFunctionMlirFailure[] = "kCompileFunctionMlirFailure";

TEST(Metrics, Phase2XlaCompilerMetric) {
  CellReader<int64_t> counter(kPhase2XlaCompilerStreamzName);

  IncrementPhase2XlaCompilerCounter(
      Phase2XlaCompilerMetric::kCompileSingleOpXlaBuilderSuccess);
  IncrementPhase2XlaCompilerCounter(
      Phase2XlaCompilerMetric::kCompileSingleOpXlaBuilderFailure);
  IncrementPhase2XlaCompilerCounter(
      Phase2XlaCompilerMetric::kCompileSingleOpMlirSuccess);
  IncrementPhase2XlaCompilerCounter(
      Phase2XlaCompilerMetric::kCompileSingleOpMlirFailure);
  IncrementPhase2XlaCompilerCounter(
      Phase2XlaCompilerMetric::kCompileFunctionXlaBuilderSuccess);
  IncrementPhase2XlaCompilerCounter(
      Phase2XlaCompilerMetric::kCompileFunctionXlaBuilderFailure);
  IncrementPhase2XlaCompilerCounter(
      Phase2XlaCompilerMetric::kCompileFunctionMlirSuccess);
  IncrementPhase2XlaCompilerCounter(
      Phase2XlaCompilerMetric::kCompileFunctionMlirFailure);

  ASSERT_EQ(counter.Read(kCompileSingleOpXlaBuilderSuccess), 1);
  ASSERT_EQ(counter.Read(kCompileSingleOpXlaBuilderFailure), 1);
  ASSERT_EQ(counter.Read(kCompileSingleOpMlirSuccess), 1);
  ASSERT_EQ(counter.Read(kCompileSingleOpMlirFailure), 1);
  ASSERT_EQ(counter.Read(kCompileFunctionXlaBuilderSuccess), 1);
  ASSERT_EQ(counter.Read(kCompileFunctionXlaBuilderFailure), 1);
  ASSERT_EQ(counter.Read(kCompileFunctionMlirSuccess), 1);
  ASSERT_EQ(counter.Read(kCompileFunctionMlirFailure), 1);
}

TEST(Metrics, Phase2ComilationStatusCounterIncremented) {
  CellReader<int64_t> counter(kPhase2CompilationStatusStreamzName);

  tensorflow::metrics::IncrementTfMlirBridgeSecondPhaseCounter(
      tensorflow::metrics::MlirBridgeSecondPhaseMetric::
          kMlirWithFallbackModeSuccess);

  ASSERT_EQ(counter.Read(kMlirWithFallbackModeSuccess), 1);
}

TEST(Metrics, Phase2ComilationStatusUntouchedCounterNotIncremented) {
  CellReader<int64_t> counter(kPhase2CompilationStatusStreamzName);

  tensorflow::metrics::IncrementTfMlirBridgeSecondPhaseCounter(
      tensorflow::metrics::MlirBridgeSecondPhaseMetric::
          kMlirWithFallbackModeFailure);

  ASSERT_EQ(counter.Read(kMlirWithFallbackModeSuccess), 0);
}

}  // namespace
