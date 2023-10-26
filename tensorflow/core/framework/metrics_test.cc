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
using ::tensorflow::monitoring::testing::CellReader;

constexpr char kPhase2CompilationStatusStreamzName[] =
    "/tensorflow/core/tf2xla/api/v2/phase2_compilation_status";
constexpr char kMlirWithFallbackModeSuccess[] = "kMlirWithFallbackModeSuccess";

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
