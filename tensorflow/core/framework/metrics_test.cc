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
#include "tensorflow/core/lib/monitoring/test_utils.h"

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

TEST(Metrics, TFDataClientGetElementAction) {
  CellReader<int64_t> counter(
      "/tensorflow/data/service/client_routing_outcome");

  tensorflow::metrics::RecordTFDataClientGetElementAction(
      "success", "client_1", "worker_1", "thread_0");
  tensorflow::metrics::RecordTFDataClientGetElementAction(
      "skip_empty_buffer", "client_1", "worker_1", "thread_0");
  tensorflow::metrics::RecordTFDataClientGetElementAction(
      "skip_empty_buffer", "client_1", "worker_1", "thread_1");
  tensorflow::metrics::RecordTFDataClientGetElementAction(
      "skip_error", "client_2", "worker_2", "thread_0");

  EXPECT_EQ(counter.Read("success", "client_1", "worker_1", "thread_0"), 1);
  EXPECT_EQ(
      counter.Read("skip_empty_buffer", "client_1", "worker_1", "thread_0"), 1);
  EXPECT_EQ(
      counter.Read("skip_empty_buffer", "client_1", "worker_1", "thread_1"), 1);
  EXPECT_EQ(counter.Read("skip_error", "client_2", "worker_2", "thread_0"), 1);
}

TEST(Metrics, TFDataPrefetchResidenceTime) {
  CellReader<tensorflow::monitoring::testing::Histogram> sampler(
      "/tensorflow/data/prefetch_residence_time_usecs");

  tensorflow::metrics::RecordTFDataPrefetchResidenceTime("node_1", 1000);
  tensorflow::metrics::RecordTFDataPrefetchResidenceTime("node_1", 2000);
  tensorflow::metrics::RecordTFDataPrefetchResidenceTime("node_2", 3000);

  EXPECT_EQ(sampler.Read("node_1").num(), 2);
  EXPECT_EQ(sampler.Read("node_2").num(), 1);
}

TEST(Metrics, TFDataPrefetchEnqueue) {
  CellReader<int64_t> counter("/tensorflow/data/prefetch_buffer");

  tensorflow::metrics::RecordTFDataPrefetchEnqueue("node_1");
  tensorflow::metrics::RecordTFDataPrefetchEnqueue("node_1");
  tensorflow::metrics::RecordTFDataPrefetchEnqueue("node_2");

  EXPECT_EQ(counter.Read("node_1", "enqueue"), 2);
  EXPECT_EQ(counter.Read("node_2", "enqueue"), 1);
}

TEST(Metrics, TFDataPrefetchDequeue) {
  CellReader<int64_t> counter("/tensorflow/data/prefetch_buffer");

  tensorflow::metrics::RecordTFDataPrefetchDequeue("node_1");
  tensorflow::metrics::RecordTFDataPrefetchDequeue("node_2");

  EXPECT_EQ(counter.Read("node_1", "dequeue"), 1);
  EXPECT_EQ(counter.Read("node_2", "dequeue"), 1);
}

TEST(Metrics, TFDataPrefetchBufferSize) {
  CellReader<int64_t> gauge("/tensorflow/data/prefetch_buffer_size");

  tensorflow::metrics::RecordTFDataPrefetchBufferSize("node_1", 5);
  EXPECT_EQ(gauge.Read("node_1"), 5);

  tensorflow::metrics::RecordTFDataPrefetchBufferSize("node_1", 3);
  EXPECT_EQ(gauge.Read("node_1"), 3);
}

TEST(Metrics, UpdateXlaCompilationTime) {
  CellReader<int64_t> start_counter(
      "/tensorflow/core/xla_compilation_start_time");
  CellReader<int64_t> end_counter("/tensorflow/core/xla_compilation_end_time");
  CellReader<int64_t> counter("/tensorflow/core/xla_compilations");
  CellReader<int64_t> time_counter(
      "/tensorflow/core/xla_compilation_time_usecs");

  tensorflow::metrics::UpdateXlaCompilationStartTime(100);
  tensorflow::metrics::UpdateXlaCompilationTime(500, 600);

  EXPECT_EQ(start_counter.Read(), 100);
  EXPECT_EQ(end_counter.Read(), 600);
  EXPECT_EQ(counter.Read(), 1);
  EXPECT_EQ(time_counter.Read(), 500);
}

}  // namespace
