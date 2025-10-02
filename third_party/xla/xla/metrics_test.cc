/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/metrics.h"

#include <cstdint>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/globals.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

class MetricsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::SetVLogLevel("metrics", -1);
    GetGlobalMetricsHolder().Clear();
  }
};

TEST_F(MetricsTest, GlobalLogging) {
  absl::SetVLogLevel("metrics", 1);
  XLA_ENABLE_METRIC_FAMILY("test");
  XLA_ADD_METRIC_I64("test", "int_a", 12);
  XLA_ADD_METRIC_F64("test", "weights", 2.3);
  XLA_ADD_METRIC_STR("test", "scheduler", "list");

  auto metrics = GetGlobalMetricsHolder().GetMergedMetrics();
  EXPECT_THAT(metrics.int_metrics,
              ElementsAre(Pair("int_a", ElementsAre(Pair(kGlobalLoggingScope,
                                                         ElementsAre(12))))));
  EXPECT_THAT(
      metrics.double_metrics,
      ElementsAre(Pair("weights", ElementsAre(Pair(kGlobalLoggingScope,
                                                   ElementsAre(2.3))))));
  EXPECT_THAT(
      metrics.string_metrics,
      ElementsAre(Pair("scheduler", ElementsAre(Pair(kGlobalLoggingScope,
                                                     ElementsAre("list"))))));
}

TEST_F(MetricsTest, ScopedLogging) {
  absl::SetVLogLevel("metrics", 1);
  XLA_ENABLE_METRIC_FAMILY("test");
  XLA_SET_METRICS_SCOPE("scope");
  XLA_ADD_SCOPED_METRIC_I64("test", "int_a", 12);
  XLA_ADD_SCOPED_METRIC_F64("test", "weights", 2.3);
  XLA_ADD_SCOPED_METRIC_STR("test", "scheduler", "list");

  auto metrics = GetGlobalMetricsHolder().GetMergedMetrics();
  EXPECT_THAT(
      metrics.int_metrics,
      ElementsAre(Pair("int_a", ElementsAre(Pair("scope", ElementsAre(12))))));
  EXPECT_THAT(metrics.double_metrics,
              ElementsAre(Pair("weights",
                               ElementsAre(Pair("scope", ElementsAre(2.3))))));
  EXPECT_THAT(
      metrics.string_metrics,
      ElementsAre(
          Pair("scheduler", ElementsAre(Pair("scope", ElementsAre("list"))))));
}

TEST_F(MetricsTest, MoreThanOneKeys) {
  absl::SetVLogLevel("metrics", 1);
  XLA_ENABLE_METRIC_FAMILY("test");
  XLA_SET_METRICS_SCOPE("scope");
  XLA_ADD_SCOPED_METRIC_I64("test", "int_a", 12);
  XLA_ADD_SCOPED_METRIC_F64("test", "weights_a", 2.3);
  XLA_ADD_SCOPED_METRIC_STR("test", "scheduler_a", "list");

  XLA_ADD_SCOPED_METRIC_I64("test", "int_b", 112);
  XLA_ADD_SCOPED_METRIC_F64("test", "weights_b", 12.3);
  XLA_ADD_SCOPED_METRIC_STR("test", "scheduler_b", "post_order");

  auto metrics = GetGlobalMetricsHolder().GetMergedMetrics();
  EXPECT_THAT(metrics.int_metrics,
              UnorderedElementsAre(
                  Pair("int_a", ElementsAre(Pair("scope", ElementsAre(12)))),
                  Pair("int_b", ElementsAre(Pair("scope", ElementsAre(112))))));
  EXPECT_THAT(
      metrics.double_metrics,
      UnorderedElementsAre(
          Pair("weights_a", ElementsAre(Pair("scope", ElementsAre(2.3)))),
          Pair("weights_b", ElementsAre(Pair("scope", ElementsAre(12.3))))));
  EXPECT_THAT(
      metrics.string_metrics,
      UnorderedElementsAre(
          Pair("scheduler_a", ElementsAre(Pair("scope", ElementsAre("list")))),
          Pair("scheduler_b",
               ElementsAre(Pair("scope", ElementsAre("post_order"))))));
}

TEST_F(MetricsTest, MoreThanOneValues) {
  absl::SetVLogLevel("metrics", 1);
  XLA_ENABLE_METRIC_FAMILY("test");
  XLA_SET_METRICS_SCOPE("scope");
  XLA_ADD_SCOPED_METRIC_I64("test", "int_a", 12);
  XLA_ADD_SCOPED_METRIC_F64("test", "weights_a", 2.3);
  XLA_ADD_SCOPED_METRIC_STR("test", "scheduler_a", "list");

  XLA_ADD_SCOPED_METRIC_I64("test", "int_a", 112);
  XLA_ADD_SCOPED_METRIC_F64("test", "weights_a", 12.3);
  XLA_ADD_SCOPED_METRIC_STR("test", "scheduler_a", "post_order");

  auto metrics = GetGlobalMetricsHolder().GetMergedMetrics();
  EXPECT_THAT(
      metrics.int_metrics,
      UnorderedElementsAre(Pair(
          "int_a", ElementsAre(Pair("scope", UnorderedElementsAre(12, 112))))));
  EXPECT_THAT(
      metrics.double_metrics,
      UnorderedElementsAre(
          Pair("weights_a",
               ElementsAre(Pair("scope", UnorderedElementsAre(2.3, 12.3))))));
  EXPECT_THAT(metrics.string_metrics,
              UnorderedElementsAre(Pair(
                  "scheduler_a",
                  ElementsAre(Pair(
                      "scope", UnorderedElementsAre("list", "post_order"))))));
}

TEST_F(MetricsTest, MoreThanOneScopes) {
  absl::SetVLogLevel("metrics", 1);
  XLA_ENABLE_METRIC_FAMILY("test");
  XLA_SET_METRICS_SCOPE("scope_a");
  XLA_ADD_SCOPED_METRIC_I64("test", "int_a", 12);
  XLA_ADD_SCOPED_METRIC_F64("test", "weights", 2.3);
  XLA_ADD_SCOPED_METRIC_STR("test", "scheduler", "list");
  XLA_SET_METRICS_SCOPE("scope_b");
  XLA_ADD_SCOPED_METRIC_I64("test", "int_a", 112);
  XLA_ADD_SCOPED_METRIC_F64("test", "weights", 12.3);
  XLA_ADD_SCOPED_METRIC_STR("test", "scheduler", "post_order");

  auto metrics = GetGlobalMetricsHolder().GetMergedMetrics();
  EXPECT_THAT(
      metrics.int_metrics,
      ElementsAre(Pair(
          "int_a", UnorderedElementsAre(Pair("scope_a", ElementsAre(12)),
                                        Pair("scope_b", ElementsAre(112))))));
}

TEST_F(MetricsTest, MoreThanOneThreads) {
  absl::SetVLogLevel("metrics", 1);
  XLA_ENABLE_METRIC_FAMILY("test");
  std::vector<std::thread> threads;
  for (int i = 1; i <= 3; ++i) {
    threads.push_back(std::thread([i]() {
      XLA_SET_METRICS_SCOPE("scope_a");
      XLA_ADD_SCOPED_METRIC_I64("test", "int_a", i);
      XLA_ADD_SCOPED_METRIC_I64("test", "int_a", i * 10);
      XLA_SET_METRICS_SCOPE("scope_b");
      XLA_ADD_SCOPED_METRIC_I64("test", "int_a", i * 100);
      XLA_ADD_SCOPED_METRIC_I64("test", "int_a", i * 1000);
    }));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  auto metrics = GetGlobalMetricsHolder().GetMergedMetrics();
  EXPECT_THAT(
      metrics.int_metrics,
      ElementsAre(Pair(
          "int_a",
          UnorderedElementsAre(
              Pair("scope_a", UnorderedElementsAre(1, 10, 2, 20, 3, 30)),
              Pair("scope_b",
                   UnorderedElementsAre(100, 1000, 200, 2000, 300, 3000))))));
}

int64_t FunctionShouldNotBeCalledI64(bool fail = true) {
  EXPECT_FALSE(fail);
  return 12;
}

double FunctionShouldNotBeCalledF64(bool fail = true) {
  EXPECT_FALSE(fail);
  return 2.3;
}

std::string FunctionShouldNotBeCalledStr(bool fail = true) {
  EXPECT_FALSE(fail);
  return "simple_example";
}

TEST_F(MetricsTest, LoggingDisabled_DoesNotEvaluateNorLog) {
  XLA_ENABLE_METRIC_FAMILY("test");
  XLA_SET_METRICS_SCOPE("scope");
  XLA_ADD_SCOPED_METRIC_I64("test", "int_a", FunctionShouldNotBeCalledI64());
  XLA_ADD_SCOPED_METRIC_F64("test", "double_a", FunctionShouldNotBeCalledF64());
  XLA_ADD_SCOPED_METRIC_STR("test", "str_a", FunctionShouldNotBeCalledStr());

  auto metrics = GetGlobalMetricsHolder().GetMergedMetrics();
  EXPECT_THAT(metrics.int_metrics, IsEmpty());
  EXPECT_THAT(metrics.double_metrics, IsEmpty());
  EXPECT_THAT(metrics.string_metrics, IsEmpty());
}

TEST_F(MetricsTest, FamilyDisabled_DoesNotEvaluateNorLog) {
  absl::SetVLogLevel("metrics", 1);
  XLA_SET_METRICS_SCOPE("scope");
  XLA_ADD_SCOPED_METRIC_I64("test", "int_a", FunctionShouldNotBeCalledI64());
  XLA_ADD_SCOPED_METRIC_F64("test", "double_a", FunctionShouldNotBeCalledF64());
  XLA_ADD_SCOPED_METRIC_STR("test", "str_a", FunctionShouldNotBeCalledStr());

  auto metrics = GetGlobalMetricsHolder().GetMergedMetrics();
  EXPECT_THAT(metrics.int_metrics, IsEmpty());
  EXPECT_THAT(metrics.double_metrics, IsEmpty());
  EXPECT_THAT(metrics.string_metrics, IsEmpty());
}

}  // namespace
}  // namespace xla
