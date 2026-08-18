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

#include "xla/tsl/lib/monitoring/callback_metric.h"

#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "xla/tsl/lib/monitoring/cell_reader.h"

namespace tsl::monitoring {
namespace {

TEST(CallbackMetricTest, TriggerAndRead) {
  auto* metric = CallbackMetric<int64_t, 1>::New(
      "/test/callback_metric", "Test callback metric", "label");

  int64_t value = 0;
  {
    CallbackTrigger trigger([&]() { metric->Set(value, "foo"); }, {metric});

    testing::CellReader<int64_t> reader("/test/callback_metric");

    value = 42;
    EXPECT_EQ(reader.Read("foo"), 42);

    value = 100;
    EXPECT_EQ(reader.Read("foo"), 100);
  }

  // After trigger is destroyed, the metric is no longer updated by the trigger.
  value = 200;
  testing::CellReader<int64_t> reader("/test/callback_metric");
  EXPECT_EQ(reader.Read("foo"), 100);

  delete metric;
}

TEST(CallbackMetricTest, MultipleMetrics) {
  auto metric1 = std::unique_ptr<CallbackMetric<int64_t, 1>>(
      CallbackMetric<int64_t, 1>::New("/test/callback_metric1",
                                      "Test callback metric 1", "label1"));
  auto metric2 = std::unique_ptr<CallbackMetric<std::string, 1>>(
      CallbackMetric<std::string, 1>::New("/test/callback_metric2",
                                          "Test callback metric 2", "label2"));

  int64_t value1 = 0;
  std::string value2 = "initial";

  CallbackTrigger trigger(
      [&]() {
        metric1->Set(value1, "a");
        metric2->Set(value2, "b");
      },
      {metric1.get(), metric2.get()});

  testing::CellReader<int64_t> reader1("/test/callback_metric1");
  testing::CellReader<std::string> reader2("/test/callback_metric2");

  value1 = 10;
  value2 = "hello";

  EXPECT_EQ(reader1.Read("a"), 10);
  EXPECT_EQ(reader2.Read("b"), "hello");
}

}  // namespace
}  // namespace tsl::monitoring
