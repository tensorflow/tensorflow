/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/monitoring/metric_def.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace monitoring {
namespace {

TEST(MetricDefTest, Simple) {
  const MetricDef<MetricKind::kCumulative, int64, 0> metric_def0(
      "/tensorflow/metric0", "An example metric with no labels.");
  const MetricDef<MetricKind::kGauge, double, 1> metric_def1(
      "/tensorflow/metric1", "An example metric with one label.", "LabelName");

  EXPECT_EQ("/tensorflow/metric0", metric_def0.name());
  EXPECT_EQ("/tensorflow/metric1", metric_def1.name());

  EXPECT_EQ(MetricKind::kCumulative, metric_def0.kind());
  EXPECT_EQ(MetricKind::kGauge, metric_def1.kind());

  EXPECT_EQ("An example metric with no labels.", metric_def0.description());
  EXPECT_EQ("An example metric with one label.", metric_def1.description());

  EXPECT_EQ(0, metric_def0.label_descriptions().size());
  ASSERT_EQ(1, metric_def1.label_descriptions().size());
  EXPECT_EQ("LabelName", metric_def1.label_descriptions()[0]);
}

}  // namespace
}  // namespace monitoring
}  // namespace tensorflow
