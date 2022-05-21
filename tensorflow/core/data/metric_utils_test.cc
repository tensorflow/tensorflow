/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/metric_utils.h"

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/lib/monitoring/test_utils.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

using tensorflow::monitoring::testing::CellReader;
using tensorflow::monitoring::testing::Histogram;

TEST(MetricUtilsTest, CollectMetrics) {
  CellReader<Histogram> latency("/tensorflow/data/getnext_duration");
  CellReader<int64_t> iterator_lifetime("/tensorflow/data/iterator_lifetime");
  CellReader<int64_t> iterator_busy("/tensorflow/data/iterator_busy");
  EXPECT_FLOAT_EQ(latency.Delta().num(), 0.0);
  EXPECT_EQ(iterator_lifetime.Delta(), 0.0);
  EXPECT_EQ(iterator_busy.Delta(), 0.0);

  IteratorMetricsCollector metrics_collector(DEVICE_CPU, *Env::Default());
  metrics_collector.RecordStart();
  absl::SleepFor(absl::Seconds(1));
  metrics_collector.RecordStop(/*output=*/{});

  Histogram latency_histogram = latency.Delta();
  EXPECT_FLOAT_EQ(latency_histogram.num(), 1.0);
  EXPECT_GT(latency_histogram.sum(), 0.0);
  EXPECT_GT(iterator_lifetime.Delta(), 0);
  EXPECT_GT(iterator_busy.Delta(), 0.0);
}

TEST(MetricUtilsTest, ShouldNotCollectMetrics) {
  CellReader<Histogram> latency("/tensorflow/data/getnext_duration");
  CellReader<int64_t> iterator_lifetime("/tensorflow/data/iterator_lifetime");
  CellReader<int64_t> iterator_busy("/tensorflow/data/iterator_busy");
  EXPECT_FLOAT_EQ(latency.Delta().num(), 0.0);
  EXPECT_EQ(iterator_lifetime.Delta(), 0.0);
  EXPECT_EQ(iterator_busy.Delta(), 0.0);

  IteratorMetricsCollector metrics_collector(DEVICE_TPU, *Env::Default());
  metrics_collector.RecordStart();
  absl::SleepFor(absl::Seconds(1));
  metrics_collector.RecordStop(/*output=*/{});

  EXPECT_FLOAT_EQ(latency.Delta().num(), 0.0);
  EXPECT_EQ(iterator_lifetime.Delta(), 0.0);
  EXPECT_EQ(iterator_busy.Delta(), 0.0);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
