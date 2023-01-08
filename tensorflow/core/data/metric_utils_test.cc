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

#include <cstdint>

#include "absl/memory/memory.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
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
  EXPECT_EQ(iterator_lifetime.Delta(), 0);
  EXPECT_EQ(iterator_busy.Delta(), 0);

  IteratorMetricsCollector metrics_collector(DEVICE_CPU, *Env::Default());
  absl::Time start_time = metrics_collector.RecordStart();
  absl::SleepFor(absl::Seconds(1));
  metrics_collector.RecordStop(start_time, /*output=*/{});

  Histogram latency_histogram = latency.Delta();
  EXPECT_FLOAT_EQ(latency_histogram.num(), 1.0);
  EXPECT_GT(latency_histogram.sum(), 0.0);
  EXPECT_GT(iterator_lifetime.Delta(), 0);
  EXPECT_GT(iterator_busy.Delta(), 0);
}

TEST(MetricUtilsTest, ShouldNotCollectMetrics) {
  CellReader<Histogram> latency("/tensorflow/data/getnext_duration");
  CellReader<int64_t> iterator_lifetime("/tensorflow/data/iterator_lifetime");
  CellReader<int64_t> iterator_busy("/tensorflow/data/iterator_busy");
  EXPECT_FLOAT_EQ(latency.Delta().num(), 0.0);
  EXPECT_EQ(iterator_lifetime.Delta(), 0);
  EXPECT_EQ(iterator_busy.Delta(), 0);

  IteratorMetricsCollector metrics_collector(DEVICE_TPU, *Env::Default());
  absl::Time start_time = metrics_collector.RecordStart();
  absl::SleepFor(absl::Seconds(1));
  metrics_collector.RecordStop(start_time, /*output=*/{});

  EXPECT_FLOAT_EQ(latency.Delta().num(), 0.0);
  EXPECT_EQ(iterator_lifetime.Delta(), 0);
  EXPECT_EQ(iterator_busy.Delta(), 0);
}

TEST(MetricUtilsTest, ConcurrentThreads) {
  CellReader<Histogram> latency("/tensorflow/data/getnext_duration");
  CellReader<int64_t> iterator_lifetime("/tensorflow/data/iterator_lifetime");
  CellReader<int64_t> iterator_busy("/tensorflow/data/iterator_busy");
  EXPECT_FLOAT_EQ(latency.Delta().num(), 0.0);
  EXPECT_EQ(iterator_lifetime.Delta(), 0);
  EXPECT_EQ(iterator_busy.Delta(), 0);

  IteratorMetricsCollector metrics_collector(DEVICE_CPU, *Env::Default());
  absl::Time start_time = metrics_collector.RecordStart();
  auto thread = absl::WrapUnique(Env::Default()->StartThread(
      /*thread_options=*/{}, /*name=*/"Concurrent metric collection thread",
      [&metrics_collector]() {
        absl::Time concurrent_start_time = metrics_collector.RecordStart();
        absl::SleepFor(absl::Seconds(1));
        metrics_collector.RecordStop(concurrent_start_time, /*output=*/{});
      }));
  absl::SleepFor(absl::Seconds(1));
  metrics_collector.RecordStop(start_time, /*output=*/{});
  thread.reset();

  Histogram latency_histogram = latency.Delta();
  EXPECT_FLOAT_EQ(latency_histogram.num(), 2.0);
  EXPECT_GT(latency_histogram.sum(),
            absl::ToInt64Microseconds(absl::Seconds(2)));
  // The iterator busy time and lifetime do not count the latency twice.
  EXPECT_GE(iterator_lifetime.Delta(),
            absl::ToInt64Microseconds(absl::Seconds(1)));
  EXPECT_LT(iterator_lifetime.Delta(),
            absl::ToInt64Microseconds(absl::Seconds(1.5)));
  EXPECT_GE(iterator_busy.Delta(), absl::ToInt64Microseconds(absl::Seconds(1)));
  EXPECT_LT(iterator_busy.Delta(),
            absl::ToInt64Microseconds(absl::Seconds(1.5)));
}

TEST(MetricUtilsTest, OverlappingThreads) {
  CellReader<Histogram> latency("/tensorflow/data/getnext_duration");
  CellReader<int64_t> iterator_lifetime("/tensorflow/data/iterator_lifetime");
  CellReader<int64_t> iterator_busy("/tensorflow/data/iterator_busy");
  EXPECT_FLOAT_EQ(latency.Delta().num(), 0.0);
  EXPECT_EQ(iterator_lifetime.Delta(), 0);
  EXPECT_EQ(iterator_busy.Delta(), 0);

  // Two overlapping threads collect metrics:
  // Thread 1 (end - start = 1 sec): |---------|
  // Thread 2 (end - start = 2 sec):      |------------------|
  // Overlap: 0.5 sec.
  // Iterator busy time: 2.5 sec.
  IteratorMetricsCollector metrics_collector(DEVICE_CPU, *Env::Default());
  absl::Time start_time = metrics_collector.RecordStart();
  absl::SleepFor(absl::Seconds(0.5));
  auto thread = absl::WrapUnique(Env::Default()->StartThread(
      /*thread_options=*/{}, /*name=*/"Concurrent metric collection thread",
      [&metrics_collector]() {
        absl::Time concurrent_start_time = metrics_collector.RecordStart();
        absl::SleepFor(absl::Seconds(2));
        metrics_collector.RecordStop(concurrent_start_time, /*output=*/{});
      }));
  absl::SleepFor(absl::Seconds(0.5));
  metrics_collector.RecordStop(start_time, /*output=*/{});
  absl::SleepFor(absl::Seconds(1.5));
  thread.reset();

  Histogram latency_histogram = latency.Delta();
  EXPECT_FLOAT_EQ(latency_histogram.num(), 2.0);
  EXPECT_GT(latency_histogram.sum(),
            absl::ToInt64Microseconds(absl::Seconds(3)));
  // The iterator busy time and lifetime should not count the overlap twice.
  EXPECT_GE(iterator_lifetime.Delta(),
            absl::ToInt64Microseconds(absl::Seconds(2.5)));
  EXPECT_LT(iterator_lifetime.Delta(),
            absl::ToInt64Microseconds(absl::Seconds(2.9)));
  EXPECT_GE(iterator_busy.Delta(),
            absl::ToInt64Microseconds(absl::Seconds(2.5)));
  EXPECT_LT(iterator_busy.Delta(),
            absl::ToInt64Microseconds(absl::Seconds(2.9)));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
