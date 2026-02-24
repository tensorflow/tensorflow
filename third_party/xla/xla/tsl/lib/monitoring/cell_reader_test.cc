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

#include "xla/tsl/lib/monitoring/cell_reader.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xla/tsl/lib/monitoring/counter.h"
#include "xla/tsl/lib/monitoring/counter_gauge.h"
#include "xla/tsl/lib/monitoring/gauge.h"
#include "xla/tsl/lib/monitoring/percentile_sampler.h"
#include "xla/tsl/lib/monitoring/sampler.h"
#include "xla/tsl/lib/monitoring/test_utils.h"
#include "xla/tsl/lib/monitoring/types.h"

namespace tsl {
namespace monitoring {
namespace testing {
namespace {

std::vector<double> GetDefaultPercentiles() {
  return {25.0, 50.0, 80.0, 90.0, 95.0, 99.0};
}

auto* test_counter = tsl::monitoring::Counter<0>::New(
    "/tsl/monitoring/test/counter", "Test counter.");

auto* test_counter_with_labels = tsl::monitoring::Counter<2>::New(
    "/tsl/monitoring/test/counter_with_labels", "Test counter with two labels.",
    "label1", "label2");

auto* test_sampler = tsl::monitoring::Sampler<0>::New(
    {"/tsl/monitoring/test/sampler", "Test sampler."},
    /*buckets=*/tsl::monitoring::Buckets::Explicit(
        {0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0}));

auto* test_sampler_with_labels = tsl::monitoring::Sampler<2>::New(
    {"/tsl/monitoring/test/sampler_with_labels", "Test sampler.", "label1",
     "label2"},
    /*buckets=*/tsl::monitoring::Buckets::Exponential(
        /*scale=*/1.0, /*growth_factor=*/10.0, /*bucket_boundary_count=*/5));

auto* test_exponential_buckets_with_explicit_domain =
    tsl::monitoring::Sampler<0>::New(
        {"/tsl/monitoring/test/exponential_buckets_with_explicit_domain",
         "Test sampler."},
        /*buckets=*/tsl::monitoring::Buckets::Exponential(
            /*scale=*/1.5, /*growth_factor=*/2.0, /*domain_max=*/{50.0}));

auto* test_exponential_buckets_with_unbounded_domain =
    tsl::monitoring::Sampler<0>::New(
        {"/tsl/monitoring/test/exponential_buckets_with_unbounded_domain",
         "Test sampler."},
        /*buckets=*/tsl::monitoring::Buckets::Exponential(
            /*scale=*/1.0, /*growth_factor=*/100.0));

auto* test_int_gauge = tsl::monitoring::Gauge<int64_t, 0>::New(
    "/tsl/monitoring/test/int_gauge", "Test gauge.");

auto* test_int_gauge_with_labels = tsl::monitoring::Gauge<int64_t, 2>::New(
    "/tsl/monitoring/test/int_gauge_with_labels", "Test gauge.", "label1",
    "label2");

auto* test_string_gauge = tsl::monitoring::Gauge<std::string, 0>::New(
    "/tsl/monitoring/test/string_gauge", "Test gauge.");

auto* test_string_gauge_with_labels =
    tsl::monitoring::Gauge<std::string, 2>::New(
        "/tsl/monitoring/test/string_gauge_with_labels", "Test gauge.",
        "label1", "label2");

auto* test_bool_gauge = tsl::monitoring::Gauge<bool, 0>::New(
    "/tsl/monitoring/test/bool_gauge", "Test gauge.");

auto* test_bool_gauge_with_labels = tsl::monitoring::Gauge<bool, 2>::New(
    "/tsl/monitoring/test/bool_gauge_with_labels", "Test gauge.", "label1",
    "label2");

auto* test_counter_gauge = tsl::monitoring::CounterGauge<0>::New(
    "/tsl/monitoring/test/counter_gauge", "Test counter gauge.");

auto* test_counter_gauge_with_labels = tsl::monitoring::CounterGauge<2>::New(
    "/tsl/monitoring/test/counter_gauge_with_labels",
    "Test counter gauge with two labels.", "label1", "label2");

auto* test_percentiles = tsl::monitoring::PercentileSampler<0>::New(
    {"/tsl/monitoring/test/percentiles", "Test percentiles."},
    GetDefaultPercentiles(), /*max_samples=*/1024,
    tsl::monitoring::UnitOfMeasure::kTime);

auto* test_percentiles_with_labels = tsl::monitoring::PercentileSampler<2>::New(
    {"/tsl/monitoring/test/percentiles_with_labels", "Test percentiles.",
     "label1", "label2"},
    GetDefaultPercentiles(), /*max_samples=*/1024,
    tsl::monitoring::UnitOfMeasure::kTime);

void IncrementLazyCounter() {
  static auto* test_counter = monitoring::Counter<0>::New(
      "/tsl/monitoring/test/lazy_counter", "Test lazy counter.");
  test_counter->GetCell()->IncrementBy(1);
}

TEST(CellReaderTest, CounterDeltaNoLabels) {
  CellReader<int64_t> cell_reader("/tsl/monitoring/test/counter");
  EXPECT_EQ(cell_reader.Delta(), 0);

  test_counter->GetCell()->IncrementBy(5);
  EXPECT_EQ(cell_reader.Delta(), 5);

  test_counter->GetCell()->IncrementBy(10);
  EXPECT_EQ(cell_reader.Delta(), 10);

  test_counter->GetCell()->IncrementBy(100);
  EXPECT_EQ(cell_reader.Delta(), 100);
}

TEST(CellReaderTest, CounterReadNoLabels) {
  CellReader<int64_t> cell_reader("/tsl/monitoring/test/counter");
  EXPECT_EQ(cell_reader.Read(), 0);

  test_counter->GetCell()->IncrementBy(5);
  EXPECT_EQ(cell_reader.Read(), 5);

  test_counter->GetCell()->IncrementBy(10);
  EXPECT_EQ(cell_reader.Read(), 15);

  test_counter->GetCell()->IncrementBy(100);
  EXPECT_EQ(cell_reader.Read(), 115);
}

TEST(CellReaderTest, CounterDeltaAndReadNoLabels) {
  CellReader<int64_t> cell_reader("/tsl/monitoring/test/counter");
  EXPECT_EQ(cell_reader.Delta(), 0);
  EXPECT_EQ(cell_reader.Read(), 0);

  test_counter->GetCell()->IncrementBy(5);
  EXPECT_EQ(cell_reader.Delta(), 5);
  EXPECT_EQ(cell_reader.Read(), 5);

  test_counter->GetCell()->IncrementBy(10);
  EXPECT_EQ(cell_reader.Delta(), 10);
  EXPECT_EQ(cell_reader.Read(), 15);

  test_counter->GetCell()->IncrementBy(100);
  EXPECT_EQ(cell_reader.Delta(), 100);
  EXPECT_EQ(cell_reader.Read(), 115);
}

TEST(CellReaderTest, CounterDeltaWithLabels) {
  CellReader<int64_t> cell_reader("/tsl/monitoring/test/counter_with_labels");
  EXPECT_EQ(cell_reader.Delta("x1", "y1"), 0);
  EXPECT_EQ(cell_reader.Delta("x1", "y2"), 0);
  EXPECT_EQ(cell_reader.Delta("x2", "y1"), 0);
  EXPECT_EQ(cell_reader.Delta("x2", "y2"), 0);

  test_counter_with_labels->GetCell("x1", "y1")->IncrementBy(1);
  test_counter_with_labels->GetCell("x1", "y2")->IncrementBy(10);
  test_counter_with_labels->GetCell("x2", "y1")->IncrementBy(100);
  EXPECT_EQ(cell_reader.Delta("x1", "y1"), 1);
  EXPECT_EQ(cell_reader.Delta("x1", "y2"), 10);
  EXPECT_EQ(cell_reader.Delta("x2", "y1"), 100);
  EXPECT_EQ(cell_reader.Delta("x2", "y2"), 0);

  test_counter_with_labels->GetCell("x1", "y2")->IncrementBy(5);
  test_counter_with_labels->GetCell("x2", "y1")->IncrementBy(50);
  test_counter_with_labels->GetCell("x2", "y2")->IncrementBy(500);
  EXPECT_EQ(cell_reader.Delta("x1", "y1"), 0);
  EXPECT_EQ(cell_reader.Delta("x1", "y2"), 5);
  EXPECT_EQ(cell_reader.Delta("x2", "y1"), 50);
  EXPECT_EQ(cell_reader.Delta("x2", "y2"), 500);

  test_counter_with_labels->GetCell("x1", "y1")->IncrementBy(1000);
  test_counter_with_labels->GetCell("x2", "y2")->IncrementBy(1000);
  EXPECT_EQ(cell_reader.Delta("x1", "y1"), 1000);
  EXPECT_EQ(cell_reader.Delta("x1", "y2"), 0);
  EXPECT_EQ(cell_reader.Delta("x2", "y1"), 0);
  EXPECT_EQ(cell_reader.Delta("x2", "y2"), 1000);
}

TEST(CellReaderTest, CounterReadWithLabels) {
  CellReader<int64_t> cell_reader("/tsl/monitoring/test/counter_with_labels");
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 0);
  EXPECT_EQ(cell_reader.Read("x1", "y2"), 0);
  EXPECT_EQ(cell_reader.Read("x2", "y1"), 0);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), 0);

  test_counter_with_labels->GetCell("x1", "y1")->IncrementBy(1);
  test_counter_with_labels->GetCell("x1", "y2")->IncrementBy(10);
  test_counter_with_labels->GetCell("x2", "y1")->IncrementBy(100);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 1);
  EXPECT_EQ(cell_reader.Read("x1", "y2"), 10);
  EXPECT_EQ(cell_reader.Read("x2", "y1"), 100);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), 0);

  test_counter_with_labels->GetCell("x1", "y2")->IncrementBy(5);
  test_counter_with_labels->GetCell("x2", "y1")->IncrementBy(50);
  test_counter_with_labels->GetCell("x2", "y2")->IncrementBy(500);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 1);
  EXPECT_EQ(cell_reader.Read("x1", "y2"), 15);
  EXPECT_EQ(cell_reader.Read("x2", "y1"), 150);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), 500);

  test_counter_with_labels->GetCell("x1", "y1")->IncrementBy(1000);
  test_counter_with_labels->GetCell("x2", "y2")->IncrementBy(1000);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 1001);
  EXPECT_EQ(cell_reader.Read("x1", "y2"), 15);
  EXPECT_EQ(cell_reader.Read("x2", "y1"), 150);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), 1500);
}

TEST(CellReaderTest, CounterDeltaAndReadWithLabels) {
  CellReader<int64_t> cell_reader("/tsl/monitoring/test/counter_with_labels");
  EXPECT_EQ(cell_reader.Delta("x1", "y1"), 0);
  EXPECT_EQ(cell_reader.Delta("x1", "y2"), 0);
  EXPECT_EQ(cell_reader.Delta("x2", "y1"), 0);
  EXPECT_EQ(cell_reader.Delta("x2", "y2"), 0);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 0);
  EXPECT_EQ(cell_reader.Read("x1", "y2"), 0);
  EXPECT_EQ(cell_reader.Read("x2", "y1"), 0);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), 0);

  test_counter_with_labels->GetCell("x1", "y1")->IncrementBy(1);
  test_counter_with_labels->GetCell("x1", "y2")->IncrementBy(10);
  test_counter_with_labels->GetCell("x2", "y1")->IncrementBy(100);
  EXPECT_EQ(cell_reader.Delta("x1", "y1"), 1);
  EXPECT_EQ(cell_reader.Delta("x1", "y2"), 10);
  EXPECT_EQ(cell_reader.Delta("x2", "y1"), 100);
  EXPECT_EQ(cell_reader.Delta("x2", "y2"), 0);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 1);
  EXPECT_EQ(cell_reader.Read("x1", "y2"), 10);
  EXPECT_EQ(cell_reader.Read("x2", "y1"), 100);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), 0);

  test_counter_with_labels->GetCell("x1", "y2")->IncrementBy(5);
  test_counter_with_labels->GetCell("x2", "y1")->IncrementBy(50);
  test_counter_with_labels->GetCell("x2", "y2")->IncrementBy(500);
  EXPECT_EQ(cell_reader.Delta("x1", "y1"), 0);
  EXPECT_EQ(cell_reader.Delta("x1", "y2"), 5);
  EXPECT_EQ(cell_reader.Delta("x2", "y1"), 50);
  EXPECT_EQ(cell_reader.Delta("x2", "y2"), 500);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 1);
  EXPECT_EQ(cell_reader.Read("x1", "y2"), 15);
  EXPECT_EQ(cell_reader.Read("x2", "y1"), 150);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), 500);

  test_counter_with_labels->GetCell("x1", "y1")->IncrementBy(1000);
  test_counter_with_labels->GetCell("x2", "y2")->IncrementBy(1000);
  EXPECT_EQ(cell_reader.Delta("x1", "y1"), 1000);
  EXPECT_EQ(cell_reader.Delta("x1", "y2"), 0);
  EXPECT_EQ(cell_reader.Delta("x2", "y1"), 0);
  EXPECT_EQ(cell_reader.Delta("x2", "y2"), 1000);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 1001);
  EXPECT_EQ(cell_reader.Read("x1", "y2"), 15);
  EXPECT_EQ(cell_reader.Read("x2", "y1"), 150);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), 1500);
}

TEST(CellReaderTest, TwoCounterReaders) {
  CellReader<int64_t> cell_reader("/tsl/monitoring/test/counter");
  CellReader<int64_t> cell_reader_with_labels(
      "/tsl/monitoring/test/counter_with_labels");
  EXPECT_EQ(cell_reader.Delta(), 0);
  EXPECT_EQ(cell_reader_with_labels.Delta("x1", "y1"), 0);
  EXPECT_EQ(cell_reader_with_labels.Delta("x2", "y2"), 0);
  EXPECT_EQ(cell_reader.Read(), 0);
  EXPECT_EQ(cell_reader_with_labels.Read("x1", "y1"), 0);
  EXPECT_EQ(cell_reader_with_labels.Read("x2", "y2"), 0);

  test_counter->GetCell()->IncrementBy(1);
  test_counter_with_labels->GetCell("x1", "y1")->IncrementBy(100);
  EXPECT_EQ(cell_reader.Delta(), 1);
  EXPECT_EQ(cell_reader_with_labels.Delta("x1", "y1"), 100);
  EXPECT_EQ(cell_reader_with_labels.Delta("x2", "y2"), 0);
  EXPECT_EQ(cell_reader.Read(), 1);
  EXPECT_EQ(cell_reader_with_labels.Read("x1", "y1"), 100);
  EXPECT_EQ(cell_reader_with_labels.Read("x2", "y2"), 0);

  test_counter->GetCell()->IncrementBy(5);
  test_counter_with_labels->GetCell("x2", "y2")->IncrementBy(500);
  EXPECT_EQ(cell_reader.Delta(), 5);
  EXPECT_EQ(cell_reader_with_labels.Delta("x1", "y1"), 0);
  EXPECT_EQ(cell_reader_with_labels.Delta("x2", "y2"), 500);
  EXPECT_EQ(cell_reader.Read(), 6);
  EXPECT_EQ(cell_reader_with_labels.Read("x1", "y1"), 100);
  EXPECT_EQ(cell_reader_with_labels.Read("x2", "y2"), 500);

  test_counter->GetCell()->IncrementBy(1);
  test_counter_with_labels->GetCell("x1", "y1")->IncrementBy(1);
  test_counter_with_labels->GetCell("x2", "y2")->IncrementBy(1);
  EXPECT_EQ(cell_reader.Delta(), 1);
  EXPECT_EQ(cell_reader_with_labels.Delta("x1", "y1"), 1);
  EXPECT_EQ(cell_reader_with_labels.Delta("x2", "y2"), 1);
  EXPECT_EQ(cell_reader.Read(), 7);
  EXPECT_EQ(cell_reader_with_labels.Read("x1", "y1"), 101);
  EXPECT_EQ(cell_reader_with_labels.Read("x2", "y2"), 501);
}

TEST(CellReaderTest, RepeatedReads) {
  CellReader<int64_t> cell_reader("/tsl/monitoring/test/counter");
  CellReader<int64_t> cell_reader_with_labels(
      "/tsl/monitoring/test/counter_with_labels");
  EXPECT_EQ(cell_reader.Delta(), 0);
  EXPECT_EQ(cell_reader_with_labels.Delta("x1", "y1"), 0);
  EXPECT_EQ(cell_reader_with_labels.Delta("x2", "y2"), 0);
  EXPECT_EQ(cell_reader.Read(), 0);
  EXPECT_EQ(cell_reader_with_labels.Read("x1", "y1"), 0);
  EXPECT_EQ(cell_reader_with_labels.Read("x2", "y2"), 0);

  test_counter->GetCell()->IncrementBy(1);
  test_counter_with_labels->GetCell("x1", "y1")->IncrementBy(100);
  EXPECT_EQ(cell_reader.Delta(), 1);
  EXPECT_EQ(cell_reader_with_labels.Delta("x1", "y1"), 100);
  EXPECT_EQ(cell_reader_with_labels.Delta("x2", "y2"), 0);
  EXPECT_EQ(cell_reader.Read(), 1);
  EXPECT_EQ(cell_reader_with_labels.Read("x1", "y1"), 100);
  EXPECT_EQ(cell_reader_with_labels.Read("x2", "y2"), 0);

  // Repeats the previous read. The values will stay the same, while the deltas
  // will be 0.
  EXPECT_EQ(cell_reader.Delta(), 0);
  EXPECT_EQ(cell_reader_with_labels.Delta("x1", "y1"), 0);
  EXPECT_EQ(cell_reader_with_labels.Delta("x2", "y2"), 0);
  EXPECT_EQ(cell_reader.Read(), 1);
  EXPECT_EQ(cell_reader_with_labels.Read("x1", "y1"), 100);
  EXPECT_EQ(cell_reader_with_labels.Read("x2", "y2"), 0);
}

TEST(CellReaderTest, SamplerDeltaNoLabels) {
  CellReader<Histogram> cell_reader("/tsl/monitoring/test/sampler");
  Histogram histogram = cell_reader.Delta();
  EXPECT_FLOAT_EQ(histogram.num(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(7), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(8), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(9), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(10), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(11), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(12), 0.0);

  test_sampler->GetCell()->Add(0.1);
  histogram = cell_reader.Delta();
  EXPECT_FLOAT_EQ(histogram.num(), 1.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.1);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.01);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 1.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(7), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(8), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(9), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(10), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(11), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(12), 0.0);

  test_sampler->GetCell()->Add(1.1);
  histogram = cell_reader.Delta();
  EXPECT_FLOAT_EQ(histogram.num(), 1.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 1.1);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 1.21);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 1.0);
  EXPECT_FLOAT_EQ(histogram.num(7), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(8), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(9), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(10), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(11), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(12), 0.0);

  test_sampler->GetCell()->Add(100);
  histogram = cell_reader.Delta();
  EXPECT_FLOAT_EQ(histogram.num(), 1.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 100);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 10000);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(7), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(8), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(9), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(10), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(11), 1.0);
  EXPECT_FLOAT_EQ(histogram.num(12), 0.0);
}

TEST(CellReaderTest, SamplerReadNoLabels) {
  CellReader<Histogram> cell_reader("/tsl/monitoring/test/sampler");
  Histogram histogram = cell_reader.Read();
  EXPECT_FLOAT_EQ(histogram.num(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(7), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(8), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(9), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(10), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(11), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(12), 0.0);

  test_sampler->GetCell()->Add(0.1);
  histogram = cell_reader.Read();
  EXPECT_FLOAT_EQ(histogram.num(), 1.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.1);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.01);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 1.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(7), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(8), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(9), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(10), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(11), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(12), 0.0);

  test_sampler->GetCell()->Add(1.1);
  histogram = cell_reader.Read();
  EXPECT_FLOAT_EQ(histogram.num(), 2.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 1.2);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 1.22);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 1.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 1.0);
  EXPECT_FLOAT_EQ(histogram.num(7), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(8), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(9), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(10), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(11), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(12), 0.0);

  test_sampler->GetCell()->Add(100);
  histogram = cell_reader.Read();
  EXPECT_FLOAT_EQ(histogram.num(), 3.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 101.2);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 10001.22);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 1.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 1.0);
  EXPECT_FLOAT_EQ(histogram.num(7), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(8), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(9), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(10), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(11), 1.0);
  EXPECT_FLOAT_EQ(histogram.num(12), 0.0);
}

TEST(CellReaderTest, SamplerDeltaAndReadNoLabels) {
  CellReader<Histogram> cell_reader("/tsl/monitoring/test/sampler");
  Histogram histogram = cell_reader.Delta();
  EXPECT_FLOAT_EQ(histogram.num(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(7), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(8), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(9), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(10), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(11), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(12), 0.0);

  histogram = cell_reader.Read();
  EXPECT_FLOAT_EQ(histogram.num(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(7), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(8), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(9), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(10), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(11), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(12), 0.0);

  test_sampler->GetCell()->Add(0.1);
  test_sampler->GetCell()->Add(0.1);
  histogram = cell_reader.Delta();
  EXPECT_FLOAT_EQ(histogram.num(), 2.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.2);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.02);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 2.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(7), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(8), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(9), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(10), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(11), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(12), 0.0);

  histogram = cell_reader.Read();
  EXPECT_FLOAT_EQ(histogram.num(), 2.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.2);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.02);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 2.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(7), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(8), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(9), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(10), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(11), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(12), 0.0);

  test_sampler->GetCell()->Add(100);
  test_sampler->GetCell()->Add(100);
  histogram = cell_reader.Delta();
  EXPECT_FLOAT_EQ(histogram.num(), 2.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 200);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 20000);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(7), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(8), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(9), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(10), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(11), 2.0);
  EXPECT_FLOAT_EQ(histogram.num(12), 0.0);

  histogram = cell_reader.Read();
  EXPECT_FLOAT_EQ(histogram.num(), 4.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 200.2);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 20000.02);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 2.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(7), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(8), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(9), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(10), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(11), 2.0);
  EXPECT_FLOAT_EQ(histogram.num(12), 0.0);
}

TEST(CellReaderTest, SamplerDeltaWithLabels) {
  CellReader<Histogram> cell_reader("/tsl/monitoring/test/sampler_with_labels");
  Histogram histogram = cell_reader.Delta("x1", "y1");
  EXPECT_FLOAT_EQ(histogram.num(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  histogram = cell_reader.Delta("x2", "y2");
  EXPECT_FLOAT_EQ(histogram.num(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);

  test_sampler_with_labels->GetCell("x1", "y1")->Add(-100);
  test_sampler_with_labels->GetCell("x1", "y1")->Add(-100);
  histogram = cell_reader.Delta("x1", "y1");
  EXPECT_FLOAT_EQ(histogram.num(), 2.0);
  EXPECT_FLOAT_EQ(histogram.sum(), -200.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 20000.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 2.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  histogram = cell_reader.Delta("x2", "y2");
  EXPECT_FLOAT_EQ(histogram.num(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);

  test_sampler_with_labels->GetCell("x2", "y2")->Add(100);
  test_sampler_with_labels->GetCell("x2", "y2")->Add(100);
  histogram = cell_reader.Delta("x1", "y1");
  EXPECT_FLOAT_EQ(histogram.num(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  histogram = cell_reader.Delta("x2", "y2");
  EXPECT_FLOAT_EQ(histogram.num(), 2.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 200.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 20000.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 2.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);

  test_sampler_with_labels->GetCell("x1", "y1")->Add(-100000000);
  test_sampler_with_labels->GetCell("x2", "y2")->Add(100000000);
  histogram = cell_reader.Delta("x1", "y1");
  EXPECT_FLOAT_EQ(histogram.num(), 1.0);
  EXPECT_FLOAT_EQ(histogram.sum(), -100000000);
  EXPECT_FLOAT_EQ(histogram.num(0), 1.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  histogram = cell_reader.Delta("x2", "y2");
  EXPECT_FLOAT_EQ(histogram.num(), 1.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 100000000);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 1.0);
}

TEST(CellReaderTest, SamplerReadWithLabels) {
  CellReader<Histogram> cell_reader("/tsl/monitoring/test/sampler_with_labels");
  Histogram histogram = cell_reader.Read("x1", "y1");
  EXPECT_FLOAT_EQ(histogram.num(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  histogram = cell_reader.Read("x2", "y2");
  EXPECT_FLOAT_EQ(histogram.num(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);

  test_sampler_with_labels->GetCell("x1", "y1")->Add(-100);
  test_sampler_with_labels->GetCell("x1", "y1")->Add(-100);
  histogram = cell_reader.Read("x1", "y1");
  EXPECT_FLOAT_EQ(histogram.num(), 2.0);
  EXPECT_FLOAT_EQ(histogram.sum(), -200.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 20000.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 2.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  histogram = cell_reader.Read("x2", "y2");
  EXPECT_FLOAT_EQ(histogram.num(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);

  test_sampler_with_labels->GetCell("x2", "y2")->Add(100);
  test_sampler_with_labels->GetCell("x2", "y2")->Add(100);
  histogram = cell_reader.Read("x1", "y1");
  EXPECT_FLOAT_EQ(histogram.num(), 2.0);
  EXPECT_FLOAT_EQ(histogram.sum(), -200.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 20000.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 2.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  histogram = cell_reader.Read("x2", "y2");
  EXPECT_FLOAT_EQ(histogram.num(), 2.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 200.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 20000.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 2.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);

  test_sampler_with_labels->GetCell("x1", "y1")->Add(-100000000);
  test_sampler_with_labels->GetCell("x2", "y2")->Add(100000000);
  histogram = cell_reader.Read("x1", "y1");
  EXPECT_FLOAT_EQ(histogram.num(), 3.0);
  EXPECT_FLOAT_EQ(histogram.sum(), -100000200);
  EXPECT_FLOAT_EQ(histogram.num(0), 3.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  histogram = cell_reader.Read("x2", "y2");
  EXPECT_FLOAT_EQ(histogram.num(), 3.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 100000200);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 2.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 1.0);
}

TEST(CellReaderTest, SamplerRepeatedReads) {
  CellReader<Histogram> cell_reader("/tsl/monitoring/test/sampler_with_labels");
  Histogram histogram = cell_reader.Read("x1", "y1");
  EXPECT_FLOAT_EQ(histogram.num(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  histogram = cell_reader.Read("x2", "y2");
  EXPECT_FLOAT_EQ(histogram.num(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);

  test_sampler_with_labels->GetCell("x1", "y1")->Add(-100);
  test_sampler_with_labels->GetCell("x1", "y1")->Add(-100);
  test_sampler_with_labels->GetCell("x2", "y2")->Add(100);
  test_sampler_with_labels->GetCell("x2", "y2")->Add(100);
  test_sampler_with_labels->GetCell("x1", "y1")->Add(-100000000);
  test_sampler_with_labels->GetCell("x2", "y2")->Add(100000000);
  histogram = cell_reader.Delta("x1", "y1");
  EXPECT_FLOAT_EQ(histogram.num(), 3.0);
  EXPECT_FLOAT_EQ(histogram.sum(), -100000200);
  EXPECT_FLOAT_EQ(histogram.num(0), 3.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  histogram = cell_reader.Delta("x2", "y2");
  EXPECT_FLOAT_EQ(histogram.num(), 3.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 100000200);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 2.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 1.0);

  histogram = cell_reader.Read("x1", "y1");
  EXPECT_FLOAT_EQ(histogram.num(), 3.0);
  EXPECT_FLOAT_EQ(histogram.sum(), -100000200);
  EXPECT_FLOAT_EQ(histogram.num(0), 3.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  histogram = cell_reader.Read("x2", "y2");
  EXPECT_FLOAT_EQ(histogram.num(), 3.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 100000200);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 2.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 1.0);

  // Repeats the previous read. The values will stay the same, while the deltas
  // will be 0.
  histogram = cell_reader.Delta("x1", "y1");
  EXPECT_FLOAT_EQ(histogram.num(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  histogram = cell_reader.Delta("x2", "y2");
  EXPECT_FLOAT_EQ(histogram.num(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);

  histogram = cell_reader.Read("x1", "y1");
  EXPECT_FLOAT_EQ(histogram.num(), 3.0);
  EXPECT_FLOAT_EQ(histogram.sum(), -100000200);
  EXPECT_FLOAT_EQ(histogram.num(0), 3.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  histogram = cell_reader.Read("x2", "y2");
  EXPECT_FLOAT_EQ(histogram.num(), 3.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 100000200);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 2.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 1.0);
}

TEST(CellReaderTest, ExponentialBucketsWithExplicitDomain) {
  CellReader<Histogram> cell_reader(
      "/tsl/monitoring/test/exponential_buckets_with_explicit_domain");
  Histogram histogram = cell_reader.Read();
  EXPECT_FLOAT_EQ(histogram.num(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(7), 0.0);

  auto* cell = test_exponential_buckets_with_explicit_domain->GetCell();
  ASSERT_NE(cell, nullptr);

  // Cell 0: (-inf, 1.5)
  cell->Add(-100000000);
  cell->Add(-1);
  cell->Add(0);
  cell->Add(1);

  // Cell 1: [1.5, 3.0)
  cell->Add(2);

  // Cell 2: [3.0, 6.0)
  cell->Add(3);
  cell->Add(4);
  cell->Add(5);

  // Cell 3: [6.0, 12.0)
  cell->Add(6);
  cell->Add(7);
  cell->Add(8);
  cell->Add(9);
  cell->Add(10);

  // Cell 4: [12.0, 24.0)
  cell->Add(20);

  // Cell 5: [24.0, 48.0)
  cell->Add(30);
  cell->Add(40);

  // Cell 6: [48.0, 96.0)
  cell->Add(50);
  cell->Add(60);
  cell->Add(70);
  cell->Add(80);
  cell->Add(90);

  // Cell 7: [96.0, +inf)
  cell->Add(100);
  cell->Add(500);
  cell->Add(1000);
  cell->Add(100000000);

  histogram = cell_reader.Read();
  EXPECT_FLOAT_EQ(histogram.num(), 25.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 2094.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 4.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 1.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 3.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 5.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 1.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 2.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 5.0);
  EXPECT_FLOAT_EQ(histogram.num(7), 4.0);
}

TEST(CellReaderTest, ExponentialBucketsWithUnboundedDomain) {
  CellReader<Histogram> cell_reader(
      "/tsl/monitoring/test/exponential_buckets_with_unbounded_domain");
  Histogram histogram = cell_reader.Read();
  EXPECT_FLOAT_EQ(histogram.num(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum(), 0.0);
  EXPECT_FLOAT_EQ(histogram.sum_squares(), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 0.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 0.0);

  auto* cell = test_exponential_buckets_with_unbounded_domain->GetCell();
  ASSERT_NE(cell, nullptr);

  // Cell 0: (-inf, 1.0)
  cell->Add(-100000000);
  cell->Add(-1);
  cell->Add(0);
  cell->Add(0.99);

  // Cell 1: [1.0, 100.0)
  cell->Add(1);
  cell->Add(5);
  cell->Add(10);
  cell->Add(50);
  cell->Add(99.99);

  // Cell 2: [100.0, 10000.0)
  cell->Add(100);
  cell->Add(500);
  cell->Add(1000);
  cell->Add(5000);
  cell->Add(9999.99);

  // Cell 3: [10000.0, 1000000.0)
  cell->Add(10000);
  cell->Add(50000);
  cell->Add(100000);
  cell->Add(500000);
  cell->Add(999999.99);

  // Cell 4: [1000000.0, 100000000.0)
  cell->Add(1000000);
  cell->Add(5000000);
  cell->Add(10000000);
  cell->Add(50000000);
  cell->Add(99999999.99);

  // Cell 5: [100000000.0, 10000000000.0)  (contains default upper bound 2^32-1)
  cell->Add(100000000);
  cell->Add(500000000);
  cell->Add(1000000000);
  cell->Add(5000000000);
  cell->Add(9999999999.99);

  // Cell 6: [10000000000.0, +inf)
  cell->Add(10000000000);
  cell->Add(1000000000000);
  cell->Add(100000000000000);

  histogram = cell_reader.Read();
  EXPECT_FLOAT_EQ(histogram.num(), 32.0);
  EXPECT_FLOAT_EQ(histogram.num(0), 4.0);
  EXPECT_FLOAT_EQ(histogram.num(1), 5.0);
  EXPECT_FLOAT_EQ(histogram.num(2), 5.0);
  EXPECT_FLOAT_EQ(histogram.num(3), 5.0);
  EXPECT_FLOAT_EQ(histogram.num(4), 5.0);
  EXPECT_FLOAT_EQ(histogram.num(5), 5.0);
  EXPECT_FLOAT_EQ(histogram.num(6), 3.0);
}

TEST(CellReaderTest, IntGaugeRead) {
  CellReader<int64_t> cell_reader("/tsl/monitoring/test/int_gauge");
  EXPECT_EQ(cell_reader.Read(), 0);

  test_int_gauge->GetCell()->Set(100);
  EXPECT_EQ(cell_reader.Read(), 100);

  test_int_gauge->GetCell()->Set(-100);
  EXPECT_EQ(cell_reader.Read(), -100);

  test_int_gauge->GetCell()->Set(0);
  EXPECT_EQ(cell_reader.Read(), 0);
}

TEST(CellReaderTest, IntGaugeReadWithLabels) {
  CellReader<int64_t> cell_reader("/tsl/monitoring/test/int_gauge_with_labels");
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 0);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), 0);

  test_int_gauge_with_labels->GetCell("x1", "y1")->Set(100000);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 100000);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), 0);

  test_int_gauge_with_labels->GetCell("x2", "y2")->Set(-100000);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 100000);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), -100000);

  test_int_gauge_with_labels->GetCell("x1", "y1")->Set(-100000);
  test_int_gauge_with_labels->GetCell("x2", "y2")->Set(100000);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), -100000);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), 100000);

  test_int_gauge_with_labels->GetCell("x1", "y1")->Set(0);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 0);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), 100000);
  test_int_gauge_with_labels->GetCell("x2", "y2")->Set(0);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), 0);
}

TEST(CellReaderTest, IntGaugeRepeatedSetAndRead) {
  CellReader<int64_t> cell_reader("/tsl/monitoring/test/int_gauge_with_labels");

  test_int_gauge_with_labels->GetCell("x1", "y1")->Set(-1);
  test_int_gauge_with_labels->GetCell("x2", "y2")->Set(1);
  test_int_gauge_with_labels->GetCell("x1", "y1")->Set(1);
  test_int_gauge_with_labels->GetCell("x2", "y2")->Set(-1);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 1);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), -1);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 1);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), -1);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 1);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), -1);

  test_int_gauge_with_labels->GetCell("x1", "y1")->Set(0);
  test_int_gauge_with_labels->GetCell("x2", "y2")->Set(500);
  test_int_gauge_with_labels->GetCell("x1", "y1")->Set(0);
  test_int_gauge_with_labels->GetCell("x2", "y2")->Set(-500);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 0);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), -500);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 0);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), -500);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), 0);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), -500);
  test_int_gauge_with_labels->GetCell("x2", "y2")->Set(0);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), 0);
}

TEST(CellReaderTest, StringGaugeRead) {
  CellReader<std::string> cell_reader("/tsl/monitoring/test/string_gauge");
  EXPECT_EQ(cell_reader.Read(), "");

  test_string_gauge->GetCell()->Set("gauge value");
  EXPECT_EQ(cell_reader.Read(), "gauge value");

  test_string_gauge->GetCell()->Set("Updated gauge value");
  EXPECT_EQ(cell_reader.Read(), "Updated gauge value");

  test_string_gauge->GetCell()->Set("");
  EXPECT_EQ(cell_reader.Read(), "");
}

TEST(CellReaderTest, StringGaugeReadWithLabels) {
  CellReader<std::string> cell_reader(
      "/tsl/monitoring/test/string_gauge_with_labels");
  EXPECT_EQ(cell_reader.Read("x1", "y1"), "");
  EXPECT_EQ(cell_reader.Read("x2", "y2"), "");

  test_string_gauge_with_labels->GetCell("x1", "y1")->Set("Value 1");
  EXPECT_EQ(cell_reader.Read("x1", "y1"), "Value 1");
  EXPECT_EQ(cell_reader.Read("x2", "y2"), "");

  test_string_gauge_with_labels->GetCell("x2", "y2")->Set("Value 2");
  EXPECT_EQ(cell_reader.Read("x1", "y1"), "Value 1");
  EXPECT_EQ(cell_reader.Read("x2", "y2"), "Value 2");

  test_string_gauge_with_labels->GetCell("x1", "y1")->Set("Value 3");
  test_string_gauge_with_labels->GetCell("x2", "y2")->Set("Value 3");
  EXPECT_EQ(cell_reader.Read("x1", "y1"), "Value 3");
  EXPECT_EQ(cell_reader.Read("x2", "y2"), "Value 3");

  test_string_gauge_with_labels->GetCell("x1", "y1")->Set("");
  test_string_gauge_with_labels->GetCell("x2", "y2")->Set("");
  EXPECT_EQ(cell_reader.Read("x1", "y1"), "");
  EXPECT_EQ(cell_reader.Read("x2", "y2"), "");
}

TEST(CellReaderTest, StringGaugeRepeatedSetAndRead) {
  CellReader<std::string> cell_reader(
      "/tsl/monitoring/test/string_gauge_with_labels");
  EXPECT_EQ(cell_reader.Read("x1", "y1"), "");
  EXPECT_EQ(cell_reader.Read("x2", "y2"), "");

  test_string_gauge_with_labels->GetCell("x1", "y1")->Set("Value 1");
  test_string_gauge_with_labels->GetCell("x2", "y2")->Set("Value 2");
  test_string_gauge_with_labels->GetCell("x1", "y1")->Set("Value 3");
  test_string_gauge_with_labels->GetCell("x2", "y2")->Set("Value 3");
  EXPECT_EQ(cell_reader.Read("x1", "y1"), "Value 3");
  EXPECT_EQ(cell_reader.Read("x2", "y2"), "Value 3");
  EXPECT_EQ(cell_reader.Read("x1", "y1"), "Value 3");
  EXPECT_EQ(cell_reader.Read("x2", "y2"), "Value 3");
  EXPECT_EQ(cell_reader.Read("x1", "y1"), "Value 3");
  EXPECT_EQ(cell_reader.Read("x2", "y2"), "Value 3");

  test_string_gauge_with_labels->GetCell("x1", "y1")->Set("");
  test_string_gauge_with_labels->GetCell("x2", "y2")->Set("");
  test_string_gauge_with_labels->GetCell("x1", "y1")->Set("-10");
  test_string_gauge_with_labels->GetCell("x2", "y2")->Set("-10");
  EXPECT_EQ(cell_reader.Read("x1", "y1"), "-10");
  EXPECT_EQ(cell_reader.Read("x2", "y2"), "-10");
  EXPECT_EQ(cell_reader.Read("x1", "y1"), "-10");
  EXPECT_EQ(cell_reader.Read("x2", "y2"), "-10");
  EXPECT_EQ(cell_reader.Read("x1", "y1"), "-10");
  EXPECT_EQ(cell_reader.Read("x2", "y2"), "-10");
  test_string_gauge_with_labels->GetCell("x1", "y1")->Set("");
  test_string_gauge_with_labels->GetCell("x2", "y2")->Set("");
  EXPECT_EQ(cell_reader.Read("x1", "y1"), "");
  EXPECT_EQ(cell_reader.Read("x2", "y2"), "");
}

TEST(CellReaderTest, BoolGaugeRead) {
  CellReader<bool> cell_reader("/tsl/monitoring/test/bool_gauge");
  EXPECT_EQ(cell_reader.Read(), false);

  test_bool_gauge->GetCell()->Set(true);
  EXPECT_EQ(cell_reader.Read(), true);

  test_bool_gauge->GetCell()->Set(false);
  EXPECT_EQ(cell_reader.Read(), false);
}

TEST(CellReaderTest, BoolGaugeReadWithLabels) {
  CellReader<bool> cell_reader("/tsl/monitoring/test/bool_gauge_with_labels");
  EXPECT_EQ(cell_reader.Read("x1", "y1"), false);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), false);

  test_bool_gauge_with_labels->GetCell("x1", "y1")->Set(true);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), true);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), false);

  test_bool_gauge_with_labels->GetCell("x2", "y2")->Set(true);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), true);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), true);

  test_bool_gauge_with_labels->GetCell("x1", "y1")->Set(false);
  test_bool_gauge_with_labels->GetCell("x2", "y2")->Set(true);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), false);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), true);

  test_bool_gauge_with_labels->GetCell("x1", "y1")->Set(false);
  test_bool_gauge_with_labels->GetCell("x2", "y2")->Set(false);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), false);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), false);
}

TEST(CellReaderTest, BoolGaugeRepeatedSetAndRead) {
  CellReader<bool> cell_reader("/tsl/monitoring/test/bool_gauge_with_labels");
  EXPECT_EQ(cell_reader.Read("x1", "y1"), false);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), false);

  test_bool_gauge_with_labels->GetCell("x1", "y1")->Set(true);
  test_bool_gauge_with_labels->GetCell("x2", "y2")->Set(false);
  test_bool_gauge_with_labels->GetCell("x1", "y1")->Set(true);
  test_bool_gauge_with_labels->GetCell("x2", "y2")->Set(true);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), true);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), true);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), true);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), true);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), true);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), true);

  test_bool_gauge_with_labels->GetCell("x1", "y1")->Set(false);
  test_bool_gauge_with_labels->GetCell("x2", "y2")->Set(true);
  test_bool_gauge_with_labels->GetCell("x1", "y1")->Set(false);
  test_bool_gauge_with_labels->GetCell("x2", "y2")->Set(true);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), false);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), true);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), false);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), true);
  EXPECT_EQ(cell_reader.Read("x1", "y1"), false);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), true);
  test_bool_gauge_with_labels->GetCell("x2", "y2")->Set(false);
  EXPECT_EQ(cell_reader.Read("x2", "y2"), false);
}

TEST(CellReaderTest, CounterGaugeRead) {
  CellReader<int64_t> cell_reader("/tsl/monitoring/test/counter_gauge");
  EXPECT_EQ(cell_reader.Read(), 0);
  test_counter_gauge->GetCell()->IncrementBy(10);
  EXPECT_EQ(cell_reader.Read(), 10);
  test_counter_gauge->GetCell()->IncrementBy(20);
  EXPECT_EQ(cell_reader.Read(), 30);
  test_counter_gauge->GetCell()->IncrementBy(-30);
  EXPECT_EQ(cell_reader.Read(), 0);
}

TEST(CellReaderTest, CounterGaugeRepeatedIncrementAndDecrement) {
  CellReader<int64_t> cell_reader("/tsl/monitoring/test/counter_gauge");
  EXPECT_EQ(cell_reader.Read(), 0);
  const int kNumIterations = 10;
  for (int i = 0; i < kNumIterations; ++i) {
    test_counter_gauge->GetCell()->Increment();
    EXPECT_EQ(cell_reader.Read(), i + 1);
  }
  for (int i = 0; i < kNumIterations; ++i) {
    test_counter_gauge->GetCell()->Decrement();
    EXPECT_EQ(cell_reader.Read(), 10 - i - 1);
  }
  for (int i = 0; i < kNumIterations; ++i) {
    test_counter_gauge->GetCell()->Increment();
    test_counter_gauge->GetCell()->Decrement();
    EXPECT_EQ(cell_reader.Read(), 0);
  }
}

TEST(CellReaderTest, PercentilesDeltaNoLabels) {
  CellReader<Percentiles> cell_reader("/tsl/monitoring/test/percentiles");
  Percentiles percentiles = cell_reader.Delta();
  EXPECT_EQ(percentiles.num(), 0);
  EXPECT_FLOAT_EQ(percentiles.sum(), 0.0);

  test_percentiles->GetCell()->Add(1.0);
  percentiles = cell_reader.Delta();
  EXPECT_EQ(percentiles.num(), 1);
  EXPECT_FLOAT_EQ(percentiles.sum(), 1.0);

  test_percentiles->GetCell()->Add(-10.0);
  percentiles = cell_reader.Delta();
  EXPECT_EQ(percentiles.num(), 1);
  EXPECT_FLOAT_EQ(percentiles.sum(), -10.0);

  test_percentiles->GetCell()->Add(1000.0);
  percentiles = cell_reader.Delta();
  EXPECT_EQ(percentiles.num(), 1);
  EXPECT_FLOAT_EQ(percentiles.sum(), 1000.0);
}

TEST(CellReaderTest, PercentilesReadNoLabels) {
  CellReader<Percentiles> cell_reader("/tsl/monitoring/test/percentiles");
  Percentiles percentiles = cell_reader.Read();
  EXPECT_EQ(percentiles.num(), 0);
  EXPECT_FLOAT_EQ(percentiles.sum(), 0.0);

  test_percentiles->GetCell()->Add(1.0);
  percentiles = cell_reader.Read();
  EXPECT_EQ(percentiles.num(), 1);
  EXPECT_FLOAT_EQ(percentiles.sum(), 1.0);

  test_percentiles->GetCell()->Add(-10.0);
  percentiles = cell_reader.Read();
  EXPECT_EQ(percentiles.num(), 2);
  EXPECT_FLOAT_EQ(percentiles.sum(), -9.0);

  test_percentiles->GetCell()->Add(1000.0);
  percentiles = cell_reader.Read();
  EXPECT_EQ(percentiles.num(), 3);
  EXPECT_FLOAT_EQ(percentiles.sum(), 991.0);
}

TEST(CellReaderTest, PercentilesWithLabels) {
  CellReader<Percentiles> cell_reader(
      "/tsl/monitoring/test/percentiles_with_labels");
  Percentiles percentiles = cell_reader.Delta("x1", "y1");
  EXPECT_EQ(percentiles.num(), 0);
  EXPECT_FLOAT_EQ(percentiles.sum(), 0.0);
  percentiles = cell_reader.Delta("x2", "y2");
  EXPECT_EQ(percentiles.num(), 0);
  EXPECT_FLOAT_EQ(percentiles.sum(), 0.0);
  percentiles = cell_reader.Read("x1", "y1");
  EXPECT_EQ(percentiles.num(), 0);
  EXPECT_FLOAT_EQ(percentiles.sum(), 0.0);
  percentiles = cell_reader.Read("x2", "y2");
  EXPECT_EQ(percentiles.num(), 0);
  EXPECT_FLOAT_EQ(percentiles.sum(), 0.0);

  test_percentiles_with_labels->GetCell("x1", "y1")->Add(-1.0);
  percentiles = cell_reader.Delta("x1", "y1");
  EXPECT_EQ(percentiles.num(), 1);
  EXPECT_FLOAT_EQ(percentiles.sum(), -1.0);
  percentiles = cell_reader.Delta("x2", "y2");
  EXPECT_EQ(percentiles.num(), 0);
  EXPECT_FLOAT_EQ(percentiles.sum(), 0.0);
  percentiles = cell_reader.Read("x1", "y1");
  EXPECT_EQ(percentiles.num(), 1);
  EXPECT_FLOAT_EQ(percentiles.sum(), -1.0);
  percentiles = cell_reader.Read("x2", "y2");
  EXPECT_EQ(percentiles.num(), 0);
  EXPECT_FLOAT_EQ(percentiles.sum(), 0.0);

  test_percentiles_with_labels->GetCell("x2", "y2")->Add(1.0);
  percentiles = cell_reader.Delta("x1", "y1");
  EXPECT_EQ(percentiles.num(), 0);
  EXPECT_FLOAT_EQ(percentiles.sum(), 0.0);
  percentiles = cell_reader.Delta("x2", "y2");
  EXPECT_EQ(percentiles.num(), 1);
  EXPECT_FLOAT_EQ(percentiles.sum(), 1.0);
  percentiles = cell_reader.Read("x1", "y1");
  EXPECT_EQ(percentiles.num(), 1);
  EXPECT_FLOAT_EQ(percentiles.sum(), -1.0);
  percentiles = cell_reader.Read("x2", "y2");
  EXPECT_EQ(percentiles.num(), 1);
  EXPECT_FLOAT_EQ(percentiles.sum(), 1.0);

  test_percentiles_with_labels->GetCell("x1", "y1")->Add(100.0);
  test_percentiles_with_labels->GetCell("x2", "y2")->Add(-100.0);
  percentiles = cell_reader.Delta("x1", "y1");
  EXPECT_EQ(percentiles.num(), 1);
  EXPECT_FLOAT_EQ(percentiles.sum(), 100.0);
  percentiles = cell_reader.Delta("x2", "y2");
  EXPECT_EQ(percentiles.num(), 1);
  EXPECT_FLOAT_EQ(percentiles.sum(), -100.0);
  percentiles = cell_reader.Read("x1", "y1");
  EXPECT_EQ(percentiles.num(), 2);
  EXPECT_FLOAT_EQ(percentiles.sum(), 99.0);
  percentiles = cell_reader.Read("x2", "y2");
  EXPECT_EQ(percentiles.num(), 2);
  EXPECT_FLOAT_EQ(percentiles.sum(), -99.0);
}

TEST(CellReaderTest, PercentilesRepeatedSetAndRead) {
  CellReader<Percentiles> cell_reader(
      "/tsl/monitoring/test/percentiles_with_labels");
  Percentiles percentiles = cell_reader.Delta("x1", "y1");
  EXPECT_EQ(percentiles.num(), 0);
  EXPECT_FLOAT_EQ(percentiles.sum(), 0.0);
  percentiles = cell_reader.Delta("x1", "y1");
  EXPECT_EQ(percentiles.num(), 0);
  EXPECT_FLOAT_EQ(percentiles.sum(), 0.0);

  test_percentiles_with_labels->GetCell("x1", "y1")->Add(1.0);
  test_percentiles_with_labels->GetCell("x2", "y2")->Add(-1.0);
  test_percentiles_with_labels->GetCell("x1", "y1")->Add(10.0);
  test_percentiles_with_labels->GetCell("x2", "y2")->Add(-10.0);
  test_percentiles_with_labels->GetCell("x1", "y1")->Add(100.0);
  test_percentiles_with_labels->GetCell("x2", "y2")->Add(-100.0);

  percentiles = cell_reader.Delta("x1", "y1");
  EXPECT_EQ(percentiles.num(), 3);
  EXPECT_FLOAT_EQ(percentiles.sum(), 111.0);
  percentiles = cell_reader.Read("x1", "y1");
  EXPECT_EQ(percentiles.num(), 3);
  EXPECT_FLOAT_EQ(percentiles.sum(), 111.0);

  // Repeats the previous read. The values will stay the same, while the deltas
  // will be 0.
  percentiles = cell_reader.Delta("x1", "y1");
  EXPECT_EQ(percentiles.num(), 0);
  EXPECT_FLOAT_EQ(percentiles.sum(), 0);
  percentiles = cell_reader.Read("x1", "y1");
  EXPECT_EQ(percentiles.num(), 3);
  EXPECT_FLOAT_EQ(percentiles.sum(), 111.0);

  percentiles = cell_reader.Delta("x2", "y2");
  EXPECT_EQ(percentiles.num(), 3);
  EXPECT_FLOAT_EQ(percentiles.sum(), -111.0);
  percentiles = cell_reader.Read("x2", "y2");
  EXPECT_EQ(percentiles.num(), 3);
  EXPECT_FLOAT_EQ(percentiles.sum(), -111.0);

  // Repeats the previous read. The values will stay the same, while the deltas
  // will be 0.
  percentiles = cell_reader.Delta("x2", "y2");
  EXPECT_EQ(percentiles.num(), 0);
  EXPECT_FLOAT_EQ(percentiles.sum(), 0);
  percentiles = cell_reader.Read("x2", "y2");
  EXPECT_EQ(percentiles.num(), 3);
  EXPECT_FLOAT_EQ(percentiles.sum(), -111.0);
}

TEST(CellReaderTest, WrongNumberOfLabels) {
  CellReader<int64_t> cell_reader("/tsl/monitoring/test/counter");
  EXPECT_EQ(cell_reader.Read(), 0);
  EXPECT_DEATH(cell_reader.Read("label1"), "has 0 labels");
  EXPECT_DEATH(cell_reader.Read("label1", "label2"), "has 0 labels");
  EXPECT_DEATH(cell_reader.Read("label1", "label2", "label3"), "has 0 labels");

  CellReader<int64_t> cell_reader_with_labels(
      "/tsl/monitoring/test/counter_with_labels");
  EXPECT_DEATH(cell_reader_with_labels.Read(), "has 2 labels");
  EXPECT_DEATH(cell_reader_with_labels.Read("label1"), "has 2 labels");
  EXPECT_EQ(cell_reader_with_labels.Read("label1", "label2"), 0);
  EXPECT_DEATH(cell_reader_with_labels.Read("label1", "label2", "label3"),
               "has 2 labels");
}

TEST(CellReaderTest, MetricIsNotFoundRead) {
  CellReader<int64_t> cell_reader("/metric/does/not/exist");
  CellReader<int64_t> empty_cell_reader("");
  EXPECT_DEATH(cell_reader.Read(), "Metric descriptor is not found");
  EXPECT_DEATH(empty_cell_reader.Read(), "Metric descriptor is not found");
}

TEST(CellReaderTest, StringGaugeDelta) {
  CellReader<std::string> cell_reader("/tsl/monitoring/test/string_gauge");
  CellReader<std::string> cell_reader_with_labels(
      "/tsl/monitoring/test/string_gauge_with_labels");
  EXPECT_DEATH(cell_reader.Delta(), "Please use `Read` instead.");
  EXPECT_DEATH(cell_reader_with_labels.Delta("x", "y"),
               "Please use `Read` instead.");
}

TEST(CellReaderTest, BoolGaugeDelta) {
  CellReader<bool> cell_reader("/tsl/monitoring/test/bool_gauge");
  CellReader<bool> cell_reader_with_labels(
      "/tsl/monitoring/test/bool_gauge_with_labels");
  EXPECT_DEATH(cell_reader.Delta(), "Please use `Read` instead.");
  EXPECT_DEATH(cell_reader_with_labels.Delta("x", "y"),
               "Please use `Read` instead.");
}

TEST(CellReaderTest, InvalidType) {
  CellReader<std::vector<int>> cell_reader("/tsl/monitoring/test/counter");
  CellReader<std::vector<int>> cell_reader_with_labels(
      "/tsl/monitoring/test/counter_with_labels");
  EXPECT_DEATH(cell_reader.Read(),
               "Tensorflow CellReader does not support type");
  EXPECT_DEATH(cell_reader_with_labels.Delta("x", "y"),
               "Tensorflow CellReader does not support type");

  test_counter->GetCell()->IncrementBy(1);
  test_counter_with_labels->GetCell("x", "y")->IncrementBy(1);
  EXPECT_DEATH(cell_reader.Read(),
               "Tensorflow CellReader does not support type");
  EXPECT_DEATH(cell_reader_with_labels.Delta("x", "y"),
               "Tensorflow CellReader does not support type");
}

TEST(CellReaderTest, MetricIsNotFoundDelta) {
  CellReader<int64_t> cell_reader("/metric/does/not/exist");
  CellReader<int64_t> empty_cell_reader("");
  EXPECT_EQ(cell_reader.Delta(), 0);
  EXPECT_EQ(empty_cell_reader.Delta(), 0);
}

TEST(CellReaderTest, LazyInitializationDelta) {
  CellReader<int64_t> cell_reader("/tsl/monitoring/test/lazy_counter");
  EXPECT_EQ(cell_reader.Delta(), 0);
}

TEST(CellReaderTest, LazyInitializationDeltaAfterIncrement) {
  CellReader<int64_t> cell_reader("/tsl/monitoring/test/lazy_counter");
  IncrementLazyCounter();
  EXPECT_EQ(cell_reader.Delta(), 1);
}

}  // namespace
}  // namespace testing
}  // namespace monitoring
}  // namespace tsl
