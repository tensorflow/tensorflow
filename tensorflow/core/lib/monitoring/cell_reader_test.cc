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
#include "tensorflow/core/lib/monitoring/cell_reader.h"

#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/lib/monitoring/test_utils.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace monitoring {
namespace testing {
namespace {

auto* test_counter = monitoring::Counter<0>::New(
    "/tensorflow/monitoring/test/counter", "Test counter.");

auto* test_counter_with_labels = monitoring::Counter<2>::New(
    "/tensorflow/monitoring/test/counter_with_labels",
    "Test counter with two labels.", "label1", "label2");

auto* test_sampler = monitoring::Sampler<0>::New(
    {"/tensorflow/monitoring/test/sampler", "Test sampler."},
    /*buckets=*/monitoring::Buckets::Explicit(
        {0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0}));

auto* test_sampler_with_labels = monitoring::Sampler<2>::New(
    {"/tensorflow/monitoring/test/sampler_with_labels", "Test sampler.",
     "label1", "label2"},
    /*buckets=*/monitoring::Buckets::Exponential(
        /*scale=*/1, /*growth_factor=*/10, /*bucket_count=*/5));

TEST(CellReaderTest, CounterDeltaNoLabels) {
  CellReader<int64_t> cell_reader("/tensorflow/monitoring/test/counter");
  EXPECT_EQ(cell_reader.Delta(), 0);

  test_counter->GetCell()->IncrementBy(5);
  EXPECT_EQ(cell_reader.Delta(), 5);

  test_counter->GetCell()->IncrementBy(10);
  EXPECT_EQ(cell_reader.Delta(), 10);

  test_counter->GetCell()->IncrementBy(100);
  EXPECT_EQ(cell_reader.Delta(), 100);
}

TEST(CellReaderTest, CounterReadNoLabels) {
  CellReader<int64_t> cell_reader("/tensorflow/monitoring/test/counter");
  EXPECT_EQ(cell_reader.Read(), 0);

  test_counter->GetCell()->IncrementBy(5);
  EXPECT_EQ(cell_reader.Read(), 5);

  test_counter->GetCell()->IncrementBy(10);
  EXPECT_EQ(cell_reader.Read(), 15);

  test_counter->GetCell()->IncrementBy(100);
  EXPECT_EQ(cell_reader.Read(), 115);
}

TEST(CellReaderTest, CounterDeltaAndReadNoLabels) {
  CellReader<int64_t> cell_reader("/tensorflow/monitoring/test/counter");
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
  CellReader<int64_t> cell_reader(
      "/tensorflow/monitoring/test/counter_with_labels");
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
  CellReader<int64_t> cell_reader(
      "/tensorflow/monitoring/test/counter_with_labels");
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
  CellReader<int64_t> cell_reader(
      "/tensorflow/monitoring/test/counter_with_labels");
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
  CellReader<int64_t> cell_reader("/tensorflow/monitoring/test/counter");
  CellReader<int64_t> cell_reader_with_labels(
      "/tensorflow/monitoring/test/counter_with_labels");
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
  CellReader<int64_t> cell_reader("/tensorflow/monitoring/test/counter");
  CellReader<int64_t> cell_reader_with_labels(
      "/tensorflow/monitoring/test/counter_with_labels");
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
  CellReader<Histogram> cell_reader("/tensorflow/monitoring/test/sampler");
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
  CellReader<Histogram> cell_reader("/tensorflow/monitoring/test/sampler");
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
  CellReader<Histogram> cell_reader("/tensorflow/monitoring/test/sampler");
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
  CellReader<Histogram> cell_reader(
      "/tensorflow/monitoring/test/sampler_with_labels");
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
  CellReader<Histogram> cell_reader(
      "/tensorflow/monitoring/test/sampler_with_labels");
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
  CellReader<Histogram> cell_reader(
      "/tensorflow/monitoring/test/sampler_with_labels");
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

#ifdef GTEST_HAS_DEATH_TEST
TEST(CellReaderTest, WrongNumberOfLabels) {
  CellReader<int64_t> cell_reader("/tensorflow/monitoring/test/counter");
  EXPECT_EQ(cell_reader.Read(), 0);
  EXPECT_DEATH(cell_reader.Read("label1"), "has 0 labels");
  EXPECT_DEATH(cell_reader.Read("label1", "label2"), "has 0 labels");
  EXPECT_DEATH(cell_reader.Read("label1", "label2", "label3"), "has 0 labels");

  CellReader<int64_t> cell_reader_with_labels(
      "/tensorflow/monitoring/test/counter_with_labels");
  EXPECT_DEATH(cell_reader_with_labels.Read(), "has 2 labels");
  EXPECT_DEATH(cell_reader_with_labels.Read("label1"), "has 2 labels");
  EXPECT_EQ(cell_reader_with_labels.Read("label1", "label2"), 0);
  EXPECT_DEATH(cell_reader_with_labels.Read("label1", "label2", "label3"),
               "has 2 labels");
}

TEST(CellReaderTest, MetricIsNotFound) {
  CellReader<int64_t> cell_reader("/metric/does/not/exist");
  CellReader<int64_t> empty_cell_reader("");
  EXPECT_DEATH(cell_reader.Read(), "Metric descriptor is not found");
  EXPECT_DEATH(empty_cell_reader.Read(), "Metric descriptor is not found");
}
#endif

}  // namespace
}  // namespace testing
}  // namespace monitoring
}  // namespace tensorflow
