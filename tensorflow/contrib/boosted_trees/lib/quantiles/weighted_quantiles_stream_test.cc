// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include "tensorflow/contrib/boosted_trees/lib/quantiles/weighted_quantiles_stream.h"

#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {
using Tuple = std::tuple<int64, int64>;

using Summary =
    boosted_trees::quantiles::WeightedQuantilesSummary<double, double>;
using SummaryEntry =
    boosted_trees::quantiles::WeightedQuantilesSummary<double,
                                                       double>::SummaryEntry;
using Stream =
    boosted_trees::quantiles::WeightedQuantilesStream<double, double>;

TEST(GetQuantileSpecs, InvalidEps) {
  EXPECT_DEATH({ Stream::GetQuantileSpecs(-0.01, 0L); }, "eps >= 0");
  EXPECT_DEATH({ Stream::GetQuantileSpecs(1.01, 0L); }, "eps < 1");
}

TEST(GetQuantileSpecs, ZeroEps) {
  EXPECT_DEATH({ Stream::GetQuantileSpecs(0.0, 0L); }, "max_elements > 0");
  EXPECT_EQ(Stream::GetQuantileSpecs(0.0, 1LL), Tuple(1LL, 2LL));
  EXPECT_EQ(Stream::GetQuantileSpecs(0.0, 20LL), Tuple(1LL, 20LL));
}

TEST(GetQuantileSpecs, NonZeroEps) {
  EXPECT_DEATH({ Stream::GetQuantileSpecs(0.01, 0L); }, "max_elements > 0");
  EXPECT_EQ(Stream::GetQuantileSpecs(0.1, 320LL), Tuple(4LL, 31LL));
  EXPECT_EQ(Stream::GetQuantileSpecs(0.01, 25600LL), Tuple(6LL, 501LL));
  EXPECT_EQ(Stream::GetQuantileSpecs(0.01, 104857600LL), Tuple(17LL, 1601LL));
  EXPECT_EQ(Stream::GetQuantileSpecs(0.1, 104857600LL), Tuple(20LL, 191LL));
  EXPECT_EQ(Stream::GetQuantileSpecs(0.01, 1LL << 40), Tuple(29LL, 2801LL));
  EXPECT_EQ(Stream::GetQuantileSpecs(0.001, 1LL << 40), Tuple(26LL, 25001LL));
}

class WeightedQuantilesStreamTest : public ::testing::Test {};

// Stream generators.
void GenerateFixedUniformSummary(int32 worker_id, int64 max_elements,
                                 double *total_weight, Stream *stream) {
  for (int64 i = 0; i < max_elements; ++i) {
    const double x = static_cast<double>(i) / max_elements;
    stream->PushEntry(x, 1.0);
    ++(*total_weight);
  }
  stream->Finalize();
}

void GenerateFixedNonUniformSummary(int32 worker_id, int64 max_elements,
                                    double *total_weight, Stream *stream) {
  for (int64 i = 0; i < max_elements; ++i) {
    const double x = static_cast<double>(i) / max_elements;
    stream->PushEntry(x, x);
    (*total_weight) += x;
  }
  stream->Finalize();
}

void GenerateRandUniformFixedWeightsSummary(int32 worker_id, int64 max_elements,
                                            double *total_weight,
                                            Stream *stream) {
  // Simulate uniform distribution stream.
  random::PhiloxRandom philox(13 + worker_id);
  random::SimplePhilox rand(&philox);
  for (int64 i = 0; i < max_elements; ++i) {
    const double x = rand.RandDouble();
    stream->PushEntry(x, 1);
    ++(*total_weight);
  }
  stream->Finalize();
}

void GenerateRandUniformRandWeightsSummary(int32 worker_id, int64 max_elements,
                                           double *total_weight,
                                           Stream *stream) {
  // Simulate uniform distribution stream.
  random::PhiloxRandom philox(13 + worker_id);
  random::SimplePhilox rand(&philox);
  for (int64 i = 0; i < max_elements; ++i) {
    const double x = rand.RandDouble();
    const double w = rand.RandDouble();
    stream->PushEntry(x, w);
    (*total_weight) += w;
  }
  stream->Finalize();
}

// Single worker tests.
void TestSingleWorkerStreams(
    double eps, int64 max_elements,
    const std::function<void(int32, int64, double *, Stream *)>
        &worker_summary_generator,
    std::initializer_list<double> expected_quantiles,
    double quantiles_matcher_epsilon) {
  // Generate single stream.
  double total_weight = 0;
  Stream stream(eps, max_elements);
  worker_summary_generator(0, max_elements, &total_weight, &stream);

  // Ensure we didn't lose track of any elements and are
  // within approximation error bound.
  EXPECT_LE(stream.ApproximationError(), eps);
  EXPECT_NEAR(stream.GetFinalSummary().TotalWeight(), total_weight, 1e-6);

  // Verify expected quantiles.
  int i = 0;
  auto actuals = stream.GenerateQuantiles(expected_quantiles.size() - 1);
  for (auto expected_quantile : expected_quantiles) {
    EXPECT_NEAR(actuals[i], expected_quantile, quantiles_matcher_epsilon);
    ++i;
  }
}

// Stream generators.
void GenerateOneValue(int32 worker_id, int64 max_elements, double *total_weight,
                      Stream *stream) {
  stream->PushEntry(10, 1);
  ++(*total_weight);
  stream->Finalize();
}

TEST(WeightedQuantilesStreamTest, OneValue) {
  const double eps = 0.01;
  const int64 max_elements = 1 << 16;
  TestSingleWorkerStreams(eps, max_elements, GenerateOneValue,
                          {10.0, 10.0, 10.0, 10.0, 10.0}, 1e-2);
}

TEST(WeightedQuantilesStreamTest, FixedUniform) {
  const double eps = 0.01;
  const int64 max_elements = 1 << 16;
  TestSingleWorkerStreams(eps, max_elements, GenerateFixedUniformSummary,
                          {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
                          1e-2);
}

TEST(WeightedQuantilesStreamTest, FixedNonUniform) {
  const double eps = 0.01;
  const int64 max_elements = 1 << 16;
  TestSingleWorkerStreams(eps, max_elements, GenerateFixedNonUniformSummary,
                          {0, std::sqrt(0.1), std::sqrt(0.2), std::sqrt(0.3),
                           std::sqrt(0.4), std::sqrt(0.5), std::sqrt(0.6),
                           std::sqrt(0.7), std::sqrt(0.8), std::sqrt(0.9), 1.0},
                          1e-2);
}

TEST(WeightedQuantilesStreamTest, RandUniformFixedWeights) {
  const double eps = 0.01;
  const int64 max_elements = 1 << 16;
  TestSingleWorkerStreams(
      eps, max_elements, GenerateRandUniformFixedWeightsSummary,
      {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, 1e-2);
}

TEST(WeightedQuantilesStreamTest, RandUniformRandWeights) {
  const double eps = 0.01;
  const int64 max_elements = 1 << 16;
  TestSingleWorkerStreams(
      eps, max_elements, GenerateRandUniformRandWeightsSummary,
      {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, 1e-2);
}

// Distributed tests.
void TestDistributedStreams(
    int32 num_workers, double eps, int64 max_elements,
    const std::function<void(int32, int64, double *, Stream *)>
        &worker_summary_generator,
    std::initializer_list<double> expected_quantiles,
    double quantiles_matcher_epsilon) {
  // Simulate streams on each worker running independently
  double total_weight = 0;
  std::vector<std::vector<SummaryEntry>> worker_summaries;
  for (int32 i = 0; i < num_workers; ++i) {
    Stream stream(eps / 2, max_elements);
    worker_summary_generator(i, max_elements / num_workers, &total_weight,
                             &stream);
    worker_summaries.push_back(stream.GetFinalSummary().GetEntryList());
  }

  // In the accumulation phase, we aggregate the summaries from each worker
  // and build an overall summary while maintaining error bounds by ensuring we
  // don't increase the error by more than eps / 2.
  Stream reducer_stream(eps, max_elements);
  for (const auto &summary : worker_summaries) {
    reducer_stream.PushSummary(summary);
  }
  reducer_stream.Finalize();

  // Ensure we didn't lose track of any elements and are
  // within approximation error bound.
  EXPECT_LE(reducer_stream.ApproximationError(), eps);
  EXPECT_NEAR(reducer_stream.GetFinalSummary().TotalWeight(), total_weight,
              total_weight);

  // Verify expected quantiles.
  int i = 0;
  auto actuals =
      reducer_stream.GenerateQuantiles(expected_quantiles.size() - 1);
  for (auto expected_quantile : expected_quantiles) {
    EXPECT_NEAR(actuals[i], expected_quantile, quantiles_matcher_epsilon);
    ++i;
  }
}

TEST(WeightedQuantilesStreamTest, FixedUniformDistributed) {
  const int32 num_workers = 10;
  const double eps = 0.01;
  const int64 max_elements = num_workers * (1 << 16);
  TestDistributedStreams(
      num_workers, eps, max_elements, GenerateFixedUniformSummary,
      {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, 1e-2);
}

TEST(WeightedQuantilesStreamTest, FixedNonUniformDistributed) {
  const int32 num_workers = 10;
  const double eps = 0.01;
  const int64 max_elements = num_workers * (1 << 16);
  TestDistributedStreams(num_workers, eps, max_elements,
                         GenerateFixedNonUniformSummary,
                         {0, std::sqrt(0.1), std::sqrt(0.2), std::sqrt(0.3),
                          std::sqrt(0.4), std::sqrt(0.5), std::sqrt(0.6),
                          std::sqrt(0.7), std::sqrt(0.8), std::sqrt(0.9), 1.0},
                         1e-2);
}

TEST(WeightedQuantilesStreamTest, RandUniformFixedWeightsDistributed) {
  const int32 num_workers = 10;
  const double eps = 0.01;
  const int64 max_elements = num_workers * (1 << 16);
  TestDistributedStreams(
      num_workers, eps, max_elements, GenerateRandUniformFixedWeightsSummary,
      {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, 1e-2);
}

TEST(WeightedQuantilesStreamTest, RandUniformRandWeightsDistributed) {
  const int32 num_workers = 10;
  const double eps = 0.01;
  const int64 max_elements = num_workers * (1 << 16);
  TestDistributedStreams(
      num_workers, eps, max_elements, GenerateRandUniformRandWeightsSummary,
      {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, 1e-2);
}

}  // namespace
}  // namespace tensorflow
