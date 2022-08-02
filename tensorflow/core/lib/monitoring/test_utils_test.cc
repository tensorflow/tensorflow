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
#include "tensorflow/core/lib/monitoring/test_utils.h"

#include <string>

#include "tensorflow/core/lib/monitoring/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace monitoring {
namespace testing {
namespace {

using ::tensorflow::testing::StatusIs;
using ::testing::HasSubstr;

template <typename MessageType>
StatusOr<MessageType> ParseTextProto(const std::string& text_proto) {
  protobuf::TextFormat::Parser parser;
  MessageType parsed_proto;
  protobuf::io::ArrayInputStream input_stream(text_proto.data(),
                                              text_proto.size());
  if (!parser.Parse(&input_stream, &parsed_proto)) {
    return errors::InvalidArgument("Could not parse text proto: ", text_proto);
  }
  return parsed_proto;
}

TEST(HistogramTest, Subtract) {
  TF_ASSERT_OK_AND_ASSIGN(HistogramProto histogram1,
                          ParseTextProto<HistogramProto>(R"pb(
                            min: 0.0
                            max: 500.0
                            num: 3.0
                            sum: 555.0
                            sum_squares: 252525.0
                            bucket_limit: 0.0
                            bucket_limit: 10.0
                            bucket_limit: 100.0
                            bucket: 0
                            bucket: 1
                            bucket: 1
                            bucket: 1
                          )pb"));

  TF_ASSERT_OK_AND_ASSIGN(HistogramProto histogram2,
                          ParseTextProto<HistogramProto>(R"pb(
                            min: 0.0
                            max: 5.0
                            num: 1.0
                            sum: 5.0
                            sum_squares: 25.0
                            bucket_limit: 0.0
                            bucket_limit: 10.0
                            bucket_limit: 100.0
                            bucket: 0
                            bucket: 1
                            bucket: 0
                            bucket: 0
                          )pb"));

  TF_ASSERT_OK_AND_ASSIGN(
      Histogram delta, Histogram(histogram1).Subtract(Histogram(histogram2)));
  EXPECT_FLOAT_EQ(delta.num(), 2.0);
  EXPECT_FLOAT_EQ(delta.sum(), 550.0);
  EXPECT_FLOAT_EQ(delta.sum_squares(), 252500.0);
  EXPECT_FLOAT_EQ(delta.num(0), 0.0);
  EXPECT_FLOAT_EQ(delta.num(1), 0.0);
  EXPECT_FLOAT_EQ(delta.num(2), 1.0);
  EXPECT_FLOAT_EQ(delta.num(3), 1.0);
}

TEST(HistogramTest, ReverseSubtract) {
  TF_ASSERT_OK_AND_ASSIGN(HistogramProto histogram1,
                          ParseTextProto<HistogramProto>(R"pb(
                            min: 0.0
                            max: 500.0
                            num: 3.0
                            sum: 555.0
                            sum_squares: 252525.0
                            bucket_limit: 0.0
                            bucket_limit: 10.0
                            bucket_limit: 100.0
                            bucket: 0
                            bucket: 1
                            bucket: 1
                            bucket: 1
                          )pb"));

  TF_ASSERT_OK_AND_ASSIGN(HistogramProto histogram2,
                          ParseTextProto<HistogramProto>(R"pb(
                            min: 0.0
                            max: 5.0
                            num: 1.0
                            sum: 5.0
                            sum_squares: 25.0
                            bucket_limit: 0.0
                            bucket_limit: 10.0
                            bucket_limit: 100.0
                            bucket: 0
                            bucket: 1
                            bucket: 0
                            bucket: 0
                          )pb"));

  EXPECT_THAT(
      Histogram(histogram2).Subtract(Histogram(histogram1)),
      StatusIs(
          error::INVALID_ARGUMENT,
          HasSubstr("Failed to subtract a histogram by a larger histogram.")));
}

TEST(HistogramTest, NegativeSubtract) {
  TF_ASSERT_OK_AND_ASSIGN(HistogramProto histogram1,
                          ParseTextProto<HistogramProto>(R"pb(
                            min: -100.0
                            max: 0.0
                            num: 5.0
                            sum: -500.0
                            sum_squares: 50000.0
                            bucket_limit: 0.0
                            bucket_limit: 10.0
                            bucket_limit: 100.0
                            bucket: 5
                            bucket: 0
                            bucket: 0
                            bucket: 0
                          )pb"));

  TF_ASSERT_OK_AND_ASSIGN(HistogramProto histogram2,
                          ParseTextProto<HistogramProto>(R"pb(
                            min: -100.0
                            max: 0.0
                            num: 2.0
                            sum: -200.0
                            sum_squares: 20000.0
                            bucket_limit: 0.0
                            bucket_limit: 10.0
                            bucket_limit: 100.0
                            bucket: 2
                            bucket: 0
                            bucket: 0
                            bucket: 0
                          )pb"));

  TF_ASSERT_OK_AND_ASSIGN(
      Histogram delta, Histogram(histogram1).Subtract(Histogram(histogram2)));
  EXPECT_FLOAT_EQ(delta.num(), 3.0);
  EXPECT_FLOAT_EQ(delta.sum(), -300.0);
  EXPECT_FLOAT_EQ(delta.sum_squares(), 30000.0);
  EXPECT_FLOAT_EQ(delta.num(0), 3.0);
  EXPECT_FLOAT_EQ(delta.num(1), 0.0);
  EXPECT_FLOAT_EQ(delta.num(2), 0.0);
  EXPECT_FLOAT_EQ(delta.num(3), 0.0);
}

TEST(HistogramTest, SingleBucketSubtract) {
  TF_ASSERT_OK_AND_ASSIGN(HistogramProto histogram1,
                          ParseTextProto<HistogramProto>(R"pb(
                            min: 0.0
                            max: 1.0
                            num: 100.0
                            sum: 100.0
                            sum_squares: 100.0
                            bucket: 100
                          )pb"));

  TF_ASSERT_OK_AND_ASSIGN(HistogramProto histogram2,
                          ParseTextProto<HistogramProto>(R"pb(
                            min: 0.0
                            max: 1.0
                            num: 50.0
                            sum: 50.0
                            sum_squares: 50.0
                            bucket: 50
                          )pb"));

  TF_ASSERT_OK_AND_ASSIGN(
      Histogram delta, Histogram(histogram1).Subtract(Histogram(histogram2)));
  EXPECT_FLOAT_EQ(delta.num(), 50.0);
  EXPECT_FLOAT_EQ(delta.sum(), 50.0);
  EXPECT_FLOAT_EQ(delta.sum_squares(), 50.0);
  EXPECT_FLOAT_EQ(delta.num(0), 50.0);
}

TEST(HistogramTest, SelfSubtract) {
  TF_ASSERT_OK_AND_ASSIGN(HistogramProto histogram,
                          ParseTextProto<HistogramProto>(R"pb(
                            min: 0.0
                            max: 500.0
                            num: 3.0
                            sum: 555.0
                            sum_squares: 252525.0
                            bucket_limit: 0.0
                            bucket_limit: 10.0
                            bucket_limit: 100.0
                            bucket: 0
                            bucket: 1
                            bucket: 1
                            bucket: 1
                          )pb"));

  TF_ASSERT_OK_AND_ASSIGN(Histogram delta,
                          Histogram(histogram).Subtract(Histogram(histogram)));
  EXPECT_FLOAT_EQ(delta.num(), 0.0);
  EXPECT_FLOAT_EQ(delta.sum(), 0.0);
  EXPECT_FLOAT_EQ(delta.sum_squares(), 0.0);
  EXPECT_FLOAT_EQ(delta.num(0), 0.0);
  EXPECT_FLOAT_EQ(delta.num(1), 0.0);
  EXPECT_FLOAT_EQ(delta.num(2), 0.0);
  EXPECT_FLOAT_EQ(delta.num(3), 0.0);
}

TEST(HistogramTest, SubtractEmptyHistogram) {
  TF_ASSERT_OK_AND_ASSIGN(HistogramProto histogram,
                          ParseTextProto<HistogramProto>(R"pb(
                            min: 0.0
                            max: 500.0
                            num: 3.0
                            sum: 555.0
                            sum_squares: 252525.0
                            bucket_limit: 0.0
                            bucket_limit: 10.0
                            bucket_limit: 100.0
                            bucket: 0
                            bucket: 1
                            bucket: 1
                            bucket: 1
                          )pb"));
  const HistogramProto empty;

  TF_ASSERT_OK_AND_ASSIGN(Histogram delta,
                          Histogram(histogram).Subtract(Histogram(empty)));
  EXPECT_FLOAT_EQ(delta.num(), 3.0);
  EXPECT_FLOAT_EQ(delta.sum(), 555.0);
  EXPECT_FLOAT_EQ(delta.sum_squares(), 252525.0);
  EXPECT_FLOAT_EQ(delta.num(0), 0.0);
  EXPECT_FLOAT_EQ(delta.num(1), 1.0);
  EXPECT_FLOAT_EQ(delta.num(2), 1.0);
  EXPECT_FLOAT_EQ(delta.num(3), 1.0);
}

TEST(HistogramTest, SubtractTwoEmptyHistograms) {
  const HistogramProto histogram1;
  const HistogramProto histogram2;

  TF_ASSERT_OK_AND_ASSIGN(
      Histogram delta, Histogram(histogram1).Subtract(Histogram(histogram2)));
  EXPECT_FLOAT_EQ(delta.num(), 0.0);
  EXPECT_FLOAT_EQ(delta.sum(), 0.0);
  EXPECT_FLOAT_EQ(delta.sum_squares(), 0.0);
  EXPECT_FLOAT_EQ(delta.num(0), 0.0);
  EXPECT_FLOAT_EQ(delta.num(1), 0.0);
  EXPECT_FLOAT_EQ(delta.num(2), 0.0);
  EXPECT_FLOAT_EQ(delta.num(3), 0.0);
}

TEST(HistogramTest, DifferentBuckets) {
  TF_ASSERT_OK_AND_ASSIGN(HistogramProto histogram1,
                          ParseTextProto<HistogramProto>(R"pb(
                            min: 0.0
                            max: 500.0
                            num: 3.0
                            sum: 555.0
                            sum_squares: 252525.0
                            bucket_limit: 0.0
                            bucket_limit: 10.0
                            bucket_limit: 100.0
                            bucket: 0
                            bucket: 1
                            bucket: 1
                            bucket: 1
                          )pb"));

  TF_ASSERT_OK_AND_ASSIGN(HistogramProto histogram2,
                          ParseTextProto<HistogramProto>(R"pb(
                            min: 0.0
                            max: 50000.0
                            num: 5.0
                            sum: 55555.0
                            sum_squares: 2525252525.0
                            bucket_limit: 0.0
                            bucket_limit: 10.0
                            bucket_limit: 100.0
                            bucket_limit: 1000.0
                            bucket: 0
                            bucket: 1
                            bucket: 1
                            bucket: 1
                            bucket: 2
                          )pb"));

  EXPECT_THAT(
      Histogram(histogram1).Subtract(Histogram(histogram2)),
      StatusIs(error::INVALID_ARGUMENT,
               HasSubstr("Subtracting a histogram with different buckets.")));
}

TEST(PercentilesTest, Percentiles) {
  tensorflow::monitoring::Percentiles percentiles_value;
  percentiles_value.total_samples = 100;
  percentiles_value.accumulator = -100;
  Percentiles percentiles(percentiles_value);
  EXPECT_EQ(percentiles.num(), 100);
  EXPECT_FLOAT_EQ(percentiles.sum(), -100);

  Percentiles delta = percentiles.Subtract(percentiles);
  EXPECT_EQ(delta.num(), 0);
  EXPECT_FLOAT_EQ(delta.sum(), 0);

  delta = delta.Subtract(percentiles);
  EXPECT_EQ(delta.num(), -100);
  EXPECT_FLOAT_EQ(delta.sum(), 100);
}

TEST(PercentilesTest, Subtract) {
  tensorflow::monitoring::Percentiles percentiles_value1;
  percentiles_value1.total_samples = 100;
  percentiles_value1.accumulator = 100;
  Percentiles percentiles1(percentiles_value1);
  EXPECT_EQ(percentiles1.num(), 100);
  EXPECT_FLOAT_EQ(percentiles1.sum(), 100);

  tensorflow::monitoring::Percentiles percentiles_value2;
  percentiles_value2.total_samples = 90;
  percentiles_value2.accumulator = 90;
  Percentiles percentiles2(percentiles_value2);
  EXPECT_EQ(percentiles2.num(), 90);
  EXPECT_FLOAT_EQ(percentiles2.sum(), 90);

  Percentiles delta = percentiles1.Subtract(percentiles2);
  EXPECT_EQ(delta.num(), 10);
  EXPECT_FLOAT_EQ(delta.sum(), 10);
}

TEST(PercentilesTest, ReverseSubtract) {
  tensorflow::monitoring::Percentiles percentiles_value1;
  percentiles_value1.total_samples = 100;
  percentiles_value1.accumulator = 100;
  Percentiles percentiles1(percentiles_value1);
  EXPECT_EQ(percentiles1.num(), 100);
  EXPECT_FLOAT_EQ(percentiles1.sum(), 100);

  tensorflow::monitoring::Percentiles percentiles_value2;
  percentiles_value2.total_samples = 90;
  percentiles_value2.accumulator = 90;
  Percentiles percentiles2(percentiles_value2);
  EXPECT_EQ(percentiles2.num(), 90);
  EXPECT_FLOAT_EQ(percentiles2.sum(), 90);

  Percentiles delta = percentiles2.Subtract(percentiles1);
  EXPECT_EQ(delta.num(), -10);
  EXPECT_FLOAT_EQ(delta.sum(), -10);
}

TEST(PercentilesTest, SubtractEmptyPercentile) {
  tensorflow::monitoring::Percentiles percentiles_value;
  percentiles_value.total_samples = 1;
  percentiles_value.accumulator = 1;
  Percentiles percentiles(percentiles_value);
  EXPECT_EQ(percentiles.num(), 1);
  EXPECT_FLOAT_EQ(percentiles.sum(), 1);

  Percentiles empty_percentile((tensorflow::monitoring::Percentiles()));
  EXPECT_EQ(empty_percentile.num(), 0);
  EXPECT_FLOAT_EQ(empty_percentile.sum(), 0);

  Percentiles delta = percentiles.Subtract(empty_percentile);
  EXPECT_EQ(delta.num(), 1);
  EXPECT_FLOAT_EQ(delta.sum(), 1);
}

TEST(PercentilesTest, EmptyPercentiles) {
  Percentiles empty_percentile((tensorflow::monitoring::Percentiles()));
  EXPECT_EQ(empty_percentile.num(), 0);
  EXPECT_FLOAT_EQ(empty_percentile.sum(), 0);

  Percentiles delta = empty_percentile.Subtract(empty_percentile);
  EXPECT_EQ(delta.num(), 0);
  EXPECT_FLOAT_EQ(delta.sum(), 0);
}

}  // namespace
}  // namespace testing
}  // namespace monitoring
}  // namespace tensorflow
