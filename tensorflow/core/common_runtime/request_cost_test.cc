/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/request_cost.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"

namespace tensorflow {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(RequestCostTest, RecordCost) {
  RequestCost request_cost;

  request_cost.RecordCost(
      {{"tpu_v1", absl::Milliseconds(1)}, {"tpu_v2", absl::Milliseconds(2)}});
  request_cost.RecordCost({{"tpu_v1", absl::Milliseconds(10)},
                           {"tpu_v2", absl::Milliseconds(20)},
                           {"cpu_v1", absl::Milliseconds(30)},
                           {"cpu_v2", absl::Milliseconds(40)}});
  EXPECT_THAT(request_cost.GetCosts(),
              UnorderedElementsAre(Pair("tpu_v1", absl::Milliseconds(11)),
                                   Pair("tpu_v2", absl::Milliseconds(22)),
                                   Pair("cpu_v1", absl::Milliseconds(30)),
                                   Pair("cpu_v2", absl::Milliseconds(40))));

  request_cost.RecordCost(
      {{"cpu_v1", absl::Milliseconds(3)}, {"cpu_v2", absl::Milliseconds(4)}});
  EXPECT_THAT(request_cost.GetCosts(),
              UnorderedElementsAre(Pair("tpu_v1", absl::Milliseconds(11)),
                                   Pair("tpu_v2", absl::Milliseconds(22)),
                                   Pair("cpu_v1", absl::Milliseconds(33)),
                                   Pair("cpu_v2", absl::Milliseconds(44))));

  request_cost.ScaleCosts(2);
  EXPECT_THAT(request_cost.GetCosts(),
              UnorderedElementsAre(Pair("tpu_v1", absl::Milliseconds(22)),
                                   Pair("tpu_v2", absl::Milliseconds(44)),
                                   Pair("cpu_v1", absl::Milliseconds(66)),
                                   Pair("cpu_v2", absl::Milliseconds(88))));
}

TEST(RequestCostTest, RecordMetrics) {
  RequestCost request_cost;

  request_cost.RecordMetrics({{"metric_v1", 1}, {"metric_v2", 3.14}});
  EXPECT_THAT(
      request_cost.GetMetrics(),
      UnorderedElementsAre(Pair("metric_v1", 1), Pair("metric_v2", 3.14)));

  request_cost.RecordMetrics({{"metric_v1", 11},
                              {"metric_v2", 3.14159},
                              {"other_metric_v1", 3},
                              {"other_metric_v2", 4}});
  EXPECT_THAT(request_cost.GetMetrics(),
              UnorderedElementsAre(
                  Pair("metric_v1", 11), Pair("metric_v2", 3.14159),
                  Pair("other_metric_v1", 3), Pair("other_metric_v2", 4)));
}

TEST(RequestCostTest, RecordBatchMetrics) {
  RequestCost request_cost;

  request_cost.RecordBatchMetrics(RequestCost::BatchMetrics{
      /*processed_size=*/8,
      /*input_size=*/8,
      /*padding_size=*/0,
      {{"gcu", absl::Milliseconds(80)}, {"tpu", absl::Milliseconds(160)}}});
  request_cost.RecordBatchMetrics(RequestCost::BatchMetrics{
      /*processed_size=*/4,
      /*input_size=*/2,
      /*padding_size=*/1,
      {{"gcu", absl::Milliseconds(40)}, {"tpu", absl::Milliseconds(80)}}});

  EXPECT_THAT(
      request_cost.GetBatchMetrics(),
      ElementsAre(
          FieldsAre(8, 8, 0,
                    UnorderedElementsAre(Pair("gcu", absl::Milliseconds(80)),
                                         Pair("tpu", absl::Milliseconds(160)))),
          FieldsAre(
              4, 2, 1,
              UnorderedElementsAre(Pair("gcu", absl::Milliseconds(40)),
                                   Pair("tpu", absl::Milliseconds(80))))));

  request_cost.ScaleBatchCosts(4);
  EXPECT_THAT(
      request_cost.GetBatchMetrics(),
      ElementsAre(
          FieldsAre(8, 8, 0,
                    UnorderedElementsAre(Pair("gcu", absl::Milliseconds(320)),
                                         Pair("tpu", absl::Milliseconds(640)))),
          FieldsAre(
              4, 2, 1,
              UnorderedElementsAre(Pair("gcu", absl::Milliseconds(160)),
                                   Pair("tpu", absl::Milliseconds(320))))));
}

}  // namespace
}  // namespace tensorflow
