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

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

namespace tensorflow {
namespace {

using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::FieldsAre;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::testing::VariantWith;

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

TEST(RequestCostTest, RecordStructuredMetrics) {
  RequestCost request_cost;

  RequestCost::StructuredMetric m1{.values = std::vector<double>{1.1, 1.2}};

  RequestCost::StructuredMetric m2{.values = std::vector<std::string>{"c"}};

  request_cost.RecordStructuredMetrics({{"metric_v1", m1}, {"metric_v2", m2}});

  auto metrics = request_cost.GetStructuredMetrics();
  EXPECT_EQ(metrics.size(), 2);
  ASSERT_TRUE(
      std::holds_alternative<std::vector<double>>(metrics["metric_v1"].values));
  EXPECT_THAT(std::get<std::vector<double>>(metrics["metric_v1"].values),
              ElementsAre(1.1, 1.2));

  ASSERT_TRUE(std::holds_alternative<std::vector<std::string>>(
      metrics["metric_v2"].values));
  EXPECT_THAT(std::get<std::vector<std::string>>(metrics["metric_v2"].values),
              ElementsAre("c"));

  const std::vector<std::pair<std::string, RequestCost::StructuredMetric>>
      lvalue_metrics = {
          {"metric_v1",
           RequestCost::StructuredMetric{.values = std::vector<double>{3.3}}},
          {"metric_v3",
           RequestCost::StructuredMetric{
               .values = std::vector<std::string>{"d", "e"}}},
      };
  request_cost.RecordStructuredMetrics(lvalue_metrics);
  EXPECT_THAT(std::get<std::vector<double>>(lvalue_metrics[0].second.values),
              ElementsAre(3.3));
  EXPECT_THAT(
      request_cost.GetStructuredMetrics(),
      UnorderedElementsAre(
          Pair("metric_v1",
               Field(&RequestCost::StructuredMetric::values,
                     VariantWith<std::vector<double>>(ElementsAre(3.3)))),
          Pair("metric_v2",
               Field(&RequestCost::StructuredMetric::values,
                     VariantWith<std::vector<std::string>>(ElementsAre("c")))),
          Pair("metric_v3", Field(&RequestCost::StructuredMetric::values,
                                  VariantWith<std::vector<std::string>>(
                                      ElementsAre("d", "e"))))));
}

TEST(RequestCostTest, RecordStructuredMetricsMoveVector) {
  RequestCost request_cost;

  // Empty rvalue vector on an initially empty RequestCost is a no-op.
  std::vector<std::pair<std::string, RequestCost::StructuredMetric>>
      empty_metrics;
  request_cost.RecordStructuredMetrics(std::move(empty_metrics));
  EXPECT_THAT(request_cost.GetStructuredMetrics(), IsEmpty());

  std::vector<std::pair<std::string, RequestCost::StructuredMetric>>
      structured_metrics = {
          {"metric_v1",
           RequestCost::StructuredMetric{.values =
                                             std::vector<double>{1.1, 1.2}}},
          {"metric_v2",
           RequestCost::StructuredMetric{
               .values = std::vector<std::string>{"block_1", "block_2"}}},
      };
  request_cost.RecordStructuredMetrics(std::move(structured_metrics));

  EXPECT_THAT(
      request_cost.GetStructuredMetrics(),
      UnorderedElementsAre(
          Pair("metric_v1",
               Field(&RequestCost::StructuredMetric::values,
                     VariantWith<std::vector<double>>(ElementsAre(1.1, 1.2)))),
          Pair("metric_v2", Field(&RequestCost::StructuredMetric::values,
                                  VariantWith<std::vector<std::string>>(
                                      ElementsAre("block_1", "block_2"))))));

  // Duplicate keys within the same moved vector and replacement of existing
  // keys both retain the latest value.
  std::vector<std::pair<std::string, RequestCost::StructuredMetric>>
      replacement_metrics = {
          {"metric_v1",
           RequestCost::StructuredMetric{.values = std::vector<double>{4.4}}},
          {"metric_v1",
           RequestCost::StructuredMetric{.values = std::vector<double>{9.9}}},
      };
  request_cost.RecordStructuredMetrics(std::move(replacement_metrics));

  // Recording an empty vector into a populated RequestCost preserves existing
  // entries.
  request_cost.RecordStructuredMetrics(
      std::vector<std::pair<std::string, RequestCost::StructuredMetric>>{});

  EXPECT_THAT(
      request_cost.GetStructuredMetrics(),
      UnorderedElementsAre(
          Pair("metric_v1",
               Field(&RequestCost::StructuredMetric::values,
                     VariantWith<std::vector<double>>(ElementsAre(9.9)))),
          Pair("metric_v2", Field(&RequestCost::StructuredMetric::values,
                                  VariantWith<std::vector<std::string>>(
                                      ElementsAre("block_1", "block_2"))))));
}

TEST(RequestCostTest, RecordStructuredMetricsFromMap) {
  RequestCost dest_cost;

  // Empty rvalue and lvalue maps on an initially empty RequestCost are no-ops.
  absl::flat_hash_map<std::string, RequestCost::StructuredMetric> empty_map;
  dest_cost.RecordStructuredMetrics(empty_map);
  dest_cost.RecordStructuredMetrics(std::move(empty_map));
  EXPECT_THAT(dest_cost.GetStructuredMetrics(), IsEmpty());

  RequestCost source_cost;
  source_cost.RecordStructuredMetrics(
      {{"metric_v1",
        RequestCost::StructuredMetric{.values = std::vector<double>{1.0, 2.0}}},
       {"metric_v2", RequestCost::StructuredMetric{
                         .values = std::vector<std::string>{"id_a", "id_b"}}}});

  // Move map into an initially empty RequestCost.
  dest_cost.RecordStructuredMetrics(source_cost.GetStructuredMetrics());
  EXPECT_THAT(
      dest_cost.GetStructuredMetrics(),
      UnorderedElementsAre(
          Pair("metric_v1",
               Field(&RequestCost::StructuredMetric::values,
                     VariantWith<std::vector<double>>(ElementsAre(1.0, 2.0)))),
          Pair("metric_v2", Field(&RequestCost::StructuredMetric::values,
                                  VariantWith<std::vector<std::string>>(
                                      ElementsAre("id_a", "id_b"))))));

  // Move map into a non-empty RequestCost (replaces existing keys and adds new
  // keys).
  absl::flat_hash_map<std::string, RequestCost::StructuredMetric> moved_map = {
      {"metric_v1",
       RequestCost::StructuredMetric{.values = std::vector<double>{5.0}}},
      {"metric_v3",
       RequestCost::StructuredMetric{.values =
                                         std::vector<std::string>{"id_c"}}},
  };
  dest_cost.RecordStructuredMetrics(std::move(moved_map));
  EXPECT_THAT(
      dest_cost.GetStructuredMetrics(),
      UnorderedElementsAre(
          Pair("metric_v1",
               Field(&RequestCost::StructuredMetric::values,
                     VariantWith<std::vector<double>>(ElementsAre(5.0)))),
          Pair("metric_v2", Field(&RequestCost::StructuredMetric::values,
                                  VariantWith<std::vector<std::string>>(
                                      ElementsAre("id_a", "id_b")))),
          Pair("metric_v3", Field(&RequestCost::StructuredMetric::values,
                                  VariantWith<std::vector<std::string>>(
                                      ElementsAre("id_c"))))));

  // Copy from a const lvalue map without modifying the source map.
  const absl::flat_hash_map<std::string, RequestCost::StructuredMetric>
      const_map = {
          {"metric_v2",
           RequestCost::StructuredMetric{
               .values = std::vector<std::string>{"id_override"}}},
      };
  dest_cost.RecordStructuredMetrics(const_map);
  EXPECT_THAT(
      std::get<std::vector<std::string>>(const_map.at("metric_v2").values),
      ElementsAre("id_override"));

  // Recording an empty rvalue map into a populated RequestCost preserves
  // existing entries.
  dest_cost.RecordStructuredMetrics(
      absl::flat_hash_map<std::string, RequestCost::StructuredMetric>{});

  EXPECT_THAT(
      dest_cost.GetStructuredMetrics(),
      UnorderedElementsAre(
          Pair("metric_v1",
               Field(&RequestCost::StructuredMetric::values,
                     VariantWith<std::vector<double>>(ElementsAre(5.0)))),
          Pair("metric_v2", Field(&RequestCost::StructuredMetric::values,
                                  VariantWith<std::vector<std::string>>(
                                      ElementsAre("id_override")))),
          Pair("metric_v3", Field(&RequestCost::StructuredMetric::values,
                                  VariantWith<std::vector<std::string>>(
                                      ElementsAre("id_c"))))));
}

TEST(RequestCostTest, RecordStructuredCosts) {
  RequestCost request_cost;

  request_cost.RecordStructuredCosts(
      {{"cost_a",
        {{"dim1", absl::Milliseconds(10)}, {"dim2", absl::Milliseconds(5)}}}});
  request_cost.RecordStructuredCosts(
      {{"cost_a",
        {{"dim1", absl::Milliseconds(20)}, {"dim2", absl::Milliseconds(10)}}},
       {"cost_b", {{"dim1", absl::Milliseconds(3)}}}});

  const std::vector<std::pair<absl::string_view, RequestCost::StructuredCost>>
      lvalue_costs = {
          {"cost_a", {{"dim1", absl::Milliseconds(5)}}},
          {"cost_c", {{"dim3", absl::Milliseconds(7)}}},
      };
  request_cost.RecordStructuredCosts(lvalue_costs);
  EXPECT_EQ(lvalue_costs[0].second.at("dim1"), absl::Milliseconds(5));

  auto costs = request_cost.GetStructuredCosts();
  ASSERT_EQ(costs.size(), 3);

  // cost_a should be summed across all calls.
  EXPECT_EQ(costs["cost_a"]["dim1"], absl::Milliseconds(35));
  EXPECT_EQ(costs["cost_a"]["dim2"], absl::Milliseconds(15));

  EXPECT_EQ(costs["cost_b"]["dim1"], absl::Milliseconds(3));
  EXPECT_EQ(costs["cost_c"]["dim3"], absl::Milliseconds(7));
}

TEST(RequestCostTest, RecordStructuredCostsMoveVector) {
  RequestCost request_cost;

  // Empty rvalue vector on an initially empty RequestCost is a no-op.
  std::vector<std::pair<absl::string_view, RequestCost::StructuredCost>>
      empty_costs;
  request_cost.RecordStructuredCosts(std::move(empty_costs));
  EXPECT_THAT(request_cost.GetStructuredCosts(), IsEmpty());

  std::vector<std::pair<absl::string_view, RequestCost::StructuredCost>>
      structured_costs = {
          {"cost_a",
           {{"dim1", absl::Milliseconds(10)}, {"dim2", absl::Milliseconds(5)}}},
          {"cost_b", {{"dim1", absl::Milliseconds(3)}}},
      };
  request_cost.RecordStructuredCosts(std::move(structured_costs));

  EXPECT_THAT(
      request_cost.GetStructuredCosts(),
      UnorderedElementsAre(
          Pair("cost_a",
               UnorderedElementsAre(Pair("dim1", absl::Milliseconds(10)),
                                    Pair("dim2", absl::Milliseconds(5)))),
          Pair("cost_b",
               UnorderedElementsAre(Pair("dim1", absl::Milliseconds(3))))));

  // Duplicate keys within the same moved vector and accumulation onto existing
  // keys both sum dimensions properly.
  std::vector<std::pair<absl::string_view, RequestCost::StructuredCost>>
      additional_costs = {
          {"cost_a", {{"dim1", absl::Milliseconds(4)}}},
          {"cost_a",
           {{"dim1", absl::Milliseconds(6)}, {"dim3", absl::Milliseconds(2)}}},
          {"cost_c", {{"dim1", absl::Milliseconds(1)}}},
          {"cost_c", {{"dim1", absl::Milliseconds(8)}}},
      };
  request_cost.RecordStructuredCosts(std::move(additional_costs));

  // Recording an empty vector into a populated RequestCost preserves existing
  // entries.
  request_cost.RecordStructuredCosts(
      std::vector<std::pair<absl::string_view, RequestCost::StructuredCost>>{});

  EXPECT_THAT(
      request_cost.GetStructuredCosts(),
      UnorderedElementsAre(
          Pair("cost_a",
               UnorderedElementsAre(Pair("dim1", absl::Milliseconds(20)),
                                    Pair("dim2", absl::Milliseconds(5)),
                                    Pair("dim3", absl::Milliseconds(2)))),
          Pair("cost_b",
               UnorderedElementsAre(Pair("dim1", absl::Milliseconds(3)))),
          Pair("cost_c",
               UnorderedElementsAre(Pair("dim1", absl::Milliseconds(9))))));
}

TEST(RequestCostTest, RecordStructuredCostsFromMap) {
  RequestCost dest_cost;

  // Empty rvalue and lvalue maps on an initially empty RequestCost are no-ops.
  absl::flat_hash_map<std::string, RequestCost::StructuredCost> empty_map;
  dest_cost.RecordStructuredCosts(empty_map);
  dest_cost.RecordStructuredCosts(std::move(empty_map));
  EXPECT_THAT(dest_cost.GetStructuredCosts(), IsEmpty());

  RequestCost source_cost;
  source_cost.RecordStructuredCosts(
      {{"cost_a",
        {{"dim1", absl::Milliseconds(10)}, {"dim2", absl::Milliseconds(5)}}},
       {"cost_b", {{"dim1", absl::Milliseconds(3)}}}});

  // Move map into an initially empty RequestCost.
  dest_cost.RecordStructuredCosts(source_cost.GetStructuredCosts());
  EXPECT_THAT(
      dest_cost.GetStructuredCosts(),
      UnorderedElementsAre(
          Pair("cost_a",
               UnorderedElementsAre(Pair("dim1", absl::Milliseconds(10)),
                                    Pair("dim2", absl::Milliseconds(5)))),
          Pair("cost_b",
               UnorderedElementsAre(Pair("dim1", absl::Milliseconds(3))))));

  // Move map into a non-empty RequestCost (sums existing keys and inserts new
  // keys).
  absl::flat_hash_map<std::string, RequestCost::StructuredCost> moved_map = {
      {"cost_a",
       {{"dim1", absl::Milliseconds(15)}, {"dim3", absl::Milliseconds(2)}}},
      {"cost_c", {{"dim2", absl::Milliseconds(8)}}},
  };
  dest_cost.RecordStructuredCosts(std::move(moved_map));
  EXPECT_THAT(
      dest_cost.GetStructuredCosts(),
      UnorderedElementsAre(
          Pair("cost_a",
               UnorderedElementsAre(Pair("dim1", absl::Milliseconds(25)),
                                    Pair("dim2", absl::Milliseconds(5)),
                                    Pair("dim3", absl::Milliseconds(2)))),
          Pair("cost_b",
               UnorderedElementsAre(Pair("dim1", absl::Milliseconds(3)))),
          Pair("cost_c",
               UnorderedElementsAre(Pair("dim2", absl::Milliseconds(8))))));

  // Copy from a const lvalue map without modifying the source map.
  const absl::flat_hash_map<std::string, RequestCost::StructuredCost>
      const_map = {
          {"cost_b", {{"dim1", absl::Milliseconds(7)}}},
      };
  dest_cost.RecordStructuredCosts(const_map);
  EXPECT_EQ(const_map.at("cost_b").at("dim1"), absl::Milliseconds(7));

  // Recording an empty rvalue map into a populated RequestCost preserves
  // existing entries.
  dest_cost.RecordStructuredCosts(
      absl::flat_hash_map<std::string, RequestCost::StructuredCost>{});

  EXPECT_THAT(
      dest_cost.GetStructuredCosts(),
      UnorderedElementsAre(
          Pair("cost_a",
               UnorderedElementsAre(Pair("dim1", absl::Milliseconds(25)),
                                    Pair("dim2", absl::Milliseconds(5)),
                                    Pair("dim3", absl::Milliseconds(2)))),
          Pair("cost_b",
               UnorderedElementsAre(Pair("dim1", absl::Milliseconds(10)))),
          Pair("cost_c",
               UnorderedElementsAre(Pair("dim2", absl::Milliseconds(8))))));
}

TEST(RequestCostTest, ToProtoAndFromProtoRoundTrip) {
  RequestCost orig_cost;
  orig_cost.RecordCost(
      {{"tpu_v1", absl::Milliseconds(11)}, {"cpu_v1", absl::Milliseconds(30)}});
  orig_cost.RecordStructuredCosts(
      {{"tpu_v1",
        {{"dim1", absl::Milliseconds(5)}, {"dim2", absl::Milliseconds(6)}}}});
  orig_cost.RecordMetrics({{"latency", 100.0}, {"throughput", 2.0}});
  orig_cost.RecordStructuredMetrics(
      {{"blockwise_cache_read", {.values = std::vector<double>{10.0, 20.0}}},
       {"blockwise_cache_ids",
        {.values = std::vector<std::string>{"id1", "id2"}}}});
  orig_cost.RecordBatchMetrics(RequestCost::BatchMetrics{
      /*processed_size=*/8,
      /*input_size=*/4,
      /*padding_size=*/1,
      {{"gcu", absl::Milliseconds(80)}, {"tpu", absl::Milliseconds(160)}}});

  RequestCostProto proto = orig_cost.ToProto();

  EXPECT_EQ(proto.costs().at("tpu_v1"), 11000.0);
  EXPECT_EQ(proto.costs().at("cpu_v1"), 30000.0);
  EXPECT_EQ(proto.structured_costs().at("tpu_v1").dimensions().at("dim1"),
            5000.0);
  EXPECT_EQ(proto.structured_costs().at("tpu_v1").dimensions().at("dim2"),
            6000.0);
  EXPECT_EQ(proto.metrics().at("latency"), 100.0);
  EXPECT_EQ(proto.metrics().at("throughput"), 2.0);
  EXPECT_THAT(
      proto.structured_metrics().at("blockwise_cache_read").double_values(),
      ElementsAre(10.0, 20.0));
  EXPECT_THAT(
      proto.structured_metrics().at("blockwise_cache_ids").bytes_values(),
      ElementsAre("id1", "id2"));
  ASSERT_EQ(proto.batch_metrics_size(), 1);
  EXPECT_EQ(proto.batch_metrics(0).processed_size(), 8);
  EXPECT_EQ(proto.batch_metrics(0).input_size(), 4);
  EXPECT_EQ(proto.batch_metrics(0).padding_size(), 1);
  EXPECT_EQ(proto.batch_metrics(0).batch_costs().at("gcu"), 80000.0);
  EXPECT_EQ(proto.batch_metrics(0).batch_costs().at("tpu"), 160000.0);

  // Deserialize into new RequestCost.
  RequestCost new_cost;
  new_cost.FromProto(proto);

  EXPECT_THAT(new_cost.GetCosts(),
              UnorderedElementsAre(Pair("tpu_v1", absl::Milliseconds(11)),
                                   Pair("cpu_v1", absl::Milliseconds(30))));
  EXPECT_THAT(new_cost.GetStructuredCosts(),
              UnorderedElementsAre(Pair(
                  "tpu_v1",
                  UnorderedElementsAre(Pair("dim1", absl::Milliseconds(5)),
                                       Pair("dim2", absl::Milliseconds(6))))));
  EXPECT_THAT(
      new_cost.GetMetrics(),
      UnorderedElementsAre(Pair("latency", 100.0), Pair("throughput", 2.0)));

  auto structured_metrics = new_cost.GetStructuredMetrics();
  EXPECT_EQ(structured_metrics.size(), 2);
  ASSERT_TRUE(std::holds_alternative<std::vector<double>>(
      structured_metrics["blockwise_cache_read"].values));
  EXPECT_THAT(std::get<std::vector<double>>(
                  structured_metrics["blockwise_cache_read"].values),
              ElementsAre(10.0, 20.0));
  ASSERT_TRUE(std::holds_alternative<std::vector<std::string>>(
      structured_metrics["blockwise_cache_ids"].values));
  EXPECT_THAT(std::get<std::vector<std::string>>(
                  structured_metrics["blockwise_cache_ids"].values),
              ElementsAre("id1", "id2"));

  EXPECT_THAT(new_cost.GetBatchMetrics(),
              ElementsAre(FieldsAre(
                  8, 4, 1,
                  UnorderedElementsAre(Pair("gcu", absl::Milliseconds(80)),
                                       Pair("tpu", absl::Milliseconds(160))))));
}

TEST(RequestCostTest, FromProtoReplacesExistingEntries) {
  RequestCost cost;
  cost.RecordCost({{"old_cost", absl::Milliseconds(10)}});
  cost.RecordStructuredCosts(
      {{"old_structured", {{"dim", absl::Milliseconds(1)}}}});
  cost.RecordMetrics({{"old_metric", 1.0}});
  cost.RecordStructuredMetrics(
      {{"old_struct_metric", {.values = std::vector<double>{1.0}}}});
  cost.RecordBatchMetrics(
      RequestCost::BatchMetrics{/*processed_size=*/1,
                                /*input_size=*/1,
                                /*padding_size=*/0,
                                {{"old_batch", absl::Milliseconds(10)}}});

  RequestCostProto proto;
  proto.mutable_costs()->insert({"new_cost", 20000.0});
  (*proto.mutable_structured_costs())["new_structured"]
      .mutable_dimensions()
      ->insert({"new_dim", 3000.0});
  proto.mutable_metrics()->insert({"new_metric", 42.0});
  (*proto.mutable_structured_metrics())["new_struct_metric"]
      .mutable_bytes_values()
      ->Add("str_val");
  auto* bm = proto.add_batch_metrics();
  bm->set_processed_size(2);
  bm->set_input_size(2);
  bm->set_padding_size(0);
  bm->mutable_batch_costs()->insert({"new_batch", 50000.0});

  cost.FromProto(proto);

  EXPECT_THAT(cost.GetCosts(),
              UnorderedElementsAre(Pair("new_cost", absl::Milliseconds(20))));
  EXPECT_THAT(cost.GetStructuredCosts(),
              UnorderedElementsAre(Pair(
                  "new_structured", UnorderedElementsAre(Pair(
                                        "new_dim", absl::Milliseconds(3))))));
  EXPECT_THAT(cost.GetMetrics(),
              UnorderedElementsAre(Pair("new_metric", 42.0)));
  auto structured_metrics = cost.GetStructuredMetrics();
  EXPECT_EQ(structured_metrics.size(), 1);
  ASSERT_TRUE(std::holds_alternative<std::vector<std::string>>(
      structured_metrics["new_struct_metric"].values));
  EXPECT_THAT(std::get<std::vector<std::string>>(
                  structured_metrics["new_struct_metric"].values),
              ElementsAre("str_val"));
  EXPECT_THAT(
      cost.GetBatchMetrics(),
      ElementsAre(FieldsAre(
          2, 2, 0,
          UnorderedElementsAre(Pair("new_batch", absl::Milliseconds(50))))));
}

}  // namespace
}  // namespace tensorflow
