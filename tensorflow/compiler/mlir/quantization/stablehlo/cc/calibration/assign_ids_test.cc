/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/assign_ids.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep for tsl::protobuf

namespace stablehlo::quantization {
namespace {

using ::tensorflow::GraphDef;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::SizeIs;
using ::tsl::protobuf::TextFormat;

TEST(AssignIdsTest, IdsAddedToCustomAggregatorOps) {
  GraphDef graph_def;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        node { op: "CustomAggregator" name: "foo" }
      )pb",
      &graph_def));

  AssignIdsToCustomAggregatorOps(graph_def);

  ASSERT_THAT(graph_def.node(), SizeIs(1));
  EXPECT_TRUE(graph_def.node()[0].attr().contains("id"));
  EXPECT_THAT(graph_def.node()[0].attr().at("id").s(), Not(IsEmpty()));
}

TEST(AssignIdsTest, IdsNotAddedForNonCustomAggregatorOps) {
  GraphDef graph_def;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        node { op: "NotCustomAggregator" name: "bar" }
      )pb",
      &graph_def));

  AssignIdsToCustomAggregatorOps(graph_def);

  ASSERT_THAT(graph_def.node(), SizeIs(1));
  EXPECT_FALSE(graph_def.node()[0].attr().contains("id"));
}

}  // namespace
}  // namespace stablehlo::quantization
