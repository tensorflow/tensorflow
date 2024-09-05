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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/graph_def.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace stablehlo::quantization {
namespace {

using ::tensorflow::GraphDef;
using ::tensorflow::NodeDef;
using ::testing::SizeIs;
using ::testing::StrEq;
using ::tsl::protobuf::TextFormat;

TEST(GraphDefTest, MutateNodeDefsMutatesTopLevelNodeDefs) {
  GraphDef graph_def;
  ASSERT_TRUE(TextFormat::ParseFromString(R"pb(
                                            node { name: "foo" }
                                          )pb",
                                          &graph_def));
  MutateNodeDefs(graph_def,
                 [](NodeDef& node_def) { node_def.set_name("bar"); });

  ASSERT_THAT(graph_def.node(), SizeIs(1));
  EXPECT_THAT(graph_def.node()[0].name(), StrEq("bar"));
}

TEST(GraphDefTest, MutateNodeDefsMutatesFunctionNodeDefs) {
  GraphDef graph_def;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        library { function { node_def { name: "foo" } } }
      )pb",
      &graph_def));

  MutateNodeDefs(graph_def,
                 [](NodeDef& node_def) { node_def.set_name("bar"); });

  ASSERT_THAT(graph_def.library().function(), SizeIs(1));
  ASSERT_THAT(graph_def.library().function()[0].node_def(), SizeIs(1));
  EXPECT_THAT(graph_def.library().function()[0].node_def()[0].name(),
              StrEq("bar"));
}

}  // namespace
}  // namespace stablehlo::quantization
