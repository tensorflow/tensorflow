/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace gpu {
namespace {

using ::testing::ElementsAre;

TEST(TaskProfileTest, EmptyGraph) {
  UsageGraph deps_graph;
  UsageGraph reallocation_graph = ReallocationGraph(deps_graph);
  EXPECT_TRUE(reallocation_graph.empty());
}

TEST(TaskProfileTest, OneDependency) {
  UsageGraph deps_graph = {{1}, {}};
  UsageGraph reallocation_graph = ReallocationGraph(deps_graph);
  ASSERT_EQ(reallocation_graph.size(), 2);
  EXPECT_TRUE(reallocation_graph[0].empty());
  EXPECT_TRUE(reallocation_graph[1].empty());
}

TEST(TaskProfileTest, ChainDependencies) {
  UsageGraph deps_graph = {{1}, {2}, {3}, {4}, {}};
  UsageGraph reallocation_graph = ReallocationGraph(deps_graph);
  ASSERT_EQ(reallocation_graph.size(), 5);
  EXPECT_THAT(reallocation_graph[0], ElementsAre(2, 3, 4));
  EXPECT_THAT(reallocation_graph[1], ElementsAre(3, 4));
  EXPECT_THAT(reallocation_graph[2], ElementsAre(0, 4));
  EXPECT_THAT(reallocation_graph[3], ElementsAre(0, 1));
  EXPECT_THAT(reallocation_graph[4], ElementsAre(0, 1, 2));
}

TEST(TaskProfileTest, ComplexGraph) {
  UsageGraph deps_graph = {{1}, {2, 3, 4}, {7},  {5},  {7},  {6},
                           {7}, {8, 9},    {10}, {10}, {11}, {}};
  UsageGraph reallocation_graph = ReallocationGraph(deps_graph);
  ASSERT_EQ(reallocation_graph.size(), 12);
  EXPECT_THAT(reallocation_graph[0],
              ElementsAre(2, 3, 4, 5, 6, 7, 8, 9, 10, 11));
  EXPECT_THAT(reallocation_graph[1], ElementsAre(7, 8, 9, 10, 11));
  EXPECT_THAT(reallocation_graph[2], ElementsAre(0, 8, 9, 10, 11));
  EXPECT_THAT(reallocation_graph[3], ElementsAre(0, 6, 7, 8, 9, 10, 11));
  EXPECT_THAT(reallocation_graph[4], ElementsAre(0, 8, 9, 10, 11));
  EXPECT_THAT(reallocation_graph[5], ElementsAre(0, 7, 8, 9, 10, 11));
  EXPECT_THAT(reallocation_graph[6], ElementsAre(0, 3, 8, 9, 10, 11));
  EXPECT_THAT(reallocation_graph[7], ElementsAre(0, 1, 3, 5, 10, 11));
  EXPECT_THAT(reallocation_graph[8], ElementsAre(0, 1, 2, 3, 4, 5, 6, 11));
  EXPECT_THAT(reallocation_graph[9], ElementsAre(0, 1, 2, 3, 4, 5, 6, 11));
  EXPECT_THAT(reallocation_graph[10], ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
  EXPECT_THAT(reallocation_graph[11],
              ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
