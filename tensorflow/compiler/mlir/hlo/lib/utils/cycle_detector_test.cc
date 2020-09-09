/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir-hlo/utils/cycle_detector.h"

#include "tensorflow/compiler/xla/test.h"

class GraphCyclesTest : public ::testing::Test {
 public:
  GraphCyclesTest() : g_(100) {}

  bool AddEdge(int x, int y) { return g_.InsertEdge(x, y); }

  void AddMultiples() {
    // For every node x > 0: add edge to 2*x, 3*x
    for (int x = 1; x < 25; x++) {
      EXPECT_TRUE(AddEdge(x, 2 * x)) << x;
      EXPECT_TRUE(AddEdge(x, 3 * x)) << x;
    }
  }

  mlir::GraphCycles g_;
};

TEST_F(GraphCyclesTest, NoCycle) { AddMultiples(); }

TEST_F(GraphCyclesTest, SimpleCycle) {
  AddMultiples();
  EXPECT_FALSE(AddEdge(8, 4));
}

TEST_F(GraphCyclesTest, IndirectCycle) {
  AddMultiples();
  EXPECT_TRUE(AddEdge(16, 9));
  EXPECT_FALSE(AddEdge(9, 2));
}

TEST_F(GraphCyclesTest, RemoveEdge) {
  EXPECT_TRUE(AddEdge(1, 2));
  EXPECT_TRUE(AddEdge(2, 3));
  EXPECT_TRUE(AddEdge(3, 4));
  EXPECT_TRUE(AddEdge(4, 5));
  g_.RemoveEdge(2, 3);
  EXPECT_FALSE(g_.HasEdge(2, 3));
}

TEST_F(GraphCyclesTest, IsReachable) {
  EXPECT_TRUE(AddEdge(1, 2));
  EXPECT_TRUE(AddEdge(2, 3));
  EXPECT_TRUE(AddEdge(3, 4));
  EXPECT_TRUE(AddEdge(4, 5));

  EXPECT_TRUE(g_.IsReachable(1, 5));
  EXPECT_FALSE(g_.IsReachable(5, 1));
}

TEST_F(GraphCyclesTest, ContractEdge) {
  ASSERT_TRUE(AddEdge(1, 2));
  ASSERT_TRUE(AddEdge(1, 3));
  ASSERT_TRUE(AddEdge(2, 3));
  ASSERT_TRUE(AddEdge(2, 4));
  ASSERT_TRUE(AddEdge(3, 4));

  // It will introduce a cycle if the edge is contracted
  EXPECT_FALSE(g_.ContractEdge(1, 3).hasValue());
  EXPECT_TRUE(g_.HasEdge(1, 3));

  // Node (2) has more edges.
  EXPECT_EQ(*g_.ContractEdge(1, 2), 2);
  EXPECT_TRUE(g_.HasEdge(2, 3));
  EXPECT_TRUE(g_.HasEdge(2, 4));
  EXPECT_TRUE(g_.HasEdge(3, 4));

  // Node (2) has more edges.
  EXPECT_EQ(*g_.ContractEdge(2, 3), 2);
  EXPECT_TRUE(g_.HasEdge(2, 4));
}
