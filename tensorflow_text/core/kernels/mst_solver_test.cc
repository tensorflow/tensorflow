// Copyright 2025 TF.Text Authors.
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

#include "tensorflow_text/core/kernels/mst_solver.h"

#include <limits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace text {

// Testing rig.
//
// Template args:
//   Solver: An instantiation of the MstSolver<> template.
template <class Solver>
class MstSolverTest : public ::testing::Test {
 protected:
  using Index = typename Solver::IndexType;
  using Score = typename Solver::ScoreType;

  // Adds directed arcs for all |num_nodes| nodes to the |solver_| with the
  // |score|.
  void AddAllArcs(Index num_nodes, Score score) {
    for (Index source = 0; source < num_nodes; ++source) {
      for (Index target = 0; target < num_nodes; ++target) {
        if (source == target) continue;
        solver_.AddArc(source, target, score);
      }
    }
  }

  // Adds root selections for all |num_nodes| nodes to the |solver_| with the
  // |score|.
  void AddAllRoots(Index num_nodes, Score score) {
    for (Index root = 0; root < num_nodes; ++root) {
      solver_.AddRoot(root, score);
    }
  }

  // Runs the |solver_| using an argmax array of size |argmax_array_size| and
  // expects it to fail with an error message that matches |error_substr|.
  void SolveAndExpectError(int argmax_array_size,
                           const std::string &error_message_substr) {
    std::vector<Index> argmax(argmax_array_size);
    EXPECT_TRUE(absl::StrContains(solver_.Solve(&argmax).ToString(),
                                  error_message_substr));
  }

  // As above, but expects success.  Does not assert anything about the solution
  // produced by the solver.
  void SolveAndExpectOk(int argmax_array_size) {
    std::vector<Index> argmax(argmax_array_size);
    TF_EXPECT_OK(solver_.Solve(&argmax));
  }

  // As above, but expects the solution to be |expected_argmax| and infers the
  // argmax array size.
  void SolveAndExpectArgmax(const std::vector<Index> &expected_argmax) {
    std::vector<Index> actual_argmax(expected_argmax.size());
    TF_ASSERT_OK(solver_.Solve(&actual_argmax));
    EXPECT_EQ(expected_argmax, actual_argmax);
  }

  // MstSolver<> instance used by the test.  Reused across all MST problems in
  // each test to exercise reuse.
  Solver solver_;
};

using Solvers =
    ::testing::Types<MstSolver<uint8, int16>, MstSolver<uint16, int32>,
                     MstSolver<uint32, int64>, MstSolver<uint16, float>,
                     MstSolver<uint32, double>>;
TYPED_TEST_SUITE(MstSolverTest, Solvers);

TYPED_TEST(MstSolverTest, FailIfNoNodes) {
  for (const bool forest : {false, true}) {
    EXPECT_TRUE(absl::StrContains(this->solver_.Init(forest, 0).ToString(),
                                  "Non-positive number of nodes"));
  }
}

TYPED_TEST(MstSolverTest, FailIfTooManyNodes) {
  // Set to a value that would overflow when doubled.
  const auto kNumNodes =
      (std::numeric_limits<typename TypeParam::IndexType>::max() / 2) + 10;
  for (const bool forest : {false, true}) {
    EXPECT_TRUE(absl::StrContains(
        this->solver_.Init(forest, kNumNodes).ToString(), "Too many nodes"));
  }
}

TYPED_TEST(MstSolverTest, InfeasibleIfNoRootsNoArcs) {
  const int kNumNodes = 10;
  for (const bool forest : {false, true}) {
    TF_ASSERT_OK(this->solver_.Init(forest, kNumNodes));
    this->SolveAndExpectError(kNumNodes, "Infeasible digraph");
  }
}

TYPED_TEST(MstSolverTest, InfeasibleIfNoRootsAllArcs) {
  const int kNumNodes = 10;
  for (const bool forest : {false, true}) {
    TF_ASSERT_OK(this->solver_.Init(forest, kNumNodes));
    this->AddAllArcs(kNumNodes, 0);
    this->SolveAndExpectError(kNumNodes, "Infeasible digraph");
  }
}

TYPED_TEST(MstSolverTest, FeasibleForForestOnlyIfAllRootsNoArcs) {
  const int kNumNodes = 10;
  for (const bool forest : {false, true}) {
    TF_ASSERT_OK(this->solver_.Init(forest, kNumNodes));
    this->AddAllRoots(kNumNodes, 0);
    if (forest) {
      this->SolveAndExpectOk(kNumNodes);  // all roots is a valid forest
    } else {
      this->SolveAndExpectError(kNumNodes, "Infeasible digraph");
    }
  }
}

TYPED_TEST(MstSolverTest, FeasibleIfAllRootsAllArcs) {
  const int kNumNodes = 10;
  for (const bool forest : {false, true}) {
    TF_ASSERT_OK(this->solver_.Init(forest, kNumNodes));
    this->AddAllRoots(kNumNodes, 0);
    this->AddAllArcs(kNumNodes, 0);
    this->SolveAndExpectOk(kNumNodes);
  }
}

TYPED_TEST(MstSolverTest, FailIfArgmaxArrayTooSmall) {
  const int kNumNodes = 10;
  for (const bool forest : {false, true}) {
    TF_ASSERT_OK(this->solver_.Init(forest, kNumNodes));
    this->AddAllRoots(kNumNodes, 0);
    this->AddAllArcs(kNumNodes, 0);
    this->SolveAndExpectError(kNumNodes - 1,  // too small
                              "Argmax array too small");
  }
}

TYPED_TEST(MstSolverTest, OkIfArgmaxArrayTooLarge) {
  const int kNumNodes = 10;
  for (const bool forest : {false, true}) {
    TF_ASSERT_OK(this->solver_.Init(forest, kNumNodes));
    this->AddAllRoots(kNumNodes, 0);
    this->AddAllArcs(kNumNodes, 0);
    this->SolveAndExpectOk(kNumNodes + 1);  // too large
  }
}

TYPED_TEST(MstSolverTest, SolveForAllRootsForestOnly) {
  const int kNumNodes = 10;
  const bool forest = true;
  TF_ASSERT_OK(this->solver_.Init(forest, kNumNodes));
  this->AddAllRoots(kNumNodes, 1);  // favor all root selections
  this->AddAllArcs(kNumNodes, 0);
  this->SolveAndExpectArgmax({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
}

TYPED_TEST(MstSolverTest, SolveForLeftToRightChain) {
  const int kNumNodes = 10;
  for (const bool forest : {false, true}) {
    TF_ASSERT_OK(this->solver_.Init(forest, kNumNodes));
    this->AddAllRoots(kNumNodes, 0);
    this->AddAllArcs(kNumNodes, 0);
    for (int target = 1; target < kNumNodes; ++target) {
      this->solver_.AddArc(target - 1, target, 1);  // favor left-to-right chain
    }
    this->SolveAndExpectArgmax({0, 0, 1, 2, 3, 4, 5, 6, 7, 8});
  }
}

TYPED_TEST(MstSolverTest, SolveForRightToLeftChain) {
  const int kNumNodes = 10;
  for (const bool forest : {false, true}) {
    TF_ASSERT_OK(this->solver_.Init(forest, kNumNodes));
    this->AddAllRoots(kNumNodes, 0);
    this->AddAllArcs(kNumNodes, 0);
    for (int source = 1; source < kNumNodes; ++source) {
      this->solver_.AddArc(source, source - 1, 1);  // favor right-to-left chain
    }
    this->SolveAndExpectArgmax({1, 2, 3, 4, 5, 6, 7, 8, 9, 9});
  }
}

TYPED_TEST(MstSolverTest, SolveForAllFromFirstTree) {
  const int kNumNodes = 10;
  for (const bool forest : {false, true}) {
    TF_ASSERT_OK(this->solver_.Init(forest, kNumNodes));
    this->AddAllRoots(kNumNodes, 0);
    this->AddAllArcs(kNumNodes, 0);
    for (int target = 1; target < kNumNodes; ++target) {
      this->solver_.AddArc(0, target, 1);  // favor first -> target
    }
    this->SolveAndExpectArgmax({0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  }
}

TYPED_TEST(MstSolverTest, SolveForAllFromLastTree) {
  const int kNumNodes = 10;
  for (const bool forest : {false, true}) {
    TF_ASSERT_OK(this->solver_.Init(forest, kNumNodes));
    this->AddAllRoots(kNumNodes, 0);
    this->AddAllArcs(kNumNodes, 0);
    for (int target = 0; target + 1 < kNumNodes; ++target) {
      this->solver_.AddArc(9, target, 1);  // favor last -> target
    }
    this->SolveAndExpectArgmax({9, 9, 9, 9, 9, 9, 9, 9, 9, 9});
  }
}

TYPED_TEST(MstSolverTest, SolveForBinaryTree) {
  const int kNumNodes = 15;
  for (const bool forest : {false, true}) {
    TF_ASSERT_OK(this->solver_.Init(forest, kNumNodes));
    this->AddAllRoots(kNumNodes, 0);
    this->AddAllArcs(kNumNodes, 0);
    for (int target = 1; target < kNumNodes; ++target) {
      this->solver_.AddArc((target - 1) / 2, target, 1);  // like a binary heap
    }
    // clang-format off
    this->SolveAndExpectArgmax({0,
                                0,          0,
                                1,    1,    2,    2,
                                3, 3, 4, 4, 5, 5, 6, 6});
    // clang-format on
  }
}

TYPED_TEST(MstSolverTest, ScoreAccessors) {
  for (const bool forest : {false, true}) {
    TF_ASSERT_OK(this->solver_.Init(forest, 10));
    this->solver_.AddArc(0, 1, 0);
    this->solver_.AddArc(1, 4, 1);
    this->solver_.AddArc(7, 6, 2);
    this->solver_.AddArc(9, 2, 3);

    this->solver_.AddRoot(0, 10);
    this->solver_.AddRoot(2, 20);
    this->solver_.AddRoot(8, 30);

    EXPECT_EQ(this->solver_.ArcScore(0, 1), 0);
    EXPECT_EQ(this->solver_.ArcScore(1, 4), 1);
    EXPECT_EQ(this->solver_.ArcScore(7, 6), 2);
    EXPECT_EQ(this->solver_.ArcScore(9, 2), 3);

    EXPECT_EQ(this->solver_.RootScore(0), 10);
    EXPECT_EQ(this->solver_.RootScore(2), 20);
    EXPECT_EQ(this->solver_.RootScore(8), 30);
  }
}

}  // namespace text
}  // namespace tensorflow
