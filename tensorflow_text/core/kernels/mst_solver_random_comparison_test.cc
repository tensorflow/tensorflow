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

#include <time.h>

#include <random>
#include <set>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_text/core/kernels/mst_solver.h"
#include "tensorflow_text/core/kernels/spanning_tree_iterator.h"

ABSL_FLAG(int64_t, seed, 0,
          "Seed for random comparison tests, or 0 for a weak random seed.");
ABSL_FLAG(int, num_trials, 3, "Number of trials for random comparison tests.");

namespace tensorflow {
namespace text {

using ::testing::Contains;

// Returns the random seed, or 0 for a weak random seed.
int64 GetSeed() { return absl::GetFlag(FLAGS_seed); }

// Returns the number of trials to run for each random comparison.
int64 GetNumTrials() { return absl::GetFlag(FLAGS_num_trials); }

// Testing rig.  Runs a comparison between a brute-force MST solver and the
// MstSolver<> on random digraphs.  When the first test parameter is true,
// solves for forests instead of trees.  The second test parameter defines the
// size of the test digraph.
class MstSolverRandomComparisonTest
    : public ::testing::TestWithParam<::testing::tuple<bool, uint32>> {
 protected:
  // Use integer scores so score comparisons are exact.
  using Solver = MstSolver<uint32, int32>;

  // An array providing a source node for each node.  Roots are self-loops.
  using SourceList = SpanningTreeIterator::SourceList;

  // A row-major n x n matrix whose i,j entry gives the score of the arc from i
  // to j, and whose i,i entry gives the score of selecting i as a root.
  using ScoreMatrix = std::vector<int32>;

  // Returns true if this should be a forest.
  bool forest() const { return ::testing::get<0>(GetParam()); }

  // Returns the number of nodes for digraphs.
  uint32 num_nodes() const { return ::testing::get<1>(GetParam()); }

  // Returns the score of the arcs in |sources| based on the |scores|.
  int32 ScoreArcs(const ScoreMatrix &scores, const SourceList &sources) const {
    CHECK_EQ(num_nodes() * num_nodes(), scores.size());
    int32 score = 0;
    for (uint32 target = 0; target < num_nodes(); ++target) {
      const uint32 source = sources[target];
      score += scores[target + source * num_nodes()];
    }
    return score;
  }

  // Returns the score of the maximum spanning tree (or forest, if the first
  // test parameter is true) of the dense digraph defined by the |scores|, and
  // sets |argmax_trees| to contain all maximal trees.
  int32 RunBruteForceMstSolver(const ScoreMatrix &scores,
                               std::set<SourceList> *argmax_trees) {
    CHECK_EQ(num_nodes() * num_nodes(), scores.size());
    int32 max_score;
    argmax_trees->clear();

    iterator_.ForEachTree(num_nodes(), [&](const SourceList &sources) {
      const int32 score = ScoreArcs(scores, sources);
      if (argmax_trees->empty() || max_score < score) {
        max_score = score;
        argmax_trees->clear();
        argmax_trees->insert(sources);
      } else if (max_score == score) {
        argmax_trees->insert(sources);
      }
    });

    return max_score;
  }

  // As above, but uses the |solver_| and extracts only one |argmax_tree|.
  int32 RunMstSolver(const ScoreMatrix &scores, SourceList *argmax_tree) {
    CHECK_EQ(num_nodes() * num_nodes(), scores.size());
    TF_CHECK_OK(solver_.Init(forest(), num_nodes()));

    // Add all roots and arcs.
    for (uint32 source = 0; source < num_nodes(); ++source) {
      for (uint32 target = 0; target < num_nodes(); ++target) {
        const int32 score = scores[target + source * num_nodes()];
        if (source == target) {
          solver_.AddRoot(target, score);
        } else {
          solver_.AddArc(source, target, score);
        }
      }
    }

    // Solve for the max spanning tree.
    argmax_tree->resize(num_nodes());
    TF_CHECK_OK(solver_.Solve(argmax_tree));
    return ScoreArcs(scores, *argmax_tree);
  }

  // Returns a random ScoreMatrix spanning num_nodes() nodes.
  ScoreMatrix RandomScores() {
    ScoreMatrix scores(num_nodes() * num_nodes());
    for (int32 &value : scores) value = static_cast<int32>(prng_() % 201) - 100;
    return scores;
  }

  // Runs a comparison between MstSolver and BruteForceMst on random digraphs of
  // num_nodes() nodes, for the specified number of trials.
  void RunComparison() {
    // Seed the PRNG, possibly non-deterministically.  Log the seed value so the
    // test results can be reproduced, even when the seed is non-deterministic.
    uint32 seed = GetSeed();
    if (seed == 0) seed = time(nullptr);
    prng_.seed(seed);
    LOG(INFO) << "seed = " << seed;

    const int num_trials = GetNumTrials();
    for (int trial = 0; trial < num_trials; ++trial) {
      const ScoreMatrix scores = RandomScores();

      std::set<SourceList> expected_argmax_trees;
      const int32 expected_max_score =
          RunBruteForceMstSolver(scores, &expected_argmax_trees);

      SourceList actual_argmax_tree;
      const int32 actual_max_score = RunMstSolver(scores, &actual_argmax_tree);

      // In case of ties, MstSolver will find a maximal spanning tree, but we
      // don't know which one.
      EXPECT_EQ(expected_max_score, actual_max_score);
      ASSERT_THAT(expected_argmax_trees, Contains(actual_argmax_tree));
    }
  }

  // Tree iterator for brute-force solver.
  SpanningTreeIterator iterator_{forest()};

  // MstSolver<> instance used by the test.  Reused across all MST invocations
  // to exercise reuse.
  Solver solver_;

  // Pseudo-random number generator.
  std::mt19937 prng_;
};

INSTANTIATE_TEST_SUITE_P(AllowForest, MstSolverRandomComparisonTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Range<uint32>(1, 9)));

TEST_P(MstSolverRandomComparisonTest, Comparison) { RunComparison(); }

}  // namespace text
}  // namespace tensorflow
