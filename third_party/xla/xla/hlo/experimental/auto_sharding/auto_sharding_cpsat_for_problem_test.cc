/* Copyright 2023 The OpenXLA Authors.
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

#include "xla/hlo/experimental/auto_sharding/auto_sharding_solver.h"

#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding.pb.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_iopddl.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/hlo/experimental/auto_sharding/iopddl.h"

namespace xla {
namespace spmd {
namespace {

using CostMatrix = std::vector<std::vector<double>>;

iopddl::Problem DefaultProblem() {
  return {
      "default",
      {{{0, 5}, {{110, 100000}, {121, 110000}, {132, 990000}, {143, 130000}}},
       {{0, 5}, {{220, 200000}, {231, 210000}, {242, 220000}}},
       {{2, 4}, {{330, 300000}, {341, 310000}, {352, 320000}, {363, 330000}}},
       {{3, 5}, {{440, 400000}, {451, 410000}, {462, 420000}, {473, 430000}}},
       {{0, 0}, {{550, 500000}, {561, 510000}, {572, 520000}}}},
      {{{0, 2},
        {{1000},
         {1100},
         {1200},
         {1300},
         {2000},
         {2100},
         {2200},
         {2300},
         {3000},
         {3100},
         {3200},
         {3300},
         {4000},
         {4100},
         {4200},
         {4300}}},
       {{1, 2},
        {{5000},
         {5100},
         {5200},
         {5300},
         {6000},
         {6100},
         {6200},
         {6300},
         {7000},
         {7100},
         {7200},
         {7300}}},
       {{1, 4},  // From the alias
        {{0},
         {kInfinityInt},
         {kInfinityInt},
         {kInfinityInt},
         {0},
         {kInfinityInt},
         {kInfinityInt},
         {kInfinityInt},
         {0}}},
       {{2, 3},  // From the follower
        {{0},
         {kInfinityInt},
         {kInfinityInt},
         {kInfinityInt},
         {kInfinityInt},
         {0},
         {kInfinityInt},
         {kInfinityInt},
         {kInfinityInt},
         {kInfinityInt},
         {0},
         {kInfinityInt},
         {kInfinityInt},
         {kInfinityInt},
         {kInfinityInt},
         {0}}}},
      {1500000}};
}

AutoShardingSolverParams DefaultParams() {
  AutoShardingSolverParams params;
  params.departure_costs = {{1.0, 0.0, 1.0, 1.0},
                            {1.0, 0.0, 1.0},
                            {1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                            {1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                            {1.0, 0.0, 1.0}};
  return params;
}

TEST(FormulateAndSolveMIPFromProblemTest, SolvesOptimally) {
  const iopddl::Problem problem = DefaultProblem();
  const AutoShardingSolverParams params = DefaultParams();

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromProblem(
                              problem, params));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 0, 0, 0};
  const double objective_value = 7650.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromProblemTest, SolvesOverbudget) {
  iopddl::Problem problem = DefaultProblem();
  AutoShardingSolverParams params = DefaultParams();
  problem.usage_limit = 100000;
  params.overbudget_coeff = 10.0;

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromProblem(problem, params));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 0, 0, 0};
  const double objective_value = 9007650.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromProblemTest, SolvesMaxDepartures) {
  const iopddl::Problem problem = DefaultProblem();
  AutoShardingSolverParams params = DefaultParams();
  params.max_departures = 3.0;

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromProblem(problem, params));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 1, 1, 0};
  const double objective_value = 7872.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromProblemTest, MinimizesDepartures) {
  const iopddl::Problem problem = DefaultProblem();
  AutoShardingSolverParams params = DefaultParams();
  params.minimize_departures =  true;

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromProblem(problem, params));

  const std::vector<NodeStrategyIdx> s_val = {1, 1, 1, 1, 1};
  const double objective_value = 0.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromProblemTest, AvoidsInfiniteNodeCosts) {
  iopddl::Problem problem = DefaultProblem();
  const AutoShardingSolverParams params = DefaultParams();
  problem.nodes[0].strategies[0].cost = kInfinityInt;
  problem.nodes[0].strategies[1].cost = kInfinityInt;
  problem.nodes[0].strategies[2].cost = kInfinityInt;

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromProblem(problem, params));

  const std::vector<NodeStrategyIdx> s_val = {3, 0, 0, 0, 0};
  const double objective_value = 10683.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromProblemTest, AvoidsInfiniteEdgeCosts) {
  iopddl::Problem problem = DefaultProblem();
  const AutoShardingSolverParams params = DefaultParams();
  problem.edges[0].strategies[0].cost = kInfinityInt;

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromProblem(problem, params));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 1, 1, 0};
  const double objective_value = 7872.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromProblemTest, HandlesFollowedEdges) {
  iopddl::Problem problem = DefaultProblem();
  const AutoShardingSolverParams params = DefaultParams();
  iopddl::Edge edge;
  edge.nodes = {1, 3};
  // Reduces to {1, 2} since node 3 follows node 2
  problem.edges.push_back(edge);
  const CostMatrix r = {{5000, 5100, 5200, 5300,
                         6000, 6100, 6200, 6300,
                         7000, 7100, 7200, 7300}};
  for (auto edge_cost : *r.begin()) {
    iopddl::Strategy strategy;
    strategy.cost = static_cast<iopddl::Cost>(edge_cost);
    problem.edges.back().strategies.push_back(strategy);
  }

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromProblem(problem, params));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 0, 0, 0};
  const double objective_value = 12650.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromProblemTest, HandlesCollapsedEdge) {
  iopddl::Problem problem = DefaultProblem();
  const AutoShardingSolverParams params = DefaultParams();
  iopddl::Edge edge;
  edge.nodes = {2, 3};
  // Both members of this edge will be collapsed into a single node.
  problem.edges.push_back(edge);
  const CostMatrix r = {{9000, 5100, 5200, 5300,
                         6000, 6100, 6200, 6300,
                         7000, 7100, 7200, 7300,
                         8000, 8100, 8200, 8300}};
  for (auto edge_cost : *r.begin()) {
    iopddl::Strategy strategy;
    strategy.cost = static_cast<iopddl::Cost>(edge_cost);
    problem.edges.back().strategies.push_back(strategy);
  }

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                        FormulateAndSolveMIPFromProblem(problem, params));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 1, 1, 0};
  const double objective_value = 13972.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromProblemTest, UsesHint) {
  const iopddl::Problem problem = DefaultProblem();
  AutoShardingSolverParams params = DefaultParams();
  const std::vector<NodeStrategyIdx> s_hint = {1, 0, 0, 0, 0};
  params.s_hint = s_hint;  // Not optimal, but close.

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                        FormulateAndSolveMIPFromProblem(problem, params));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 0, 0, 0};
  const double objective_value = 7650.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(DISABLED_FormulateAndSolveMIPFromProblemTest, HonorsMaxCost) {
  const iopddl::Problem problem = DefaultProblem();
  AutoShardingSolverParams params = DefaultParams();
  params.max_cost = 7600.0;  // Best possible is 7650

  const absl::StatusOr<AutoShardingSolverOutput> result =
      FormulateAndSolveMIPFromProblem(problem,
                                            params);

  EXPECT_TRUE(absl::IsInternal(result.status()));
}

TEST(FormulateAndSolveMIPFromProblemTest, HandlesExtremelyHighMaxCost) {
  const iopddl::Problem problem = DefaultProblem();
  AutoShardingSolverParams params = DefaultParams();
  params.max_cost = 1e19;

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromProblem(problem, params));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 0, 0, 0};
  const double objective_value = 7650.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

}  // namespace
}  // namespace spmd
}  // namespace xla
