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

#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_solver.h"

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"

namespace xla {
namespace spmd {
namespace {

// clang-format off

AutoShardingSolverRequest DefaultAutoShardingSolverRequest() {
  AutoShardingSolverRequest request;
  // The problem below is partially inspired by 'DotLHSTwoNonContractingDims'
  request.num_nodes = 5;
  request.memory_budget = 1500000;
  request.s_len = {4, 3, 4, 4, 3};
  request.s_follow = {-1, -1, -1, 2, -1};
  request.e = {{0, 2}, {1, 2}};
  request.live = {{1, 0},
                  {1, 0},
                  {1, 2, 0},
                  {1, 2, 3, 0},
                  {1, 3, 0}};
  request.c = {{10, 11, 12, 13},
               {20, 21, 22},
               {30, 31, 32, 33},
               {40, 41, 42, 43},
               {50, 51, 52, 53}};
  request.d = {{100, 110, 120, 130},
               {200, 210, 220},
               {300, 310, 320, 330},
               {400, 410, 420, 430},
               {500, 510, 520}};
  request.m = {{100000, 110000, 990000, 130000},
               {200000, 210000, 220000},
               {300000, 310000, 320000, 330000},
               {400000, 410000, 420000, 430000},
               {500000, 510000, 520000}};
  request.r = {{1000, 1100, 1200, 1300,
                2000, 2100, 2200, 2300,
                3000, 3100, 3200, 3300,
                4000, 4100, 4200, 4300},
               {5000, 5100, 5200, 5300,
                6000, 6100, 6200, 6300,
                7000, 7100, 7200, 7300}};
  request.a = {{1, 4}};
  request.v = {{0, 1, 1,
                1, 0, 1,
                1, 1, 0}};
  request.instruction_names = {"A", "B", "C", "D", "E"};
  return request;
}

TEST(CallORToolsSolverTest, SolvesOptimally) {
  const AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();

  const AutoShardingSolverResult result = CallORToolsSolver(request);

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 0, 0, 0};
  const std::vector<EdgeStrategyIdx> e_val = {0, 0};
  const double objective_value = 7650.0;
  const AutoShardingSolverResult expected_result = {
      std::make_tuple(
          std::move(s_val), std::move(e_val), objective_value), false};
  EXPECT_EQ(result, expected_result);
}

TEST(CallORToolsSolverTest, AvoidsInfiniteNodeCosts) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.c[0][0] = request.c[0][1] = request.c[0][2] = kInfinityCost;

  const AutoShardingSolverResult result = CallORToolsSolver(request);

  const std::vector<NodeStrategyIdx> s_val = {3, 0, 0, 0, 0};
  const std::vector<EdgeStrategyIdx> e_val = {12, 0};
  const double objective_value = 10683.0;
  const AutoShardingSolverResult expected_result = {
      std::make_tuple(
          std::move(s_val), std::move(e_val), objective_value), false};
  EXPECT_EQ(result, expected_result);
}

TEST(CallORToolsSolverTest, AvoidsInfiniteEdgeCosts) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.r[0][0] = kInfinityCost;

  const AutoShardingSolverResult result = CallORToolsSolver(request);

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 1, 1, 0};
  const std::vector<EdgeStrategyIdx> e_val = {1, 1};
  const double objective_value = 7872.0;
  const AutoShardingSolverResult expected_result = {
      std::make_tuple(
          std::move(s_val), std::move(e_val), objective_value), false};
  EXPECT_EQ(result, expected_result);
}

TEST(CallORToolsSolverTest, HandlesFollowedEdges) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.e.push_back({1, 3});  // Reduces to {1, 2} since node 3 follows node 2
  request.r.push_back({5000, 5100, 5200, 5300,
                       6000, 6100, 6200, 6300,
                       7000, 7100, 7200, 7300});

  const AutoShardingSolverResult result = CallORToolsSolver(request);

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 0, 0, 0};
  const std::vector<EdgeStrategyIdx> e_val = {0, 0, 0};
  const double objective_value = 12650.0;
  const AutoShardingSolverResult expected_result = {
      std::make_tuple(
          std::move(s_val), std::move(e_val), objective_value), false};
  EXPECT_EQ(result, expected_result);
}

TEST(CallORToolsSolverTest, UsesHint) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.s_hint = {1, 0, 0, 0, 0};  // Not optimal, but close.

  const AutoShardingSolverResult result = CallORToolsSolver(request);

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 0, 0, 0};
  const std::vector<EdgeStrategyIdx> e_val = {0, 0};
  const double objective_value = 7650.0;
  const AutoShardingSolverResult expected_result = {
      std::make_tuple(
          std::move(s_val), std::move(e_val), objective_value), false};
  EXPECT_EQ(result, expected_result);
}

TEST(AutoShardingEvaluatorTest, NoViolations) {
  const AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  const std::vector<NodeStrategyIdx> s_val = {3, 1, 2, 2, 1};
  const std::vector<EdgeStrategyIdx> e_val = {14, 6};
  const double objective_value = 12149.0;
  const AutoShardingSolverResult result = {
      std::make_tuple(
          std::move(s_val), std::move(e_val), objective_value), false};

  const AutoShardingEvaluation evaluation = Evaluate(request, result);

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.total_computation_cost = 159.0;  // 13+21+32+42+51
  expected_evaluation.total_communication_cost = 1590.0;  // 130+210+320+420+510
  expected_evaluation.total_resharding_cost = 10400.0;  // 4200+6200
  expected_evaluation.total_cost = 12149.0;  // 159+1590+10400
  expected_evaluation.lower_bound_computation_cost = 150.0;
  expected_evaluation.lower_bound_communication_cost = 1500.0;
  expected_evaluation.lower_bound_resharding_cost = 6000.0;
  expected_evaluation.lower_bound_cost = 7650.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorTest, ViolatesFollower) {
  const AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  const std::vector<NodeStrategyIdx> s_val = {3, 1, 2, 1 /* violates */, 1};
  const std::vector<EdgeStrategyIdx> e_val = {14, 6};
  const double objective_value = 12138.0;
  const AutoShardingSolverResult result = {
      std::make_tuple(
          std::move(s_val), std::move(e_val), objective_value), false};

  const AutoShardingEvaluation evaluation = Evaluate(request, result);

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.violation_codes = {kFollowerViolationCode};
  expected_evaluation.total_computation_cost = 158.0;  // 13+21+32+41+51
  expected_evaluation.total_communication_cost = 1580.0;  // 130+210+320+410+510
  expected_evaluation.total_resharding_cost = 10400.0;  // 4200+6200
  expected_evaluation.total_cost = 12138.0;  // 158+1580+10400
  expected_evaluation.lower_bound_computation_cost = 150.0;
  expected_evaluation.lower_bound_communication_cost = 1500.0;
  expected_evaluation.lower_bound_resharding_cost = 6000.0;
  expected_evaluation.lower_bound_cost = 7650.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorTest, ViolatesAlias) {
  const AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  const std::vector<NodeStrategyIdx> s_val = {3, 1, 2, 2, 0 /* violates */};
  const std::vector<EdgeStrategyIdx> e_val = {14, 6};
  const double objective_value = 12138.0;
  const AutoShardingSolverResult result = {
      std::make_tuple(
          std::move(s_val), std::move(e_val), objective_value), false};

  const AutoShardingEvaluation evaluation = Evaluate(request, result);

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.violation_codes = {kAliasViolationCode};
  expected_evaluation.total_computation_cost = 158.0;  // 13+21+32+42+50
  expected_evaluation.total_communication_cost = 1580.0;  // 130+210+320+420+500
  expected_evaluation.total_resharding_cost = 10400.0;  // 4200+6200
  expected_evaluation.total_cost = 12138.0;  // 158+1580+10400
  expected_evaluation.lower_bound_computation_cost = 150.0;
  expected_evaluation.lower_bound_communication_cost = 1500.0;
  expected_evaluation.lower_bound_resharding_cost = 6000.0;
  expected_evaluation.lower_bound_cost = 7650.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorTest, ViolatesMemory) {
  const AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  const std::vector<NodeStrategyIdx> s_val = {2 /* violates */, 1, 2, 2, 1};
  const std::vector<EdgeStrategyIdx> e_val = {10, 6};
  const double objective_value = 11138.0;
  const AutoShardingSolverResult result = {
      std::make_tuple(
          std::move(s_val), std::move(e_val), objective_value), false};

  const AutoShardingEvaluation evaluation = Evaluate(request, result);

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.violation_codes = {kMemoryViolationCode};
  expected_evaluation.total_computation_cost = 158.0;  // 12+21+32+42+51
  expected_evaluation.total_communication_cost = 1580.0;  // 120+210+320+420+510
  expected_evaluation.total_resharding_cost = 9400.0;  // 3200+6200
  expected_evaluation.total_cost = 11138.0;  // 158+1580+9400
  expected_evaluation.lower_bound_computation_cost = 150.0;
  expected_evaluation.lower_bound_communication_cost = 1500.0;
  expected_evaluation.lower_bound_resharding_cost = 6000.0;
  expected_evaluation.lower_bound_cost = 7650.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorTest, ViolatesInfiniteCostForNode) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.c[0][0] = request.c[0][1] = request.c[0][2] = kInfinityCost;
  const std::vector<NodeStrategyIdx> s_val = {0 /* violates */, 1, 2, 2, 1};
  const std::vector<EdgeStrategyIdx> e_val = {2, 6};
  const double objective_value = 1e+20;
  const AutoShardingSolverResult result = {
      std::make_tuple(
          std::move(s_val), std::move(e_val), objective_value), false};

  const AutoShardingEvaluation evaluation = Evaluate(request, result);

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.violation_codes = {kInfiniteCostViolationCode};
  expected_evaluation.total_computation_cost = 1e+20;  // infinite cost
  expected_evaluation.total_communication_cost = 1560.0;  // 100+210+320+420+510
  expected_evaluation.total_resharding_cost = 7400.0;  // 1200+6200
  expected_evaluation.total_cost = 1e+20;  // infinite cost
  expected_evaluation.lower_bound_computation_cost = 153.0;
  expected_evaluation.lower_bound_communication_cost = 1500.0;
  expected_evaluation.lower_bound_resharding_cost = 6000.0;
  expected_evaluation.lower_bound_cost = 7653.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorTest, ViolatesInfiniteCostForEdge) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.r[0][2] = kInfinityCost;
  const std::vector<NodeStrategyIdx> s_val = {0, 1, 2, 2, 1};
  const std::vector<EdgeStrategyIdx> e_val = {2 /* violates */, 6};
  const double objective_value = 1e+20;
  const AutoShardingSolverResult result = {
      std::make_tuple(
          std::move(s_val), std::move(e_val), objective_value), false};

  const AutoShardingEvaluation evaluation = Evaluate(request, result);

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.violation_codes = {kInfiniteCostViolationCode};
  expected_evaluation.total_computation_cost = 156.0;  // 10+21+32+42+51
  expected_evaluation.total_communication_cost = 1560.0;  // 100+210+320+420+510
  expected_evaluation.total_resharding_cost = 1e+20;  // infinite cost
  expected_evaluation.total_cost = 1e+20;  // infinite cost
  expected_evaluation.lower_bound_computation_cost = 150.0;
  expected_evaluation.lower_bound_communication_cost = 1500.0;
  expected_evaluation.lower_bound_resharding_cost = 6000.0;
  expected_evaluation.lower_bound_cost = 7650.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingRationalizerTest, RationalizesProperly) {
  const AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  const std::vector<NodeStrategyIdx> s_val = {0, 1, 2, 2, 1};
  const std::vector<EdgeStrategyIdx> e_val = {2, 6};
  const double objective_value = 9116.0;
  const AutoShardingSolverResult result = {
      std::make_tuple(
          std::move(s_val), std::move(e_val), objective_value), false};
  const std::vector<NodeStrategyIdx> s_subopt = {3, 1, 2, 2, 1};
  const std::vector<EdgeStrategyIdx> e_subopt = {14, 6};
  const double subopt_value = 12149.0;
  const AutoShardingSolverResult subopt = {
      std::make_tuple(
          std::move(s_subopt), std::move(e_subopt), subopt_value), false};

  const std::vector<std::string> rationales =
      Rationalize(request, result, subopt);

  const std::vector<std::string> expected_rationales = {
      "strategy changes for A (0 -> 3)",
      "communication cost increases for A (100 -> 130)",
      "computation cost increases for A (10 -> 13)",
      "resharding cost increases for A and C (1200 -> 4200)"};
  EXPECT_EQ(rationales, expected_rationales);
}

// clang-format on

}  // namespace
}  // namespace spmd
}  // namespace xla
