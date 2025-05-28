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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding.pb.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_iopddl.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/hlo/experimental/auto_sharding/iopddl.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace spmd {
namespace {

using CostMatrix = std::vector<std::vector<double>>;
using NodeMatrix = std::vector<std::vector<int64_t>>;
using EdgeMatrix = std::vector<std::vector<int64_t>>;

void AddCosts(proto2::RepeatedPtrField<AutoShardingSolverRequest_Costs>* costs,
              const CostMatrix& cost_matrix) {
  for (const auto& cost_row : cost_matrix) {
    AutoShardingSolverRequest_Costs cost;
    cost.mutable_costs()->Add(cost_row.begin(), cost_row.end());
    costs->Add(std::move(cost));
  }
}

void AddEdges(proto2::RepeatedPtrField<AutoShardingSolverRequest_Edges>* edges,
              const EdgeMatrix& edge_matrix) {
  for (const auto& edge_row : edge_matrix) {
    AutoShardingSolverRequest_Edges edge;
    edge.mutable_edges()->Add(edge_row.begin(), edge_row.end());
    edges->Add(std::move(edge));
  }
}

void AddIntervals(
    proto2::RepeatedPtrField<AutoShardingSolverRequest_Pair>* pairs,
    const std::vector<std::pair<int64_t, int64_t>>& intervals) {
  for (const auto& interval : intervals) {
    AutoShardingSolverRequest_Pair pair;
    pair.set_first(interval.first);
    pair.set_second(interval.second);
    pairs->Add(std::move(pair));
  }
}

void AddGroups(
    proto2::RepeatedPtrField<AutoShardingSolverRequest_Group>* groups,
    const std::vector<std::vector<int64_t>>& reduced_groups) {
  for (const auto& reduced_group : reduced_groups) {
    AutoShardingSolverRequest_Group group;
    group.mutable_prims()->Add(reduced_group.begin(), reduced_group.end());
    groups->Add(std::move(group));
  }
}

AutoShardingSolverRequest DefaultAutoShardingSolverRequest() {
  // The problem below is partially inspired by 'DotLHSTwoNonContractingDims'
  const auto s_len = {4, 3, 4, 4, 3};
  const auto s_follow = {-1, -1, -1, 2, -1};
  AutoShardingSolverRequest_Pair edge1, edge2;
  edge1.set_first(0);
  edge1.set_second(2);
  edge2.set_first(1);
  edge2.set_second(2);
  const auto edges = {edge1, edge2};
  const std::vector<std::pair<int64_t, int64_t>> node_intervals =
      {{0, 4}, {0, 4}, {2, 3}, {3, 4}, {100, -1}};
  const std::vector<std::pair<int64_t, int64_t>> edge_intervals =
      {{1, 2}, {2, 3}};
  const CostMatrix c = {{10, 11, 12, 13},
                        {20, 21, 22},
                        {30, 31, 32, 33},
                        {40, 41, 42, 43},
                        {50, 51, 52}};
  const CostMatrix d = {{100, 110, 120, 130},
                        {200, 210, 220},
                        {300, 310, 320, 330},
                        {400, 410, 420, 430},
                        {500, 510, 520}};
  const CostMatrix m = {{100000, 110000, 990000, 130000},
                        {200000, 210000, 220000},
                        {300000, 310000, 320000, 330000},
                        {400000, 410000, 420000, 430000},
                        {500000, 510000, 520000}};
  const CostMatrix p = {{1.0, 0.0, 1.0, 1.0},
                        {1.0, 0.0, 1.0},
                        {1.0, 0.0, 1.0, 1.0},
                        {1.0, 0.0, 1.0, 1.0},
                        {1.0, 0.0, 1.0}};
  const CostMatrix r = {{1000, 1100, 1200, 1300,
                         2000, 2100, 2200, 2300,
                         3000, 3100, 3200, 3300,
                         4000, 4100, 4200, 4300},
                        {5000, 5100, 5200, 5300,
                         6000, 6100, 6200, 6300,
                         7000, 7100, 7200, 7300}};
  const CostMatrix t = {{73000, 72000, 71000, 70000,
                         63000, 62000, 61000, 60000,
                         53000, 52000, 51000, 50000,
                         43000, 42000, 41000, 40000},
                        {33000, 32000, 31000, 30000,
                         23000, 22000, 21000, 20000,
                         13000, 12000, 11000, 10000}};
  AutoShardingSolverRequest_Pair alias;
  alias.set_first(1);
  alias.set_second(4);
  const auto aliases = {alias};
  const CostMatrix v = {{0, 1, 1,
                         1, 0, 1,
                         1, 1, 0}};
  const std::vector<std::string> instruction_names = {"A", "B", "C", "D", "E"};
  const std::vector<std::string> metadata_source_files = {"attention.py",
                                                          "convolution.py",
                                                          "layers.py",
                                                          "logits.py",
                                                          "pipeline.py"};

  AutoShardingSolverRequest request;
  request.set_num_nodes(5);
  request.set_memory_budget(1500000);
  request.mutable_s_len()->Add(s_len.begin(), s_len.end());
  request.mutable_s_follow()->Add(s_follow.begin(), s_follow.end());
  request.mutable_edges()->Add(edges.begin(), edges.end());
  AddIntervals(request.mutable_node_intervals(), node_intervals);
  AddIntervals(request.mutable_edge_intervals(), edge_intervals);
  AddCosts(request.mutable_computation_costs(), c);
  AddCosts(request.mutable_communication_costs(), d);
  AddCosts(request.mutable_memory_costs(), m);
  AddCosts(request.mutable_departure_costs(), p);
  AddCosts(request.mutable_resharding_costs(), r);
  AddCosts(request.mutable_duration_costs(), t);
  request.mutable_aliases()->Add(aliases.begin(), aliases.end());
  AddCosts(request.mutable_value_costs(), v);
  request.mutable_instruction_names()->Add(instruction_names.begin(),
                                           instruction_names.end());
  request.mutable_metadata_source_files()->Add(metadata_source_files.begin(),
                                               metadata_source_files.end());
  return request;
}

AutoShardingSolverRequest AutoShardingSolverRequestWithEquivalences() {
  const auto s_len = {4, 3, 7, 7, 3};
  const auto s_follow = {-1, -1, -1, 2, -1};
  AutoShardingSolverRequest_Pair edge1, edge2;
  edge1.set_first(0);
  edge1.set_second(2);
  edge2.set_first(1);
  edge2.set_second(2);
  const auto edges = {edge1, edge2};
  const std::vector<std::pair<int64_t, int64_t>> node_intervals =
      {{0, 4}, {0, 4}, {2, 3}, {3, 4}, {100, -1}};
  const std::vector<std::pair<int64_t, int64_t>> edge_intervals =
      {{1, 2}, {2, 3}};
  const CostMatrix c = {{10, 10, 10, 10},
                        {20, 20, 20},
                        {30, 30, 31, 30, 30, 30, 30},
                        {40, 40, 40, 40, 40, 40, 40},
                        {50, 50, 50}};
  const CostMatrix d = {{100, 100, 100, 100},
                        {200, 200, 200},
                        {300, 300, 300, 300, 300, 300, 300},
                        {400, 400, 400, 400, 400, 400, 410},
                        {500, 500, 500}};
  const CostMatrix m = {{10000, 10000, 10000, 10000},
                        {20000, 20000, 20000},
                        {30000, 30000, 30000, 31000, 30000, 30000, 30000},
                        {40000, 40000, 40000, 40000, 40000, 40000, 40000},
                        {50000, 50000, 50000}};
  const CostMatrix p = {{1.0, 0.0, 1.0, 1.0},
                        {1.0, 0.0, 1.0},
                        {1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                        {1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                        {1.0, 0.0, 1.0}};
  const CostMatrix r = {{1000, 1000, 1000, 1000, 1000, 1000, 1000,
                         2000, 2000, 2000, 2000, 2000, 2000, 2000,
                         3000, 3000, 3000, 3000, 3100, 3000, 3000,
                         4000, 4000, 4000, 4000, 4000, 4000, 4000},
                        {5000, 5000, 5000, 5000, 5000, 5000, 5000,
                         6000, 6000, 6000, 6000, 6000, 6000, 6000,
                         7000, 7000, 7000, 7000, 7000, 7000, 7000}};
  const CostMatrix t = {{70000, 70000, 70000, 70000, 70000, 70000, 70000,
                         60000, 60000, 60000, 60000, 60000, 60000, 60000,
                         50000, 50000, 50000, 50000, 50000, 50000, 50000,
                         40000, 40000, 40000, 40000, 40000, 40000, 40000},
                        {30000, 30000, 30000, 30000, 30000, 30000, 30000,
                         20000, 20000, 20000, 20000, 20000, 20000, 20000,
                         10000, 10000, 10000, 10000, 10000, 10000, 10000}};
  AutoShardingSolverRequest_Pair alias;
  alias.set_first(2);
  alias.set_second(4);
  const auto aliases = {alias};
  const CostMatrix v = {{0, 1, 0,
                         0, 1, 0,
                         0, 1, 0,
                         0, 1, 0,
                         0, 1, 0,
                         1, 0, 1,
                         0, 1, 0}};
  const std::vector<std::string> instruction_names = {"A", "B", "C", "D", "E"};

  AutoShardingSolverRequest request;
  request.set_num_nodes(5);
  request.set_memory_budget(1500000);
  request.mutable_s_len()->Add(s_len.begin(), s_len.end());
  request.mutable_s_follow()->Add(s_follow.begin(), s_follow.end());
  request.mutable_edges()->Add(edges.begin(), edges.end());
  AddIntervals(request.mutable_node_intervals(), node_intervals);
  AddIntervals(request.mutable_edge_intervals(), edge_intervals);
  AddCosts(request.mutable_computation_costs(), c);
  AddCosts(request.mutable_communication_costs(), d);
  AddCosts(request.mutable_memory_costs(), m);
  AddCosts(request.mutable_departure_costs(), p);
  AddCosts(request.mutable_resharding_costs(), r);
  AddCosts(request.mutable_duration_costs(), t);
  request.mutable_aliases()->Add(aliases.begin(), aliases.end());
  AddCosts(request.mutable_value_costs(), v);
  request.mutable_instruction_names()->Add(instruction_names.begin(),
                                           instruction_names.end());
  return request;
}

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
        {{kInfinityInt},
         {kInfinityInt},
         {0},
         {kInfinityInt},
         {0},
         {kInfinityInt},
         {0},
         {kInfinityInt},
         {kInfinityInt}}},
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

TEST(FormulateAndSolveMIPFromSolverRequestTest, SolvesOptimally) {
  const AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromSolverRequest(
                              request, GetParams(request)));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 0, 0, 0};
  const double objective_value = 7650.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromSolverRequestTest, SolvesOverbudget) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.set_memory_budget(100000);
  request.mutable_overbudget_coeff()->set_coeff(10.0);

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromSolverRequest(
                              request, GetParams(request)));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 0, 0, 0};
  const double objective_value = 9007650.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromSolverRequestTest, SolvesMaxDepartures) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.mutable_max_departures()->set_coeff(3.0);

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromSolverRequest(
                              request, GetParams(request)));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 1, 1, 0};
  const double objective_value = 7872.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromSolverRequestTest, MinimizesDepartures) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.set_minimize_departures(true);

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromSolverRequest(
                              request, GetParams(request)));

  const std::vector<NodeStrategyIdx> s_val = {1, 1, 1, 1, 1};
  const double objective_value = 0.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromSolverRequestTest, AvoidsInfiniteNodeCosts) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.mutable_computation_costs(0)->set_costs(0, kInfinityCost);
  request.mutable_computation_costs(0)->set_costs(1, kInfinityCost);
  request.mutable_computation_costs(0)->set_costs(2, kInfinityCost);

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromSolverRequest(
                              request, GetParams(request)));

  const std::vector<NodeStrategyIdx> s_val = {3, 0, 0, 0, 0};
  const double objective_value = 10683.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromSolverRequestTest, AvoidsInfiniteEdgeCosts) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.mutable_resharding_costs(0)->set_costs(0, kInfinityCost);

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromSolverRequest(
                              request, GetParams(request)));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 1, 1, 0};
  const double objective_value = 7872.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromSolverRequestTest, HandlesFollowedEdges) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  AutoShardingSolverRequest_Pair edge;
  edge.set_first(1);
  edge.set_second(3);
  // Reduces to {1, 2} since node 3 follows node 2
  *request.mutable_edges()->Add() = edge;
  const CostMatrix r = {{5000, 5100, 5200, 5300,
                         6000, 6100, 6200, 6300,
                         7000, 7100, 7200, 7300}};
  AddCosts(request.mutable_resharding_costs(), r);
  const CostMatrix t = {{50000, 51000, 52000, 53000,
                         60000, 61000, 62000, 63000,
                         70000, 71000, 72000, 73000}};
  AddCosts(request.mutable_duration_costs(), t);

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromSolverRequest(
                              request, GetParams(request)));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 0, 0, 0};
  const double objective_value = 12650.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromSolverRequestTest, HandlesCollapsedEdge) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  AutoShardingSolverRequest_Pair edge;
  edge.set_first(2);
  edge.set_second(3);
  // Both members of this edge will be collapsed into a single node.
  *request.mutable_edges()->Add() = edge;
  const CostMatrix r = {{9000, 5100, 5200, 5300,
                         6000, 6100, 6200, 6300,
                         7000, 7100, 7200, 7300,
                         8000, 8100, 8200, 8300}};
  AddCosts(request.mutable_resharding_costs(), r);
  const CostMatrix t = {{50000, 51000, 52000, 53000,
                         60000, 61000, 62000, 63000,
                         70000, 71000, 72000, 73000,
                         80000, 81000, 82000, 83000}};
  AddCosts(request.mutable_duration_costs(), t);

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                        FormulateAndSolveMIPFromSolverRequest(
                            request, GetParams(request)));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 1, 1, 0};
  const double objective_value = 13972.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromSolverRequestTest, UsesHint) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  const auto s_hint = {1, 0, 0, 0, 0};  // Not optimal, but close.
  request.mutable_s_hint()->Add(s_hint.begin(), s_hint.end());

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                        FormulateAndSolveMIPFromSolverRequest(
                            request, GetParams(request)));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 0, 0, 0};
  const double objective_value = 7650.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromSolverRequestTest, HonorsMaxCost) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.mutable_max_cost()->set_coeff(7600.0);  // Best possible is 7650.0

  const absl::StatusOr<AutoShardingSolverOutput> result =
      FormulateAndSolveMIPFromSolverRequest(request,
                                            GetParams(request));

  EXPECT_TRUE(absl::IsInternal(result.status()));
}

TEST(FormulateAndSolveMIPFromSolverRequestTest, HandlesExtremelyHighMaxCost) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.mutable_max_cost()->set_coeff(1e19);

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromSolverRequest(
                              request, GetParams(request)));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 0, 0, 0};
  const double objective_value = 7650.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(DISABLED_FormulateAndSolveMIPFromSolverRequestTest,
     HandlesMemoryEdgeCosts) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  const EdgeMatrix live_edges = {{}, {0}, {0, 1}, {1}, {}};
  const CostMatrix memory_edge_costs = {{1000000, 1100, 1200, 1300,
                                         2000, 2100, 2200, 2300,
                                         3000, 3100, 3200, 3300,
                                         4000, 4100, 4200, 4300},
                                        {5000000, 5100, 5200, 5300,
                                         6000, 6100, 6200, 6300,
                                         7000, 7100, 7200, 7300}};
  AddEdges(request.mutable_live_edges(), live_edges);
  AddCosts(request.mutable_memory_edge_costs(), memory_edge_costs);
  request.set_enable_memory_edge_costs(true);

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromSolverRequest(
                              request, GetParams(request)));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 1, 1, 0};
  const double objective_value = 7872.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(DISABLED_FormulateAndSolveMIPFromSolverRequestTest,
     HandlesIntervals) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  const CostMatrix memory_edge_costs = {{1000000, 1100, 1200, 1300,
                                         2000, 2100, 2200, 2300,
                                         3000, 3100, 3200, 3300,
                                         4000, 4100, 4200, 4300},
                                        {5000000, 5100, 5200, 5300,
                                         6000, 6100, 6200, 6300,
                                         7000, 7100, 7200, 7300}};
  AddCosts(request.mutable_memory_edge_costs(), memory_edge_costs);
  request.set_enable_memory_edge_costs(true);

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromSolverRequest(
                              request, GetParams(request)));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 1, 1, 0};
  const double objective_value = 7872.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(DISABLED_FormulateAndSolveMIPFromSolverRequestTest,
     HandlesReducedIntervalsAndGroups) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  const std::vector<std::pair<int64_t, int64_t>> node_intervals =
      {{5, -1}, {5, -1}, {2, 3}, {3, 4}, {100, -1}, {0, 4}};
  const std::vector<std::pair<int64_t, int64_t>> edge_intervals =
      {{1, 2}, {2, 3}};
  const std::vector<std::vector<int64_t>> node_groups = {{0, 1}};
  const std::vector<std::vector<int64_t>> edge_groups = {};
  const CostMatrix memory_edge_costs = {{1000000, 1100, 1200, 1300,
                                         2000, 2100, 2200, 2300,
                                         3000, 3100, 3200, 3300,
                                         4000, 4100, 4200, 4300},
                                        {5000000, 5100, 5200, 5300,
                                         6000, 6100, 6200, 6300,
                                         7000, 7100, 7200, 7300}};
  request.clear_node_intervals();
  request.clear_edge_intervals();
  AddIntervals(request.mutable_node_intervals(), node_intervals);
  AddIntervals(request.mutable_edge_intervals(), edge_intervals);
  AddGroups(request.mutable_node_groups(), node_groups);
  AddGroups(request.mutable_edge_groups(), edge_groups);
  AddCosts(request.mutable_memory_edge_costs(), memory_edge_costs);
  request.set_enable_memory_edge_costs(true);

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromSolverRequest(
                              request, GetParams(request)));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 1, 1, 0};
  const double objective_value = 7872.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(DISABLED_FormulateAndSolveMIPFromSolverRequestTest,
     HandlesReducedIntervalsAndGroupsNoMemoryEdgeCosts) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  const std::vector<std::pair<int64_t, int64_t>> node_intervals =
      {{5, -1}, {5, -1}, {2, 3}, {3, 4}, {100, -1}, {0, 4}};
  const std::vector<std::vector<int64_t>> node_groups = {{0, 1}};
  request.clear_node_intervals();
  request.clear_edge_intervals();
  AddIntervals(request.mutable_node_intervals(), node_intervals);
  AddGroups(request.mutable_node_groups(), node_groups);
  request.set_enable_memory_edge_costs(false);

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromSolverRequest(
                              request, GetParams(request)));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 0, 0, 0};
  const double objective_value = 7650.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(DISABLED_FormulateAndSolveMIPFromSolverRequestTest,
     HandlesGroupsWithTinyMemoryCosts) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  const std::vector<std::pair<int64_t, int64_t>> node_intervals =
      {{5, -1}, {5, -1}, {2, 3}, {3, 4}, {100, -1}, {0, 4}};
  const std::vector<std::pair<int64_t, int64_t>> edge_intervals =
      {{1, 2}, {2, 3}};
  const std::vector<std::vector<int64_t>> node_groups = {{0, 1}};
  const std::vector<std::vector<int64_t>> edge_groups = {};
  const CostMatrix memory_costs = {{1, 1, 1, 1},  // These values are tiny and
                                   {2, 2, 2},     // shouldn't be rounded up.
                                   {300, 300, 300, 300, 300, 300, 300},
                                   {4000, 4000, 4000, 4000, 4000, 4000, 4000},
                                   {50000, 50000, 50000}};
  const CostMatrix memory_edge_costs = {{0, 0, 0, 0,
                                         0, 0, 0, 0,
                                         0, 0, 0, 0,
                                         0, 0, 0, 0},
                                        {0, 0, 0, 0,
                                         0, 0, 0, 0,
                                         0, 0, 0, 0}};
  request.clear_node_intervals();
  request.clear_edge_intervals();
  request.clear_memory_costs();
  AddIntervals(request.mutable_node_intervals(), node_intervals);
  AddIntervals(request.mutable_edge_intervals(), edge_intervals);
  AddGroups(request.mutable_node_groups(), node_groups);
  AddGroups(request.mutable_edge_groups(), edge_groups);
  AddCosts(request.mutable_memory_costs(), memory_costs);
  AddCosts(request.mutable_memory_edge_costs(), memory_edge_costs);
  request.set_enable_memory_edge_costs(true);
  request.set_memory_budget(4321);

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromSolverRequest(
                              request, GetParams(request)));

  const std::vector<NodeStrategyIdx> s_val = {0, 0, 0, 0, 0};
  const double objective_value = 7650.0;
  const AutoShardingSolverOutput expected_output = {s_val, objective_value};
  EXPECT_EQ(result, expected_output);
}

TEST(FormulateAndSolveMIPFromSolverRequestTest, SolvesWithEquivalences) {
  const AutoShardingSolverRequest request =
      AutoShardingSolverRequestWithEquivalences();

  TF_ASSERT_OK_AND_ASSIGN(const AutoShardingSolverOutput result,
                          FormulateAndSolveMIPFromSolverRequest(
                              request, GetParams(request)));

  const double objective_value = 7650.0;
  EXPECT_EQ(result.cost, objective_value);  // Note: multiple solutions possible
}

TEST(AutoShardingEvaluatorTest, NoViolations) {
  const AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  const std::vector<NodeStrategyIdx> s_val = {3, 1, 2, 2, 1};
  const double objective_value = 12149.0;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation =
      Evaluate(request, output, GetParams(request));

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.total.computation_cost = 159.0;  // 13+21+32+42+51
  expected_evaluation.total.communication_cost = 1590.0;  // 130+210+320+420+510
  expected_evaluation.total.resharding_cost = 10400.0;  // 4200+6200
  expected_evaluation.total.max_memory = 1080000.0;
  expected_evaluation.lower_bound.computation_cost = 150.0;
  expected_evaluation.lower_bound.communication_cost = 1500.0;
  expected_evaluation.lower_bound.resharding_cost = 6000.0;
  expected_evaluation.lower_bound.max_memory = 1000000.0;
  expected_evaluation.total_departures = 3.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorTest, EvaluatesOverbudget) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.set_memory_budget(100000);
  request.mutable_overbudget_coeff()->set_coeff(10.0);
  const std::vector<NodeStrategyIdx> s_val = {2 /* violates */, 1, 2, 2, 1};
  const double objective_value = 11138.0;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation =
      Evaluate(request, output, GetParams(request));

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.total.computation_cost = 158.0;  // 12+21+32+42+51
  expected_evaluation.total.communication_cost = 1580.0;  // 120+210+320+420+510
  expected_evaluation.total.resharding_cost = 9400.0;  // 3200+6200
  expected_evaluation.total.overbudget_cost = 18400000.0;  // 10*1840000
  expected_evaluation.total.max_memory = 1940000.0;
  expected_evaluation.lower_bound.computation_cost = 150.0;
  expected_evaluation.lower_bound.communication_cost = 1500.0;
  expected_evaluation.lower_bound.resharding_cost = 6000.0;
  expected_evaluation.lower_bound.overbudget_cost = 9000000.0;
  expected_evaluation.lower_bound.max_memory = 1000000.0;
  expected_evaluation.total_departures = 3.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorTest, EvaluatesOverbudgetWithIntervals) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.set_memory_budget(100000);
  request.mutable_overbudget_coeff()->set_coeff(10.0);
  const std::vector<NodeStrategyIdx> s_val = {2 /* violates */, 1, 2, 2, 1};
  const double objective_value = 11138.0;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation =
      Evaluate(request, output, GetParams(request));

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.total.computation_cost = 158.0;  // 12+21+32+42+51
  expected_evaluation.total.communication_cost = 1580.0;  // 120+210+320+420+510
  expected_evaluation.total.resharding_cost = 9400.0;  // 3200+6200
  expected_evaluation.total.overbudget_cost = 18400000.0;  // 10*1840000
  expected_evaluation.total.max_memory = 1940000.0;
  expected_evaluation.lower_bound.computation_cost = 150.0;
  expected_evaluation.lower_bound.communication_cost = 1500.0;
  expected_evaluation.lower_bound.resharding_cost = 6000.0;
  expected_evaluation.lower_bound.overbudget_cost = 9000000.0;
  expected_evaluation.lower_bound.max_memory = 1000000.0;
  expected_evaluation.total_departures = 3.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(DISABLED_AutoShardingEvaluatorTest,
     EvaluatesOverbudgetWithReducedIntervalsAndGroups) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  const std::vector<std::pair<int64_t, int64_t>> node_intervals =
      {{5, -1}, {5, -1}, {2, 3}, {3, 4}, {100, -1}, {0, 4}};
  const std::vector<std::vector<int64_t>> node_groups = {{0, 1}};
  request.set_memory_budget(100000);
  request.mutable_overbudget_coeff()->set_coeff(10.0);
  request.clear_node_intervals();
  request.clear_edge_intervals();
  AddIntervals(request.mutable_node_intervals(), node_intervals);
  AddGroups(request.mutable_node_groups(), node_groups);
  const std::vector<NodeStrategyIdx> s_val = {2 /* violates */, 1, 2, 2, 1};
  const double objective_value = 11138.0;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation =
      Evaluate(request, output, GetParams(request));

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.total.computation_cost = 158.0;  // 12+21+32+42+51
  expected_evaluation.total.communication_cost = 1580.0;  // 120+210+320+420+510
  expected_evaluation.total.resharding_cost = 9400.0;  // 3200+6200
  expected_evaluation.total.overbudget_cost = 18400000.0;  // 10*1840000
  expected_evaluation.total.max_memory = 1940000.0;
  expected_evaluation.lower_bound.computation_cost = 150.0;
  expected_evaluation.lower_bound.communication_cost = 1500.0;
  expected_evaluation.lower_bound.resharding_cost = 6000.0;
  expected_evaluation.lower_bound.overbudget_cost = 9000000.0;
  expected_evaluation.lower_bound.max_memory = 1000000.0;
  expected_evaluation.total_departures = 3.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorTest, ViolatesFollower) {
  const AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  const std::vector<NodeStrategyIdx> s_val = {3, 1, 2, 1 /* violates */, 1};
  const double objective_value = 12138.0;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation =
      Evaluate(request, output, GetParams(request));

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.violation_codes = {kFollowerViolationCode};
  expected_evaluation.total.computation_cost = 158.0;  // 13+21+32+41+51
  expected_evaluation.total.communication_cost = 1580.0;  // 130+210+320+410+510
  expected_evaluation.total.resharding_cost = 10400.0;  // 4200+6200
  expected_evaluation.total.max_memory = 1070000.0;
  expected_evaluation.lower_bound.computation_cost = 150.0;
  expected_evaluation.lower_bound.communication_cost = 1500.0;
  expected_evaluation.lower_bound.resharding_cost = 6000.0;
  expected_evaluation.lower_bound.max_memory = 1000000.0;
  expected_evaluation.total_departures = 2.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorTest, ViolatesAlias) {
  const AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  const std::vector<NodeStrategyIdx> s_val = {3, 1, 2, 2, 0 /* violates */};
  const double objective_value = 12138.0;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation =
      Evaluate(request, output, GetParams(request));

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.violation_codes = {kAliasViolationCode};
  expected_evaluation.total.computation_cost = 158.0;  // 13+21+32+42+50
  expected_evaluation.total.communication_cost = 1580.0;  // 130+210+320+420+500
  expected_evaluation.total.resharding_cost = 10400.0;  // 4200+6200
  expected_evaluation.total.max_memory = 1080000.0;
  expected_evaluation.lower_bound.computation_cost = 150.0;
  expected_evaluation.lower_bound.communication_cost = 1500.0;
  expected_evaluation.lower_bound.resharding_cost = 6000.0;
  expected_evaluation.lower_bound.max_memory = 1000000.0;
  expected_evaluation.total_departures = 4.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorTest, ViolatesMemory) {
  const AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  const std::vector<NodeStrategyIdx> s_val = {2 /* violates */, 1, 2, 2, 1};
  const double objective_value = 11138.0;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation =
      Evaluate(request, output, GetParams(request));

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.violation_codes = {kMemoryViolationCode};
  expected_evaluation.total.computation_cost = 158.0;  // 12+21+32+42+51
  expected_evaluation.total.communication_cost = 1580.0;  // 120+210+320+420+510
  expected_evaluation.total.resharding_cost = 9400.0;  // 3200+6200
  expected_evaluation.total.max_memory = 1940000.0;
  expected_evaluation.lower_bound.computation_cost = 150.0;
  expected_evaluation.lower_bound.communication_cost = 1500.0;
  expected_evaluation.lower_bound.resharding_cost = 6000.0;
  expected_evaluation.lower_bound.max_memory = 1000000.0;
  expected_evaluation.total_departures = 3.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorTest, ViolatesInfiniteCostForNode) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.mutable_computation_costs(0)->set_costs(0, kInfinityCost);
  request.mutable_computation_costs(0)->set_costs(1, kInfinityCost);
  request.mutable_computation_costs(0)->set_costs(2, kInfinityCost);
  const std::vector<NodeStrategyIdx> s_val = {0 /* violates */, 1, 2, 2, 1};
  const double objective_value = 1e+20;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation =
      Evaluate(request, output, GetParams(request));

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.violation_codes = {kInfiniteCostViolationCode};
  expected_evaluation.total.computation_cost = 1e+20;  // infinite cost
  expected_evaluation.total.communication_cost = 1560.0;  // 100+210+320+420+510
  expected_evaluation.total.resharding_cost = 7400.0;  // 1200+6200
  expected_evaluation.total.max_memory = 1050000.0;
  expected_evaluation.lower_bound.computation_cost = 153.0;
  expected_evaluation.lower_bound.communication_cost = 1500.0;
  expected_evaluation.lower_bound.resharding_cost = 6000.0;
  expected_evaluation.lower_bound.max_memory = 1000000.0;
  expected_evaluation.total_departures = 3.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorTest, ViolatesInfiniteCostForEdge) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.mutable_resharding_costs(0)->set_costs(2, kInfinityCost);
  const std::vector<NodeStrategyIdx> s_val = {0, 1, 2, 2, 1};
  const double objective_value = 1e+20;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation =
      Evaluate(request, output, GetParams(request));

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.violation_codes = {kInfiniteCostViolationCode};
  expected_evaluation.total.computation_cost = 156.0;  // 10+21+32+42+51
  expected_evaluation.total.communication_cost = 1560.0;  // 100+210+320+420+510
  expected_evaluation.total.resharding_cost = 1e+20;  // infinite cost
  expected_evaluation.total.max_memory = 1050000.0;
  expected_evaluation.lower_bound.computation_cost = 150.0;
  expected_evaluation.lower_bound.communication_cost = 1500.0;
  expected_evaluation.lower_bound.resharding_cost = 6000.0;
  expected_evaluation.lower_bound.max_memory = 1000000.0;
  expected_evaluation.total_departures = 3.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorTest, ViolatesMaxDepartures) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  request.mutable_max_departures()->set_coeff(2.0);
  const std::vector<NodeStrategyIdx> s_val = {3, 1, 2, 2, 1};
  const double objective_value = 12149.0;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation =
      Evaluate(request, output, GetParams(request));

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.violation_codes = {kMaxDeparturesViolationCode};
  expected_evaluation.total.computation_cost = 159.0;  // 13+21+32+42+51
  expected_evaluation.total.communication_cost = 1590.0;  // 130+210+320+420+510
  expected_evaluation.total.resharding_cost = 10400.0;  // 4200+6200
  expected_evaluation.total.max_memory = 1080000.0;
  expected_evaluation.lower_bound.computation_cost = 150.0;
  expected_evaluation.lower_bound.communication_cost = 1500.0;
  expected_evaluation.lower_bound.resharding_cost = 6000.0;
  expected_evaluation.lower_bound.max_memory = 1000000.0;
  expected_evaluation.total_departures = 3.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorForProblemTest, NoViolations) {
  const iopddl::Problem problem = DefaultProblem();
  const AutoShardingSolverParams params = DefaultParams();
  const std::vector<NodeStrategyIdx> s_val = {3, 1, 2, 2, 1};
  const double objective_value = 12149.0;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation = Evaluate(problem, output, params);

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.total.node_cost = 1749;   // 1590+159
  expected_evaluation.total.edge_cost = 10400;  // 4200+6200
  expected_evaluation.total.max_usage = 1080000;
  expected_evaluation.lower_bound.node_cost = 1650;
  expected_evaluation.lower_bound.edge_cost = 6000;
  expected_evaluation.lower_bound.max_usage = 1000000;
  expected_evaluation.total_departures = 3.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorForProblemTest, EvaluatesOverbudget) {
  iopddl::Problem problem = DefaultProblem();
  AutoShardingSolverParams params = DefaultParams();
  problem.usage_limit = 100000;
  params.overbudget_coeff = 10.0;
  const std::vector<NodeStrategyIdx> s_val = {2 /* violates */, 1, 2, 2, 1};
  const double objective_value = 11138.0;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation = Evaluate(problem, output, params);

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.total.node_cost = 1738;             // 1580+158
  expected_evaluation.total.edge_cost = 9400;             // 3200+6200
  expected_evaluation.total.overbudget_usage = 18400000;  // 10*1840000
  expected_evaluation.total.max_usage = 1940000;
  expected_evaluation.lower_bound.node_cost = 1650;
  expected_evaluation.lower_bound.edge_cost = 6000;
  expected_evaluation.lower_bound.overbudget_usage = 9000000;
  expected_evaluation.lower_bound.max_usage = 1000000;
  expected_evaluation.total_departures = 3.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorForProblemTest, ViolatesFollower) {
  const iopddl::Problem problem = DefaultProblem();
  const AutoShardingSolverParams params = DefaultParams();
  const std::vector<NodeStrategyIdx> s_val = {3, 1, 2, 1 /* violates */, 1};
  const double objective_value = 12138.0;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation = Evaluate(problem, output, params);

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.violation_codes = {kFollowerViolationCode};
  expected_evaluation.total.node_cost = 1738;   // 1580+158
  expected_evaluation.total.edge_cost = 10400;  // 4200+6200
  expected_evaluation.total.max_usage = 1070000;
  expected_evaluation.lower_bound.node_cost = 1650;
  expected_evaluation.lower_bound.edge_cost = 6000;
  expected_evaluation.lower_bound.max_usage = 1000000;
  expected_evaluation.total_departures = 2.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorForProblemTest, ViolatesAlias) {
  const iopddl::Problem problem = DefaultProblem();
  const AutoShardingSolverParams params = DefaultParams();
  const std::vector<NodeStrategyIdx> s_val = {3, 1, 2, 2, 0 /* violates */};
  const double objective_value = 12138.0;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation = Evaluate(problem, output, params);

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.violation_codes = {kAliasViolationCode};
  expected_evaluation.total.node_cost = 1738;   // 1580+158
  expected_evaluation.total.edge_cost = 10400;  // 4200+6200
  expected_evaluation.total.max_usage = 1080000;
  expected_evaluation.lower_bound.node_cost = 1650;
  expected_evaluation.lower_bound.edge_cost = 6000;
  expected_evaluation.lower_bound.max_usage = 1000000;
  expected_evaluation.total_departures = 4.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorForProblemTest, ViolatesMemory) {
  const iopddl::Problem problem = DefaultProblem();
  const AutoShardingSolverParams params = DefaultParams();
  const std::vector<NodeStrategyIdx> s_val = {2 /* violates */, 1, 2, 2, 1};
  const double objective_value = 11138.0;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation = Evaluate(problem, output, params);

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.violation_codes = {kMemoryViolationCode};
  expected_evaluation.total.node_cost = 1738;  // 1580+158
  expected_evaluation.total.edge_cost = 9400;  // 3200+6200
  expected_evaluation.total.max_usage = 1940000;
  expected_evaluation.lower_bound.node_cost = 1650;
  expected_evaluation.lower_bound.edge_cost = 6000;
  expected_evaluation.lower_bound.max_usage = 1000000;
  expected_evaluation.total_departures = 3.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorForProblemTest, ViolatesInfiniteCostForNode) {
  iopddl::Problem problem = DefaultProblem();
  const AutoShardingSolverParams params = DefaultParams();
  problem.nodes[0].strategies[0].cost = kInfinityInt;
  problem.nodes[0].strategies[1].cost = kInfinityInt;
  problem.nodes[0].strategies[2].cost = kInfinityInt;
  const std::vector<NodeStrategyIdx> s_val = {0 /* violates */, 1, 2, 2, 1};
  const double objective_value = 1e+20;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation = Evaluate(problem, output, params);

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.violation_codes = {kInfiniteCostViolationCode};
  expected_evaluation.total.node_cost = 1000000000000001606;
  expected_evaluation.total.edge_cost = 7400;  // 1200+6200
  expected_evaluation.total.max_usage = 1050000;
  expected_evaluation.lower_bound.node_cost = 1683;
  expected_evaluation.lower_bound.edge_cost = 6000;
  expected_evaluation.lower_bound.max_usage = 1000000;
  expected_evaluation.total_departures = 3.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorForProblemTest, ViolatesInfiniteCostForEdge) {
  iopddl::Problem problem = DefaultProblem();
  const AutoShardingSolverParams params = DefaultParams();
  problem.edges[0].strategies[2].cost = kInfinityInt;
  const std::vector<NodeStrategyIdx> s_val = {0, 1, 2, 2, 1};
  const double objective_value = 1e+20;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation = Evaluate(problem, output, params);

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.violation_codes = {kInfiniteCostViolationCode};
  expected_evaluation.total.node_cost = 1716;  // 1560+156
  expected_evaluation.total.edge_cost = 1000000000000006200;
  expected_evaluation.total.max_usage = 1050000;
  expected_evaluation.lower_bound.node_cost = 1650;
  expected_evaluation.lower_bound.edge_cost = 6000;
  expected_evaluation.lower_bound.max_usage = 1000000;
  expected_evaluation.total_departures = 3.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(AutoShardingEvaluatorForProblemTest, ViolatesMaxDepartures) {
  const iopddl::Problem problem = DefaultProblem();
  AutoShardingSolverParams params = DefaultParams();
  params.max_departures = 2.0;
  const std::vector<NodeStrategyIdx> s_val = {3, 1, 2, 2, 1};
  const double objective_value = 12149.0;
  const AutoShardingSolverOutput output = {s_val, objective_value};

  const AutoShardingEvaluation evaluation = Evaluate(problem, output, params);

  AutoShardingEvaluation expected_evaluation;
  expected_evaluation.violation_codes = {kMaxDeparturesViolationCode};
  expected_evaluation.total.node_cost = 1749;   // 1590+159
  expected_evaluation.total.edge_cost = 10400;  // 4200+6200
  expected_evaluation.total.max_usage = 1080000;
  expected_evaluation.lower_bound.node_cost = 1650;
  expected_evaluation.lower_bound.edge_cost = 6000;
  expected_evaluation.lower_bound.max_usage = 1000000;
  expected_evaluation.total_departures = 3.0;
  EXPECT_EQ(evaluation, expected_evaluation);
}

TEST(MinimumMemoryBudgetRequiredTest, HandlesLiveMatrix) {
  const AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  EXPECT_EQ(MinimumMemoryBudgetRequired(request), 1000000.0);
}

TEST(DISABLED_MinimumMemoryBudgetRequiredTest,
     HandlesReducedIntervalsAndGroups) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  const std::vector<std::pair<int64_t, int64_t>> node_intervals =
      {{5, -1}, {5, -1}, {2, 3}, {3, 4}, {100, -1}, {0, 4}};
  const std::vector<std::vector<int64_t>> node_groups = {{0, 1}};
  request.clear_node_intervals();
  request.clear_edge_intervals();
  AddIntervals(request.mutable_node_intervals(), node_intervals);
  AddGroups(request.mutable_node_groups(), node_groups);
  EXPECT_EQ(MinimumMemoryBudgetRequired(request), 1000000.0);
}

TEST(StableMapTest, IterationOrderDeterminism) {
  StableMap<int, int> map;
  std::vector<int> insertion_order = {6, 3, 1, 2, 4, 5, 10, 0, 7, 9, 8};
  for (int key : insertion_order) {
    map[key] = key;
  }

  std::vector<int> iteration_order;
  for (const auto& [key, value] : map) {
    iteration_order.push_back(key);
  }
  EXPECT_THAT(iteration_order,
              ::testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
}

TEST(ComputeShardingCostTest, HandlesNegativeViolationCodes) {
  AutoShardingSolverRequest request = DefaultAutoShardingSolverRequest();
  // Require that all nodes are active for one time step.
  const std::vector<NodeIdx> live = {0, 1, 2, 3, 4};
  request.mutable_live()->Add()->mutable_nodes()->Add(live.begin(), live.end());
  const std::vector<NodeStrategyIdx> s_val = {0, 0, 0, 0, 0};

  EXPECT_EQ(ComputeShardingCost(request, s_val), 7650.0);

  request.set_memory_budget(0);
  EXPECT_EQ(ComputeShardingCost(request, s_val), -1 * kMemoryViolationCode);
  EXPECT_EQ(ComputeShardingCost(request, s_val,
                                /*use_negative_violation_codes=*/false),
            7650.0);
}

TEST(ValidateRequestTest, AcceptsAutoShardingSolverRequest) {
  CHECK_OK(ValidateRequest(DefaultAutoShardingSolverRequest()));
}

}  // namespace
}  // namespace spmd
}  // namespace xla
