/*
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "xla/hlo/experimental/auto_sharding/iopddl.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "xla/hlo/experimental/auto_sharding/solver.h"
#include "xla/tsl/platform/status_matchers.h"

namespace iopddl {
namespace {

using ::tsl::testing::IsOkAndHolds;

Problem GetExampleProblem() {
  return {
      .name = "example",
      .nodes =
          {
              // Node 0
              {.interval = {30, 70}, .strategies = {{.cost = 15, .usage = 10}}},
              // Node 1
              {.interval = {40, 70},
               .strategies = {{.cost = 55, .usage = 25},
                              {.cost = 65, .usage = 25}}},
              // Node 2
              {.interval = {50, 120},
               .strategies = {{.cost = 25, .usage = 15},
                              {.cost = 45, .usage = 20},
                              {.cost = 35, .usage = 15}}},
              // Node 3
              {.interval = {110, 140},
               .strategies = {{.cost = 85, .usage = 10},
                              {.cost = 75, .usage = 10}}},
              // Node 4
              {.interval = {110, 150},
               .strategies = {{.cost = 95, .usage = 15}}},
          },
      .edges =
          {
              {.nodes = {0, 1}, .strategies = {{.cost = 30}, {.cost = 40}}},
              {.nodes = {0, 2},
               .strategies = {{.cost = 50}, {.cost = 10}, {.cost = 40}}},
              {.nodes = {1, 3},
               .strategies =
                   {{.cost = 90}, {.cost = 10}, {.cost = 20}, {.cost = 80}}},
              {.nodes = {2, 4},
               .strategies = {{.cost = 60}, {.cost = 20}, {.cost = 30}}},
              {.nodes = {3, 4}, .strategies = {{.cost = 70}, {.cost = 60}}},
          },
      .usage_limit = 50};
}

TEST(EvaluateTest, LegalSolution) {
  // Node costs: 15 + 65 + 35 + 85 + 95 = 295
  // Edge costs: 40 + 40 + 20 + 30 + 70 = 200
  EXPECT_THAT(Evaluate(GetExampleProblem(), {0, 1, 2, 0, 0}),
              IsOkAndHolds(495));
}

TEST(EvaluateTest, LegalSolutionNoUsageLimit) {
  Problem problem = GetExampleProblem();
  problem.usage_limit.reset();
  // Node costs: 15 + 55 + 45 + 75 + 95 = 285
  // Edge costs: 30 + 10 + 10 + 20 + 60 = 130
  EXPECT_THAT(Evaluate(problem, {0, 0, 1, 1, 0}), IsOkAndHolds(415));
}

TEST(EvaluateTest, IllegalSolutionEclipsesUsageLimit) {
  EXPECT_EQ(Evaluate(GetExampleProblem(), {0, 0, 1, 1, 0}).status().code(),
            absl::StatusCode::kResourceExhausted);
}

TEST(EvaluateTest, IllegalSolutionHasTooManyTerms) {
  EXPECT_EQ(Evaluate(GetExampleProblem(), {0, 0, 0, 0, 0, 0}).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(EvaluateTest, IllegalSolutionHasTooFewTerms) {
  EXPECT_EQ(Evaluate(GetExampleProblem(), {0, 0, 0, 0}).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(EvaluateTest, IllegalSolutionHasNegativeStrategyIndex) {
  EXPECT_EQ(Evaluate(GetExampleProblem(), {0, 0, -1, 0, 0}).status().code(),
            absl::StatusCode::kOutOfRange);
}

TEST(EvaluateTest, IllegalSolutionHasBogusStrategyIndex) {
  EXPECT_EQ(Evaluate(GetExampleProblem(), {0, 0, 4, 0, 0}).status().code(),
            absl::StatusCode::kOutOfRange);
}

TEST(SolveTest, FindsOptimalSolution) {
  EXPECT_THAT(Solver().Solve(GetExampleProblem(), absl::Seconds(1)),
              IsOkAndHolds(Solution{0, 0, 2, 1, 0}));
}

TEST(SolveTest, NoSolutionFound) {
  Problem problem = GetExampleProblem();
  problem.usage_limit = 0;
  EXPECT_EQ(Solver().Solve(problem, absl::Seconds(1)).status().code(),
            absl::StatusCode::kNotFound);
}

}  // namespace
}  // namespace iopddl
