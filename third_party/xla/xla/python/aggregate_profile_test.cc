/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/python/aggregate_profile.h"

#include <map>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tsl/platform/test.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace xla {
namespace {

using tensorflow::profiler::ProfiledInstructionsProto;

TEST(AggregateProfiledInstructionsProtoTest, aggregateAndGetPercentile) {
  tensorflow::profiler::ProfiledInstructionsProto profile_a;
  {
    auto *cost_a = profile_a.add_costs();
    cost_a->set_cost_us(10);
    cost_a->set_name("reduce");
  }
  {
    auto *cost_a = profile_a.add_costs();
    cost_a->set_cost_us(30);
    cost_a->set_name("copy");
  }

  tensorflow::profiler::ProfiledInstructionsProto profile_c;
  {
    auto *cost_c = profile_c.add_costs();
    cost_c->set_cost_us(30);
    cost_c->set_name("reduce");
  }

  std::vector<tensorflow::profiler::ProfiledInstructionsProto> profiles = {
      profile_a, profile_c};

  std::vector<int> custom_call_costs = {0,  10, 20, 30, 40, 50,
                                        60, 70, 80, 90, 100};
  for (int cost : custom_call_costs) {
    tensorflow::profiler::ProfiledInstructionsProto profile_custom_call;
    {
      auto *cost_c = profile_custom_call.add_costs();
      cost_c->set_cost_us(cost);
      cost_c->set_name("custom-call");
    }

    profiles.push_back(profile_custom_call);
  }
  tensorflow::profiler::ProfiledInstructionsProto result_90th;
  AggregateProfiledInstructionsProto(
      absl::Span<const tensorflow::profiler::ProfiledInstructionsProto>(
          profiles.data(), profiles.size()),
      90, &result_90th);

  EXPECT_EQ(result_90th.costs_size(), 3);
  std::map<std::string, float> costs;
  for (const auto &cost : result_90th.costs()) {
    costs[cost.name()] = cost.cost_us();
  }
  EXPECT_EQ(costs["copy"], 30);
  EXPECT_EQ(costs["custom-call"], 90);
  EXPECT_EQ(costs["reduce"], 10);

  tensorflow::profiler::ProfiledInstructionsProto result_10th;
  AggregateProfiledInstructionsProto(
      absl::Span<const tensorflow::profiler::ProfiledInstructionsProto>(
          profiles.data(), profiles.size()),
      10, &result_10th);

  EXPECT_EQ(result_10th.costs_size(), 3);
  for (const auto &cost : result_10th.costs()) {
    costs[cost.name()] = cost.cost_us();
  }
  EXPECT_EQ(costs["copy"], 30);
  EXPECT_EQ(costs["custom-call"], 10);
  EXPECT_EQ(costs["reduce"], 10);
}

TEST(AggregateProfiledInstructionsProtoTest, getIncorrectPercentile) {
  tensorflow::profiler::ProfiledInstructionsProto profile_a;
  {
    auto *cost_a = profile_a.add_costs();
    cost_a->set_cost_us(10);
    cost_a->set_name("reduce");
  }

  std::vector<tensorflow::profiler::ProfiledInstructionsProto> profiles = {
      profile_a};
  tensorflow::profiler::ProfiledInstructionsProto result;
  AggregateProfiledInstructionsProto(
      absl::Span<const tensorflow::profiler::ProfiledInstructionsProto>(
          profiles.data(), profiles.size()),
      -1, &result);
  EXPECT_EQ(result.costs_size(), 0);
  AggregateProfiledInstructionsProto(
      absl::Span<const tensorflow::profiler::ProfiledInstructionsProto>(
          profiles.data(), profiles.size()),
      101, &result);
  EXPECT_EQ(result.costs_size(), 0);

  AggregateProfiledInstructionsProto(
      absl::Span<const tensorflow::profiler::ProfiledInstructionsProto>(
          profiles.data(), profiles.size()),
      100, &result);
  EXPECT_EQ(result.costs_size(), 1);
}
}  // namespace
}  // namespace xla
