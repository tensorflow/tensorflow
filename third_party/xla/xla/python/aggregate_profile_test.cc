/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

namespace xla {
namespace {

TEST(AggregateProfiledInstructionsProtoTest, aggregateAndGetAverage) {
  tensorflow::profiler::ProfiledInstructionsProto profile_a;
  {
    auto *cost_a = profile_a.add_costs();
    cost_a->set_cost_us(20);
    cost_a->set_name("custom-call");
  }
  {
    auto *cost_a = profile_a.add_costs();
    cost_a->set_cost_us(20);
    cost_a->set_name("reduce");
  }
  {
    auto *cost_a = profile_a.add_costs();
    cost_a->set_cost_us(30);
    cost_a->set_name("copy");
  }

  tensorflow::profiler::ProfiledInstructionsProto profile_b;
  {
    auto *cost_b = profile_b.add_costs();
    cost_b->set_cost_us(20);
    cost_b->set_name("custom-call");
  }

  tensorflow::profiler::ProfiledInstructionsProto profile_c;
  {
    auto *cost_c = profile_c.add_costs();
    cost_c->set_cost_us(20);
    cost_c->set_name("custom-call");
  }
  {
    auto *cost_c = profile_c.add_costs();
    cost_c->set_cost_us(30);
    cost_c->set_name("reduce");
  }

  tensorflow::profiler::ProfiledInstructionsProto result;
  std::vector<tensorflow::profiler::ProfiledInstructionsProto> profiles = {
      profile_a, profile_b, profile_c};
  AggregateProfiledInstructionsProto(
      absl::Span<const tensorflow::profiler::ProfiledInstructionsProto>(
          profiles.data(), profiles.size()),
      &result);

  EXPECT_EQ(result.costs_size(), 3);
  std::map<std::string, float> costs;
  for (const auto &cost : result.costs()) {
    costs[cost.name()] = cost.cost_us();
  }
  EXPECT_EQ(costs["copy"], 30);
  EXPECT_EQ(costs["custom-call"], 20);
  EXPECT_EQ(costs["reduce"], 25);
}

}  // namespace
}  // namespace xla
