/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/all_reduce_simplifier.h"

#include <memory>
#include <utility>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/types.h"
#include "xla/window_util.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace m = match;

using AllReduceSimplifierTest = HloTestBase;

TEST_F(AllReduceSimplifierTest, ReplicatedParameters) {
  const char* kModuleStr = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

max {
  a.1 = f32[] parameter(0)
  b.1 = f32[] parameter(1)
  ROOT max = f32[] maximum(a.1, b.1)
}

min {
  a.2 = f32[] parameter(0)
  b.2 = f32[] parameter(1)
  ROOT min = f32[] minimum(a.2, b.2)
}

sum.1 {
  a.3 = f32[] parameter(0)
  b.3 = f32[] parameter(1)
  ROOT add.1 = f32[] add(a.3, b.3)
}

test {
  p0 = f32[8,16] parameter(0), parameter_replication={true}
  p1 = f32[8,16] parameter(1), parameter_replication={false}
  p2 = f32[] parameter(2), parameter_replication={true}
  all-reduce = f32[8,16] all-reduce(p0), replica_groups={}, to_apply=sum
  all-reduce.1 = f32[8,16] all-reduce(p0), replica_groups={}, to_apply=max
  all-reduce.2 = f32[8,16] all-reduce(p1), replica_groups={}, to_apply=min
  all-reduce.3 = f32[] all-reduce(p2), replica_groups={}, to_apply=sum.1
  ROOT tuple = (f32[8,16], f32[8,16], f32[8,16], f32[]) tuple(all-reduce, all-reduce.1, all-reduce.2, all-reduce.3)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleStr, /*replica_count=*/8));
  AllReduceSimplifier simplifier(/*replica_count=*/8);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::MultiplyAnyOrder(m::Parameter(0),
                              m::Broadcast(m::Convert(m::ConstantScalar(8)))),
          m::Parameter(0), m::AllReduce(m::Parameter(1)),
          m::MultiplyAnyOrder(m::Parameter(2),
                              m::Convert(m::ConstantScalar(8))))));
}

TEST_F(AllReduceSimplifierTest, AllReduceAfterAllReduce) {
  const char* kModuleStr = R"(
HloModule m

max {
  a.1 = f32[] parameter(0)
  b.1 = f32[] parameter(1)
  ROOT max = f32[] maximum(a.1, b.1)
}

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

test {
  p0 = f32[8,16] parameter(0), parameter_replication={false}
  all-reduce = f32[8,16] all-reduce(p0), replica_groups={}, to_apply=max
  ROOT all-reduce.1 = f32[8,16] all-reduce(all-reduce), replica_groups={}, to_apply=sum
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleStr, /*replica_count=*/8));
  AllReduceSimplifier simplifier(/*replica_count=*/8);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::MultiplyAnyOrder(
                  m::AllReduce(m::Parameter(0)),
                  m::Broadcast(m::Convert(m::ConstantScalar(8))))));
}

TEST_F(AllReduceSimplifierTest, SubgroupAllReduce) {
  const char* kModuleStr = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

max {
  a.1 = f32[] parameter(0)
  b.1 = f32[] parameter(1)
  ROOT max = f32[] maximum(a.1, b.1)
}

min {
  a.2 = f32[] parameter(0)
  b.2 = f32[] parameter(1)
  ROOT min = f32[] minimum(a.2, b.2)
}

test {
  p0 = f32[8,16] parameter(0), parameter_replication={true}
  p1 = f32[8,16] parameter(1), parameter_replication={false}
  all-reduce = f32[8,16] all-reduce(p0), replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=sum
  all-reduce.1 = f32[8,16] all-reduce(p0), replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=max
  all-reduce.2 = f32[8,16] all-reduce(p1), replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=min
  ROOT tuple = (f32[8,16], f32[8,16], f32[8,16]) tuple(all-reduce, all-reduce.1, all-reduce.2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleStr, /*replica_count=*/8));
  AllReduceSimplifier simplifier(/*replica_count=*/8);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::MultiplyAnyOrder(m::Parameter(0),
                              m::Broadcast(m::Convert(m::ConstantScalar(4)))),
          m::Parameter(0), m::AllReduce(m::Parameter(1)))));
}

TEST_F(AllReduceSimplifierTest, TrivialSubgroupAllReduce) {
  const char* kModuleStr = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}


test {
  p0 = f32[8,16] parameter(0), parameter_replication={false}
  ROOT all-reduce = f32[8,16] all-reduce(p0),
    replica_groups={{0},{1},{2},{3},{4},{5},{6},{7}},
    to_apply=sum
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleStr, /*replica_count=*/8));
  AllReduceSimplifier simplifier(/*replica_count=*/8);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}
}  // namespace
}  // namespace xla
