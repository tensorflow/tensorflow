/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/while_loop_fusible_sinking.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/hlo/utils/hlo_matchers.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;
using ::testing::_;
using WhileLoopFusibleSinkingTest = HloHardwareIndependentTestBase;

TEST_F(WhileLoopFusibleSinkingTest, SinkOneFusible) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[2],f32[2]) parameter(0)
  p_body.0 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=0
  p_body.1 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=1

  add.0 = f32[2] add(p_body.0, p_body.1)
  ROOT root = (f32[2],f32[2]) tuple(add.0, p_body.1)
}

condition {
  p_cond = (f32[2],f32[2]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[2] parameter(0)
  const_1 = f32[2] iota(), iota_dimension=0
  while_init = (f32[2],f32[2]) tuple(const_0, const_1)
  ROOT while = (f32[2],f32[2]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopFusibleSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_body = module->GetComputationWithName("body");
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::Add(_, op::Iota()), _));
}

TEST_F(WhileLoopFusibleSinkingTest, SinkMask) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[5,7],f32[5,7]) parameter(0)
  p_body.0 = get-tuple-element(p_body), index=0
  p_body.1 = get-tuple-element(p_body), index=1

  add.0 = add(p_body.0, p_body.1)
  ROOT root = tuple(add.0, p_body.1)
}

condition {
  p_cond = (f32[5,7],f32[5,7]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[5,7] parameter(0)
  p = f32[5] parameter(1)
  a = f32[5,7] iota(), iota_dimension=0
  b = f32[5,7] iota(), iota_dimension=1
  c = add(a, b)
  d = f32[5,7] broadcast(p), dimensions={0}
  mask = multiply(c,d)
  while_init = tuple(const_0, mask)
  ROOT while = while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopFusibleSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_body = module->GetComputationWithName("body");
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::Add(_, op::Multiply(op::Add(op::Iota(), op::Iota()),
                                                op::Broadcast())),
                        _, _));
}

TEST_F(WhileLoopFusibleSinkingTest, NoSinkSlicedMask) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[5,7],f32[5,7]) parameter(0)
  p_body.0 = get-tuple-element(p_body), index=0
  p_body.1 = get-tuple-element(p_body), index=1
  z = s32[] constant(0)
  j = s32[] constant(3)
  ds = f32[1,7] dynamic-slice(p_body.1, j, z), dynamic_slice_sizes={1,7}
  r = f32[7] reshape(ds)
  b = f32[5,7] broadcast(r), dimensions={1}
  a = add(b, p_body.0)
  add.0 = add(a, p_body.1)
  ROOT root = tuple(add.0, p_body.1)
}

condition {
  p_cond = (f32[5,7],f32[5,7]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[5,7] parameter(0)
  p = f32[5] parameter(1)
  a = f32[5,7] iota(), iota_dimension=0
  b = f32[5,7] iota(), iota_dimension=1
  c = add(a, b)
  d = f32[5,7] broadcast(p), dimensions={0}
  mask = multiply(c,d)
  while_init = tuple(const_0, mask)
  ROOT while = while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopFusibleSinking{}.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopFusibleSinkingTest, TestPlumbSingleBroadcast) {
  const std::string hlo_string_before = R"(
  HloModule test

  loop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[4,3,5]{2,1,0} get-tuple-element(loop_var.1), index=2
    bitcast.12855 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} bitcast(get-tuple-element.3)
    add.40974 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.2, bitcast.12855)
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(idx, add.40974, get-tuple-element.3)
  }

  loop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.3 = s32[]{:T(128)} get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant(4)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }

  ENTRY %main {
    param.1 = s32[4,3,5]{2,1,0} iota(), iota_dimension=0
    zero = s32[]{:T(128)} constant(0)
    zeros32 = s32[]{:T(128)} constant(0)
    broadcast = s32[1,1,1,4,3,5]{5,4,3,2,1,0} broadcast(zeros32)
    input = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(zero, broadcast, param.1)
    ROOT while = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) while(input), condition=loop.condition, body=loop.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_before,
                          ParseAndReturnVerifiedModule(hlo_string_before));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopFusibleSinking{}.Run(module_before.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module_before.get(), "while"),
              op::While(op::Tuple(_, op::CustomCall(), _, _)));
}

TEST_F(WhileLoopFusibleSinkingTest, TestDontSinkBroadcast) {
  const std::string hlo_string_before = R"(
  HloModule test

  loop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[4,3,5]{2,1,0} get-tuple-element(loop_var.1), index=2
    bitcast.12855 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} bitcast(get-tuple-element.3)
    add.40974 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.2, bitcast.12855)
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(idx, add.40974, get-tuple-element.3)
  }

  loop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.3 = s32[]{:T(128)} get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant(4)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }

  ENTRY %main {
    param.1 = s32[4,3,5]{2,1,0} parameter(0)
    zero = s32[]{:T(128)} constant(0)
    zeros32 = s32[]{:T(128)} constant(0)
    broadcast = s32[1,1,1,4,3,5]{5,4,3,2,1,0} broadcast(zeros32)
    input = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(zero, broadcast, param.1)
    ROOT while = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) while(input), condition=loop.condition, body=loop.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_before,
                          ParseAndReturnVerifiedModule(hlo_string_before));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      WhileLoopFusibleSinking(/*sink_broadcast_of_constant=*/false)
          .Run(module_before.get()));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopFusibleSinkingTest,
       TestPlumbSingleBroadcastNotFlattenCallGraph) {
  const std::string hlo_string_before = R"(
  HloModule test

  loop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[4,3,5]{2,1,0} get-tuple-element(loop_var.1), index=2
    bitcast.12855 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} bitcast(get-tuple-element.3)
    add.40974 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.2, bitcast.12855)
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(idx, add.40974, get-tuple-element.3)
  }

  loop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.3 = s32[]{:T(128)} get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant(4)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }

  ENTRY %main {
    param.1 = s32[4,3,5]{2,1,0} iota(), iota_dimension=0
    zero = s32[]{:T(128)} constant(0)
    zeros32 = s32[]{:T(128)} constant(0)
    broadcast = s32[1,1,1,4,3,5]{5,4,3,2,1,0} broadcast(zeros32)
    input = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(zero, broadcast, param.1)
    while1 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) while(input), condition=loop.condition, body=loop.body
    input2 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(zero, broadcast, param.1)
    ROOT while2 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) while(input2), condition=loop.condition, body=loop.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_before,
                          ParseAndReturnVerifiedModule(hlo_string_before));
  CHECK_OK(FlattenCallGraph{}.Run(module_before.get()).status());
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopFusibleSinking{}.Run(module_before.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module_before.get(), "while1"),
              op::While(op::Tuple(_, op::CustomCall(), _, _)));
  EXPECT_THAT(FindInstruction(module_before.get(), "while2"),
              op::While(op::Tuple(_, op::CustomCall(), _, _)));
}

TEST_F(WhileLoopFusibleSinkingTest, TestPlumbMultipleBroadcast) {
  const std::string hlo_string_before = R"(
  HloModule test

  loop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.4 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=2
    get-tuple-element.3 = s32[4,3,5]{2,1,0} get-tuple-element(loop_var.1), index=3
    bitcast.12855 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} bitcast(get-tuple-element.3)
    add.40974 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.2, bitcast.12855)
    add.1 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.4, add.40974)
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(idx, add.40974, add.1, get-tuple-element.3)
  }

  loop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.3 = s32[]{:T(128)} get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant(4)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }

  ENTRY %main {
    param.1 = s32[4,3,5]{2,1,0} iota(), iota_dimension=0
    zero = s32[]{:T(128)} constant(0)
    zeros32 = s32[]{:T(128)} constant(0)
    broadcast = s32[1,1,1,4,3,5]{5,4,3,2,1,0} broadcast(zeros32)
    input = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(zero, broadcast, broadcast, param.1)
    ROOT while = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) while(input), condition=loop.condition, body=loop.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_before,
                          ParseAndReturnVerifiedModule(hlo_string_before));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopFusibleSinking{}.Run(module_before.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      FindInstruction(module_before.get(), "while"),
      op::While(op::Tuple(_, op::CustomCall(), op::CustomCall(), _, _)));
}

TEST_F(WhileLoopFusibleSinkingTest, TestNoPlumbWithBadCondition) {
  const std::string hlo_string_before = R"(
  HloModule test
  tmp {
    x = s32[] parameter(0)
    y = s32[] parameter(1)
    ROOT add = s32[] add(x, y)
  }

  loop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.4 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=2
    get-tuple-element.3 = s32[4,3,5]{2,1,0} get-tuple-element(loop_var.1), index=3
    bitcast.12855 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} bitcast(get-tuple-element.3)
    add.40974 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.2, bitcast.12855)
    add.1 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.4, add.40974)
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(idx, add.40974, add.1, get-tuple-element.3)
  }

  loop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.3 = s32[]{:T(128)} get-tuple-element(loop_var.2), index=0
    get-tuple-element.4 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.2), index=1
    z = s32[]{:T(128)} constant(0)
    r = s32[]{:T(128)} reduce(get-tuple-element.4, z), dimensions={0,1,2,3,4,5}, to_apply=tmp
    ROOT less-than = pred[] compare(get-tuple-element.3, r), direction=LT
  }

  ENTRY %main {
    param.1 = s32[4,3,5]{2,1,0} parameter(0)
    zero = s32[]{:T(128)} constant(0)
    zeros32 = s32[]{:T(128)} constant(0)
    broadcast = s32[1,1,1,4,3,5]{5,4,3,2,1,0} broadcast(zeros32)
    input = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(zero, broadcast, broadcast, param.1)
    ROOT while = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) while(input), condition=loop.condition, body=loop.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_before,
                          ParseAndReturnVerifiedModule(hlo_string_before));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopFusibleSinking{}.Run(module_before.get()));
  EXPECT_FALSE(changed);
}

TEST_F(WhileLoopFusibleSinkingTest, TestNoPlumbWithUnknonwnTripCount) {
  const std::string hlo_string_before = R"(
  HloModule test
  tmp {
    x = s32[] parameter(0)
    y = s32[] parameter(1)
    ROOT add = s32[] add(x, y)
  }

  loop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.4 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=2
    get-tuple-element.3 = s32[4,3,5]{2,1,0} get-tuple-element(loop_var.1), index=3
    bitcast.12855 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} bitcast(get-tuple-element.3)
    add.40974 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.2, bitcast.12855)
    add.1 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.4, add.40974)
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(idx, add.40974, add.1, get-tuple-element.3)
  }

  loop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) parameter(0)
    get-tuple-element.3 = s32[]{:T(128)} get-tuple-element(loop_var.2), index=0
    m = s32[] constant(0)
    v = s32[] constant(10000)
    rng = s32[] rng(m, v), distribution=rng_uniform
    ROOT less-than = pred[] compare(get-tuple-element.3, rng), direction=LT
  }

  ENTRY %main {
    param.1 = s32[4,3,5]{2,1,0} parameter(0)
    zero = s32[]{:T(128)} constant(0)
    zeros32 = s32[]{:T(128)} constant(0)
    broadcast = s32[1,1,1,4,3,5]{5,4,3,2,1,0} broadcast(zeros32)
    input = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) tuple(zero, broadcast, broadcast, param.1)
    ROOT while = (s32[]{:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}, s32[4,3,5]{2,1,0}) while(input), condition=loop.condition, body=loop.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_before,
                          ParseAndReturnVerifiedModule(hlo_string_before));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopFusibleSinking{}.Run(module_before.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla
