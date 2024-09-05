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

#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;
using ::testing::_;
using WhileLoopFusibleSinkingTest = HloTestBase;

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

}  // namespace
}  // namespace xla
