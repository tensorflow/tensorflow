/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tools/parser/hlo_parser.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;
using ::testing::_;

class WhileLoopConstantSinkingTest : public ::testing::Test {};

TEST_F(WhileLoopConstantSinkingTest, SinkOneConstant) {
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
  const_0 = f32[2] constant({1, 2})
  const_1 = f32[2] constant({2, 1})
  while_init = (f32[2],f32[2]) tuple(const_0, const_1)
  ROOT while = (f32[2],f32[2]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          tools::Parse(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConstantSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_body = module->GetComputationWithName("body");
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::Add(_, op::Constant()), _));
}

TEST_F(WhileLoopConstantSinkingTest, KeepConstantsLoopInvariant) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_body = (f32[2],f32[2],f32[2]) parameter(0)
  p_body.0 = f32[2] get-tuple-element((f32[2],f32[2],f32[2]) p_body), index=0
  p_body.1 = f32[2] get-tuple-element((f32[2],f32[2],f32[2]) p_body), index=1
  p_body.2 = f32[2] get-tuple-element((f32[2],f32[2],f32[2]) p_body), index=2

  add.0 = f32[2] add(p_body.1, p_body.2)
  ROOT root = (f32[2],f32[2],f32[2]) tuple(add.0, p_body.1, p_body.2)
}

condition {
  p_cond = (f32[2],f32[2],f32[2]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[2] constant({1, 2})
  const_1 = f32[2] constant({2, 1})
  const_2 = f32[2] constant({3, 1})
  while_init = (f32[2],f32[2],f32[2]) tuple(const_0, const_1, const_2)
  ROOT while = (f32[2],f32[2],f32[2]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          tools::Parse(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConstantSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_body = module->GetComputationWithName("body");
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::Add(op::Constant(), op::Constant()),
                        op::GetTupleElement(op::Parameter(0)),
                        op::GetTupleElement(op::Parameter(0))));
}

TEST_F(WhileLoopConstantSinkingTest, TupleShapedConstants) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_b = (f32[2],(f32[2],f32[2])) parameter(0)
  p_b.0 = f32[2] get-tuple-element((f32[2],f32[2],f32[2]) p_b), index=0
  p_b.1 = (f32[2],f32[2]) get-tuple-element((f32[2],(f32[2],f32[2])) p_b), index=1

  p_b.1.1 = f32[2] get-tuple-element(p_b.1), index=0

  ROOT root = (f32[2],f32[2],f32[2]) tuple(p_b.1.1, p_b.1)
}

condition {
  p_cond = (f32[2],(f32[2],f32[2])) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[2] constant({1, 2})
  const_1 = (f32[2], f32[2]) constant((f32[2], f32[2]) ({2, 1},{3,1}))
  while_init = (f32[2],(f32[2],f32[2])) tuple(const_0, const_1)
  ROOT while = (f32[2],(f32[2],f32[2])) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          tools::Parse(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConstantSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_body = module->GetComputationWithName("body");
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Constant(), 0),
                        op::GetTupleElement(op::Parameter(0))));
}

TEST_F(WhileLoopConstantSinkingTest, DuplicateGTEs) {
  // This test shows that the pass fails to optimize non-canonical IR.
  //
  // Even though the input IR has a constant value for p_b.2.dup,
  // WhileLoopConstantSinking doesn't try to detect this.  Instead, it relies on
  // prior runs of HLO CSE to have commoned these identical GTE instructions.

  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p_b = (f32[2],f32[2],f32[2]) parameter(0)

  p_b.1     = f32[2] get-tuple-element((f32[2],f32[2],f32[2]) p_b), index=1
  p_b.2     = f32[2] get-tuple-element((f32[2],f32[2],f32[2]) p_b), index=2
  p_b.2.dup = f32[2] get-tuple-element((f32[2],f32[2],f32[2]) p_b), index=2

  add.0 = f32[2] add(p_b.1, p_b.2.dup)
  ROOT root = (f32[2],f32[2],f32[2]) tuple(add.0, p_b.1, p_b.2)
}

condition {
  p_cond = (f32[2],f32[2],f32[2]) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const_0 = f32[2] constant({1, 2})
  const_1 = f32[2] constant({2, 1})
  const_2 = f32[2] constant({3, 1})
  while_init = (f32[2],f32[2],f32[2]) tuple(const_0, const_1, const_2)
  ROOT while = (f32[2],f32[2],f32[2]) while(while_init), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          tools::Parse(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          WhileLoopConstantSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);

  auto* while_body = module->GetComputationWithName("body");
  EXPECT_THAT(while_body->root_instruction(),
              op::Tuple(op::Add(op::Constant(), ::testing::Not(op::Constant())),
                        op::GetTupleElement(op::Parameter(0)),
                        op::GetTupleElement(op::Parameter(0))));
}
}  // namespace
}  // namespace xla
