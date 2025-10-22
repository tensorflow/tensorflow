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

#include "xla/service/conditional_sinking.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;
using ConditionalSinkingTest = HloHardwareIndependentTestBase;

TEST_F(ConditionalSinkingTest, HoistDUS) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

on_true {
  p_body = (f32[2],f32[2]) parameter(0)
  p_body.0 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=0
  p_body.1 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=1

  add.0 = f32[2] add(p_body.0, p_body.1)
  mul.0 = f32[2] multiply(p_body.0, p_body.1)
  ROOT root = (f32[2],f32[2]) tuple(add.0, mul.0)
}

on_false {
  p_body_false = () parameter(0)
  c = f32[] constant(0)
  add.0 = f32[2] broadcast(c), dimensions={}
  ROOT root = (f32[2],f32[2]) tuple(add.0, add.0)
}


ENTRY entry {
  p_cond = pred[] parameter(0)
  const_0 = f32[2] parameter(1)
  const_1 = f32[2] iota(), iota_dimension=0
  p2 = f32[100] parameter(2)
  p3 = f32[100] parameter(3)
  index = s32[] parameter(4)
  init_true = (f32[2],f32[2]) tuple(const_0, const_1)
  init_false = () tuple()
  c = (f32[2],f32[2]) conditional(p_cond, init_true, init_false), true_computation=on_true, false_computation=on_false
  r0 = get-tuple-element(c), index=0
  r1 = get-tuple-element(c), index=1
  d0 = f32[100] dynamic-update-slice(p2, r0, index)
  d1 = f32[100] dynamic-update-slice(p3, r1, index)
  ROOT root = tuple(d0, d1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, ConditionalSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(
          op::GetTupleElement(op::Conditional(
              op::Parameter(0),
              op::Tuple(op::Parameter(1), op::Iota(), op::Parameter(2),
                        op::Parameter(4), op::Parameter(3)),
              op::Tuple(op::Parameter(2), op::Parameter(4), op::Parameter(3)))),
          op::GetTupleElement(op::Conditional())));
}

TEST_F(ConditionalSinkingTest, HoistDUSChain) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

on_true {
  p_body = (f32[2],f32[2]) parameter(0)
  p_body.0 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=0
  p_body.1 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=1

  add.0 = f32[2] add(p_body.0, p_body.1)
  mul.0 = f32[2] multiply(p_body.0, p_body.1)
  ROOT root = (f32[2],f32[2]) tuple(add.0, mul.0)
}

on_false {
  p_body_false = () parameter(0)
  c = f32[] constant(0)
  add.0 = f32[2] broadcast(c), dimensions={}
  ROOT root = (f32[2],f32[2]) tuple(add.0, add.0)
}


ENTRY entry {
  p_cond = pred[] parameter(0)
  const_0 = f32[2] parameter(1)
  const_1 = f32[2] iota(), iota_dimension=0
  p2 = f32[100,2] parameter(2)
  p3 = f32[100,2] parameter(3)
  index = s32[] parameter(4)
  init_true = (f32[2],f32[2]) tuple(const_0, const_1)
  init_false = () tuple()
  c = (f32[2],f32[2]) conditional(p_cond, init_true, init_false), true_computation=on_true, false_computation=on_false
  r0 = get-tuple-element(c), index=0
  b0 = f32[1,2] bitcast(r0)
  r1 = get-tuple-element(c), index=1
  b1 = f32[1,2] bitcast(r1)
  iz = s32[] constant(0)
  d0 = f32[100,2] dynamic-update-slice(p2, b0, index, iz)
  d1 = f32[100,2] dynamic-update-slice(p3, b1, index, iz)
  ROOT root = tuple(d0, d1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, ConditionalSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(
          op::GetTupleElement(op::Conditional(
              op::Parameter(0),
              op::Tuple(op::Parameter(1), op::Iota(), op::Parameter(2),
                        op::Parameter(4), op::Parameter(3)),
              op::Tuple(op::Parameter(2), op::Parameter(4), op::Parameter(3)))),
          op::GetTupleElement(op::Conditional())));
}

TEST_F(ConditionalSinkingTest, SinkDS) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

on_true {
  p_body = (f32[2],f32[2]) parameter(0)
  p_body.0 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=0
  p_body.1 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=1

  add.0 = f32[2] add(p_body.0, p_body.1)
  mul.0 = f32[2] multiply(p_body.0, p_body.1)
  ROOT root = (f32[2],f32[2]) tuple(add.0, mul.0)
}

on_false {
  p_body_false = () parameter(0)
  c = f32[] constant(0)
  add.0 = f32[2] broadcast(c), dimensions={}
  ROOT root = (f32[2],f32[2]) tuple(add.0, add.0)
}


ENTRY entry {
  p_cond = pred[] parameter(0)
  p2 = f32[100] parameter(1)
  p3 = f32[100] parameter(2)
  index = s32[] parameter(3)

  const_0 = f32[2] dynamic-slice(p2, index), dynamic_slice_sizes={2}
  const_1 = f32[2] dynamic-slice(p3, index), dynamic_slice_sizes={2}
  init_true = (f32[2],f32[2]) tuple(const_0, const_1)
  init_false = () tuple()
  c = (f32[2],f32[2]) conditional(p_cond, init_true, init_false), true_computation=on_true, false_computation=on_false
  r0 = get-tuple-element(c), index=0
  r1 = get-tuple-element(c), index=1
  ROOT root = tuple(r0, r1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, ConditionalSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(
          op::GetTupleElement(op::Conditional(
              op::Parameter(0),
              op::Tuple(op::Parameter(1), op::Parameter(2), op::Parameter(3)),
              op::Tuple())),
          op::GetTupleElement(op::Conditional())));
}

TEST_F(ConditionalSinkingTest, SinkDSChain) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

on_true {
  p_body = (f32[2],f32[2]) parameter(0)
  p_body.0 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=0
  p_body.1 = f32[2] get-tuple-element((f32[2],f32[2]) p_body), index=1

  add.0 = f32[2] add(p_body.0, p_body.1)
  mul.0 = f32[2] multiply(p_body.0, p_body.1)
  ROOT root = (f32[2],f32[2]) tuple(add.0, mul.0)
}

on_false {
  p_body_false = () parameter(0)
  c = f32[] constant(0)
  add.0 = f32[2] broadcast(c), dimensions={}
  ROOT root = (f32[2],f32[2]) tuple(add.0, add.0)
}


ENTRY entry {
  p_cond = pred[] parameter(0)
  p2 = f32[100,2,1] parameter(1)
  p3 = f32[100,2,1] parameter(2)
  index = s32[] parameter(3)
  iz = s32[] constant(0)

  ds0 = f32[1,2,1] dynamic-slice(p2, iz, index, iz), dynamic_slice_sizes={1,2,1}
  ds1 = f32[1,2,1] dynamic-slice(p3, iz, index, iz), dynamic_slice_sizes={1,2,1}
  b0 = f32[2] bitcast(ds0)
  b1 = f32[2] bitcast(ds1)
  init_true = (f32[2],f32[2]) tuple(b0, b1)
  init_false = () tuple()
  c = (f32[2],f32[2]) conditional(p_cond, init_true, init_false), true_computation=on_true, false_computation=on_false
  r0 = get-tuple-element(c), index=0
  r1 = get-tuple-element(c), index=1
  ROOT root = tuple(r0, r1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, ConditionalSinking{}.Run(module.get()));
  ASSERT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(
          op::GetTupleElement(op::Conditional(
              op::Parameter(0),
              op::Tuple(op::Parameter(1), op::Parameter(2), op::Parameter(3)),
              op::Tuple())),
          op::GetTupleElement(op::Conditional())));
}
}  // namespace
}  // namespace xla
