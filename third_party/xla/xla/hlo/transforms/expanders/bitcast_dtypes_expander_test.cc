/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/bitcast_dtypes_expander.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class BitcastDtypesExpanderTest : public HloHardwareIndependentTestBase {};

TEST_F(BitcastDtypesExpanderTest, S32toS8) {
  absl::string_view hlo_string = R"(
HloModule bitcast_to_smaller

ENTRY main {
  p = s32[10] parameter(0)
  ROOT out = s8[10,4] bitcast-convert(p)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  BitcastDtypesExpander expander;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));

  EXPECT_TRUE(changed);
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
// CHECK: HloModule bitcast_to_smaller
// CHECK: %xla.bitcast_convert_s32_10__2_s8_10_4_.17 ([[VAL_0:a.*]]: s32[10]) -> s8[10,4] {
// CHECK:  %[[VAL_0]] = s32[10]{0} parameter(0)
// CHECK:  %[[VAL_1:.*]] = s32[10,1]{1,0} reshape(%[[VAL_0]])
// CHECK:  %[[VAL_2:.*]] = s32[10,1]{1,0} broadcast(%[[VAL_1]]), dimensions={0,1}
// CHECK:  %[[VAL_3:.*]] = s32[10]{0} reshape(%[[VAL_2]])
// CHECK:  %[[VAL_4:.*]] = s32[10,4]{1,0} broadcast(%[[VAL_3]]), dimensions={0}
// CHECK:  %[[VAL_5:.*]] = u32[10,4]{1,0} bitcast-convert(%[[VAL_4]])
// CHECK:  %[[VAL_6:.*]] = u32[] constant(8)
// CHECK:  %[[VAL_7:.*]] = u32[10,4]{1,0} broadcast(%[[VAL_6]]), dimensions={}
// CHECK:  %[[VAL_8:.*]] = u32[10,4]{1,0} iota(), iota_dimension=1
// CHECK:  %[[VAL_9:.*]] = u32[10,4]{1,0} multiply(%[[VAL_7]], %[[VAL_8]])
// CHECK:  %[[VAL_10:.*]] = u32[10,4]{1,0} shift-right-logical(%[[VAL_5]], %[[VAL_9]])
// CHECK:  %[[VAL_11:.*]] = u32[] constant(255)
// CHECK:  %[[VAL_12:.*]] = u32[10,4]{1,0} broadcast(%[[VAL_11]]), dimensions={}
// CHECK:  %[[VAL_13:.*]] = u32[10,4]{1,0} and(%[[VAL_10]], %[[VAL_12]])
// CHECK:  %[[VAL_14:.*]] = u8[10,4]{1,0} convert(%[[VAL_13]])
// CHECK:  ROOT %[[VAL_15:.*]] = s8[10,4]{1,0} bitcast-convert(%[[VAL_14]])
// CHECK: }
// CHECK: ENTRY %main (p: s32[10]) -> s8[10,4] {
// CHECK:  %[[VAL_16:.*]] = s32[10]{0} parameter(0)
// CHECK:  ROOT %[[VAL_17:.*]] = s8[10,4]{1,0} call(%[[VAL_16]]), to_apply=%[[VAL_18:.*]]
// CHECK: }
)"));
}

TEST_F(BitcastDtypesExpanderTest, S64toS32) {
  absl::string_view hlo_string = R"(
HloModule bitcast_to_smaller

ENTRY main {
  p = s64[10] parameter(0)
  ROOT out = s32[10,2] bitcast-convert(p)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  BitcastDtypesExpander expander;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));

  EXPECT_TRUE(changed);
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
// CHECK: HloModule bitcast_to_smaller, entry_computation_layout={(s64[10]{0})->s32[10,2]{1,0}}
// CHECK: %xla.bitcast_convert_s64_10__2_s32_10_2_.17 ([[VAL_0:a.*]]: s64[10]) -> s32[10,2] {
// CHECK:   %[[VAL_0]] = s64[10]{0} parameter(0)
// CHECK:   %[[VAL_1:.*]] = s64[10,1]{1,0} reshape(%[[VAL_0]])
// CHECK:   %[[VAL_2:.*]] = s64[10,1]{1,0} broadcast(%[[VAL_1]]), dimensions={0,1}
// CHECK:   %[[VAL_3:.*]] = s64[10]{0} reshape(%[[VAL_2]])
// CHECK:   %[[VAL_4:.*]] = s64[10,2]{1,0} broadcast(%[[VAL_3]]), dimensions={0}
// CHECK:   %[[VAL_5:.*]] = u64[10,2]{1,0} bitcast-convert(%[[VAL_4]])
// CHECK:   %[[VAL_6:.*]] = u64[] constant(32)
// CHECK:   %[[VAL_7:.*]] = u64[10,2]{1,0} broadcast(%[[VAL_6]]), dimensions={}
// CHECK:   %[[VAL_8:.*]] = u64[10,2]{1,0} iota(), iota_dimension=1
// CHECK:   %[[VAL_9:.*]] = u64[10,2]{1,0} multiply(%[[VAL_7]], %[[VAL_8]])
// CHECK:   %[[VAL_10:.*]] = u64[10,2]{1,0} shift-right-logical(%[[VAL_5]], %[[VAL_9]])
// CHECK:   %[[VAL_11:.*]] = u64[] constant(4294967295)
// CHECK:   %[[VAL_12:.*]] = u64[10,2]{1,0} broadcast(%[[VAL_11]]), dimensions={}
// CHECK:   %[[VAL_13:.*]] = u64[10,2]{1,0} and(%[[VAL_10]], %[[VAL_12]])
// CHECK:   %[[VAL_14:.*]] = u32[10,2]{1,0} convert(%[[VAL_13]])
// CHECK:   ROOT %[[VAL_15:.*]] = s32[10,2]{1,0} bitcast-convert(%[[VAL_14]])
// CHECK: }
// CHECK: ENTRY %main (p: s64[10]) -> s32[10,2] {
// CHECK:   %[[VAL_16:.*]] = s64[10]{0} parameter(0)
// CHECK:   ROOT %[[VAL_17:.*]] = s32[10,2]{1,0} call(%[[VAL_16]]), to_apply=%[[VAL_18:.*]]
// CHECK: }
)"));
}

TEST_F(BitcastDtypesExpanderTest, S8toS32) {
  absl::string_view hlo_string = R"(
HloModule bitcast_to_larger

ENTRY main {
  p = s8[10,4] parameter(0)
  ROOT out = s32[10] bitcast-convert(p)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  BitcastDtypesExpander expander;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));

  // NB: Correctness will be checked by `bitcast_convert_test`,
  // and the fact that we have registered the converter on all platforms.
  EXPECT_TRUE(changed);
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
// CHECK: HloModule bitcast_to_larger
// CHECK: %[[OR:or_U32.*]] ([[VAL_0:lhs.*]]: u32[], [[VAL_1:rhs.*]]: u32[]) -> u32[] {
// CHECK:  %[[VAL_0]] = u32[] parameter(0)
// CHECK:  %[[VAL_1]] = u32[] parameter(1)
// CHECK:  ROOT %[[VAL_2:.*]] = u32[] or(%[[VAL_0]], %[[VAL_1]])
// CHECK: }
// CHECK: %[[BITCAST_CONVERT:xla.bitcast_convert_.*]] ([[VAL_3:a.*]]: s8[10,4]) -> s32[10] {
// CHECK:  %[[VAL_3]] = s8[10,4]{1,0} parameter(0)
// CHECK:  %[[VAL_4:.*]] = u8[10,4]{1,0} bitcast-convert(%[[VAL_3]])
// CHECK:  %[[VAL_5:.*]] = u32[10,4]{1,0} convert(%[[VAL_4]])
// CHECK:  %[[VAL_6:.*]] = u32[] constant(8)
// CHECK:  %[[VAL_7:.*]] = u32[10,4]{1,0} broadcast(%[[VAL_6]]), dimensions={}
// CHECK:  %[[VAL_8:.*]] = u32[10,4]{1,0} iota(), iota_dimension=1
// CHECK:  %[[VAL_9:.*]] = u32[10,4]{1,0} multiply(%[[VAL_7]], %[[VAL_8]])
// CHECK:  %[[VAL_10:.*]] = u32[10,4]{1,0} shift-left(%[[VAL_5]], %[[VAL_9]])
// CHECK:  %[[VAL_11:.*]] = u32[] constant(0)
// CHECK:  %[[VAL_12:.*]] = u32[10]{0} reduce(%[[VAL_10]], %[[VAL_11]]), dimensions={1}, to_apply=%[[OR]]
// CHECK:  ROOT %[[VAL_14:.*]] = s32[10]{0} bitcast-convert(%[[VAL_12]])
// CHECK: }
// CHECK: ENTRY %main (p: s8[10,4]) -> s32[10] {
// CHECK:  %[[VAL_15:.*]] = s8[10,4]{1,0} parameter(0)
// CHECK:  ROOT %[[VAL_16:.*]] = s32[10]{0} call(%[[VAL_15]]), to_apply=%[[BITCAST_CONVERT]]
// CHECK: }
)"));
}

TEST_F(BitcastDtypesExpanderTest, RewriteInsideWhileTest) {
  absl::string_view hlo_string = R"(
HloModule module

body {
  p_body = (f32[2], s32[]) parameter(0)
  val1 = f32[2] get-tuple-element(p_body), index=0
  val2 = s32[] get-tuple-element(p_body), index=1
  const = s32[] constant(42)
  converted_val2 = s8[4] bitcast-convert(val2)
  converted_const = s8[4] bitcast-convert(const)
  add = s8[4] add(converted_val2, converted_const)
  out_add = s32[] bitcast-convert(add)
  ROOT root = (f32[2], s32[]) tuple(val1, out_add)
}

condition {
  p_cond = (f32[2], s32[]) parameter(0)
  gte = s32[] get-tuple-element(p_cond), index=1
  const = s32[] constant(42)
  ROOT result = pred[] compare(gte, const), direction=EQ
}

ENTRY entry {
  param.0 = f32[2] parameter(0)
  param.1 = s32[] parameter(1)
  while_init = (f32[2], s32[]) tuple(param.0, param.1)
  ROOT while = (f32[2], s32[]) while(while_init), condition=condition, body=body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Check that we do the rewrite and do not crash in the process.
  BitcastDtypesExpander expander;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);
}

}  // namespace
}  // namespace xla
