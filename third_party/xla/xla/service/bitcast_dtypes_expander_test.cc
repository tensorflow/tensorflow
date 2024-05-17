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

#include "xla/service/bitcast_dtypes_expander.h"

#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class BitcastDtypesExpanderTest : public HloTestBase {};

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
// CHECK: %xla.bitcast_convert_s32_10__2_s8_10_4_.17 (a.1: s32[10]) -> s8[10,4] {
// CHECK:  %[[VAL_0:.*]] = s32[10]{0} parameter(0)
// CHECK:  %[[VAL_1:.*]] = s32[10,1]{1,0} reshape(s32[10]{0} %[[VAL_0]])
// CHECK:  %[[VAL_2:.*]] = s32[10,1]{1,0} broadcast(s32[10,1]{1,0} %[[VAL_1]]), dimensions={0,1}
// CHECK:  %[[VAL_3:.*]] = s32[10]{0} reshape(s32[10,1]{1,0} %[[VAL_2]])
// CHECK:  %[[VAL_4:.*]] = s32[10,4]{1,0} broadcast(s32[10]{0} %[[VAL_3]]), dimensions={0}
// CHECK:  %[[VAL_5:.*]] = u32[10,4]{1,0} bitcast-convert(s32[10,4]{1,0} %[[VAL_4]])
// CHECK:  %[[VAL_6:.*]] = u32[] constant(8)
// CHECK:  %[[VAL_7:.*]] = u32[10,4]{1,0} broadcast(u32[] %[[VAL_6]]), dimensions={}
// CHECK:  %[[VAL_8:.*]] = u32[10,4]{1,0} iota(), iota_dimension=1
// CHECK:  %[[VAL_9:.*]] = u32[10,4]{1,0} multiply(u32[10,4]{1,0} %[[VAL_7]], u32[10,4]{1,0} %[[VAL_8]])
// CHECK:  %[[VAL_10:.*]] = u32[10,4]{1,0} shift-right-logical(u32[10,4]{1,0} %[[VAL_5]], u32[10,4]{1,0} %[[VAL_9]])
// CHECK:  %[[VAL_11:.*]] = u32[] constant(255)
// CHECK:  %[[VAL_12:.*]] = u32[10,4]{1,0} broadcast(u32[] %[[VAL_11]]), dimensions={}
// CHECK:  %[[VAL_13:.*]] = u32[10,4]{1,0} and(u32[10,4]{1,0} %[[VAL_10]], u32[10,4]{1,0} %[[VAL_12]])
// CHECK:  %[[VAL_14:.*]] = u8[10,4]{1,0} convert(u32[10,4]{1,0} %[[VAL_13]])
// CHECK:  ROOT %[[VAL_15:.*]] = s8[10,4]{1,0} bitcast-convert(u8[10,4]{1,0} %[[VAL_14]])
// CHECK: }
// CHECK: ENTRY %main (p: s32[10]) -> s8[10,4] {
// CHECK:  %[[VAL_16:.*]] = s32[10]{0} parameter(0)
// CHECK:  ROOT %[[VAL_17:.*]] = s8[10,4]{1,0} call(s32[10]{0} %[[VAL_16]]), to_apply=%[[VAL_18:.*]]
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
// CHECK: %xla.bitcast_convert_s64_10__2_s32_10_2_.17 (a.1: s64[10]) -> s32[10,2] {
// CHECK:   %[[VAL_0:.*]] = s64[10]{0} parameter(0)
// CHECK:   %[[VAL_1:.*]] = s64[10,1]{1,0} reshape(s64[10]{0} %[[VAL_0]])
// CHECK:   %[[VAL_2:.*]] = s64[10,1]{1,0} broadcast(s64[10,1]{1,0} %[[VAL_1]]), dimensions={0,1}
// CHECK:   %[[VAL_3:.*]] = s64[10]{0} reshape(s64[10,1]{1,0} %[[VAL_2]])
// CHECK:   %[[VAL_4:.*]] = s64[10,2]{1,0} broadcast(s64[10]{0} %[[VAL_3]]), dimensions={0}
// CHECK:   %[[VAL_5:.*]] = u64[10,2]{1,0} bitcast-convert(s64[10,2]{1,0} %[[VAL_4]])
// CHECK:   %[[VAL_6:.*]] = u64[] constant(32)
// CHECK:   %[[VAL_7:.*]] = u64[10,2]{1,0} broadcast(u64[] %[[VAL_6]]), dimensions={}
// CHECK:   %[[VAL_8:.*]] = u64[10,2]{1,0} iota(), iota_dimension=1
// CHECK:   %[[VAL_9:.*]] = u64[10,2]{1,0} multiply(u64[10,2]{1,0} %[[VAL_7]], u64[10,2]{1,0} %[[VAL_8]])
// CHECK:   %[[VAL_10:.*]] = u64[10,2]{1,0} shift-right-logical(u64[10,2]{1,0} %[[VAL_5]], u64[10,2]{1,0} %[[VAL_9]])
// CHECK:   %[[VAL_11:.*]] = u64[] constant(4294967295)
// CHECK:   %[[VAL_12:.*]] = u64[10,2]{1,0} broadcast(u64[] %[[VAL_11]]), dimensions={}
// CHECK:   %[[VAL_13:.*]] = u64[10,2]{1,0} and(u64[10,2]{1,0} %[[VAL_10]], u64[10,2]{1,0} %[[VAL_12]])
// CHECK:   %[[VAL_14:.*]] = u32[10,2]{1,0} convert(u64[10,2]{1,0} %[[VAL_13]])
// CHECK:   ROOT %[[VAL_15:.*]] = s32[10,2]{1,0} bitcast-convert(u32[10,2]{1,0} %[[VAL_14]])
// CHECK: }
// CHECK: ENTRY %main (p: s64[10]) -> s32[10,2] {
// CHECK:   %[[VAL_16:.*]] = s64[10]{0} parameter(0)
// CHECK:   ROOT %[[VAL_17:.*]] = s32[10,2]{1,0} call(s64[10]{0} %[[VAL_16]]), to_apply=%[[VAL_18:.*]]
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
// CHECK: %or_U32.10 (lhs.11: u32[], rhs.12: u32[]) -> u32[] {
// CHECK:  %[[VAL_0:.*]] = u32[] parameter(0)
// CHECK:  %[[VAL_1:.*]] = u32[] parameter(1)
// CHECK:  ROOT %[[VAL_2:.*]] = u32[] or(u32[] %[[VAL_0]], u32[] %[[VAL_1]])
// CHECK: }
// CHECK: %xla.bitcast_convert_s8_10_4__2_s32_10_.16 (a.1: s8[10,4]) -> s32[10] {
// CHECK:  %[[VAL_3:.*]] = s8[10,4]{1,0} parameter(0)
// CHECK:  %[[VAL_4:.*]] = u8[10,4]{1,0} bitcast-convert(s8[10,4]{1,0} %[[VAL_3]])
// CHECK:  %[[VAL_5:.*]] = u32[10,4]{1,0} convert(u8[10,4]{1,0} %[[VAL_4]])
// CHECK:  %[[VAL_6:.*]] = u32[] constant(8)
// CHECK:  %[[VAL_7:.*]] = u32[10,4]{1,0} broadcast(u32[] %[[VAL_6]]), dimensions={}
// CHECK:  %[[VAL_8:.*]] = u32[10,4]{1,0} iota(), iota_dimension=1
// CHECK:  %[[VAL_9:.*]] = u32[10,4]{1,0} multiply(u32[10,4]{1,0} %[[VAL_7]], u32[10,4]{1,0} %[[VAL_8]])
// CHECK:  %[[VAL_10:.*]] = u32[10,4]{1,0} shift-left(u32[10,4]{1,0} %[[VAL_5]], u32[10,4]{1,0} %[[VAL_9]])
// CHECK:  %[[VAL_11:.*]] = u32[] constant(0)
// CHECK:  %[[VAL_12:.*]] = u32[10]{0} reduce(u32[10,4]{1,0} %[[VAL_10]], u32[] %[[VAL_11]]), dimensions={1}, to_apply=%[[VAL_13:.*]]
// CHECK:  ROOT %[[VAL_14:.*]] = s32[10]{0} bitcast-convert(u32[10]{0} %[[VAL_12]])
// CHECK: }
// CHECK: ENTRY %main (p: s8[10,4]) -> s32[10] {
// CHECK:  %[[VAL_15:.*]] = s8[10,4]{1,0} parameter(0)
// CHECK:  ROOT %[[VAL_16:.*]] = s32[10]{0} call(s8[10,4]{1,0} %[[VAL_15]]), to_apply=%[[VAL_17:.*]]
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
