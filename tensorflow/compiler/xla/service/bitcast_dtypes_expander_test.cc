/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/bitcast_dtypes_expander.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/statusor.h"

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
// CHECK: %xla.bitcast_convert_s32_10__2_s8_10_4_.17. (a.1: s32[10]) -> s8[10,4] {
// CHECK:  %a.1 = s32[10]{0} parameter(0)
// CHECK:  %reshape = s32[10,1]{1,0} reshape(s32[10]{0} %a.1)
// CHECK:  %broadcast = s32[10,1]{1,0} broadcast(s32[10,1]{1,0} %reshape), dimensions={0,1}
// CHECK:  %reshape.1 = s32[10]{0} reshape(s32[10,1]{1,0} %broadcast)
// CHECK:  %broadcast.1 = s32[10,4]{1,0} broadcast(s32[10]{0} %reshape.1), dimensions={0}
// CHECK:  %bitcast-convert = u32[10,4]{1,0} bitcast-convert(s32[10,4]{1,0} %broadcast.1)
// CHECK:  %constant = u32[] constant(8)
// CHECK:  %broadcast.2 = u32[10,4]{1,0} broadcast(u32[] %constant), dimensions={}
// CHECK:  %iota = u32[10,4]{1,0} iota(), iota_dimension=1
// CHECK:  %multiply = u32[10,4]{1,0} multiply(u32[10,4]{1,0} %broadcast.2, u32[10,4]{1,0} %iota)
// CHECK:  %shift-right-logical = u32[10,4]{1,0} shift-right-logical(u32[10,4]{1,0} %bitcast-convert, u32[10,4]{1,0} %multiply)
// CHECK:  %constant.1 = u32[] constant(255)
// CHECK:  %broadcast.3 = u32[10,4]{1,0} broadcast(u32[] %constant.1), dimensions={}
// CHECK:  %and = u32[10,4]{1,0} and(u32[10,4]{1,0} %shift-right-logical, u32[10,4]{1,0} %broadcast.3)
// CHECK:  %convert = u8[10,4]{1,0} convert(u32[10,4]{1,0} %and)
// CHECK:  ROOT %bitcast-convert.1 = s8[10,4]{1,0} bitcast-convert(u8[10,4]{1,0} %convert)
// CHECK: }
// CHECK: ENTRY %main (p: s32[10]) -> s8[10,4] {
// CHECK:  %p = s32[10]{0} parameter(0)
// CHECK:  ROOT %call = s8[10,4]{1,0} call(s32[10]{0} %p), to_apply=%xla.bitcast_convert_s32_10__2_s8_10_4_.17.
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
// CHECK: %xla.bitcast_convert_s8_10_4__2_s32_10_.16. (a.1: s8[10,4]) -> s32[10] {
// CHECK:  %a.1 = s8[10,4]{1,0} parameter(0)
// CHECK:  %bitcast-convert = u8[10,4]{1,0} bitcast-convert(s8[10,4]{1,0} %a.1)
// CHECK:  %convert = u32[10,4]{1,0} convert(u8[10,4]{1,0} %bitcast-convert)
// CHECK:  %constant = u32[] constant(8)
// CHECK:  %broadcast = u32[10,4]{1,0} broadcast(u32[] %constant), dimensions={}
// CHECK:  %iota = u32[10,4]{1,0} iota(), iota_dimension=1
// CHECK:  %multiply = u32[10,4]{1,0} multiply(u32[10,4]{1,0} %broadcast, u32[10,4]{1,0} %iota)
// CHECK:  %shift-left = u32[10,4]{1,0} shift-left(u32[10,4]{1,0} %convert, u32[10,4]{1,0} %multiply)
// CHECK:  %constant.1 = u32[] constant(0)
// CHECK:  %reduce = u32[10]{0} reduce(u32[10,4]{1,0} %shift-left, u32[] %constant.1), dimensions={1}, to_apply=%or_U32.10.
// CHECK:  ROOT %bitcast-convert.1 = s32[10]{0} bitcast-convert(u32[10]{0} %reduce)
// CHECK: }
// CHECK: ENTRY %main (p: s8[10,4]) -> s32[10] {
// CHECK:  %p = s8[10,4]{1,0} parameter(0)
// CHECK:  ROOT %call = s32[10]{0} call(s8[10,4]{1,0} %p), to_apply=%xla.bitcast_convert_s8_10_4__2_s32_10_.16.
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
