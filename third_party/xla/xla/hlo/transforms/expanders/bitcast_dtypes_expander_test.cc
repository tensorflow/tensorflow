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

#include <gmock/gmock.h>
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  BitcastDtypesExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));

  EXPECT_TRUE(changed);
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
// CHECK: HloModule bitcast_to_smaller
// CHECK: ENTRY %main (p: s32[10]) -> s8[10,4] {
// CHECK:  %[[VAL_0:.*]] = s32[10]{0} parameter(0)
// CHECK:  %[[VAL_1:.*]] = s32[10,4]{1,0} broadcast(%[[VAL_0]]), dimensions={0}
// CHECK:  %[[VAL_2:.*]] = u32[10,4]{1,0} bitcast-convert(%[[VAL_1]])
// CHECK:  %[[VAL_3:.*]] = u32[] constant(8)
// CHECK:  %[[VAL_4:.*]] = u32[10,4]{1,0} broadcast(%[[VAL_3]]), dimensions={}
// CHECK:  %[[VAL_5:.*]] = u32[10,4]{1,0} iota(), iota_dimension=1
// CHECK:  %[[VAL_6:.*]] = u32[10,4]{1,0} multiply(%[[VAL_4]], %[[VAL_5]])
// CHECK:  %[[VAL_7:.*]] = u32[10,4]{1,0} shift-right-logical(%[[VAL_2]], %[[VAL_6]])
// CHECK:  %[[VAL_8:.*]] = u32[] constant(255)
// CHECK:  %[[VAL_9:.*]] = u32[10,4]{1,0} broadcast(%[[VAL_8]]), dimensions={}
// CHECK:  %[[VAL_10:.*]] = u32[10,4]{1,0} and(%[[VAL_7]], %[[VAL_9]])
// CHECK:  %[[VAL_11:.*]] = u8[10,4]{1,0} convert(%[[VAL_10]])
// CHECK:  ROOT %[[VAL_12:.*]] = s8[10,4]{1,0} bitcast-convert(%[[VAL_11]])
// CHECK: }
)"));
}

TEST_F(BitcastDtypesExpanderTest, S32toPred) {
  absl::string_view hlo_string = R"(
HloModule bitcast_to_smaller

ENTRY main {
  p = s32[10] parameter(0)
  ROOT out = pred[10,4] bitcast-convert(p)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  BitcastDtypesExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));

  EXPECT_TRUE(changed);
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
// CHECK: HloModule bitcast_to_smaller
// CHECK: ENTRY %main (p: s32[10]) -> pred[10,4] {
// CHECK:  %[[VAL_0:.*]] = s32[10]{0} parameter(0)
// CHECK:  %[[VAL_1:.*]] = s32[10,4]{1,0} broadcast(%[[VAL_0]]), dimensions={0}
// CHECK:  %[[VAL_2:.*]] = u32[10,4]{1,0} bitcast-convert(%[[VAL_1]])
// CHECK:  %[[VAL_3:.*]] = u32[] constant(8)
// CHECK:  %[[VAL_4:.*]] = u32[10,4]{1,0} broadcast(%[[VAL_3]]), dimensions={}
// CHECK:  %[[VAL_5:.*]] = u32[10,4]{1,0} iota(), iota_dimension=1
// CHECK:  %[[VAL_6:.*]] = u32[10,4]{1,0} multiply(%[[VAL_4]], %[[VAL_5]])
// CHECK:  %[[VAL_7:.*]] = u32[10,4]{1,0} shift-right-logical(%[[VAL_2]], %[[VAL_6]])
// CHECK:  %[[VAL_8:.*]] = u32[] constant(255)
// CHECK:  %[[VAL_9:.*]] = u32[10,4]{1,0} broadcast(%[[VAL_8]]), dimensions={}
// CHECK:  %[[VAL_10:.*]] = u32[10,4]{1,0} and(%[[VAL_7]], %[[VAL_9]])
// CHECK:  %[[VAL_11:.*]] = u8[10,4]{1,0} convert(%[[VAL_10]])
// CHECK:  ROOT %[[VAL_12:.*]] = pred[10,4]{1,0} convert(%[[VAL_11]])
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  BitcastDtypesExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));

  EXPECT_TRUE(changed);
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
// CHECK: HloModule bitcast_to_smaller, entry_computation_layout={(s64[10]{0})->s32[10,2]{1,0}}
// CHECK: ENTRY %main (p: s64[10]) -> s32[10,2] {
// CHECK:   %[[VAL_0:.*]] = s64[10]{0} parameter(0)
// CHECK:   %[[VAL_1:.*]] = s64[10,2]{1,0} broadcast(%[[VAL_0]]), dimensions={0}
// CHECK:   %[[VAL_2:.*]] = u64[10,2]{1,0} bitcast-convert(%[[VAL_1]])
// CHECK:   %[[VAL_3:.*]] = u64[] constant(32)
// CHECK:   %[[VAL_4:.*]] = u64[10,2]{1,0} broadcast(%[[VAL_3]]), dimensions={}
// CHECK:   %[[VAL_5:.*]] = u64[10,2]{1,0} iota(), iota_dimension=1
// CHECK:   %[[VAL_6:.*]] = u64[10,2]{1,0} multiply(%[[VAL_4]], %[[VAL_5]])
// CHECK:   %[[VAL_7:.*]] = u64[10,2]{1,0} shift-right-logical(%[[VAL_2]], %[[VAL_6]])
// CHECK:   %[[VAL_8:.*]] = u64[] constant(4294967295)
// CHECK:   %[[VAL_9:.*]] = u64[10,2]{1,0} broadcast(%[[VAL_8]]), dimensions={}
// CHECK:   %[[VAL_10:.*]] = u64[10,2]{1,0} and(%[[VAL_7]], %[[VAL_9]])
// CHECK:   %[[VAL_11:.*]] = u32[10,2]{1,0} convert(%[[VAL_10]])
// CHECK:   ROOT %[[VAL_12:.*]] = s32[10,2]{1,0} bitcast-convert(%[[VAL_11]])
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  BitcastDtypesExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));

  // NB: Correctness will be checked by `bitcast_convert_test`,
  // and the fact that we have registered the converter on all platforms.
  EXPECT_TRUE(changed);
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
// CHECK: ENTRY %main (p: s8[10,4]) -> s32[10] {
// CHECK:   %[[P:.*]] = s8[10,4]{1,0} parameter(0)
// CHECK:   %[[RESHAPE:.*]] = s8[40]{0} reshape(%[[P]])
// CHECK:   %[[SLICE_0:.*]] = s8[10]{0} slice(%[[RESHAPE]]), slice={[0:37:4]}
// CHECK:   %[[BC_0:.*]] = u8[10]{0} bitcast-convert(%[[SLICE_0]])
// CHECK:   %[[CONV_0:.*]] = u32[10]{0} convert(%[[BC_0]])
// CHECK:   %[[SLICE_1:.*]] = s8[10]{0} slice(%[[RESHAPE]]), slice={[1:38:4]}
// CHECK:   %[[BC_1:.*]] = u8[10]{0} bitcast-convert(%[[SLICE_1]])
// CHECK:   %[[CONV_1:.*]] = u32[10]{0} convert(%[[BC_1]])
// CHECK:   %[[C_8:.*]] = u32[] constant(8)
// CHECK:   %[[BCAST_8:.*]] = u32[10]{0} broadcast(%[[C_8]]), dimensions={}
// CHECK:   %[[SHL_0:.*]] = u32[10]{0} shift-left(%[[CONV_1]], %[[BCAST_8]])
// CHECK:   %[[OR_0:.*]] = u32[10]{0} or(%[[CONV_0]], %[[SHL_0]])
// CHECK:   %[[SLICE_2:.*]] = s8[10]{0} slice(%[[RESHAPE]]), slice={[2:39:4]}
// CHECK:   %[[BC_2:.*]] = u8[10]{0} bitcast-convert(%[[SLICE_2]])
// CHECK:   %[[CONV_2:.*]] = u32[10]{0} convert(%[[BC_2]])
// CHECK:   %[[C_16:.*]] = u32[] constant(16)
// CHECK:   %[[BCAST_16:.*]] = u32[10]{0} broadcast(%[[C_16]]), dimensions={}
// CHECK:   %[[SHL_1:.*]] = u32[10]{0} shift-left(%[[CONV_2]], %[[BCAST_16]])
// CHECK:   %[[OR_1:.*]] = u32[10]{0} or(%[[OR_0]], %[[SHL_1]])
// CHECK:   %[[SLICE_3:.*]] = s8[10]{0} slice(%[[RESHAPE]]), slice={[3:40:4]}
// CHECK:   %[[BC_3:.*]] = u8[10]{0} bitcast-convert(%[[SLICE_3]])
// CHECK:   %[[CONV_3:.*]] = u32[10]{0} convert(%[[BC_3]])
// CHECK:   %[[C_24:.*]] = u32[] constant(24)
// CHECK:   %[[BCAST_24:.*]] = u32[10]{0} broadcast(%[[C_24]]), dimensions={}
// CHECK:   %[[SHL_2:.*]] = u32[10]{0} shift-left(%[[CONV_3]], %[[BCAST_24]])
// CHECK:   %[[OR_2:.*]] = u32[10]{0} or(%[[OR_1]], %[[SHL_2]])
// CHECK:   ROOT %[[OUT:.*]] = s32[10]{0} bitcast-convert(%[[OR_2]])
// CHECK: }
)"));
}

TEST_F(BitcastDtypesExpanderTest, PredtoS32) {
  absl::string_view hlo_string = R"(
HloModule bitcast_to_larger

ENTRY main {
  p = pred[10,4] parameter(0)
  ROOT out = s32[10] bitcast-convert(p)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  BitcastDtypesExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));

  // NB: Correctness will be checked by `bitcast_convert_test`,
  // and the fact that we have registered the converter on all platforms.
  EXPECT_TRUE(changed);
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
// CHECK: ENTRY %main (p: pred[10,4]) -> s32[10] {
// CHECK:   %[[P:.*]] = pred[10,4]{1,0} parameter(0)
// CHECK:   %[[RESHAPE:.*]] = pred[40]{0} reshape(%[[P]])
// CHECK:   %[[SLICE_0:.*]] = pred[10]{0} slice(%[[RESHAPE]]), slice={[0:37:4]}
// CHECK:   %[[BC_0:.*]] = u8[10]{0} convert(%[[SLICE_0]])
// CHECK:   %[[CONV_0:.*]] = u32[10]{0} convert(%[[BC_0]])
// CHECK:   %[[SLICE_1:.*]] = pred[10]{0} slice(%[[RESHAPE]]), slice={[1:38:4]}
// CHECK:   %[[BC_1:.*]] = u8[10]{0} convert(%[[SLICE_1]])
// CHECK:   %[[CONV_1:.*]] = u32[10]{0} convert(%[[BC_1]])
// CHECK:   %[[C_8:.*]] = u32[] constant(8)
// CHECK:   %[[BCAST_8:.*]] = u32[10]{0} broadcast(%[[C_8]]), dimensions={}
// CHECK:   %[[SHL_0:.*]] = u32[10]{0} shift-left(%[[CONV_1]], %[[BCAST_8]])
// CHECK:   %[[OR_0:.*]] = u32[10]{0} or(%[[CONV_0]], %[[SHL_0]])
// CHECK:   %[[SLICE_2:.*]] = pred[10]{0} slice(%[[RESHAPE]]), slice={[2:39:4]}
// CHECK:   %[[BC_2:.*]] = u8[10]{0} convert(%[[SLICE_2]])
// CHECK:   %[[CONV_2:.*]] = u32[10]{0} convert(%[[BC_2]])
// CHECK:   %[[C_16:.*]] = u32[] constant(16)
// CHECK:   %[[BCAST_16:.*]] = u32[10]{0} broadcast(%[[C_16]]), dimensions={}
// CHECK:   %[[SHL_1:.*]] = u32[10]{0} shift-left(%[[CONV_2]], %[[BCAST_16]])
// CHECK:   %[[OR_1:.*]] = u32[10]{0} or(%[[OR_0]], %[[SHL_1]])
// CHECK:   %[[SLICE_3:.*]] = pred[10]{0} slice(%[[RESHAPE]]), slice={[3:40:4]}
// CHECK:   %[[BC_3:.*]] = u8[10]{0} convert(%[[SLICE_3]])
// CHECK:   %[[CONV_3:.*]] = u32[10]{0} convert(%[[BC_3]])
// CHECK:   %[[C_24:.*]] = u32[] constant(24)
// CHECK:   %[[BCAST_24:.*]] = u32[10]{0} broadcast(%[[C_24]]), dimensions={}
// CHECK:   %[[SHL_2:.*]] = u32[10]{0} shift-left(%[[CONV_3]], %[[BCAST_24]])
// CHECK:   %[[OR_2:.*]] = u32[10]{0} or(%[[OR_1]], %[[SHL_2]])
// CHECK:   ROOT %[[OUT:.*]] = s32[10]{0} bitcast-convert(%[[OR_2]])
// CHECK: }
)"));
}

TEST_F(BitcastDtypesExpanderTest, S16toS32Scalar) {
  absl::string_view hlo_string = R"(
HloModule bitcast_to_larger
ENTRY main {
  p = s16[2] parameter(0)
  ROOT out = s32[] bitcast-convert(p)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  BitcastDtypesExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));

  // NB: Correctness will be checked by `bitcast_convert_test`,
  // and the fact that we have registered the converter on all platforms.
  EXPECT_TRUE(changed);
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
// CHECK: ENTRY %main (p: s16[2]) -> s32[] {
// CHECK:   %[[P:.*]] = s16[2]{0} parameter(0)
// CHECK:   %[[SLICE_0:.*]] = s16[1]{0} slice(%[[P]]), slice={[0:1:2]}
// CHECK:   %[[BC_0:.*]] = u16[1]{0} bitcast-convert(%[[SLICE_0]])
// CHECK:   %[[CONV_0:.*]] = u32[1]{0} convert(%[[BC_0]])
// CHECK:   %[[SLICE_1:.*]] = s16[1]{0} slice(%[[P]]), slice={[1:2:2]}
// CHECK:   %[[BC_1:.*]] = u16[1]{0} bitcast-convert(%[[SLICE_1]])
// CHECK:   %[[CONV_1:.*]] = u32[1]{0} convert(%[[BC_1]])
// CHECK:   %[[C_16:.*]] = u32[] constant(16)
// CHECK:   %[[BCAST_16:.*]] = u32[1]{0} broadcast(%[[C_16]]), dimensions={}
// CHECK:   %[[SHL:.*]] = u32[1]{0} shift-left(%[[CONV_1]], %[[BCAST_16]])
// CHECK:   %[[OR:.*]] = u32[1]{0} or(%[[CONV_0]], %[[SHL]])
// CHECK:   %[[RESHAPE:.*]] = u32[] reshape(%[[OR]])
// CHECK:   ROOT %[[OUT:.*]] = s32[] bitcast-convert(%[[RESHAPE]])
// CHECK: }
)"));
}

TEST_F(BitcastDtypesExpanderTest, RewriteInsideWhileTest) {
  absl::string_view hlo_string = R"(
HloModule module

body {
  p_body = (s32[], s32[10]) parameter(0)
  loop_ctr = s32[] get-tuple-element(p_body), index=0
  val = s32[10] get-tuple-element(p_body), index=1
  const_ctr = s32[] constant(1)
  next_loop_ctr = s32[] add(loop_ctr, const_ctr)

  converted_val = s8[10, 4] bitcast-convert(val)
  const = s32[10] constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  converted_const = s8[10, 4] bitcast-convert(const)
  add = s8[10, 4] add(converted_val, converted_const)
  out_add = s32[10] bitcast-convert(add)
  ROOT root = (s32[], s32[10]) tuple(next_loop_ctr, out_add)
}

condition {
  p_cond = (s32[], s32[10]) parameter(0)
  loop_ctr = s32[] get-tuple-element(p_cond), index=0
  limit = s32[] constant(42)
  ROOT result = pred[] compare(loop_ctr, limit), direction=EQ
}

ENTRY entry {
  param.0 = s32[] parameter(0)
  param.1 = s32[10] parameter(1)
  while_init = (s32[], s32[10]) tuple(param.0, param.1)
  ROOT while = (s32[], s32[10]) while(while_init), condition=condition, body=body
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  // Check that we do the rewrite and do not crash in the process.
  BitcastDtypesExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);
}

}  // namespace
}  // namespace xla
