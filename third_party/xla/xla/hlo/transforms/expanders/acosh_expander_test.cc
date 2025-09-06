/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/acosh_expander.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/pattern_matcher.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace m = match;

class AcoshExpanderTest : public HloHardwareIndependentTestBase {};

TEST_F(AcoshExpanderTest, ExpandWith) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = f32[2,3] parameter(0)
      ROOT r = f32[2,3] acosh(p)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));

  auto computation = m->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAcosh);
  AcoshExpander acosh_expander;
  ASSERT_TRUE(acosh_expander.Run(m.get()).value());
  root = computation->root_instruction();

  EXPECT_TRUE(*RunFileCheck(root->GetModule()->ToString(), R"(
// CHECK: %[[VAL_0:.*]] = f32[2,3]{1,0} parameter(0)
// CHECK: %[[VAL_1:.*]] = f32[] constant(3.40282347e+38)
// CHECK: %[[VAL_2:.*]] = f32[2,3]{1,0} broadcast(%[[VAL_1]]), dimensions={}
// CHECK: %[[VAL_3:.*]] = f32[] constant(2)
// CHECK: %[[VAL_4:.*]] = f32[2,3]{1,0} broadcast(%[[VAL_3]]), dimensions={}
// CHECK: %[[VAL_5:.*]] = f32[2,3]{1,0} divide(%[[VAL_2]], %[[VAL_4]])
// CHECK: %[[VAL_6:.*]] = pred[2,3]{1,0} compare(%[[VAL_0]], %[[VAL_5]]), direction=GE
// CHECK: %[[VAL_7:.*]] = f32[2,3]{1,0} log(%[[VAL_4]])
// CHECK: %[[VAL_8:.*]] = f32[2,3]{1,0} log(%[[VAL_0]])
// CHECK: %[[VAL_9:.*]] = f32[2,3]{1,0} add(%[[VAL_7]], %[[VAL_8]])
// CHECK: %[[VAL_10:.*]] = f32[] constant(1)
// CHECK: %[[VAL_11:.*]] = f32[2,3]{1,0} broadcast(%[[VAL_10]]), dimensions={}
// CHECK: %[[VAL_12:.*]] = f32[2,3]{1,0} subtract(%[[VAL_0]], %[[VAL_11]])
// CHECK: %[[VAL_13:.*]] = f32[2,3]{1,0} sqrt(%[[VAL_12]])
// CHECK: %[[VAL_14:.*]] = f32[2,3]{1,0} add(%[[VAL_0]], %[[VAL_13]])
// CHECK: %[[VAL_15:.*]] = f32[2,3]{1,0} sqrt(%[[VAL_14]])
// CHECK: %[[VAL_16:.*]] = f32[2,3]{1,0} add(%[[VAL_15]], %[[VAL_13]])
// CHECK: %[[VAL_17:.*]] = f32[2,3]{1,0} multiply(%[[VAL_13]], %[[VAL_16]])
// CHECK: %[[VAL_18:.*]] = f32[2,3]{1,0} log-plus-one(%[[VAL_17]])
// CHECK: f32[2,3]{1,0} select(%[[VAL_6]], %[[VAL_9]], %[[VAL_18]])
)"));
}

}  // namespace
}  // namespace xla
