/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_opcode.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace {

// This test verifies that an example HloOpcode stringifies as expected.
TEST(HloOpcodeTest, StringifyMultiply) {
  ASSERT_EQ("multiply", HloOpcodeString(HloOpcode::kMultiply));
}

TEST(HloOpcodeTest, OpcodeProperties) {
  // Test counting macro.
#define SOME_LIST(X) \
  X(One)             \
  X(Two)             \
  X(Three)
  EXPECT_EQ(3, HLO_XLIST_LENGTH(SOME_LIST));
#undef SOME_LIST

  for (int i = 0; i < HloOpcodeCount(); ++i) {
    auto opcode = static_cast<HloOpcode>(i);
    // Test round-trip conversion to and from string.
    EXPECT_EQ(opcode, StringToHloOpcode(HloOpcodeString(opcode)).ValueOrDie());

    // Test some properties.
    switch (opcode) {
      case HloOpcode::kEq:
      case HloOpcode::kNe:
      case HloOpcode::kGt:
      case HloOpcode::kLt:
      case HloOpcode::kGe:
      case HloOpcode::kLe:
        EXPECT_TRUE(HloOpcodeIsComparison(opcode));
        break;
      default:
        EXPECT_FALSE(HloOpcodeIsComparison(opcode));
    }
    switch (opcode) {
      case HloOpcode::kCall:
      case HloOpcode::kConcatenate:
      case HloOpcode::kFusion:
      case HloOpcode::kMap:
      case HloOpcode::kGenerateToken:
      case HloOpcode::kTuple:
        EXPECT_TRUE(HloOpcodeIsVariadic(opcode));
        break;
      default:
        EXPECT_FALSE(HloOpcodeIsVariadic(opcode));
    }
  }
}

}  // namespace
}  // namespace xla
