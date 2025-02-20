/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_opcode.h"

#include <cstdint>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

namespace xla {
namespace {

TEST(HloOpcodeTest, ExampleUsage) {
  ASSERT_EQ(HloOpcodeString(HloOpcode::kMultiply), "multiply");
  ASSERT_EQ(HloOpcodeArity(HloOpcode::kAdd), 2);
}

TEST(HloOpcodeTest, HloXList) {
#define SOME_LIST(X) \
  X(One)             \
  X(Two)             \
  X(Three)
  EXPECT_EQ(3, HLO_XLIST_LENGTH(SOME_LIST));
#undef SOME_LIST
}

std::vector<HloOpcode> GetAllCodes() {
  std::vector<HloOpcode> test_cases;
  for (int i = 0; i < HloOpcodeCount(); ++i) {
    test_cases.push_back(static_cast<HloOpcode>(i));
  }
  return test_cases;
}

class HloOpcodeTestP : public ::testing::TestWithParam<HloOpcode> {};

TEST_P(HloOpcodeTestP, OpcodePropertiesNew) {
  HloOpcode opcode = GetParam();
  EXPECT_EQ(opcode, StringToHloOpcode(HloOpcodeString(opcode)).value());
  std::optional<int64_t> arity = HloOpcodeArity(opcode);
  if (arity.has_value()) {
    EXPECT_GE(arity.value(), 0);
  }
}

INSTANTIATE_TEST_SUITE_P(HloOpcodeTestSuite, HloOpcodeTestP,
                         testing::ValuesIn(GetAllCodes()));

}  // namespace
}  // namespace xla
