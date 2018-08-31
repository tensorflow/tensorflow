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

// Test for parse_flags_from_env.cc

#include "tensorflow/compiler/xla/legacy_flags/debug_options_parsers.h"

#include <unordered_map>
#include <vector>

#include "tensorflow/core/platform/test.h"

namespace xla {
namespace legacy_flags {

// Test that the xla_backend_extra_options flag is parsed correctly.
TEST(DebugOptionsFlags, ParseXlaBackendExtraOptions) {
  std::unordered_map<string, string> test_map;
  string test_string = "aa=bb,cc,dd=,ee=ff=gg";
  impl::parse_xla_backend_extra_options(&test_map, test_string);
  EXPECT_EQ(test_map.size(), 4);
  EXPECT_EQ(test_map.at("aa"), "bb");
  EXPECT_EQ(test_map.at("cc"), "");
  EXPECT_EQ(test_map.at("dd"), "");
  EXPECT_EQ(test_map.at("ee"), "ff=gg");
}

// Test that the xla_reduce_precision flag is parsed correctly.
TEST(DebugOptionsFlags, ParseXlaReducePrecisionOptionNoStrings) {
  HloReducePrecisionOptions proto;
  string test_string = "OP_OUTPUTS=5,10:add,dot";
  EXPECT_TRUE(impl::parse_xla_reduce_precision_option(&proto, test_string));
  EXPECT_EQ(proto.location(), HloReducePrecisionOptions::OP_OUTPUTS);
  EXPECT_EQ(proto.exponent_bits(), 5);
  EXPECT_EQ(proto.mantissa_bits(), 10);
  EXPECT_EQ(proto.opcodes_to_suffix_size(), 2);
  EXPECT_EQ(static_cast<HloOpcode>(proto.opcodes_to_suffix(0)),
            HloOpcode::kAdd);
  EXPECT_EQ(static_cast<HloOpcode>(proto.opcodes_to_suffix(1)),
            HloOpcode::kDot);
  EXPECT_EQ(proto.opname_substrings_to_suffix_size(), 0);
}

TEST(DebugOptionsFlags, ParseXlaReducePrecisionOptionNoStringsSemicolon) {
  HloReducePrecisionOptions proto;
  string test_string = "OP_OUTPUTS=5,10:add,dot;";
  EXPECT_TRUE(impl::parse_xla_reduce_precision_option(&proto, test_string));
  EXPECT_EQ(proto.location(), HloReducePrecisionOptions::OP_OUTPUTS);
  EXPECT_EQ(proto.exponent_bits(), 5);
  EXPECT_EQ(proto.mantissa_bits(), 10);
  EXPECT_EQ(proto.opcodes_to_suffix_size(), 2);
  EXPECT_EQ(static_cast<HloOpcode>(proto.opcodes_to_suffix(0)),
            HloOpcode::kAdd);
  EXPECT_EQ(static_cast<HloOpcode>(proto.opcodes_to_suffix(1)),
            HloOpcode::kDot);
  EXPECT_EQ(proto.opname_substrings_to_suffix_size(), 0);
}

TEST(DebugOptionsFlags, ParseXlaReducePrecisionOptionNoOpcodes) {
  HloReducePrecisionOptions proto;
  string test_string = "UNFUSED_OP_OUTPUTS=5,10:;foo,bar/baz";
  EXPECT_TRUE(impl::parse_xla_reduce_precision_option(&proto, test_string));
  EXPECT_EQ(proto.location(), HloReducePrecisionOptions::UNFUSED_OP_OUTPUTS);
  EXPECT_EQ(proto.exponent_bits(), 5);
  EXPECT_EQ(proto.mantissa_bits(), 10);
  EXPECT_EQ(proto.opcodes_to_suffix_size(), HloOpcodeCount());
  EXPECT_EQ(proto.opname_substrings_to_suffix_size(), 2);
  EXPECT_EQ(proto.opname_substrings_to_suffix(0), "foo");
  EXPECT_EQ(proto.opname_substrings_to_suffix(1), "bar/baz");
}

TEST(DebugOptionsFlags, ParseXlaReducePrecisionOptionBoth) {
  HloReducePrecisionOptions proto;
  string test_string = "UNFUSED_OP_OUTPUTS=5,10:subtract;foo,bar/baz";
  EXPECT_TRUE(impl::parse_xla_reduce_precision_option(&proto, test_string));
  EXPECT_EQ(proto.location(), HloReducePrecisionOptions::UNFUSED_OP_OUTPUTS);
  EXPECT_EQ(proto.exponent_bits(), 5);
  EXPECT_EQ(proto.mantissa_bits(), 10);
  EXPECT_EQ(proto.opcodes_to_suffix_size(), 1);
  EXPECT_EQ(static_cast<HloOpcode>(proto.opcodes_to_suffix(0)),
            HloOpcode::kSubtract);
  EXPECT_EQ(proto.opname_substrings_to_suffix_size(), 2);
  EXPECT_EQ(proto.opname_substrings_to_suffix(0), "foo");
  EXPECT_EQ(proto.opname_substrings_to_suffix(1), "bar/baz");
}

}  // namespace legacy_flags
}  // namespace xla

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
