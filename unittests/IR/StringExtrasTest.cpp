//===- StringExtrasTest.cpp - Tests for utility methods in StringExtras.h -===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/Support/StringExtras.h"
#include "gtest/gtest.h"

using namespace mlir;

static void testConvertToSnakeCase(llvm::StringRef input,
                                   llvm::StringRef expected) {
  EXPECT_EQ(convertToSnakeCase(input), expected.str());
}

TEST(StringExtras, ConvertToSnakeCase) {
  testConvertToSnakeCase("OpName", "op_name");
  testConvertToSnakeCase("opName", "op_name");
  testConvertToSnakeCase("_OpName", "_op_name");
  testConvertToSnakeCase("Op_Name", "op_name");
  testConvertToSnakeCase("", "");
  testConvertToSnakeCase("A", "a");
  testConvertToSnakeCase("_", "_");
  testConvertToSnakeCase("a", "a");
  testConvertToSnakeCase("op_name", "op_name");
  testConvertToSnakeCase("_op_name", "_op_name");
  testConvertToSnakeCase("__op_name", "__op_name");
  testConvertToSnakeCase("op__name", "op__name");
}

template <bool capitalizeFirst>
static void testConvertToCamelCase(llvm::StringRef input,
                                   llvm::StringRef expected) {
  EXPECT_EQ(convertToCamelCase(input, capitalizeFirst), expected.str());
}

TEST(StringExtras, ConvertToCamelCase) {
  testConvertToCamelCase<false>("op_name", "opName");
  testConvertToCamelCase<false>("_op_name", "_opName");
  testConvertToCamelCase<false>("__op_name", "_OpName");
  testConvertToCamelCase<false>("op__name", "op_Name");
  testConvertToCamelCase<false>("", "");
  testConvertToCamelCase<false>("A", "A");
  testConvertToCamelCase<false>("_", "_");
  testConvertToCamelCase<false>("a", "a");
  testConvertToCamelCase<false>("OpName", "OpName");
  testConvertToCamelCase<false>("opName", "opName");
  testConvertToCamelCase<false>("_OpName", "_OpName");
  testConvertToCamelCase<false>("Op_Name", "Op_Name");
  testConvertToCamelCase<true>("op_name", "OpName");
  testConvertToCamelCase<true>("_op_name", "_opName");
  testConvertToCamelCase<true>("__op_name", "_OpName");
  testConvertToCamelCase<true>("op__name", "Op_Name");
  testConvertToCamelCase<true>("", "");
  testConvertToCamelCase<true>("A", "A");
  testConvertToCamelCase<true>("_", "_");
  testConvertToCamelCase<true>("a", "A");
  testConvertToCamelCase<true>("OpName", "OpName");
  testConvertToCamelCase<true>("_OpName", "_OpName");
  testConvertToCamelCase<true>("Op_Name", "Op_Name");
  testConvertToCamelCase<true>("opName", "OpName");
}
