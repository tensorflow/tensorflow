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

#include "xla/backends/cpu/codegen/symbol_name_util.h"

#include <gtest/gtest.h>
#include "xla/tsl/platform/statusor.h"

namespace {

TEST(SymbolNameUtilTest, NoChange) {
  TF_ASSERT_OK_AND_ASSIGN(auto result, xla::cpu::ConvertToCName("foo"));
  EXPECT_EQ(result, "foo");
}

TEST(SymbolNameUtilTest, Dot) {
  TF_ASSERT_OK_AND_ASSIGN(auto result, xla::cpu::ConvertToCName("foo.bar"));
  EXPECT_EQ(result, "foo_bar");
}

TEST(SymbolNameUtilTest, Dash) {
  TF_ASSERT_OK_AND_ASSIGN(auto result, xla::cpu::ConvertToCName("foo-bar"));
  EXPECT_EQ(result, "foo_bar");
}

TEST(SymbolNameUtilTest, Colon) {
  TF_ASSERT_OK_AND_ASSIGN(auto result, xla::cpu::ConvertToCName("foo:bar"));
  EXPECT_EQ(result, "foo_bar");
}

TEST(SymbolNameUtilTest, CantConvertToCNameInvalid) {
  EXPECT_FALSE(xla::cpu::ConvertToCName("1_test").ok());
}

}  // namespace
