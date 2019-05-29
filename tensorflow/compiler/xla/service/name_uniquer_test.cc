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

#include "tensorflow/compiler/xla/service/name_uniquer.h"

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class NameUniquerTest : public ::testing::Test {};

TEST_F(NameUniquerTest, SimpleUniquer) {
  NameUniquer uniquer;

  EXPECT_EQ("foo", uniquer.GetUniqueName("foo"));
  EXPECT_EQ("foo__1", uniquer.GetUniqueName("foo"));
  EXPECT_EQ("foo__2", uniquer.GetUniqueName("foo"));
  EXPECT_EQ("bar", uniquer.GetUniqueName("bar"));
  EXPECT_EQ("foo__3", uniquer.GetUniqueName("foo"));
  EXPECT_EQ("bar__1", uniquer.GetUniqueName("bar"));
  EXPECT_EQ("qux", uniquer.GetUniqueName("qux"));
}

TEST_F(NameUniquerTest, DifferentSeparator) {
  NameUniquer uniquer(".");

  EXPECT_EQ("foo", uniquer.GetUniqueName("foo"));
  EXPECT_EQ("foo.1", uniquer.GetUniqueName("foo"));
  EXPECT_EQ("foo.2", uniquer.GetUniqueName("foo"));
  EXPECT_EQ("bar", uniquer.GetUniqueName("bar"));
  EXPECT_EQ("foo.3", uniquer.GetUniqueName("foo"));
  EXPECT_EQ("bar.1", uniquer.GetUniqueName("bar"));
}

TEST_F(NameUniquerTest, NumericSuffixes) {
  NameUniquer uniquer(".");

  EXPECT_EQ("foo", uniquer.GetUniqueName("foo"));
  EXPECT_EQ("foo.54", uniquer.GetUniqueName("foo.54"));
  EXPECT_EQ("foo.1", uniquer.GetUniqueName("foo"));
  EXPECT_EQ("foo.55.1", uniquer.GetUniqueName("foo.55.1"));
  EXPECT_EQ("foo.55.0", uniquer.GetUniqueName("foo.55.1"));
  EXPECT_EQ("bar.1000", uniquer.GetUniqueName("bar.1000"));
  EXPECT_EQ("bar.2000", uniquer.GetUniqueName("bar.2000"));
  EXPECT_EQ("bar.-2000", uniquer.GetUniqueName("bar.-2000"));
  EXPECT_EQ("bar.1", uniquer.GetUniqueName("bar.1"));
}

TEST_F(NameUniquerTest, PrefixHasSuffix) {
  NameUniquer uniquer(".");

  EXPECT_EQ("foo.11.0", uniquer.GetUniqueName("foo.11.0"));
  EXPECT_EQ("foo.11", uniquer.GetUniqueName("foo.11"));
}

TEST_F(NameUniquerTest, Sanitize) {
  NameUniquer uniquer("_");

  EXPECT_EQ("foo", uniquer.GetUniqueName("foo"));
  EXPECT_EQ("foo_1", uniquer.GetUniqueName("foo"));
  EXPECT_EQ("foo.54", uniquer.GetUniqueName("foo.54"));
  EXPECT_EQ("foo_54", uniquer.GetUniqueName("foo_54"));
  EXPECT_EQ("foo_54.1", uniquer.GetUniqueName("foo_54.1"));
  EXPECT_EQ("foo_2", uniquer.GetUniqueName("foo"));

  // Invalid characters will be replaced with '_'.
  EXPECT_EQ("bar_1000", uniquer.GetUniqueName("bar<1000"));
  EXPECT_EQ("bar_2000", uniquer.GetUniqueName("bar<2000"));
  EXPECT_EQ("bar_1", uniquer.GetUniqueName("bar_1"));

  // Separator is only recognized in the middle of the prefix.
  EXPECT_EQ("_10", uniquer.GetUniqueName(
                       ".10"));  // the leading '.' is replaced with '_'.
  EXPECT_EQ("_10_1", uniquer.GetUniqueName(".10"));
  EXPECT_EQ("_10_2", uniquer.GetUniqueName("_10"));
  EXPECT_EQ("foobar_", uniquer.GetUniqueName("foobar_"));
  EXPECT_EQ("foobar__1", uniquer.GetUniqueName("foobar_"));
}

TEST_F(NameUniquerTest, KeepNamesInRandomOrder) {
  NameUniquer uniquer(".");

  EXPECT_EQ("foo.11", uniquer.GetUniqueName("foo.11"));
  EXPECT_EQ("foo.10", uniquer.GetUniqueName("foo.10"));
  EXPECT_EQ("foo.1", uniquer.GetUniqueName("foo.1"));
  EXPECT_EQ("foo.12", uniquer.GetUniqueName("foo.12"));
  EXPECT_EQ("foo.3", uniquer.GetUniqueName("foo.3"));
}

TEST_F(NameUniquerTest, AvoidKeywords) {
  NameUniquer uniquer(".");

  EXPECT_EQ("f32_", uniquer.GetUniqueName("f32"));
  EXPECT_EQ("s64_", uniquer.GetUniqueName("s64"));
  EXPECT_EQ("pred_", uniquer.GetUniqueName("pred"));

  // Name prefix __xla_ is preserved.
  EXPECT_NE(uniquer.GetUniqueName("__xla_").find("__xla_"), std::string::npos);
  // Other form of __ prefixes is not preserved to avoid using name prefixes
  // reserved by backends.
  EXPECT_EQ(uniquer.GetUniqueName("__abx").find("__"), std::string::npos);

  // Though a primitive type, "tuple" is not a keyword.
  EXPECT_EQ("tuple", uniquer.GetUniqueName("tuple"));

  // Keywords are not capitalized.
  EXPECT_EQ("F32", uniquer.GetUniqueName("F32"));
  EXPECT_EQ("S32", uniquer.GetUniqueName("S32"));
  EXPECT_EQ("Pred", uniquer.GetUniqueName("Pred"));
}

}  // namespace
}  // namespace xla
