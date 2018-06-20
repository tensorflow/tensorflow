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
  EXPECT_EQ("foo.55", uniquer.GetUniqueName("foo"));
  EXPECT_EQ("foo.55.1", uniquer.GetUniqueName("foo.55.1"));
  EXPECT_EQ("foo.55.2", uniquer.GetUniqueName("foo.55.1"));
  EXPECT_EQ("bar.0", uniquer.GetUniqueName("bar.-1000"));
  EXPECT_EQ("bar.1", uniquer.GetUniqueName("bar.-2000"));
  EXPECT_EQ("bar.2", uniquer.GetUniqueName("bar.1"));
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
  EXPECT_EQ("foo_55", uniquer.GetUniqueName("foo"));

  // Invalid characters will be replaced with '_'.
  EXPECT_EQ("bar_0", uniquer.GetUniqueName("bar<-1000"));
  EXPECT_EQ("bar_1", uniquer.GetUniqueName("bar<-2000"));
  EXPECT_EQ("bar_2", uniquer.GetUniqueName("bar_1"));

  // Separator is only recognized in the middle of the prefix.
  EXPECT_EQ("_10", uniquer.GetUniqueName(
                       ".10"));  // the leading '.' is replaced with '_'.
  EXPECT_EQ("_10_1", uniquer.GetUniqueName(".10"));
  EXPECT_EQ("_10_2", uniquer.GetUniqueName("_10"));
  EXPECT_EQ("foobar_", uniquer.GetUniqueName("foobar_"));
  EXPECT_EQ("foobar__1", uniquer.GetUniqueName("foobar_"));
}

}  // namespace
}  // namespace xla
