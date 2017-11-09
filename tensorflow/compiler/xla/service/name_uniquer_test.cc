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
  EXPECT_EQ("bar", uniquer.GetUniqueName("bar.-1000"));
  EXPECT_EQ("bar.1", uniquer.GetUniqueName("bar.-2000"));
  EXPECT_EQ("bar.2", uniquer.GetUniqueName("bar.1"));

  // Separator is only recognized in the middle of the prefix.
  EXPECT_EQ(".10", uniquer.GetUniqueName(".10"));
  EXPECT_EQ(".10.1", uniquer.GetUniqueName(".10"));
  EXPECT_EQ("foobar.", uniquer.GetUniqueName("foobar."));
  EXPECT_EQ("foobar..1", uniquer.GetUniqueName("foobar."));
}

}  // namespace
}  // namespace xla
