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

// TODO(b/80179519): Fix open source build for real.
#if 0
#include "tensorflow/compiler/xla/scanner.h"

#include <string>

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/env.h"

namespace xla {
namespace {

TEST(Scanner, Empty) {
  Scanner scanner("");

  EXPECT_EQ(scanner.PeekChar(), EOF);
  EXPECT_TRUE(scanner.MatchEof());
  EXPECT_TRUE(scanner.Match(""));
  EXPECT_FALSE(scanner.Match("1"));
  EXPECT_TRUE(scanner.ok());
}

TEST(Scanner, Prefix) {
  Scanner scanner("1234 5");
  EXPECT_FALSE(scanner.MatchEof());
  EXPECT_TRUE(scanner.Match("12"));
  EXPECT_TRUE(scanner.Match("34 "));
  EXPECT_FALSE(scanner.MatchEof());
  EXPECT_FALSE(scanner.Match("5 "));
  EXPECT_TRUE(scanner.Match("5"));
  EXPECT_TRUE(scanner.MatchEof());
}

TEST(Scanner, Whitespace) {
  Scanner scanner(" \t\n\r 1\t2\n\n");

  EXPECT_FALSE(scanner.Match(" "));
  EXPECT_TRUE(scanner.Match("1"));
  EXPECT_TRUE(scanner.Match("2"));
  EXPECT_TRUE(scanner.MatchEof());
  EXPECT_TRUE(scanner.ok());
}

TEST(Scanner, Fail) {
  Scanner scanner("153 4q");

  scanner.Expect("5");
  EXPECT_FALSE(scanner.ok());
  EXPECT_FALSE(scanner.status().ok());

  EXPECT_TRUE(scanner.MatchEof());
}

TEST(Scanner, Identifier) {
  Scanner scanner("1 q1  _1_ _1a= qqb");

  string identifier = "foo";
  EXPECT_FALSE(scanner.MatchReadIdentifier(&identifier));
  EXPECT_EQ(identifier, "foo");
  scanner.Match("1");

  EXPECT_TRUE(scanner.MatchReadIdentifier(&identifier));
  EXPECT_EQ(identifier, "q1");

  scanner.ExpectIdentifier("_1_");
  EXPECT_TRUE(scanner.ok());

  scanner.ExpectIdentifier("_1a");
  EXPECT_TRUE(scanner.ok());

  // The = after _1a is not included in the identifier.
  scanner.Expect("=");

  // The expected identifier matches a prefix but is not the full identifier in
  // the input.
  EXPECT_TRUE(scanner.ok());
  scanner.ExpectIdentifier("qq");
  EXPECT_FALSE(scanner.ok());
}

TEST(Scanner, Int) {
  Scanner scanner("1_2 3% -1 124345 -363 0 -0");
  EXPECT_EQ(1, scanner.ReadInt());
  EXPECT_TRUE(scanner.Match("_"));
  EXPECT_EQ(2, scanner.ReadInt());
  EXPECT_EQ(3, scanner.ReadInt());
  EXPECT_TRUE(scanner.Match("%"));
  EXPECT_EQ(-1, scanner.ReadInt());
  EXPECT_EQ(124345, scanner.ReadInt());
  EXPECT_EQ(-363, scanner.ReadInt());
  EXPECT_EQ(0, scanner.ReadInt());
  EXPECT_EQ(0, scanner.ReadInt());
  EXPECT_TRUE(scanner.MatchEof());
}

TEST(Scanner, IntVector) {
  Scanner scanner("()(0) (-1,2) ( 3 , 4 )");
  EXPECT_THAT(scanner.ReadIntVector(), testing::IsEmpty());
  EXPECT_THAT(scanner.ReadIntVector(), testing::ElementsAre(0));
  EXPECT_THAT(scanner.ReadIntVector(), testing::ElementsAre(-1, 2));
  EXPECT_THAT(scanner.ReadIntVector(), testing::ElementsAre(3, 4));
  EXPECT_TRUE(scanner.MatchEof());
  EXPECT_TRUE(scanner.ok());
}

}  // namespace
}  // namespace xla
#endif
