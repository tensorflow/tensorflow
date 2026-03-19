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

#include "xla/tsl/util/proto/parse_text_proto.h"

#include <gtest/gtest.h>
#include "xla/tsl/util/proto/proto_matchers_test_protos.pb.h"

namespace tsl::proto_testing {
namespace {

TEST(ParseTextProtoOrDieTest, ParsesValidTextProto) {
  Foo foo = ParseTextProtoOrDie<Foo>(R"pb(
    s1: "hello" i2: 42 r3: "a" r3: "b"
  )pb");
  EXPECT_EQ(foo.s1(), "hello");
  EXPECT_EQ(foo.i2(), 42);
  ASSERT_EQ(foo.r3_size(), 2);
  EXPECT_EQ(foo.r3(0), "a");
  EXPECT_EQ(foo.r3(1), "b");
}

TEST(ParseTextProtoOrDieDeathTest, DiesOnInvalidTextProto) {
  EXPECT_DEATH(ParseTextProtoOrDie<Foo>(R"pb(
                 invalid_field: "hello"
               )pb"),
               "Failed to parse text proto");
}

}  // namespace
}  // namespace tsl::proto_testing
