/* Copyright 2020 The TensorFlow Authors All Rights Reserved.

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
#include "tsl/profiler/lib/traceme_encode.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"

namespace tsl {
namespace profiler {
namespace {

TEST(TraceMeEncodeTest, NoArgTest) {
  EXPECT_EQ(TraceMeEncode("Hello!", {}), "Hello!");
}

TEST(TraceMeEncodeTest, OneArgTest) {
  EXPECT_EQ(TraceMeEncode("Hello", {{"context", "World"}}),
            "Hello#context=World#");
}

TEST(TraceMeEncodeTest, TwoArgsTest) {
  EXPECT_EQ(TraceMeEncode("Hello", {{"context", "World"}, {"request_id", 42}}),
            "Hello#context=World,request_id=42#");
}

TEST(TraceMeEncodeTest, ThreeArgsTest) {
  EXPECT_EQ(TraceMeEncode("Hello", {{"context", "World"},
                                    {"request_id", 42},
                                    {"addr", absl::Hex(0xdeadbeef)}}),
            "Hello#context=World,request_id=42,addr=deadbeef#");
}

#if !defined(PLATFORM_WINDOWS)
TEST(TraceMeEncodeTest, TemporaryStringTest) {
  EXPECT_EQ(TraceMeEncode("Hello", {{std::string("context"),
                                     absl::StrCat("World:", 2020)}}),
            "Hello#context=World:2020#");
}
#endif

// This can be removed when the absl version has been updated to include
// AbslStringify for open source builds.
#if defined(PLATFORM_GOOGLE)

struct Point {
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Point& p) {
    absl::Format(&sink, "(%d, %d)", p.x, p.y);
  }

  int x;
  int y;
};

TEST(TraceMeEncodeTest, AbslStringifyTest) {
  EXPECT_EQ(TraceMeEncode("Plot", {{"point", Point{10, 20}}}),
            "Plot#point=(10, 20)#");
}

#endif

TEST(TraceMeEncodeTest, NoNameTest) {
  EXPECT_EQ(TraceMeEncode({{"context", "World"}, {"request_id", 42}}),
            "#context=World,request_id=42#");
}

}  // namespace

void BM_TraceMeEncode(::testing::benchmark::State& state) {
  for (auto s : state) {
    TraceMeEncode(
        "MyTestEvent",
        {{"Lorem ipsum dolor sit amet", 1},
         {"consectetur adipiscing elit", 2},
         {"sed do eiusmod tempor incididunt", 3.52},
         {"ut labore et dolore magna aliqua", "Ut enim ad minim veniam"},
         {"quis nostrud exercitation ullamco", "laboris nisi ut aliquip ex"},
         {"ea commodo consequat.", 11111.1111},
         {"Duis aute", 1234567890},
         {"irure dolor in", " reprehenderit in voluptate"},
         {"velit esse cillum dolore", "eu fugiat nulla pariatur."},
         {"Excepteur sint", "occaecat cupidatat non proident, sunt in"},
         {"culpa qui officia", "deserunt mollit anim id est laborum."}});
  }
}
BENCHMARK(BM_TraceMeEncode);

}  // namespace profiler
}  // namespace tsl
