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

#include "xla/compact_string.h"

#include <string>

#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/test.h"

namespace xla {

TEST(CompactString, Simple) {
  CompactString s;
  EXPECT_EQ(s.size(), 0);
  EXPECT_EQ(s.view(), "");
  s.set("hello");
  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s.view(), "hello");
  s.set("second assignment");
  EXPECT_EQ(s.size(), strlen("second assignment"));
  EXPECT_EQ(s.view(), "second assignment");
  CompactString s2 = s;
  EXPECT_EQ(s2.view(), s.view());
  EXPECT_NE(s2.view().data(), s.view().data());  // Actually copied
}

TEST(CompactString, StringConversions) {
  std::string str("string");
  CompactString s = str;
  EXPECT_EQ(s.view(), "string");
  s = str + "more";
  EXPECT_EQ(s.view(), "stringmore");
}

TEST(CompactString, StringViewConversions) {
  absl::string_view sview("stringview");
  CompactString s = sview;
  EXPECT_EQ(s.view(), "stringview");
  s = absl::string_view("stringviewtwo");
  EXPECT_EQ(s.view(), "stringviewtwo");

  absl::string_view sview_dst = s;
  EXPECT_EQ(sview_dst, "stringviewtwo");
}

}  // namespace xla
