/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/profiler/gpu/string_deduper.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace xla {
namespace profiler {
namespace {

TEST(StringDeduperTest, EmptyStringReturnsEmpty) {
  StringDeduper deduper;
  EXPECT_TRUE(deduper.Dedup("").empty());
  EXPECT_EQ(deduper.Size(), 0);
}

TEST(StringDeduperTest, DeduplicationReturnsSamePointer) {
  StringDeduper deduper;
  std::string s1 = "kernel_launch_event_tag";
  std::string s2 = "kernel_launch_event_tag";
  absl::string_view sv1 = deduper.Dedup(s1);
  absl::string_view sv2 = deduper.Dedup(s2);
  EXPECT_EQ(sv1, "kernel_launch_event_tag");
  EXPECT_EQ(sv2, "kernel_launch_event_tag");
  EXPECT_EQ(sv1.data(), sv2.data());
  EXPECT_EQ(deduper.Size(), 1);
}

TEST(StringDeduperTest, MultipleUniqueStrings) {
  StringDeduper deduper;
  absl::string_view a = deduper.Dedup("alpha");
  absl::string_view b = deduper.Dedup("beta");
  EXPECT_EQ(a, "alpha");
  EXPECT_EQ(b, "beta");
  EXPECT_NE(a.data(), b.data());
  EXPECT_EQ(deduper.Size(), 2);
}

TEST(StringDeduperTest, MaxUniqueCountLimit) {
  StringDeduper deduper;
  absl::string_view a = deduper.Dedup("alpha", /*max_unique_count=*/1);
  EXPECT_EQ(a, "alpha");
  EXPECT_EQ(deduper.Size(), 1);

  // Exceeding max count should return empty string_view
  absl::string_view b = deduper.Dedup("beta", /*max_unique_count=*/1);
  EXPECT_TRUE(b.empty());
  EXPECT_EQ(deduper.Size(), 1);

  // Existing string should still be returned
  absl::string_view a_again = deduper.Dedup("alpha", /*max_unique_count=*/1);
  EXPECT_EQ(a_again.data(), a.data());
}

TEST(StringDeduperTest, ClearResetsStorage) {
  StringDeduper deduper;
  deduper.Dedup("alpha");
  EXPECT_EQ(deduper.Size(), 1);
  deduper.Clear();
  EXPECT_EQ(deduper.Size(), 0);
}

}  // namespace
}  // namespace profiler
}  // namespace xla
