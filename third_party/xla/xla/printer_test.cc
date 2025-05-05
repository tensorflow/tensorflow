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

#include "xla/printer.h"

#include <gtest/gtest.h>
#include "xla/hlo/testlib/test.h"

namespace xla {
namespace {

TEST(HighwayHashPrinterTest, Simple) {
  HighwayHashPrinter p1;
  p1.Append("hello");

  HighwayHashPrinter p2;
  p2.Append("world");
  EXPECT_NE(p1.ToFingerprint(), p2.ToFingerprint());
}

TEST(HighwayHashPrinterTest, Empty) {
  HighwayHashPrinter p1;

  HighwayHashPrinter p2;
  p2.Append("");
  EXPECT_EQ(p1.ToFingerprint(), p2.ToFingerprint());
}

TEST(HighwayHashPrinterTest, Concat) {
  HighwayHashPrinter p1;
  p1.Append("hello");

  HighwayHashPrinter p2;
  p2.Append("hel");
  EXPECT_NE(p1.ToFingerprint(), p2.ToFingerprint());
  p2.Append("lo");
  EXPECT_EQ(p1.ToFingerprint(), p2.ToFingerprint());
}

}  // namespace
}  // namespace xla
