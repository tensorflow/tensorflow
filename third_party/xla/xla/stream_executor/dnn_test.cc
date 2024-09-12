/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/stream_executor/dnn.h"

#include <tuple>
#include <vector>

#include "tsl/platform/test.h"

namespace stream_executor {
namespace {

TEST(DnnTest, AlgorithmDescToString) {
  dnn::AlgorithmDesc desc(17, {{12, 1}, {1, 0}, {3, 1}}, 0);
  EXPECT_EQ(desc.ToString(), "eng17{k1=0,k3=1,k12=1}");
}

TEST(DnnTest, VersionInfoComparisonOperators) {
  std::vector<std::tuple<int, int, int>> vs;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        vs.push_back(std::make_tuple(i, j, k));
      }
    }
  }
  for (const auto& a : vs) {
    for (const auto& b : vs) {
      auto [a1, a2, a3] = a;
      auto [b1, b2, b3] = b;
      dnn::VersionInfo va(a1, a2, a3);
      dnn::VersionInfo vb(b1, b2, b3);

      EXPECT_EQ((a == b), va == vb);
      EXPECT_EQ((a != b), va != vb);
      EXPECT_EQ((a < b), va < vb);
      EXPECT_EQ((a <= b), va <= vb);
      EXPECT_EQ((a > b), va > vb);
      EXPECT_EQ((a >= b), va >= vb);
    }
  }
}

}  // namespace
}  // namespace stream_executor
