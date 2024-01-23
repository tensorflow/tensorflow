/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/ef57.h"

#include <cmath>
#include <limits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log_streamer.h"
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "xla/test.h"

namespace xla {
namespace {

TEST(Ef57Test, DoubleMax) {
  // Overflowing the F32 exponent in SplitF64ToF32 should result in a pair of
  // [∞,0].
  auto [high, low] = SplitF64ToF32(std::numeric_limits<double>::max());
  EXPECT_EQ(high, std::numeric_limits<float>::infinity());
  EXPECT_EQ(low, 0.0f);
}

TEST(Ef57Test, Overflow) {
  auto [high, low] = SplitF64ToF32(0x1.ffffffp+127);
  EXPECT_EQ(high, std::numeric_limits<float>::infinity());
  EXPECT_EQ(low, 0.0f);
}

TEST(Ef57Test, CheckPrecision) {
  auto [high, low] = SplitF64ToF32(2.0 - 0x1p-52);
  EXPECT_EQ(high, 2.0f);
  EXPECT_EQ(low, -0x1p-52f);
}

TEST(Ef57Test, SimpleArray) {
  std::vector<double> inputs(127);

  absl::BitGen gen;
  for (double& input : inputs) {
    input = absl::Uniform<float>(gen, 0.0f, 1.0f);
  }

  std::vector<float> outputs(inputs.size() * 2);
  ConvertF64ToEf57(inputs, absl::MakeSpan(outputs));
  for (int i = 0; i < inputs.size(); ++i) {
    EXPECT_EQ(outputs[i * 2], inputs[i]);
    EXPECT_EQ(outputs[i * 2 + 1], 0.0f);
  }
}

TEST(Ef57Test, RelativeSplit) {
  const float distance = std::scalbnf(1.0f, std::numeric_limits<float>::digits);
  std::vector<double> inputs(127);

  absl::BitGen gen;
  for (double& input : inputs) {
    input = absl::Uniform<double>(gen, 0.0, 1.0);
  }

  std::vector<float> outputs(inputs.size() * 2);
  ConvertF64ToEf57(inputs, absl::MakeSpan(outputs));
  for (int i = 0; i < outputs.size(); i += 2) {
    auto most_significant = outputs[i];
    auto least_significant = outputs[i + 1];
    auto most_significant_mag = std::fabs(most_significant);
    auto least_significant_mag = std::fabs(least_significant);
    EXPECT_FALSE(std::isnan(most_significant_mag));
    if (most_significant_mag == 0.0f) {
      EXPECT_EQ(least_significant_mag, 0.0f);
    } else {
      EXPECT_GT(most_significant_mag, least_significant_mag * distance);
    }
  }
}

}  // namespace
}  // namespace xla
