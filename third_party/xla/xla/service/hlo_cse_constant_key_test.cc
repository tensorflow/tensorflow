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

#include "xla/service/hlo_cse_constant_key.h"

#include <array>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

template <typename kIsLayoutSensitive>
class HloCseConstantKeyTest : public ::testing::Test {
 public:
  using ConstantKey = CseConstantKey<kIsLayoutSensitive::value>;
};

class NameGenerator {
 public:
  template <typename T>
  static std::string GetName(int) {
    if constexpr (T::value) {
      return "layout_sensitive";
    }
    return "not_layout_sensitive";
  }
};

using HloCseConstantKeyTestTypes =
    ::testing::Types<std::true_type, std::false_type>;
TYPED_TEST_SUITE(HloCseConstantKeyTest, HloCseConstantKeyTestTypes,
                 NameGenerator);

TYPED_TEST(HloCseConstantKeyTest, VerifyAbslHashIsImplementedCorrectly) {
  using ConstantKey = typename TestFixture::ConstantKey;

  // Values that we will combine to create the constant keys to check.
  // Combinations need not make inherent sense. We just want to check that
  // AbslHash is implemented correctly.
  const std::array<Literal, 3> literals = {
      LiteralUtil::CreateR3WithLayout({{{0.0f}}},
                                      LayoutUtil::MakeLayout({2, 1, 0})),
      LiteralUtil::CreateR3WithLayout({{{0.0f}}},
                                      LayoutUtil::MakeLayout({1, 2, 0})),
      LiteralUtil::CreateR3WithLayout({{{-0.0f}}},
                                      LayoutUtil::MakeLayout({1, 2, 0})),
  };
  const std::array<std::optional<Shape>, 2> shapes = {
      std::nullopt,  // nullopt represents the scenario where we take the shape
                     // of the literal rather than emulating the HLO constant
                     // instruction's shape.
      ShapeUtil::MakeShape(F32, {3, 4, 5}),
  };
  constexpr std::array<int, 3> kDomains = {0, 1, 2};

  std::vector<ConstantKey> keys;
  keys.reserve(literals.size() * kDomains.size() * shapes.size());
  for (const Literal& literal : literals) {
    for (const std::optional<Shape>& shape : shapes) {
      for (const int domain : kDomains) {
        keys.push_back(ConstantKey{
            literal, shape.has_value() ? *shape : literal.shape(), domain});
      }
    }
  }
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(keys));
}

}  // namespace
}  // namespace xla
