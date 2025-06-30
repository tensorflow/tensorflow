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

#include "xla/backends/cpu/xnn_fusion.h"

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla::cpu {
namespace {

class XnnFusionTest : public ::testing::Test {};

TEST_F(XnnFusionTest, UnaryEltwiseOpMap) {
  const absl::flat_hash_map<HloOpcode, xnn_unary_operator>* unary_map =
      GetXnnUnaryOpMap();

  auto check = [&](const HloOpcode opcode, const xnn_unary_operator expected) {
    auto result = unary_map->find(opcode);
    EXPECT_NE(result, unary_map->end());
    EXPECT_EQ(result->second, expected);
  };

  // Supported unary ops.
  check(HloOpcode::kAbs, xnn_unary_abs);
  check(HloOpcode::kExp, xnn_unary_exp);
  check(HloOpcode::kFloor, xnn_unary_floor);
  check(HloOpcode::kSqrt, xnn_unary_square_root);

  // Unsupported unary ops.
  EXPECT_EQ(unary_map->find(HloOpcode::kErf), unary_map->end());
  EXPECT_EQ(unary_map->find(HloOpcode::kSort), unary_map->end());
}

TEST_F(XnnFusionTest, BinaryEltwiseOpMap) {
  const absl::flat_hash_map<HloOpcode, xnn_binary_operator>* binary_map =
      GetXnnBinaryOpMap();

  auto check = [&](const HloOpcode opcode, const xnn_binary_operator expected) {
    auto result = binary_map->find(opcode);
    EXPECT_NE(result, binary_map->end());
    EXPECT_EQ(result->second, expected);
  };

  // Supported unary ops.
  check(HloOpcode::kAdd, xnn_binary_add);
  check(HloOpcode::kMultiply, xnn_binary_multiply);
  check(HloOpcode::kSubtract, xnn_binary_subtract);
  check(HloOpcode::kDivide, xnn_binary_divide);

  // Unsupported unary ops.
  EXPECT_EQ(binary_map->find(HloOpcode::kAtan2), binary_map->end());
  EXPECT_EQ(binary_map->find(HloOpcode::kComplex), binary_map->end());
}

}  // namespace
}  // namespace xla::cpu
