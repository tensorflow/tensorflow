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

#include "xla/service/gpu/model/experimental/symbolic_map.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/service/gpu/model/experimental/symbolic_expr.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;

TEST(SymbolicMapTest, IsEmpty) {
  SymbolicExprContext ctx;
  EXPECT_TRUE(SymbolicMap::get(&ctx, 0, 0, {}).isEmpty());
  EXPECT_TRUE(SymbolicMap::get(&ctx, 2, 1, {}).isEmpty());
  EXPECT_FALSE(SymbolicMap::get(&ctx, 1, 0, {ctx.CreateVariable(0)}).isEmpty());
}

TEST(SymbolicMapTest, IsIdentity) {
  SymbolicExprContext ctx;

  SymbolicMap true_identity = SymbolicMap::get(
      &ctx, 2, 0, {ctx.CreateVariable(0), ctx.CreateVariable(1)});
  EXPECT_TRUE(true_identity.isIdentity());

  SymbolicMap true_identity_with_symbols = SymbolicMap::get(
      &ctx, 2, 1, {ctx.CreateVariable(0), ctx.CreateVariable(1)});
  EXPECT_TRUE(true_identity_with_symbols.isIdentity());

  SymbolicMap too_few_results =
      SymbolicMap::get(&ctx, 2, 0, {ctx.CreateVariable(0)});
  EXPECT_FALSE(too_few_results.isIdentity());

  SymbolicMap too_many_results = SymbolicMap::get(
      &ctx, 1, 0, {ctx.CreateVariable(0), ctx.CreateVariable(1)});
  EXPECT_FALSE(too_many_results.isIdentity());

  SymbolicMap wrong_expr_type = SymbolicMap::get(
      &ctx, 2, 0, {ctx.CreateVariable(0), ctx.CreateConstant(1)});
  EXPECT_FALSE(wrong_expr_type.isIdentity());

  SymbolicMap unordered_variable_id = SymbolicMap::get(
      &ctx, 2, 0, {ctx.CreateVariable(1), ctx.CreateVariable(0)});
  EXPECT_FALSE(unordered_variable_id.isIdentity());
}

TEST(SymbolicMapTest, GetConstantResults) {
  SymbolicExprContext ctx;

  SymbolicMap all_constants_map = SymbolicMap::get(
      &ctx, 0, 0, {ctx.CreateConstant(5), ctx.CreateConstant(10)});
  EXPECT_TRUE(all_constants_map.isConstant());
  EXPECT_THAT(all_constants_map.getConstantResults(), ElementsAre(5, 10));

  SymbolicMap mixed_map = SymbolicMap::get(
      &ctx, 1, 0, {ctx.CreateConstant(5), ctx.CreateVariable(0)});
  EXPECT_FALSE(mixed_map.isConstant());
  EXPECT_DEATH(mixed_map.getConstantResults(),
               "Cannot get constant results from a non-constant map");

  SymbolicMap no_results_map = SymbolicMap::get(&ctx, 0, 0, {});
  EXPECT_TRUE(no_results_map.isConstant());
  EXPECT_THAT(no_results_map.getConstantResults(), ElementsAre());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
