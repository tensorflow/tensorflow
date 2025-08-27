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
  EXPECT_TRUE(SymbolicMap::Get(&ctx, 0, 0, {}).IsEmpty());
  EXPECT_TRUE(SymbolicMap::Get(&ctx, 2, 1, {}).IsEmpty());
  EXPECT_FALSE(SymbolicMap::Get(&ctx, 1, 0, {ctx.CreateVariable(0)}).IsEmpty());
}

TEST(SymbolicMapTest, IsIdentity) {
  SymbolicExprContext ctx;

  SymbolicMap true_identity = SymbolicMap::Get(
      &ctx, 2, 0, {ctx.CreateVariable(0), ctx.CreateVariable(1)});
  EXPECT_TRUE(true_identity.IsIdentity());

  SymbolicMap true_identity_with_symbols = SymbolicMap::Get(
      &ctx, 2, 1, {ctx.CreateVariable(0), ctx.CreateVariable(1)});
  EXPECT_TRUE(true_identity_with_symbols.IsIdentity());

  SymbolicMap few_results =
      SymbolicMap::Get(&ctx, 2, 0, {ctx.CreateVariable(0)});
  EXPECT_FALSE(few_results.IsIdentity());

  SymbolicMap too_many_results = SymbolicMap::Get(
      &ctx, 1, 0, {ctx.CreateVariable(0), ctx.CreateVariable(1)});
  EXPECT_FALSE(too_many_results.IsIdentity());

  SymbolicMap wrong_expr_type = SymbolicMap::Get(
      &ctx, 2, 0, {ctx.CreateVariable(0), ctx.CreateConstant(1)});
  EXPECT_FALSE(wrong_expr_type.IsIdentity());

  SymbolicMap unordered_variable_id = SymbolicMap::Get(
      &ctx, 2, 0, {ctx.CreateVariable(1), ctx.CreateVariable(0)});
  EXPECT_FALSE(unordered_variable_id.IsIdentity());
}

TEST(SymbolicMapTest, GetConstantResults) {
  SymbolicExprContext ctx;

  SymbolicMap all_constants_map = SymbolicMap::Get(
      &ctx, 0, 0, {ctx.CreateConstant(5), ctx.CreateConstant(10)});
  EXPECT_TRUE(all_constants_map.IsConstant());
  EXPECT_THAT(all_constants_map.GetConstantResults(), ElementsAre(5, 10));

  SymbolicMap mixed_map = SymbolicMap::Get(
      &ctx, 1, 0, {ctx.CreateConstant(5), ctx.CreateVariable(0)});
  EXPECT_FALSE(mixed_map.IsConstant());
  EXPECT_DEATH(mixed_map.GetConstantResults(),
               "Cannot get constant results from a non-constant map");

  SymbolicMap no_results_map = SymbolicMap::Get(&ctx, 0, 0, {});
  EXPECT_TRUE(no_results_map.IsConstant());
  EXPECT_THAT(no_results_map.GetConstantResults(), ElementsAre());
}

TEST(SymbolicMapTest, ReplaceDimsAndSymbols) {
  SymbolicExprContext ctx;
  SymbolicExpr d0 = ctx.CreateVariable(0);
  SymbolicExpr d1 = ctx.CreateVariable(1);
  SymbolicExpr s0 = ctx.CreateVariable(2);
  SymbolicExpr s1 = ctx.CreateVariable(3);
  SymbolicExpr c1 = ctx.CreateConstant(10);
  SymbolicExpr c2 = ctx.CreateConstant(20);
  SymbolicExpr c3 = ctx.CreateConstant(30);

  SymbolicMap map_basic = SymbolicMap::Get(&ctx, 2, 2, {d0 + s0, d1 * s1});
  SymbolicMap replaced_basic = map_basic.ReplaceDimsAndSymbols(
      {c1, c2}, {c3, d0}, map_basic.GetNumDims(), map_basic.GetNumSymbols());
  EXPECT_THAT(replaced_basic.GetResults(), ElementsAre(c1 + c3, c2 * d0));

  SymbolicMap map_empty = SymbolicMap::Get(&ctx, 0, 0, {});
  SymbolicMap replaced_empty = map_empty.ReplaceDimsAndSymbols({}, {}, 0, 0);
  EXPECT_TRUE(replaced_empty.IsEmpty());

  SymbolicMap map_change_dims = SymbolicMap::Get(&ctx, 1, 1, {d0 + s0 * c2});
  // Replacements in the context of the NEW map (2 dims, 1 symbol)
  SymbolicExpr new_d0 = ctx.CreateVariable(0);
  SymbolicExpr new_d1 = ctx.CreateVariable(1);
  SymbolicExpr new_s0 = ctx.CreateVariable(2);
  SymbolicMap replaced_change_dims = map_change_dims.ReplaceDimsAndSymbols(
      {new_d0 * c1 + new_d1}, {new_s0}, 2, 1);
  EXPECT_EQ(replaced_change_dims.GetNumDims(), 2);
  EXPECT_EQ(replaced_change_dims.GetNumSymbols(), 1);
  EXPECT_THAT(replaced_change_dims.GetResults(),
              ElementsAre((new_d0 * c1 + new_d1) + new_s0 * c2));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
