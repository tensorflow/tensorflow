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

TEST(SymbolicMapTest, Compose) {
  SymbolicExprContext ctx;
  SymbolicExpr d0 = ctx.CreateVariable(0);
  SymbolicExpr d1 = ctx.CreateVariable(1);

  // Composition without Symbols
  SymbolicMap map1_no_symbols = SymbolicMap::Get(&ctx, 1, 0, {d0 * 2});
  SymbolicMap map2_no_symbols = SymbolicMap::Get(&ctx, 1, 0, {d0 + 5});
  SymbolicMap composed_no_symbols = map1_no_symbols.Compose(map2_no_symbols);
  EXPECT_THAT(composed_no_symbols.GetResults(), ElementsAre((d0 + 5) * 2));

  // Composition with Symbols
  SymbolicExpr s0_map1 = ctx.CreateVariable(/*map1_dims*/ 2);
  SymbolicExpr s0_map2 = ctx.CreateVariable(/*map2_dims*/ 1);
  SymbolicMap map1_symbols =
      SymbolicMap::Get(&ctx, 2, 1, {d0 + s0_map1, d1 * 2});
  SymbolicMap map2_symbols =
      SymbolicMap::Get(&ctx, 1, 1, {d0 - 10, d0 + s0_map2});
  SymbolicMap compose_with_symbols = map1_symbols.Compose(map2_symbols);
  EXPECT_EQ(compose_with_symbols.GetNumDims(), 1);
  EXPECT_EQ(compose_with_symbols.GetNumSymbols(), 2);
  SymbolicExpr new_d0 = d0;
  SymbolicExpr new_s0_map1 = ctx.CreateVariable(/*compose_dims*/ 1);
  SymbolicExpr new_s0_map2 =
      ctx.CreateVariable(/*compose_dims + map1_symbols.GetNumSymbols()*/ 2);
  EXPECT_THAT(
      compose_with_symbols.GetResults(),
      ElementsAre((new_d0 - 10) + new_s0_map1, (new_d0 + new_s0_map2) * 2));

  // Composition with identity
  SymbolicMap id_2dim = SymbolicMap::Get(&ctx, 2, 0, {d0, d1});
  EXPECT_EQ(map1_symbols, map1_symbols.Compose(id_2dim));

  SymbolicMap id_2dim_1sym = SymbolicMap::Get(&ctx, 2, 1, {d0, d1});
  SymbolicMap compose_with_id2dim_1sym = map1_symbols.Compose(id_2dim_1sym);
  EXPECT_EQ(compose_with_id2dim_1sym.GetNumSymbols(), 2);
  EXPECT_EQ(compose_with_id2dim_1sym.GetNumDims(), map1_symbols.GetNumDims());
  EXPECT_EQ(compose_with_id2dim_1sym.GetResults(), map1_symbols.GetResults());

  SymbolicMap compose_left_with_id2dim_1sym =
      id_2dim_1sym.Compose(map1_symbols);
  EXPECT_EQ(compose_left_with_id2dim_1sym.GetNumDims(), 2);
  EXPECT_EQ(compose_left_with_id2dim_1sym.GetNumSymbols(), 2);
  SymbolicExpr reindexed_map1_s0 =
      ctx.CreateVariable(compose_left_with_id2dim_1sym.GetNumDims() +
                         id_2dim_1sym.GetNumSymbols());
  EXPECT_THAT(compose_left_with_id2dim_1sym.GetResults(),
              ElementsAre(d0 + reindexed_map1_s0, d1 * 2));
}

TEST(SymbolicMapTest, Replace) {
  SymbolicExprContext ctx;
  SymbolicExpr d0 = ctx.CreateVariable(0);
  SymbolicExpr d1 = ctx.CreateVariable(1);
  SymbolicExpr c2 = ctx.CreateConstant(2);
  SymbolicExpr c5 = ctx.CreateConstant(5);

  SymbolicExpr expr0 = (d0 + c2) * d1;
  SymbolicExpr expr1 = d1 + c2;
  SymbolicMap map = SymbolicMap::Get(&ctx, 2, 0, {expr0, expr1});

  SymbolicMap replaced_both_exprs = map.Replace(c2, d0);
  EXPECT_THAT(replaced_both_exprs.GetResults(),
              ElementsAre((d0 + d0) * d1, d1 + d0));

  SymbolicMap replaced_just_one = map.Replace(d1 + c2, c5);
  EXPECT_THAT(replaced_just_one.GetResults(), ElementsAre(expr0, c5));

  SymbolicMap no_replacement_map = map.Replace(ctx.CreateVariable(99), c5);
  EXPECT_EQ(no_replacement_map, map);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
