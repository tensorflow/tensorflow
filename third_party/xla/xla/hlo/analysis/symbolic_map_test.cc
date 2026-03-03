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

#include "xla/hlo/analysis/symbolic_map.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;

struct SymbolicMapTest : public ::testing::Test {
  static constexpr int kSampleDims = 2;
  static constexpr int kSampleSymbols = 2;
  SymbolicMapTest() {
    RegisterSymbolicExprStorage(&ctx);
    d0 = CreateDimExpr(0, &ctx);
    d1 = CreateDimExpr(1, &ctx);
    s0 = CreateSymbolExpr(0, kSampleDims, &ctx);
    s1 = CreateSymbolExpr(1, kSampleDims, &ctx);
    s2 = CreateSymbolExpr(2, kSampleDims, &ctx);
    c2 = CreateSymbolicConstant(2, &ctx);
    c10 = CreateSymbolicConstant(10, &ctx);
    sample_map =
        SymbolicMap::Get(&ctx, kSampleDims, kSampleSymbols, {d0 + s0, d1 * s1});
  }

  mlir::MLIRContext ctx;
  SymbolicExpr d0;
  SymbolicExpr d1;
  SymbolicExpr s0;
  SymbolicExpr s1;
  SymbolicExpr s2;
  SymbolicExpr c2;
  SymbolicExpr c10;
  SymbolicMap sample_map;
};

TEST_F(SymbolicMapTest, GetSymbolAndDimExpressions) {
  EXPECT_EQ(sample_map.GetSymbolExpression(0), s0);
  EXPECT_EQ(sample_map.GetSymbolExpression(1), s1);
  EXPECT_EQ(sample_map.GetDimExpression(0), d0);
  EXPECT_EQ(sample_map.GetDimExpression(1), d1);
}

TEST_F(SymbolicMapTest, ToString) {
  EXPECT_EQ(sample_map.ToString(),
            "(d0, d1)[s0, s1] -> ((d0 + s0), (d1 * s1))");

  SymbolicMap empty_map = SymbolicMap::Get(&ctx, 0, 0, {});
  EXPECT_EQ(empty_map.ToString(), "()[] -> ()");

  SymbolicMap dims_only = SymbolicMap::Get(&ctx, kSampleDims, 0, {d0, d1});
  EXPECT_EQ(dims_only.ToString(), "(d0, d1)[] -> (d0, d1)");

  SymbolicExpr s0_no_dims =
      CreateSymbolExpr(/*symbol_id=*/0, /*num_dims=*/0, &ctx);
  SymbolicExpr s1_no_dims =
      CreateSymbolExpr(/*symbol_id=*/1, /*num_dims=*/0, &ctx);
  SymbolicMap symbols_only =
      SymbolicMap::Get(&ctx, 0, kSampleSymbols, {s0_no_dims, s1_no_dims});
  EXPECT_EQ(symbols_only.ToString(), "()[s0, s1] -> (s0, s1)");
}

TEST_F(SymbolicMapTest, IsEmpty) {
  EXPECT_TRUE(SymbolicMap::Get(&ctx, 0, 0, {}).IsEmpty());
  EXPECT_TRUE(SymbolicMap::Get(&ctx, 2, 1, {}).IsEmpty());
  EXPECT_FALSE(
      SymbolicMap::Get(&ctx, 1, 0, {CreateDimExpr(0, &ctx)}).IsEmpty());
}

TEST_F(SymbolicMapTest, IsIdentity) {
  SymbolicMap true_identity = SymbolicMap::Get(
      &ctx, 2, 0, {CreateDimExpr(0, &ctx), CreateDimExpr(1, &ctx)});
  EXPECT_TRUE(true_identity.IsIdentity());

  SymbolicMap true_identity_with_symbols = SymbolicMap::Get(
      &ctx, 2, 1, {CreateDimExpr(0, &ctx), CreateDimExpr(1, &ctx)});
  EXPECT_TRUE(true_identity_with_symbols.IsIdentity());

  SymbolicMap few_results =
      SymbolicMap::Get(&ctx, 2, 0, {CreateDimExpr(0, &ctx)});
  EXPECT_FALSE(few_results.IsIdentity());

  SymbolicMap too_many_results = SymbolicMap::Get(
      &ctx, 1, 0, {CreateDimExpr(0, &ctx), CreateDimExpr(1, &ctx)});
  EXPECT_FALSE(too_many_results.IsIdentity());

  SymbolicMap wrong_expr_type = SymbolicMap::Get(
      &ctx, 2, 0, {CreateDimExpr(0, &ctx), CreateSymbolicConstant(1, &ctx)});
  EXPECT_FALSE(wrong_expr_type.IsIdentity());

  SymbolicMap unordered_variable_id = SymbolicMap::Get(
      &ctx, 2, 0, {CreateDimExpr(1, &ctx), CreateDimExpr(0, &ctx)});
  EXPECT_FALSE(unordered_variable_id.IsIdentity());
}

TEST_F(SymbolicMapTest, GetConstantResults) {
  SymbolicMap all_constants_map = SymbolicMap::Get(
      &ctx, 0, 0,
      {CreateSymbolicConstant(5, &ctx), CreateSymbolicConstant(10, &ctx)});
  EXPECT_TRUE(all_constants_map.IsConstant());
  EXPECT_THAT(all_constants_map.GetConstantResults(), ElementsAre(5, 10));

  SymbolicMap mixed_map = SymbolicMap::Get(
      &ctx, 1, 0,
      {CreateSymbolicConstant(5, &ctx), CreateSymbolicVariable(0, &ctx)});
  EXPECT_FALSE(mixed_map.IsConstant());
  EXPECT_DEATH(mixed_map.GetConstantResults(),
               "Cannot get constant results from a non-constant map");

  SymbolicMap no_results_map = SymbolicMap::Get(&ctx, 0, 0, {});
  EXPECT_TRUE(no_results_map.IsConstant());
  EXPECT_THAT(no_results_map.GetConstantResults(), ElementsAre());
}

TEST_F(SymbolicMapTest, ReplaceDimsAndSymbols) {
  SymbolicExpr c3 = CreateSymbolicConstant(30, &ctx);

  SymbolicMap replaced_basic = sample_map.ReplaceDimsAndSymbols(
      {d1, c2}, {c3, d0}, sample_map.GetNumDims(), sample_map.GetNumSymbols());
  EXPECT_THAT(replaced_basic.GetResults(), ElementsAre(d1 + c3, c2 * d0));

  SymbolicMap map_empty = SymbolicMap::Get(&ctx, 0, 0, {});
  SymbolicMap replaced_empty = map_empty.ReplaceDimsAndSymbols({}, {}, 0, 0);
  EXPECT_TRUE(replaced_empty.IsEmpty());

  SymbolicMap map_change_dims = SymbolicMap::Get(&ctx, 1, 1, {d0 + s0 * c2});
  // Replacements in the context of the NEW map (2 dims, 1 symbol)
  SymbolicExpr new_d0 = CreateDimExpr(0, &ctx);
  SymbolicExpr new_d1 = CreateDimExpr(1, &ctx);
  SymbolicExpr new_s0 = CreateSymbolExpr(/*symbol_id=*/0, /*num_dims=*/2, &ctx);
  SymbolicMap replaced_change_dims = map_change_dims.ReplaceDimsAndSymbols(
      {new_d0 * c10 + new_d1}, {new_s0}, 2, 1);
  EXPECT_EQ(replaced_change_dims.GetNumDims(), 2);
  EXPECT_EQ(replaced_change_dims.GetNumSymbols(), 1);
  EXPECT_THAT(replaced_change_dims.GetResults(),
              ElementsAre((new_d0 * c10 + new_d1) + new_s0 * c2));
}

TEST_F(SymbolicMapTest, ReplaceDimsAndSymbolsOnlyDims) {
  SymbolicMap replaced = sample_map.ReplaceDimsAndSymbols(
      /*dim_replacements=*/{c10, c2}, /*sym_replacements=*/{},
      sample_map.GetNumDims(), sample_map.GetNumSymbols());
  EXPECT_THAT(replaced.GetResults(), ElementsAre(c10 + s0, c2 * s1));
}

TEST_F(SymbolicMapTest, ReplaceDimsAndSymbolsOnlySymbols) {
  SymbolicMap replaced = sample_map.ReplaceDimsAndSymbols(
      /*dim_replacements=*/{}, /*sym_replacements=*/{c10, c2},
      sample_map.GetNumDims(), sample_map.GetNumSymbols());
  EXPECT_THAT(replaced.GetResults(), ElementsAre(d0 + c10, d1 * c2));
}

TEST_F(SymbolicMapTest, Compose) {
  // Composition without Symbols
  SymbolicMap map1_no_symbols = SymbolicMap::Get(&ctx, 1, 0, {d0 * 2});
  SymbolicMap map2_no_symbols = SymbolicMap::Get(&ctx, 1, 0, {d0 + 5});
  SymbolicMap composed_no_symbols = map1_no_symbols.Compose(map2_no_symbols);
  EXPECT_THAT(composed_no_symbols.GetResults(), ElementsAre((d0 + 5) * 2));

  // Composition with Symbols
  SymbolicExpr s0_map1 =
      CreateSymbolExpr(/*symbol_id=*/0, /*num_dims=*/2, &ctx);
  SymbolicExpr s0_map2 =
      CreateSymbolExpr(/*symbol_id=*/0, /*num_dims=*/1, &ctx);
  SymbolicMap map1_symbols =
      SymbolicMap::Get(&ctx, 2, 1, {d0 + s0_map1, d1 * 2});
  SymbolicMap map2_symbols =
      SymbolicMap::Get(&ctx, 1, 1, {d0 - 10, d0 + s0_map2});
  SymbolicMap compose_with_symbols = map1_symbols.Compose(map2_symbols);
  EXPECT_EQ(compose_with_symbols.GetNumDims(), 1);
  EXPECT_EQ(compose_with_symbols.GetNumSymbols(), 2);
  SymbolicExpr new_d0 = d0;
  SymbolicExpr new_s0_map1 =
      CreateSymbolExpr(/*symbol_id=*/0, /*num_dims=*/1, &ctx);
  SymbolicExpr new_s0_map2 =
      CreateSymbolExpr(/*symbol_id=*/1, /*num_dims=*/1, &ctx);
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
  // The composed map has 2 dims and 2 symbols:
  //    d0 and d1 (from map1_symbols)
  //    s0 (from id_2dim_1sym) and s0 (from map1_symbols)
  // The reindexed symbol from map1_symbols is the second symbol in the composed
  // map.
  SymbolicExpr reindexed_map1_s0 =
      CreateSymbolExpr(/*symbol_id=*/1, /*num_dims=*/2, &ctx);
  EXPECT_THAT(compose_left_with_id2dim_1sym.GetResults(),
              ElementsAre(d0 + reindexed_map1_s0, d1 * 2));
}

TEST_F(SymbolicMapTest, Replace) {
  SymbolicExpr c5 = CreateSymbolicConstant(5, &ctx);

  SymbolicExpr expr0 = (d0 + c2) * d1;
  SymbolicExpr expr1 = d1 + c2;
  SymbolicMap map = SymbolicMap::Get(&ctx, 2, 0, {expr0, expr1});

  SymbolicMap replaced_both_exprs = map.Replace(c2, d0);
  EXPECT_THAT(replaced_both_exprs.GetResults(),
              ElementsAre((d0 + d0) * d1, d1 + d0));

  SymbolicMap replaced_just_one = map.Replace(d1 + c2, c5);
  EXPECT_THAT(replaced_just_one.GetResults(), ElementsAre(expr0, c5));

  SymbolicMap no_replacement_map =
      map.Replace(CreateSymbolicVariable(99, &ctx), c5);
  EXPECT_EQ(no_replacement_map, map);
}

TEST_F(SymbolicMapTest, ReplaceWithMap) {
  SymbolicExpr expr = (d0 + 1) * (d1 + 2);
  SymbolicMap map = SymbolicMap::Get(&ctx, 2, 0, {expr});

  llvm::DenseMap<SymbolicExpr, SymbolicExpr> replacements;
  replacements[d0 + 1] = c10;
  replacements[d1] = d0;
  SymbolicMap replaced1 = map.Replace(replacements, 1, 0);
  EXPECT_THAT(replaced1.GetResults(), ElementsAre(c10 * (d0 + 2)));
  EXPECT_EQ(replaced1.GetNumDims(), 1);
}

TEST_F(SymbolicMapTest, GetUnusedVariables) {
  [[maybe_unused]] SymbolicExpr d2 = CreateDimExpr(2, &ctx);
  // d2 is unused.
  [[maybe_unused]] SymbolicExpr s0_3dims =
      CreateSymbolExpr(/*symbol_id=*/0, /*num_dims=*/3, &ctx);
  SymbolicExpr s1_3dims =
      CreateSymbolExpr(/*symbol_id=*/1, /*num_dims=*/3, &ctx);

  // Map with used and unused dims and symbols.
  SymbolicMap map = SymbolicMap::Get(&ctx, 3, 2, {d0 + s1_3dims, d1 * c2});

  llvm::SmallBitVector unused_dims = GetUnusedDimensionsBitVector(map);
  EXPECT_EQ(unused_dims.size(), 3);
  EXPECT_FALSE(unused_dims[0]);  // d0 is used
  EXPECT_FALSE(unused_dims[1]);  // d1 is used
  EXPECT_TRUE(unused_dims[2]);   // d2 is unused

  llvm::SmallBitVector unused_symbols = GetUnusedSymbolsBitVector(map);
  EXPECT_EQ(unused_symbols.size(), 2);
  EXPECT_TRUE(unused_symbols[0]);   // s0 is unused
  EXPECT_FALSE(unused_symbols[1]);  // s1 is used

  // Empty map
  SymbolicMap empty_map = SymbolicMap::Get(&ctx, 2, 1, {});
  llvm::SmallBitVector empty_dims = GetUnusedDimensionsBitVector(empty_map);
  EXPECT_EQ(empty_dims.size(), 2);
  EXPECT_TRUE(empty_dims.all());
  llvm::SmallBitVector empty_symbols = GetUnusedSymbolsBitVector(empty_map);
  EXPECT_EQ(empty_symbols.size(), 1);
  EXPECT_TRUE(empty_symbols.all());

  // Map with only dims
  SymbolicMap no_symbols_map = SymbolicMap::Get(&ctx, 2, 0, {d0 - d1});
  llvm::SmallBitVector no_sym_dims =
      GetUnusedDimensionsBitVector(no_symbols_map);
  EXPECT_EQ(no_sym_dims.size(), 2);
  EXPECT_FALSE(no_sym_dims[0]);
  EXPECT_FALSE(no_sym_dims[1]);
  llvm::SmallBitVector no_sym_symbols =
      GetUnusedSymbolsBitVector(no_symbols_map);
  EXPECT_EQ(no_sym_symbols.size(), 0);

  // Map with only symbols
  s0 = CreateSymbolExpr(/*symbol_id=*/0, /*num_dims=*/0, &ctx);
  s1 = CreateSymbolExpr(/*symbol_id=*/1, /*num_dims=*/0, &ctx);
  SymbolicMap no_dims_map = SymbolicMap::Get(&ctx, 0, 2, {s0 * s1});
  llvm::SmallBitVector no_dim_dims = GetUnusedDimensionsBitVector(no_dims_map);
  EXPECT_EQ(no_dim_dims.size(), 0);
  llvm::SmallBitVector no_dim_symbols = GetUnusedSymbolsBitVector(no_dims_map);
  EXPECT_EQ(no_dim_symbols.size(), 2);
  EXPECT_FALSE(no_dim_symbols[0]);
  EXPECT_FALSE(no_dim_symbols[1]);
}

TEST_F(SymbolicMapTest, CompressDims) {
  SymbolicExpr d0 = CreateDimExpr(0, &ctx);
  [[maybe_unused]] SymbolicExpr d1 = CreateDimExpr(1, &ctx);  // Unused
  SymbolicExpr d2 = CreateDimExpr(2, &ctx);
  SymbolicExpr s0 = CreateSymbolExpr(/*symbol_id=*/0, /*num_dims=*/3, &ctx);

  // Map: (d0, d1, d2)[s0] -> {d0 + d2, s0 * 5}
  SymbolicMap map = SymbolicMap::Get(&ctx, 3, 1, {d0 + d2, s0 * 5});

  // Remove d1
  llvm::SmallBitVector unused_dims = GetUnusedDimensionsBitVector(map);
  SymbolicMap compressed = CompressDims(map, unused_dims);

  EXPECT_EQ(compressed.GetNumDims(), 2);
  EXPECT_EQ(compressed.GetNumSymbols(), 1);

  SymbolicExpr new_d0 = CreateDimExpr(0, &ctx);
  SymbolicExpr new_d1 = CreateDimExpr(1, &ctx);
  SymbolicExpr new_s0 = CreateSymbolExpr(/*symbol_id=*/0, /*num_dims=*/2, &ctx);
  EXPECT_THAT(compressed.GetResults(),
              ElementsAre(new_d0 + new_d1, new_s0 * 5));

  // Check that we can't remove used dimensions.
  unused_dims.reset();
  unused_dims[0] = true;
  EXPECT_DEATH(CompressDims(map, unused_dims),
               "Attempting to compress a used dimension: 0");
}

TEST_F(SymbolicMapTest, CompressSymbols) {
  SymbolicExpr d0 = CreateDimExpr(0, &ctx);
  SymbolicExpr s0 = CreateSymbolExpr(/*symbol_id=*/0, /*num_dims=*/1, &ctx);
  [[maybe_unused]] SymbolicExpr s1 =
      CreateSymbolExpr(/*symbol_id=*/1, /*num_dims=*/1, &ctx);  // Unused
  SymbolicExpr s2 = CreateSymbolExpr(/*symbol_id=*/2, /*num_dims=*/1, &ctx);

  // Map: (d0)[s0, s1, s2] -> {d0 + s2, s0 * 5}
  SymbolicMap map = SymbolicMap::Get(&ctx, 1, 3, {d0 + s2, s0 * 5});

  // Remove s1 (the only unused symbol)
  llvm::SmallBitVector unused_symbols = GetUnusedSymbolsBitVector(map);
  SymbolicMap compressed = CompressSymbols(map, unused_symbols);

  EXPECT_EQ(compressed.GetNumDims(), 1);
  EXPECT_EQ(compressed.GetNumSymbols(), 2);

  SymbolicExpr new_d0 = CreateDimExpr(0, &ctx);
  SymbolicExpr new_s0 = CreateSymbolExpr(/*symbol_id=*/0, /*num_dims=*/1, &ctx);
  SymbolicExpr new_s1 =
      CreateSymbolExpr(/*symbol_id=*/1, /*num_dims=*/1, &ctx);  // Original s2
  EXPECT_THAT(compressed.GetResults(),
              ElementsAre(new_d0 + new_s1, new_s0 * 5));

  // Check that we can't remove used symbols.
  unused_symbols.reset();
  unused_symbols[2] = true;
  EXPECT_DEATH(CompressSymbols(map, unused_symbols),
               "Attempting to compress a used symbol: 2");
}

TEST_F(SymbolicMapTest, Hashing) {
  absl::flat_hash_set<SymbolicMap> set;

  SymbolicExpr d0 = CreateDimExpr(0, &ctx);
  SymbolicExpr d1 = CreateDimExpr(1, &ctx);
  SymbolicExpr s0 = CreateSymbolExpr(/*symbol_id=*/0, /*num_dims=*/2, &ctx);
  SymbolicExpr c42 = CreateSymbolicConstant(42, &ctx);
  SymbolicExpr c99 = CreateSymbolicConstant(99, &ctx);

  SymbolicMap map1 = SymbolicMap::Get(&ctx, 2, 1, {d0 + s0, d1 * c42});
  SymbolicMap map2 = SymbolicMap::Get(&ctx, 2, 1, {d0 + s0, d1 * c42});
  SymbolicMap map3 = SymbolicMap::Get(&ctx, 2, 1, {d0 + s0, d1 * c99});

  set.insert(map1);
  EXPECT_EQ(set.size(), 1);
  set.insert(map2);
  EXPECT_EQ(set.size(), 1);
  set.insert(map3);
  EXPECT_EQ(set.size(), 2);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
