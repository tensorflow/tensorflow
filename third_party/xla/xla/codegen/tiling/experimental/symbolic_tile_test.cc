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

#include "xla/codegen/tiling/experimental/symbolic_tile.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::gpu::experimental {
namespace {

using ::mlir::AffineExpr;

using SymbolicTileTest = HloHardwareIndependentTestBase;

TilingSpace GetFakeTilingSpace(int64_t num_dims, int64_t num_rt_vars) {
  TilingSpace tiling_space;
  for (int64_t i = 0; i < num_dims; ++i) {
    tiling_space.AppendDimension(nullptr, i, 1,
                                 TilingSpace::DimensionSemantics::kParallel);
  }
  for (int64_t i = 0; i < num_rt_vars; ++i) {
    tiling_space.AppendRTVar(nullptr, i, nullptr, 1);
  }
  return tiling_space;
}

TEST_F(SymbolicTileTest, StringFormat) {
  mlir::MLIRContext mlir_context;
  AffineExpr tid0, tid1, ts0, ts1, rt;
  mlir::bindDims(&mlir_context, tid0, tid1);
  mlir::bindSymbols(&mlir_context, ts0, ts1, rt);
  auto c1 = mlir::getAffineConstantExpr(1, &mlir_context);
  auto c16 = mlir::getAffineConstantExpr(16, &mlir_context);
  auto c32 = mlir::getAffineConstantExpr(32, &mlir_context);

  TilingSpace tiling_space =
      GetFakeTilingSpace(/*num_dims=*/2, /*num_rt_vars=*/1);
  SymbolicTile tile{tiling_space,
                    {DimTile{tid0 * ts0, ts0, c1, c16},
                     DimTile{rt + tid1 * ts1, ts1, c1, c32}}};

  EXPECT_THAT(tile.ToString(), MatchIndexingString(R"(
    (tid_0, tid_1)[ts_0, ts_1]{rt_0} ->
      offsets [tid_0 * ts_0, tid_1 * ts_1 + rt_0]
      sizes [ts_0, ts_1]
      strides [1, 1]
      upper bounds [16, 32]
  )"));
}

}  // namespace
}  // namespace xla::gpu::experimental
