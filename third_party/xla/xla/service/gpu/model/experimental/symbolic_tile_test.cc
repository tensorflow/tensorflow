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

#include "xla/service/gpu/model/experimental/symbolic_tile.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::gpu {
namespace {

using ::mlir::AffineExpr;
using ::mlir::AffineMap;

using SymbolicTileTest = HloHardwareIndependentTestBase;

TEST_F(SymbolicTileTest, StringFormat) {
  mlir::MLIRContext mlir_context;
  AffineExpr tid0, tid1, ts0, ts1, rt;
  mlir::bindDims(&mlir_context, tid0, tid1);
  mlir::bindSymbols(&mlir_context, ts0, ts1, rt);
  auto c1 = mlir::getAffineConstantExpr(1, &mlir_context);

  auto map = AffineMap::get(
      2, 3, {tid0 * ts0, rt + tid1 * ts1, ts0, ts1, c1, c1}, &mlir_context);
  ExperimentalSymbolicTile tile{map, {nullptr}};

  EXPECT_THAT(tile.ToString(), MatchIndexingString(R"(
    (tid_0, tid_1)[ts_0, ts_1]{rt_0} ->
      [tid_0 * ts_0, tid_1 * ts_1 + rt_0]
      [ts_0, ts_1]
      [1, 1]
      rt_0: nullptr
  )"));
}

}  // namespace
}  // namespace xla::gpu
