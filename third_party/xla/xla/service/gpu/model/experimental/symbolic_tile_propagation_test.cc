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

#include "xla/service/gpu/model/experimental/symbolic_tile_propagation.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/model/experimental/symbolic_tile.h"

namespace xla::gpu {
namespace {

using ::llvm::SmallVector;
using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::mlir::MLIRContext;
using ::testing::Optional;

MATCHER_P(MatchString, symbolic_tile_string, "") {
  return ExplainMatchResult(
      true, ApproximateMatch(symbolic_tile_string, arg.ToString()),
      result_listener);
}

class SymbolicTilePropagationTest : public HloHardwareIndependentTestBase {
 public:
  HloInstruction* ParseAndGetRoot(absl::string_view hlo_string) {
    auto module_or = ParseAndReturnVerifiedModule(hlo_string);
    CHECK_OK(module_or);
    module_ = std::move(module_or.value());
    return module_->entry_computation()->root_instruction();
  }

  mlir::MLIRContext mlir_context_;
  std::unique_ptr<VerifiedHloModule> module_;
};

ExperimentalSymbolicTile GetTestSymbolicTile(int64_t rank,
                                             MLIRContext* mlir_context) {
  SmallVector<AffineExpr> offsets, strides, sizes;
  offsets.reserve(rank);
  sizes.reserve(rank);
  strides.reserve(rank);
  CHECK(mlir_context);
  for (int64_t i = 0; i < rank; ++i) {
    auto tid = getAffineDimExpr(i, mlir_context);
    auto ts = getAffineSymbolExpr(i, mlir_context);
    offsets.push_back(tid * ts);
    sizes.push_back(ts);
    strides.push_back(mlir::getAffineConstantExpr(i + 1, mlir_context));
  }
  SmallVector<AffineExpr> results(offsets);
  results.append(sizes);
  results.append(strides);
  return ExperimentalSymbolicTile{
      AffineMap::get(rank, rank, results, mlir_context), {}};
}

TEST_F(SymbolicTilePropagationTest, ElementwiseOp) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = f32[10, 20] parameter(0)
      p1 = f32[10, 20] parameter(1)
      ROOT add0 = f32[10, 20] add(p0, p1)
    }
  )");
  MLIRContext mlir_context;
  std::optional<TiledOperands> tiled_operands =
      PropagateTileToInput(*root, GetTestSymbolicTile(2, &mlir_context), 0);
  EXPECT_THAT(tiled_operands, Optional(MatchString(R"(
    0) (tid_0, tid_1)[ts_0, ts_1]
      -> [tid_0 * ts_0, tid_1 * ts_1] [ts_0, ts_1] [1, 2]
    1) (tid_0, tid_1)[ts_0, ts_1]
      -> [tid_0 * ts_0, tid_1 * ts_1] [ts_0, ts_1] [1, 2]
  )")));
}

}  // namespace
}  // namespace xla::gpu
