/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/codegen/tiling/tiled_hlo_fusion_instruction.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::HasSubstr;

class TiledHloFusionInstructionTest : public HloHardwareIndependentTestBase {
 public:
  TiledHloFusionInstructionTest() {
    RegisterSymbolicExprStorage(&mlir_context_);
  }

  mlir::MLIRContext mlir_context_;
};

TEST_F(TiledHloFusionInstructionTest,
       TileSizesAndStridesShouldMatchHloShapeRank) {
  std::unique_ptr<HloInstruction> hlo = HloInstruction::CreateParameter(
      /*parameter_number=*/0,
      ShapeUtil::MakeShape(PrimitiveType::F32, {32, 64}), "p0");

  IndexingMap tile_offsets_indexing = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0) -> (d0 floordiv 16, (d0 mod 16) * 16)",
                     &mlir_context_),
      /*dim_upper_bounds=*/{8},
      /*symbol_upper_bounds=*/{});

  EXPECT_THAT(TiledHloFusionInstruction::Create(
                  hlo.get(), /*operands=*/{}, /*runtime_variables=*/{},
                  /*called_computation=*/nullptr,
                  /*tile_sizes=*/{16},
                  /*tile_strides=*/{1, 1}, tile_offsets_indexing)
                  .status()
                  .message(),
              HasSubstr("Number of tile sizes must be equal to the rank"));

  EXPECT_THAT(TiledHloFusionInstruction::Create(
                  hlo.get(), /*operands=*/{}, /*runtime_variables=*/{},
                  /*called_computation=*/nullptr,
                  /*tile_sizes=*/{16, 16},
                  /*tile_strides=*/{1, 1, 1}, tile_offsets_indexing)
                  .status()
                  .message(),
              HasSubstr("Number of tile strides must be equal to the rank"));
}

}  // namespace

}  // namespace xla
