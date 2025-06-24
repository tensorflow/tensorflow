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

#include "xla/service/gpu/model/tiled_hlo_instruction.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"  // IWYU pragma: keep
#include "xla/shape_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::HasSubstr;
using ::tsl::testing::IsOk;

class TiledHloInstructionTest : public HloHardwareIndependentTestBase {
 public:
  mlir::MLIRContext mlir_context_;
};

TEST_F(TiledHloInstructionTest, TileSizesAndStridesShouldMatchHloShapeRank) {
  std::unique_ptr<HloInstruction> hlo = HloInstruction::CreateParameter(
      /*parameter_number=*/0,
      ShapeUtil::MakeShape(PrimitiveType::F32, {32, 64}), "p0");

  IndexingMap tile_offsets_indexing = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0) -> (d0 floordiv 16, (d0 mod 16) * 16)",
                     &mlir_context_),
      /*dim_upper_bounds=*/{8},
      /*symbol_upper_bounds=*/{});

  EXPECT_THAT(TiledHloInstruction::Create(
                  hlo.get(), /*operands=*/{}, /*runtime_variables=*/{},
                  /*tile_sizes=*/{16},
                  /*tile_strides=*/{1, 1}, tile_offsets_indexing)
                  .status()
                  .message(),
              HasSubstr("Number of tile sizes must be equal to the rank"));

  EXPECT_THAT(TiledHloInstruction::Create(
                  hlo.get(), /*operands=*/{}, /*runtime_variables=*/{},
                  /*tile_sizes=*/{16, 16},
                  /*tile_strides=*/{1, 1, 1}, tile_offsets_indexing)
                  .status()
                  .message(),
              HasSubstr("Number of tile strides must be equal to the rank"));
}

TEST_F(TiledHloInstructionTest,
       ShouldReturnErrorIfBlockIdToTileOffsetsIndexingIsInvalid) {
  std::unique_ptr<HloInstruction> hlo = HloInstruction::CreateParameter(
      /*parameter_number=*/0,
      ShapeUtil::MakeShape(PrimitiveType::F32, {32, 64}), "p0");

  IndexingMap tile_offsets_indexing = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0) -> (2 * d0)", &mlir_context_),
      /*dim_upper_bounds=*/{2},
      /*symbol_upper_bounds=*/{});

  EXPECT_THAT(
      TiledHloInstruction::Create(
          hlo.get(), /*operands=*/{}, /*runtime_variables=*/{},
          /*tile_sizes=*/{16, 16},
          /*tile_strides=*/{1, 1}, tile_offsets_indexing)
          .status()
          .message(),
      HasSubstr(
          "must have the same number of results as the rank of the hlo shape"));

  IndexingMap tile_offsets_indexing2 = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0, d1) -> (d0, d1)", &mlir_context_),
      /*dim_upper_bounds=*/{8, 4},
      /*symbol_upper_bounds=*/{});

  EXPECT_THAT(TiledHloInstruction::Create(
                  hlo.get(), /*operands=*/{}, /*runtime_variables=*/{},
                  /*tile_sizes=*/{16, 16},
                  /*tile_strides=*/{1, 1}, tile_offsets_indexing2)
                  .status()
                  .message(),
              ::testing::HasSubstr("must have 1 dim"));
}

TEST_F(TiledHloInstructionTest,
       ShouldReturnErrorIfWrongNumberOfRuntimeVariablesIsProvided) {
  std::unique_ptr<HloInstruction> p0 = HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(PrimitiveType::F32, {32}),
      "p0");

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TiledHloInstruction> rt0,
      TiledHloInstruction::Create(
          p0.get(), /*operands=*/{},
          /*runtime_variables=*/{},
          /*tile_sizes=*/{16},
          /*tile_strides=*/{1},
          IndexingMap::FromTensorSizes(
              ParseAffineMap("(d0) -> (d0)", &mlir_context_),
              /*dim_upper_bounds=*/{4},
              /*symbol_upper_bounds=*/{})));

  IndexingMap indexing_map(
      ParseAffineMap("(d0)[rt0] -> (d0 + rt0)", &mlir_context_),
      /*dimensions=*/
      {IndexingMap::Variable{0, 32, "d0"}},
      /*range_vars=*/{},
      /*rt_vars=*/{IndexingMap::Variable{0, 4, "rt0"}});

  EXPECT_THAT(
      TiledHloInstruction::Create(p0.get(), /*operands=*/{},
                                  /*runtime_variables=*/{},
                                  /*tile_sizes=*/{16},
                                  /*tile_strides=*/{1}, indexing_map)
          .status()
          .message(),
      ::testing::HasSubstr("tile_offsets_indexing has 1 runtime variables, but "
                           "0 runtime variables were provided"));
  EXPECT_THAT(TiledHloInstruction::Create(p0.get(), /*operands=*/{},
                                          /*runtime_variables=*/{rt0.get()},
                                          /*tile_sizes=*/{16},
                                          /*tile_strides=*/{1}, indexing_map),
              IsOk());
  EXPECT_THAT(
      TiledHloInstruction::Create(p0.get(), /*operands=*/{},
                                  /*runtime_variables=*/{rt0.get(), rt0.get()},
                                  /*tile_sizes=*/{16},
                                  /*tile_strides=*/{1}, indexing_map)
          .status()
          .message(),
      ::testing::HasSubstr("tile_offsets_indexing has 1 runtime variables, but "
                           "2 runtime variables were provided"));
}

using TiledHloFusionInstructionTest = TiledHloInstructionTest;

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

}  // namespace gpu
}  // namespace xla
