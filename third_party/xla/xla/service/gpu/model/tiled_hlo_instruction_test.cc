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
#include "absl/container/flat_hash_set.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class TiledHloInstructionTest : public HloTestBase {
 public:
  mlir::MLIRContext mlir_context_;
};

TEST_F(TiledHloInstructionTest, PtrHashAndPtrEqualWorkCorrectly) {
  std::unique_ptr<HloInstruction> hlo = HloInstruction::CreateParameter(
      /*parameter_number=*/0,
      ShapeUtil::MakeShape(PrimitiveType::F32, {32, 64}), "p0");

  IndexingMap block_id_to_tile_offsets_indexing = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0) -> (d0 floordiv 16, (d0 mod 16) * 16)",
                     &mlir_context_),
      /*dim_upper_bounds=*/{8},
      /*symbol_upper_bounds=*/{});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TiledHloInstruction> tiled_hlo1,
      TiledHloInstruction::Create(hlo.get(), /*tile_sizes=*/{16, 16},
                                  /*tile_strides=*/{1, 1},
                                  block_id_to_tile_offsets_indexing));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TiledHloInstruction> tiled_hlo2,
      TiledHloInstruction::Create(hlo.get(), /*tile_sizes=*/{16, 16},
                                  /*tile_strides=*/{1, 1},
                                  block_id_to_tile_offsets_indexing));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TiledHloInstruction> tiled_hlo3,
      TiledHloInstruction::Create(hlo.get(), /*tile_sizes=*/{16, 32},
                                  /*tile_strides=*/{1, 1},
                                  block_id_to_tile_offsets_indexing));

  EXPECT_EQ(*tiled_hlo1, *tiled_hlo2);
  EXPECT_NE(*tiled_hlo1, *tiled_hlo3);

  absl::flat_hash_set<TiledHloInstruction*, TiledHloInstruction::PtrHash,
                      TiledHloInstruction::PtrEqual>
      tiled_hlo_set = {tiled_hlo1.get(), tiled_hlo2.get(), tiled_hlo3.get()};
  EXPECT_EQ(tiled_hlo_set.size(), 2);
}

TEST_F(TiledHloInstructionTest, TileSizesAndStridesShouldMatchHloShapeRank) {
  std::unique_ptr<HloInstruction> hlo = HloInstruction::CreateParameter(
      /*parameter_number=*/0,
      ShapeUtil::MakeShape(PrimitiveType::F32, {32, 64}), "p0");

  IndexingMap block_id_to_tile_offsets_indexing = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0) -> (d0 floordiv 16, (d0 mod 16) * 16)",
                     &mlir_context_),
      /*dim_upper_bounds=*/{8},
      /*symbol_upper_bounds=*/{});

  EXPECT_THAT(
      TiledHloInstruction::Create(hlo.get(), /*tile_sizes=*/{16},
                                  /*tile_strides=*/{1, 1},
                                  block_id_to_tile_offsets_indexing)
          .status()
          .message(),
      ::testing::HasSubstr("Number of tile sizes must be equal to the rank"));

  EXPECT_THAT(
      TiledHloInstruction::Create(hlo.get(), /*tile_sizes=*/{16, 16},
                                  /*tile_strides=*/{1, 1, 1},
                                  block_id_to_tile_offsets_indexing)
          .status()
          .message(),
      ::testing::HasSubstr("Number of tile strides must be equal to the rank"));
}

TEST_F(TiledHloInstructionTest,
       ShouldReturnErrorIfBlockIdToTileOffsetsIndexingIsInvalid) {
  std::unique_ptr<HloInstruction> hlo = HloInstruction::CreateParameter(
      /*parameter_number=*/0,
      ShapeUtil::MakeShape(PrimitiveType::F32, {32, 64}), "p0");

  IndexingMap block_id_to_tile_offsets_indexing1 = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0) -> (d0 floordiv 16)", &mlir_context_),
      /*dim_upper_bounds=*/{8},
      /*symbol_upper_bounds=*/{});

  EXPECT_THAT(
      TiledHloInstruction::Create(hlo.get(), /*tile_sizes=*/{16, 16},
                                  /*tile_strides=*/{1, 1},
                                  block_id_to_tile_offsets_indexing1)
          .status()
          .message(),
      ::testing::HasSubstr(
          "must have the same number of results as the rank of the hlo shape"));

  IndexingMap block_id_to_tile_offsets_indexing2 = IndexingMap::FromTensorSizes(
      ParseAffineMap("(d0)[s0] -> (d0 + s0, d0 floordiv 16)", &mlir_context_),
      /*dim_upper_bounds=*/{8},
      /*symbol_upper_bounds=*/{8});

  EXPECT_THAT(TiledHloInstruction::Create(hlo.get(), /*tile_sizes=*/{16, 16},
                                          /*tile_strides=*/{1, 1},
                                          block_id_to_tile_offsets_indexing2)
                  .status()
                  .message(),
              ::testing::HasSubstr("must have 1 dim and 0 symbols"));
}

}  // namespace

}  // namespace gpu
}  // namespace xla
