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

#include "xla/codegen/tiling/tiled_hlo_instruction.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map_serialization.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::HasSubstr;

class TiledHloInstructionTest : public HloHardwareIndependentTestBase {
 public:
  TiledHloInstructionTest() { RegisterSymbolicExprStorage(&mlir_context_); }
  mlir::MLIRContext mlir_context_;
};

TEST_F(TiledHloInstructionTest, TileSizesAndStridesShouldMatchHloShapeRank) {
  std::unique_ptr<HloInstruction> hlo = HloInstruction::CreateParameter(
      /*parameter_number=*/0,
      ShapeUtil::MakeShape(PrimitiveType::F32, {32, 64}), "p0");

  IndexingMap tile_offsets_indexing = IndexingMap::FromTensorSizes(
      ParseSymbolicMap("(d0) -> (d0 floordiv 16, (d0 mod 16) * 16)",
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
      ParseSymbolicMap("(d0) -> (2 * d0)", &mlir_context_),
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
      ParseSymbolicMap("(d0, d1) -> (d0, d1)", &mlir_context_),
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
              ParseSymbolicMap("(d0) -> (d0)", &mlir_context_),
              /*dim_upper_bounds=*/{4},
              /*symbol_upper_bounds=*/{})));

  IndexingMap indexing_map(
      ParseSymbolicMap("(d0)[rt0] -> (d0 + rt0)", &mlir_context_),
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
              absl_testing::IsOk());
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

TEST_F(TiledHloInstructionTest, ToString) {
  auto create_simple_tiled_hlo = [&](int64_t number)
      -> absl::StatusOr<std::pair<std::unique_ptr<HloInstruction>,
                                  std::unique_ptr<TiledHloInstruction>>> {
    std::unique_ptr<HloInstruction> hlo = HloInstruction::CreateParameter(
        /*parameter_number=*/number,
        ShapeUtil::MakeShape(PrimitiveType::F32, {4}),
        absl::StrCat("p", number));
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<TiledHloInstruction> tiled_hlo,
        TiledHloInstruction::Create(
            hlo.get(), /*operands=*/{},
            /*runtime_variables=*/{},
            /*tile_sizes=*/{4},
            /*tile_strides=*/{1},
            IndexingMap::FromTensorSizes(
                ParseSymbolicMap("(d0) -> (d0)", &mlir_context_),
                /*dim_upper_bounds=*/{0},
                /*symbol_upper_bounds=*/{})));
    return std::make_pair(std::move(hlo), std::move(tiled_hlo));
  };
  TF_ASSERT_OK_AND_ASSIGN(auto p0, create_simple_tiled_hlo(0));
  auto [p0_hlo, tiled_p0] = std::move(p0);
  TF_ASSERT_OK_AND_ASSIGN(auto p1, create_simple_tiled_hlo(1));
  auto [p1_hlo, tiled_p1] = std::move(p1);
  TF_ASSERT_OK_AND_ASSIGN(auto p2, create_simple_tiled_hlo(2));
  auto [p2_hlo, tiled_p2] = std::move(p2);

  IndexingMap indexing_map(
      ParseSymbolicMap("(d0)[s0] -> (d0 * 16 + s0)", &mlir_context_),
      /*dimensions=*/
      {IndexingMap::Variable{0, 1}},
      /*range_vars=*/{},
      /*rt_vars=*/{IndexingMap::Variable{0, 3}});

  std::vector<std::unique_ptr<TiledHloInstruction>> region;
  region.push_back(std::move(tiled_p2));
  llvm::SmallVector<std::vector<std::unique_ptr<TiledHloInstruction>>> regions;
  regions.push_back(std::move(region));
  std::unique_ptr<HloInstruction> p3_hlo = HloInstruction::CreateParameter(
      /*parameter_number=*/3, ShapeUtil::MakeShape(PrimitiveType::F32, {32}),
      "p3");
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TiledHloInstruction> tiled_p3,
      TiledHloInstruction::Create(p3_hlo.get(), /*operands=*/{tiled_p0.get()},
                                  /*runtime_variables=*/{tiled_p1.get()},
                                  /*tile_sizes=*/{16},
                                  /*tile_strides=*/{1}, indexing_map,
                                  std::move(regions)));

  EXPECT_EQ(tiled_p3->ToString(),
            R"""(	hlo: %p3 = f32[32]{0} parameter(3)
	tile_sizes: (16)
	tile_strides: (1)
	tile_offsets_indexing: (d0){rt0} -> (d0 * 16 + rt0), domain: d0 in [0, 1], rt0 in [0, 3]
	operands:
		%p0 = parameter()
	runtime variables:
			hlo: %p1 = f32[4]{0} parameter(1)
	tile_sizes: (4)
	tile_strides: (1)
	tile_offsets_indexing: KNOWN EMPTY


	regions: (
		#0 size 1)
)""");
}

}  // namespace

}  // namespace xla
