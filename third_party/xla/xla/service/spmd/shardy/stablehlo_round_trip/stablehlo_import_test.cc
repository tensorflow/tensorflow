/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_import.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "xla/hlo/ir/hlo_sharding.h"

namespace mlir::sdy {

namespace {

TEST(StablehloImportTest, FullyReplicatedEmptyMesh) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto mesh = sdy::MeshAttr::get(&context, /*axes=*/{});

  TensorShardingAttr sharding = xla::sdy::convertToSdySharding(
      /*hloSharding=*/xla::HloSharding::Replicate(),
      /*globalMesh=*/mesh,
      /*deviceIdToMaximalMeshName=*/
      llvm::SmallDenseMap<int64_t, mlir::StringRef>(), /*rank=*/2,
      /*openDims=*/true);
  EXPECT_EQ(attributeToString(sharding), "#sdy.sharding<@mesh, [{?}, {?}]>");
}

TEST(StablehloImportTest, FullyReplicatedNonEmptyMesh) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  SmallVector<sdy::MeshAxisAttr> axes;
  axes.emplace_back(mlir::sdy::MeshAxisAttr::get(&context, "x", 2));
  axes.emplace_back(mlir::sdy::MeshAxisAttr::get(&context, "y", 4));
  auto mesh = sdy::MeshAttr::get(&context, axes);

  TensorShardingAttr sharding = xla::sdy::convertToSdySharding(
      /*hloSharding=*/xla::HloSharding::Replicate(),
      /*globalMesh=*/mesh,
      /*deviceIdToMaximalMeshName=*/
      llvm::SmallDenseMap<int64_t, mlir::StringRef>(), /*rank=*/2,
      /*openDims=*/true);
  EXPECT_EQ(attributeToString(sharding), "#sdy.sharding<@mesh, [{?}, {?}]>");
}

TEST(StablehloImportTest, SkipFirstAxisOfSize1) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  SmallVector<sdy::MeshAxisAttr> axes;
  axes.emplace_back(mlir::sdy::MeshAxisAttr::get(&context, "x", 1));
  axes.emplace_back(mlir::sdy::MeshAxisAttr::get(&context, "y", 4));
  axes.emplace_back(mlir::sdy::MeshAxisAttr::get(&context, "z", 2));
  auto mesh = sdy::MeshAttr::get(&context, axes);

  TensorShardingAttr sharding = xla::sdy::convertToSdySharding(
      /*hloSharding=*/xla::HloSharding::IotaTile({4, 2}),
      /*globalMesh=*/mesh,
      /*deviceIdToMaximalMeshName=*/
      llvm::SmallDenseMap<int64_t, mlir::StringRef>(), /*rank=*/2,
      /*openDims=*/true);
  EXPECT_EQ(attributeToString(sharding),
            "#sdy.sharding<@mesh, [{\"y\", ?}, {\"z\", ?}]>");
}

// As above, but the middle axis is the one with size 1.
TEST(StablehloImportTest, SkipSecondAxisOfSize1) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  SmallVector<sdy::MeshAxisAttr> axes;
  axes.emplace_back(mlir::sdy::MeshAxisAttr::get(&context, "y", 4));
  axes.emplace_back(mlir::sdy::MeshAxisAttr::get(&context, "x", 1));
  axes.emplace_back(mlir::sdy::MeshAxisAttr::get(&context, "z", 2));
  auto mesh = sdy::MeshAttr::get(&context, axes);

  TensorShardingAttr sharding = xla::sdy::convertToSdySharding(
      /*hloSharding=*/xla::HloSharding::IotaTile({4, 2}),
      /*globalMesh=*/mesh,
      /*deviceIdToMaximalMeshName=*/
      llvm::SmallDenseMap<int64_t, mlir::StringRef>(), /*rank=*/2,
      /*openDims=*/true);
  EXPECT_EQ(attributeToString(sharding),
            "#sdy.sharding<@mesh, [{\"y\", ?}, {\"z\", ?}]>");
}

TEST(StablehloImportTest, TransposedWithReplicatedAxis) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  SmallVector<sdy::MeshAxisAttr> axes;
  axes.emplace_back(mlir::sdy::MeshAxisAttr::get(&context, "x", 2));
  axes.emplace_back(mlir::sdy::MeshAxisAttr::get(&context, "y", 4));
  axes.emplace_back(mlir::sdy::MeshAxisAttr::get(&context, "z", 2));
  auto mesh = sdy::MeshAttr::get(&context, axes);

  TensorShardingAttr sharding = xla::sdy::convertToSdySharding(
      /*hloSharding=*/xla::HloSharding::PartialTile(
          xla::TileAssignment({2, 2, 4}, {2, 4, 2}, {2, 0, 1})),
      /*globalMesh=*/mesh,
      /*deviceIdToMaximalMeshName=*/
      llvm::SmallDenseMap<int64_t, mlir::StringRef>(), /*rank=*/2,
      /*openDims=*/false);
  EXPECT_EQ(attributeToString(sharding),
            "#sdy.sharding<@mesh, [{\"z\"}, {\"x\"}]>");
}

TEST(StablehloImportTest, IotaTileRequiresSubAxes) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  SmallVector<sdy::MeshAxisAttr> axes;
  axes.emplace_back(mlir::sdy::MeshAxisAttr::get(&context, "x", 4));
  axes.emplace_back(mlir::sdy::MeshAxisAttr::get(&context, "y", 2));
  auto mesh = sdy::MeshAttr::get(&context, axes);

  TensorShardingAttr sharding = xla::sdy::convertToSdySharding(
      /*hloSharding=*/xla::HloSharding::IotaTile({2, 4}),
      /*globalMesh=*/mesh,
      /*deviceIdToMaximalMeshName=*/
      llvm::SmallDenseMap<int64_t, mlir::StringRef>(), /*rank=*/2,
      /*openDims=*/false);
  EXPECT_EQ(attributeToString(sharding),
            "#sdy.sharding<@mesh, [{\"x\":(1)2}, {\"x\":(2)2, \"y\"}]>");
}

TEST(StablehloImportTest, IotaTileRequiresSubAxes2) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  SmallVector<sdy::MeshAxisAttr> axes;
  axes.emplace_back(mlir::sdy::MeshAxisAttr::get(&context, "x", 16));
  auto mesh = sdy::MeshAttr::get(&context, axes);

  TensorShardingAttr sharding = xla::sdy::convertToSdySharding(
      /*hloSharding=*/xla::HloSharding::IotaTile({4, 4}),
      /*globalMesh=*/mesh,
      /*deviceIdToMaximalMeshName=*/
      llvm::SmallDenseMap<int64_t, mlir::StringRef>(), /*rank=*/2,
      /*openDims=*/false);
  EXPECT_EQ(attributeToString(sharding),
            "#sdy.sharding<@mesh, [{\"x\":(1)4}, {\"x\":(4)4}]>");
}

TEST(StablehloImportTest, TransposedWithReplicatedRequiresSubAxes) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  SmallVector<sdy::MeshAxisAttr> axes;
  axes.emplace_back(mlir::sdy::MeshAxisAttr::get(&context, "x", 2));
  axes.emplace_back(mlir::sdy::MeshAxisAttr::get(&context, "y", 4));
  auto mesh = sdy::MeshAttr::get(&context, axes);

  TensorShardingAttr sharding = xla::sdy::convertToSdySharding(
      /*hloSharding=*/xla::HloSharding::PartialTile(
          xla::TileAssignment({2, 2, 2}, {2, 2, 2}, {2, 0, 1})),
      /*globalMesh=*/mesh,
      /*deviceIdToMaximalMeshName=*/
      llvm::SmallDenseMap<int64_t, mlir::StringRef>(), /*rank=*/2,
      /*openDims=*/false);
  EXPECT_EQ(attributeToString(sharding),
            "#sdy.sharding<@mesh, [{\"y\":(2)2}, {\"x\"}]>");
}

}  // namespace
}  // namespace mlir::sdy
