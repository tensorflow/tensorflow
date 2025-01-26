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

#include "xla/service/spmd/shardy/mhlo_round_trip/mhlo_import.h"

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
#include "tsl/platform/test.h"

namespace mlir::sdy {

namespace {

TEST(MhloImportTest, SkipFirstAxisOfSize1) {
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
TEST(MhloImportTest, SkipSecondAxisOfSize1) {
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

}  // namespace
}  // namespace mlir::sdy
