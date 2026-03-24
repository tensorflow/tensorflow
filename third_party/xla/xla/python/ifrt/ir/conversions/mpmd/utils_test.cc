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
#include "xla/python/ifrt/ir/conversions/mpmd/utils.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/support/sharding_conversions.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_import.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

using ::llvm::SmallVector;
using ::mlir::ArrayRef;
using ::mlir::DenseSet;
using ::mlir::MLIRContext;
using ::mlir::StringRef;
using ::mlir::mpmd::MeshTensorType;
using ::xla::HloSharding;
using ::xla::TileAssignment;

namespace xla::ifrt::mpmd {
namespace {

struct ShardingParamConversionTestStruct {
  HloSharding hlo_sharding;
  ArrayRef<int64_t> tensor_shape;
  ArrayRef<StringRef> axes_names;
  ArrayRef<int64_t> axes_sizes;
};

using ShardingParamConversionTest =
    ::testing::TestWithParam<ShardingParamConversionTestStruct>;

TEST_P(ShardingParamConversionTest, MeshTensorTypeToShardingParam) {
  // The test verifies the conversion MeshTensorType to ShardingParam
  // by checking whether the conversion to HloSharding matches.
  // We assume that the following conversions are correct:
  // MeshTensorType => HloSharding <=> ShardingParam. No useful
  // information is lost in the process of conversion. Thus, to verify
  // MeshTensorType => ShardingParam, we can convert both to HloSharding
  // and verify equality of the HloShardings obtained.
  const auto& param = GetParam();

  MLIRContext context;
  context.loadDialect<mlir::sdy::SdyDialect, mlir::mpmd::MpmdDialect>();

  SmallVector<mlir::sdy::MeshAxisAttr> mesh_axes;
  mesh_axes.reserve(param.axes_names.size());
  for (const auto& [name, size] :
       llvm::zip(param.axes_names, param.axes_sizes)) {
    mesh_axes.push_back(mlir::sdy::MeshAxisAttr::get(&context, name, size));
  }
  mlir::sdy::MeshAttr mesh = mlir::sdy::MeshAttr::get(&context, mesh_axes);

  mlir::RankedTensorType ranked_tensor_type = mlir::RankedTensorType::get(
      param.tensor_shape, mlir::Float32Type::get(&context));
  MeshTensorType mesh_tensor_type = MeshTensorType::get(
      mesh.getContext(), "mesh",
      /*ranked_tensor_type=*/ranked_tensor_type, /*sharding=*/
      xla::sdy::convertToSdySharding(
          param.hlo_sharding, mesh,
          /*deviceIdToMaximalMeshName=*/
          llvm::SmallDenseMap<int64_t, mlir::StringRef>(),
          ranked_tensor_type.getRank()));

  // Check the HloSharding obtained from the MeshTensorType is the same as the
  // original HloSharding.
  EXPECT_EQ(GetHloSharding(mesh_tensor_type, mesh).ToString(),
            param.hlo_sharding.ToString());

  // Convert MeshTensorType to ShardingParam.
  TF_ASSERT_OK_AND_ASSIGN(
      auto actual_sharding_param,
      MeshTensorTypeToShardingParam(mesh_tensor_type, mesh));
  TF_EXPECT_OK(actual_sharding_param.verify());

  // Convert ShardingParam to HloSharding.
  TF_ASSERT_OK_AND_ASSIGN(
      auto actual_hlo_sharding,
      xla::ifrt::support::ToHloSharding(actual_sharding_param));

  EXPECT_EQ(param.hlo_sharding, actual_hlo_sharding);
}

INSTANTIATE_TEST_SUITE_P(
    HloShardingToShardingParamTests, ShardingParamConversionTest,
    testing::ValuesIn<ShardingParamConversionTestStruct>({
        {HloSharding::IotaTile({4, 2}), {8, 4}, {"x", "y"}, {4, 2}},
        {HloSharding::IotaTile({2, 4}, {4, 2}, {1, 0}),
         {8, 4},
         {"x", "y"},
         {4, 2}},
        {HloSharding::IotaTile({8, 1}), {8, 4}, {"x", "y"}, {4, 2}},
        {HloSharding::IotaTile({8, 1}, {4, 2}, {1, 0}),
         {8, 4},
         {"x", "y"},
         {4, 2}},
        {HloSharding::PartialTile(TileAssignment({4, 1, 2}, {8}, {0})),
         {8, 4},
         {"x", "y"},
         {4, 2}},
        {HloSharding::PartialTile(TileAssignment({2, 1, 4}, {4, 2}, {1, 0})),
         {8, 4},
         {"x", "y"},
         {4, 2}},
        {HloSharding::PartialTile(TileAssignment({1, 4, 2}, {8}, {0})),
         {8, 4},
         {"x", "y"},
         {4, 2}},
        {HloSharding::PartialTile(TileAssignment({1, 2, 4}, {4, 2}, {1, 0})),
         {8, 4},
         {"x", "y"},
         {4, 2}},
        {HloSharding::PartialTile(TileAssignment({4, 3, 2}, {2, 3, 4},
                                                 {2, 1, 0})),
         {120, 96},
         {"x", "y", "z"},
         {2, 3, 4}},
        {HloSharding::PartialTile(TileAssignment({4, 2, 3}, {6, 4}, {1, 0})),
         {120, 96},
         {"x", "y", "z"},
         {2, 3, 4}},
        {HloSharding::PartialTile(TileAssignment({6, 1, 4}, {24}, {0})),
         {120, 96},
         {"x", "y", "z"},
         {2, 3, 4}},
        {HloSharding::PartialTile(TileAssignment({12, 1, 2}, {2, 12}, {1, 0})),
         {120, 96},
         {"x", "y", "z"},
         {2, 3, 4}},
        {HloSharding::PartialTile(TileAssignment({8, 1, 3}, {6, 4}, {1, 0})),
         {120, 96},
         {"x", "y", "z"},
         {2, 3, 4}},
        {HloSharding::PartialTile(TileAssignment({2, 1, 12}, {24}, {0})),
         {120, 96},
         {"x", "y", "z"},
         {2, 3, 4}},
        {HloSharding::PartialTile(TileAssignment({3, 1, 8}, {2, 3, 4},
                                                 {1, 0, 2})),
         {120, 96},
         {"x", "y", "z"},
         {2, 3, 4}},
        {HloSharding::PartialTile(TileAssignment({1, 4, 6}, {6, 4}, {1, 0})),
         {120, 96},
         {"x", "y", "z"},
         {2, 3, 4}},
        {HloSharding::PartialTile(TileAssignment({1, 12, 2}, {2, 12}, {1, 0})),
         {120, 96},
         {"x", "y", "z"},
         {2, 3, 4}},
        {HloSharding::PartialTile(TileAssignment({3, 2, 1, 4}, {2, 3, 4},
                                                 {1, 0, 2})),
         {120, 96, 72},
         {"x", "y", "z"},
         {2, 3, 4}},
        {HloSharding::PartialTile(TileAssignment({2, 4, 1, 3}, {2, 3, 4},
                                                 {0, 2, 1})),
         {120, 96, 72},
         {"x", "y", "z"},
         {2, 3, 4}},
        {HloSharding::PartialTile(TileAssignment({4, 3, 1, 2}, {2, 3, 4},
                                                 {2, 1, 0})),
         {120, 96, 72},
         {"x", "y", "z"},
         {2, 3, 4}},
        {HloSharding::PartialTile(TileAssignment({12, 1, 1, 2}, {2, 12},
                                                 {1, 0})),
         {120, 96, 72},
         {"x", "y", "z"},
         {2, 3, 4}},
        {HloSharding::Replicate(), {120, 96, 72}, {"x", "y", "z"}, {2, 3, 4}},
    }));

}  // namespace
}  // namespace xla::ifrt::mpmd
