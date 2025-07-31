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
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"

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

TEST(AddSdyShardingsToEntryComputationTest, NonTupleArgs) {
  const char* const hloString = R"(
    HloModule main__.18

    %main.6 (Arg_0.7: f32[10,20]) -> f32[20,10] {
      %Arg_0.7 = f32[10,20]{1,0} parameter(0), sharding={devices=[1,2]<=[2]}, frontend_attributes={xla.sdy.sharding="#sdy.sharding<mesh<[\"x\"=2]>, [{}, {\"x\"}]>"}
      ROOT %custom-call.10 = f32[20,10]{1,0} custom-call(%Arg_0.7), custom_call_target="xla.sdy.FuncResultSharding", custom_call_has_side_effect=true, frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<mesh<[\"x\"=2]>, [{\"x\"}, {}]>]>"}
    }

    ENTRY %main__.18 (Arg_0.8: f32[10,20]) -> f32[20,10] {
      %Arg_0.8 = f32[10,20]{1,0:T(8,128)} parameter(0), sharding={devices=[1,2]<=[2]}
      ROOT %call.13 = f32[20,10]{1,0:T(8,128)} call(%Arg_0.8), to_apply=%main.6
    })";

  const char* const expected = R"(
    // CHECK:       HloModule main__.18
    // CHECK-NOT:     frontend_attributes
    //
    // CHECK:       %main.6 (Arg_0.7: f32[10,20]) -> f32[20,10] {
    // CHECK-NEXT:    %Arg_0.7 = f32[10,20]{1,0} parameter(0), sharding={devices=[1,2]<=[2]}, frontend_attributes={xla.sdy.sharding="#sdy.sharding<mesh<[\"x\"=2]>, [{}, {\"x\"}]>"}
    // CHECK-NEXT:    ROOT %custom-call.10 = f32[20,10]{1,0} custom-call(%Arg_0.7), custom_call_target="xla.sdy.FuncResultSharding", custom_call_has_side_effect=true, frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<mesh<[\"x\"=2]>, [{\"x\"}, {}]>]>"}
    //
    // CHECK:       ENTRY %main__.18 (Arg_0.8: f32[10,20]) -> f32[20,10] {
    // CHECK-NEXT:    %Arg_0.8 = f32[10,20]{1,0:T(8,128)} parameter(0), sharding={devices=[1,2]<=[2]}, frontend_attributes={xla.sdy.sharding="#sdy.sharding<mesh<[\"x\"=2]>, [{}, {\"x\"}]>"}
    // CHECK-NEXT:    ROOT %call.13 = f32[20,10]{1,0:T(8,128)} call(%Arg_0.8), to_apply=%main.6
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          xla::ParseAndReturnUnverifiedModule(hloString));
  TF_CHECK_OK(xla::sdy::addSdyShardingsToEntryComputation(module.get()));
  EXPECT_TRUE(
      *xla::RunFileCheck(module->ToString(xla::HloPrintOptions{}), expected));
}

TEST(AddSdyShardingsToEntryComputationTest, NonTupleArgsWithResultSharding) {
  const char* const hloString = R"(
    HloModule main__.18

    %main.6 (Arg_0.7: f32[10,20]) -> f32[20,10] {
      %Arg_0.7 = f32[10,20]{1,0} parameter(0), sharding={devices=[1,2]<=[2]}, frontend_attributes={xla.sdy.sharding="#sdy.sharding<mesh<[\"x\"=2]>, [{}, {\"x\"}]>"}
      ROOT %custom-call.10 = f32[20,10]{1,0} custom-call(%Arg_0.7), custom_call_target="xla.sdy.FuncResultSharding", custom_call_has_side_effect=true, frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<mesh<[\"x\"=2]>, [{\"x\"}, {}]>]>"}
    }

    ENTRY %main__.18 (Arg_0.8: f32[10,20]) -> f32[20,10] {
      %Arg_0.8 = f32[10,20]{1,0:T(8,128)} parameter(0), sharding={devices=[1,2]<=[2]}
      ROOT %call.13 = f32[20,10]{1,0:T(8,128)} call(%Arg_0.8), to_apply=%main.6, sharding={devices=[2,1]<=[2]}
    })";

  const char* const expected = R"(
    // CHECK:       HloModule main__.18, entry_computation_layout={{.*}}, frontend_attributes=
    // CHECK-SAME:    {xla.sdy.tuple_results_shardings="#sdy.sharding_per_value<[<mesh<[\"x\"=2]>, [{\"x\"}, {}]>]>"}
    //
    // CHECK:       %main.6 (Arg_0.7: f32[10,20]) -> f32[20,10] {
    // CHECK-NEXT:    %Arg_0.7 = f32[10,20]{1,0} parameter(0), sharding={devices=[1,2]<=[2]}, frontend_attributes={xla.sdy.sharding="#sdy.sharding<mesh<[\"x\"=2]>, [{}, {\"x\"}]>"}
    // CHECK-NEXT:    ROOT %custom-call.10 = f32[20,10]{1,0} custom-call(%Arg_0.7), custom_call_target="xla.sdy.FuncResultSharding", custom_call_has_side_effect=true, frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<mesh<[\"x\"=2]>, [{\"x\"}, {}]>]>"}
    //
    // CHECK:       ENTRY %main__.18 (Arg_0.8: f32[10,20]) -> f32[20,10] {
    // CHECK-NEXT:    %Arg_0.8 = f32[10,20]{1,0:T(8,128)} parameter(0), sharding={devices=[1,2]<=[2]}, frontend_attributes={xla.sdy.sharding="#sdy.sharding<mesh<[\"x\"=2]>, [{}, {\"x\"}]>"}
    // CHECK-NEXT:    ROOT %call.13 = f32[20,10]{1,0:T(8,128)} call(%Arg_0.8), to_apply=%main.6, sharding={devices=[2,1]<=[2]}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          xla::ParseAndReturnUnverifiedModule(hloString));
  TF_CHECK_OK(xla::sdy::addSdyShardingsToEntryComputation(module.get()));
  EXPECT_TRUE(
      *xla::RunFileCheck(module->ToString(xla::HloPrintOptions{}), expected));
}

TEST(AddSdyShardingsToEntryComputationTest, TupleArgs) {
  const char* const hloString = R"(
    HloModule main__.18

    %main.6 (Arg_0.7: f32[10,20], Arg_1.8: f32[20,10]) -> f32[20,10] {
      %Arg_0.7 = f32[10,20]{1,0} parameter(0), sharding={devices=[1,2]<=[2]}, frontend_attributes={xla.sdy.sharding="#sdy.sharding<mesh<[\"x\"=2]>, [{}, {\"x\"}]>"}
      %Arg_1.8 = f32[20,10]{1,0} parameter(1), sharding={devices=[2,1]<=[2]}, frontend_attributes={xla.sdy.sharding="#sdy.sharding<mesh<[\"x\"=2]>, [{\"x\"}, {}]>"}
      ROOT %custom-call.10 = f32[20,10]{1,0} custom-call(%Arg_0.7, %Arg_1.8), custom_call_target="xla.sdy.FuncResultSharding", custom_call_has_side_effect=true, frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<mesh<[\"x\"=2]>, [{\"x\"}, {}]>]>"}
    }

    ENTRY %main__.18 (arg_tuple.1: (f32[10,20], f32[20,10])) -> (f32[20,10]) {
      %arg_tuple.1 = (f32[10,20]{1,0:T(8,128)}, f32[20,10]{1,0:T(8,128)}) parameter(0), sharding={{devices=[1,2]<=[2]}, {devices=[2,1]<=[2]}}
      %get-tuple-element.2 = f32[10,20]{1,0:T(8,128)} get-tuple-element(%arg_tuple.1), index=0, sharding={devices=[1,2]<=[2]}
      %get-tuple-element.3 = f32[20,10]{1,0:T(8,128)} get-tuple-element(%arg_tuple.1), index=1, sharding={devices=[2,1]<=[2]}
      %call.13 = f32[20,10]{1,0} call(%get-tuple-element.2, %get-tuple-element.3), to_apply=%main.6
      ROOT %tuple.17 = (f32[20,10]{1,0:T(8,128)}) tuple(%call.13), sharding={{devices=[2,1]<=[2]}}
    })";

  const char* const expected = R"(
    // CHECK:                HloModule main__.18, entry_computation_layout={{.*}}, frontend_attributes={
    // CHECK-SAME:             xla.sdy.tuple_args_shardings="#sdy.sharding_per_value<[<mesh<[\"x\"=2]>, [{}, {\"x\"}]>, <mesh<[\"x\"=2]>, [{\"x\"}, {}]>]>",
    // CHECK-SAME:             xla.sdy.tuple_results_shardings="#sdy.sharding_per_value<[<mesh<[\"x\"=2]>, [{\"x\"}, {}]>]>",
    // CHECK-SAME:             xla.sdy.use_tuple_args="True"}
    //
    // CHECK:                %main.6 (Arg_0.7: f32[10,20], Arg_1.8: f32[20,10]) -> f32[20,10] {
    // CHECK-NEXT:             %Arg_0.7 = f32[10,20]{1,0} parameter(0), sharding={devices=[1,2]<=[2]}, frontend_attributes={xla.sdy.sharding="#sdy.sharding<mesh<[\"x\"=2]>, [{}, {\"x\"}]>"}
    // CHECK-NEXT:             %Arg_1.8 = f32[20,10]{1,0} parameter(1), sharding={devices=[2,1]<=[2]}, frontend_attributes={xla.sdy.sharding="#sdy.sharding<mesh<[\"x\"=2]>, [{\"x\"}, {}]>"}
    // CHECK-NEXT:             ROOT %custom-call.10 = f32[20,10]{1,0} custom-call(%Arg_0.7, %Arg_1.8), custom_call_target="xla.sdy.FuncResultSharding", custom_call_has_side_effect=true, frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<mesh<[\"x\"=2]>, [{\"x\"}, {}]>]>"}
    //
    // CHECK:                ENTRY %main__.18 (arg_tuple.1: (f32[10,20], f32[20,10])) -> (f32[20,10]) {
    // CHECK-NEXT{LITERAL}:    %arg_tuple.1 = (f32[10,20]{1,0:T(8,128)}, f32[20,10]{1,0:T(8,128)}) parameter(0), sharding={{devices=[1,2]<=[2]}, {devices=[2,1]<=[2]}}
    // CHECK-NEXT:             %get-tuple-element.2 = f32[10,20]{1,0:T(8,128)} get-tuple-element(%arg_tuple.1), index=0, sharding={devices=[1,2]<=[2]}, frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<mesh<[\"x\"=2]>, [{}, {\"x\"}]>]>"}
    // CHECK-NEXT:             %get-tuple-element.3 = f32[20,10]{1,0:T(8,128)} get-tuple-element(%arg_tuple.1), index=1, sharding={devices=[2,1]<=[2]}, frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<mesh<[\"x\"=2]>, [{\"x\"}, {}]>]>"}
    // CHECK-NEXT:             %call.13 = f32[20,10]{1,0} call(%get-tuple-element.2, %get-tuple-element.3), to_apply=%main.6
    // CHECK-NEXT{LITERAL}:    ROOT %tuple.17 = (f32[20,10]{1,0:T(8,128)}) tuple(%call.13), sharding={{devices=[2,1]<=[2]}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          xla::ParseAndReturnUnverifiedModule(hloString));
  TF_CHECK_OK(xla::sdy::addSdyShardingsToEntryComputation(module.get()));
  EXPECT_TRUE(
      *xla::RunFileCheck(module->ToString(xla::HloPrintOptions{}), expected));
}

TEST(AddSdyShardingsToEntryComputationTest, AlreadyHasSdyShardings) {
  const char* const hloString = R"(
    HloModule jit_predict
    ENTRY main.8 {
      Arg_0.1 = f32[16,128]{1,0} parameter(0), sharding={devices=[4,1,2]<=[8] last_tile_dim_replicate}, frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<mesh<[\"data\"=4, \"model\"=2]>, [{\"data\"}, {}]>]>"}
      tanh.4 = f32[16,128]{1,0} tanh(Arg_0.1)
      Arg_1.2 = f32[128,256]{1,0} parameter(1), sharding={devices=[1,2,4]<=[4,2]T(1,0) last_tile_dim_replicate}, frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<mesh<[\"data\"=4, \"model\"=2]>, [{}, {\"model\"}]>]>"}
      dot.5 = f32[16,256]{1,0} dot(tanh.4, Arg_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      Arg_2.3 = f32[256,10]{1,0} parameter(2)
      dot.6 = f32[16,10]{1,0} dot(dot.5, Arg_2.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT sine.7 = f32[16,10]{1,0} sine(dot.6)
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          xla::ParseAndReturnUnverifiedModule(hloString));
  absl::Status status =
      xla::sdy::addSdyShardingsToEntryComputation(module.get());
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      testing::HasSubstr("Instruction Arg_0.1 already has sdy.shardings"));
}

TEST(AddSdyShardingsToEntryComputationTest, NoSdyShardingsUseFakeMesh) {
  const char* const hloString = R"(
    HloModule jit_predict, entry_computation_layout={(f32[16,128]{1,0}, f32[128,256]{1,0}, f32[256,10]{1,0})->f32[16,10]{1,0}}
    ENTRY main.8 {
      Arg_0.1 = f32[16,128]{1,0} parameter(0), sharding={devices=[4,1,2]<=[8] last_tile_dim_replicate}
      tanh.4 = f32[16,128]{1,0} tanh(Arg_0.1)
      Arg_1.2 = f32[128,256]{1,0} parameter(1), sharding={devices=[1,2,4]<=[4,2]T(1,0) last_tile_dim_replicate}
      dot.5 = f32[16,256]{1,0} dot(tanh.4, Arg_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      Arg_2.3 = f32[256,10]{1,0} parameter(2)
      dot.6 = f32[16,10]{1,0} dot(dot.5, Arg_2.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT sine.7 = f32[16,10]{1,0} sine(dot.6)
    })";

  const char* const expected = R"(
    // CHECK:       HloModule jit_predict
    // CHECK-NOT:     frontend_attributes
    //
    // CHECK:       ENTRY %main.8 (Arg_0.1: f32[16,128], Arg_1.2: f32[128,256], Arg_2.3: f32[256,10]) -> f32[16,10] {
    // CHECK-NEXT:    %Arg_0.1 = f32[16,128]{1,0} parameter(0), sharding={devices=[4,1,2]<=[8] last_tile_dim_replicate}, frontend_attributes={xla.sdy.sharding="#sdy.sharding<mesh<[\"_axis_0\"=4, \"_axis_1\"=2]>, [{\"_axis_0\"}, {}]>"}
    // CHECK-NEXT:    %tanh.4 = f32[16,128]{1,0} tanh(%Arg_0.1)
    // CHECK-NEXT:    %Arg_1.2 = f32[128,256]{1,0} parameter(1), sharding={devices=[1,2,4]<=[4,2]T(1,0) last_tile_dim_replicate}, frontend_attributes={xla.sdy.sharding="#sdy.sharding<mesh<[\"_axis_0\"=4, \"_axis_1\"=2]>, [{}, {\"_axis_1\"}]>"}
    // CHECK-NEXT:    %dot.5 = f32[16,256]{1,0} dot(%tanh.4, %Arg_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    // CHECK-NEXT:    %Arg_2.3 = f32[256,10]{1,0} parameter(2)
    // CHECK-NEXT:    %dot.6 = f32[16,10]{1,0} dot(%dot.5, %Arg_2.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    // CHECK-NEXT:    ROOT %sine.7 = f32[16,10]{1,0} sine(%dot.6)
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          xla::ParseAndReturnUnverifiedModule(hloString));
  TF_CHECK_OK(xla::sdy::addSdyShardingsToEntryComputation(module.get()));
  EXPECT_TRUE(
      *xla::RunFileCheck(module->ToString(xla::HloPrintOptions{}), expected));
}

TEST(AddSdyShardingsToEntryComputationTest,
     NonParamaterRootInstructionWithTupleShardings) {
  const char* const hloString = R"(
    HloModule main__.18

    ENTRY %main__.18 (arg_tuple.1: (f32[10,20], f32[20,10])) -> (f32[20,10]) {
      %arg_tuple.1 = (f32[10,20]{1,0:T(8,128)}, f32[20,10]{1,0:T(8,128)}) parameter(0), sharding={{devices=[1,2]<=[2]}, {devices=[2,1]<=[2]}}
      %get-tuple-element.2 = f32[10,20]{1,0:T(8,128)} get-tuple-element(%arg_tuple.1), index=0, sharding={devices=[1,2]<=[2]}
      %get-tuple-element.3 = f32[20,10]{1,0:T(8,128)} get-tuple-element(%arg_tuple.1), index=1, sharding={devices=[2,1]<=[2]}
      %tuple.4 = (f32[10,20]{1,0:T(8,128)}, f32[20,10]{1,0:T(8,128)}) tuple(%get-tuple-element.2, %get-tuple-element.3), sharding={{devices=[1,2]<=[2]}, {devices=[2,1]<=[2]}}
      %get-tuple-element.5 = f32[10,20]{1,0:T(8,128)} get-tuple-element(%tuple.4), index=0
      %get-tuple-element.6 = f32[20,10]{1,0:T(8,128)} get-tuple-element(%tuple.4), index=1
      ROOT %tuple.17 = (f32[20,10]{1,0:T(8,128)}) tuple(%get-tuple-element.6), sharding={{devices=[2,1]<=[2]}}
    })";

  const char* const expected = R"(
    // CHECK:                HloModule main__.18, entry_computation_layout={{.*}}, frontend_attributes={
    // CHECK-SAME:             xla.sdy.tuple_args_shardings="#sdy.sharding_per_value<[<mesh<[\"_axis_0\"=2]>, [{}, {\"_axis_0\"}]>, <mesh<[\"_axis_0\"=2]>, [{\"_axis_0\"}, {}]>]>",
    // CHECK-SAME:             xla.sdy.tuple_results_shardings="#sdy.sharding_per_value<[<mesh<[\"_axis_0\"=2]>, [{\"_axis_0\"}, {}]>]>",
    // CHECK-SAME:             xla.sdy.use_tuple_args="True"}
    //
    // CHECK:                ENTRY %main__.18 (arg_tuple.1: (f32[10,20], f32[20,10])) -> (f32[20,10]) {
    // CHECK-NEXT{LITERAL}:    %arg_tuple.1 = (f32[10,20]{1,0:T(8,128)}, f32[20,10]{1,0:T(8,128)}) parameter(0), sharding={{devices=[1,2]<=[2]}, {devices=[2,1]<=[2]}}
    // CHECK-NEXT:             %get-tuple-element.2 = f32[10,20]{1,0:T(8,128)} get-tuple-element(%arg_tuple.1), index=0, sharding={devices=[1,2]<=[2]}, frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<mesh<[\"_axis_0\"=2]>, [{}, {\"_axis_0\"}]>]>"}
    // CHECK-NEXT:             %get-tuple-element.3 = f32[20,10]{1,0:T(8,128)} get-tuple-element(%arg_tuple.1), index=1, sharding={devices=[2,1]<=[2]}, frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<mesh<[\"_axis_0\"=2]>, [{\"_axis_0\"}, {}]>]>"}
    // CHECK-NEXT{LITERAL}:    %tuple.4 = (f32[10,20]{1,0:T(8,128)}, f32[20,10]{1,0:T(8,128)}) tuple(%get-tuple-element.2, %get-tuple-element.3), sharding={{devices=[1,2]<=[2]}, {devices=[2,1]<=[2]}}, frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<mesh<[\"_axis_0\"=2]>, [{}, {\"_axis_0\"}]>, <mesh<[\"_axis_0\"=2]>, [{\"_axis_0\"}, {}]>]>"}
    // CHECK-NEXT:             %get-tuple-element.5 = f32[10,20]{1,0:T(8,128)} get-tuple-element(%tuple.4), index=0
    // CHECK-NEXT:             %get-tuple-element.6 = f32[20,10]{1,0:T(8,128)} get-tuple-element(%tuple.4), index=1
    // CHECK-NEXT{LITERAL}:    ROOT %tuple.17 = (f32[20,10]{1,0:T(8,128)}) tuple(%get-tuple-element.6), sharding={{devices=[2,1]<=[2]}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          xla::ParseAndReturnUnverifiedModule(hloString));
  TF_CHECK_OK(xla::sdy::addSdyShardingsToEntryComputation(module.get()));
  EXPECT_TRUE(
      *xla::RunFileCheck(module->ToString(xla::HloPrintOptions{}), expected));
}

TEST(AddSdyShardingsToEntryComputationTest, ArgsWithAndWithoutShardings) {
  const char* const hloString = R"(
    HloModule main__.18

    ENTRY %main__.18 (param_tuple.1: (f32[10,20], f32[20,10]), param.2: f32[10,20], param_tuple.3: (f32[10,20], f32[20,10])) -> (f32[20,10]) {
      %param_tuple.1 = (f32[10,20]{1,0:T(8,128)}, f32[20,10]{1,0:T(8,128)}) parameter(0), sharding={{devices=[1,2]<=[2]}, {devices=[2,1]<=[2]}}
      %param.2 = f32[10,20]{1,0:T(8,128)} parameter(1)
      %param_tuple.3 = (f32[10,20]{1,0:T(8,128)}, f32[20,10]{1,0:T(8,128)}) parameter(2), sharding={{devices=[1,2]<=[2]}, {devices=[2,1]<=[2]}}
      ROOT %tuple.9 = (f32[20,10]{1,0:T(8,128)}) tuple(%param.2)
    })";

  const char* const expected = R"(
    // CHECK:                HloModule main__.18, entry_computation_layout={{.*}}, frontend_attributes={
    // CHECK-SAME:             xla.sdy.tuple_args_shardings="#sdy.sharding_per_value<[<mesh<[\"_axis_0\"=2]>, [{}, {\"_axis_0\"}]>, <mesh<[\"_axis_0\"=2]>, [{\"_axis_0\"}, {}]>,
    // CHECK-SAME:             <mesh<[\"_axis_0\"=2]>, [{}, {}]>, <mesh<[\"_axis_0\"=2]>, [{}, {\"_axis_0\"}]>, <mesh<[\"_axis_0\"=2]>, [{\"_axis_0\"}, {}]>]>"
    // CHECK-SAME:             xla.sdy.use_tuple_args="True"}
    //
    // CHECK:                ENTRY %main__.18 (param_tuple.1: (f32[10,20], f32[20,10]), param.2: f32[10,20], param_tuple.3: (f32[10,20], f32[20,10])) -> (f32[20,10]) {
    // CHECK-NEXT{LITERAL}:    %param_tuple.1 = (f32[10,20]{1,0:T(8,128)}, f32[20,10]{1,0:T(8,128)}) parameter(0), sharding={{devices=[1,2]<=[2]}, {devices=[2,1]<=[2]}}
    // CHECK-NEXT{LITERAL}:    %param_tuple.3 = (f32[10,20]{1,0:T(8,128)}, f32[20,10]{1,0:T(8,128)}) parameter(2), sharding={{devices=[1,2]<=[2]}, {devices=[2,1]<=[2]}}
    // CHECK-NEXT:             %param.2 = f32[10,20]{1,0:T(8,128)} parameter(1)
    // CHECK-NEXT:             ROOT %tuple.9 = (f32[20,10]{1,0:T(8,128)}) tuple(%param.2)
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          xla::ParseAndReturnUnverifiedModule(hloString));
  TF_CHECK_OK(xla::sdy::addSdyShardingsToEntryComputation(module.get()));
  EXPECT_TRUE(
      *xla::RunFileCheck(module->ToString(xla::HloPrintOptions{}), expected));
}

}  // namespace
}  // namespace mlir::sdy
