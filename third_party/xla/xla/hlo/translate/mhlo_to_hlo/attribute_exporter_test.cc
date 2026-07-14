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

#include "xla/hlo/translate/mhlo_to_hlo/attribute_exporter.h"

#include <cstdint>
#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/Register.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/replica_group.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::NotNull;
using ::testing::Optional;
using ::tsl::proto_testing::EqualsProto;

class AttributeExporterTest : public ::testing::Test {
 protected:
  AttributeExporterTest() {
    registry_.insert<mlir::func::FuncDialect, mlir::sdy::SdyDialect,
                     mlir::mhlo::MhloDialect>();
    mlir::stablehlo::registerAllDialects(registry_);
    context_ = std::make_unique<mlir::MLIRContext>(registry_);
    context_->loadDialect<mlir::mhlo::MhloDialect>();
  }

  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseMlirModule(
      absl::string_view mlir_source) {
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(context_.get());
    auto module =
        mlir::parseSourceString<mlir::ModuleOp>(mlir_source, context_.get());
    RETURN_IF_ERROR(diagnostic_handler.ConsumeStatus());
    return module;
  }

  mlir::DialectRegistry registry_;
  std::unique_ptr<mlir::MLIRContext> context_;
};

TEST_F(AttributeExporterTest, ExtractShardyArgShardingFromFrontendAttrs) {
  constexpr absl::string_view mlir_source = R"mlir(
    module attributes {mhlo.frontend_attributes = {xla.sdy.meshes =
      "{mesh = #sdy.mesh<[\"x\"=2, \"y\"=2]>}"
    }} {
      func.func @main(
      %arg0: tensor<8x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\"x\"}, {}]>"}},
      %arg1: tensor<8x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\"y\"}]>"}}
      ) -> tensor<8x8xf32> {
        %0 = stablehlo.add %arg0, %arg1 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
        return %0 : tensor<8x8xf32>
      }
    }
  )mlir";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModule(mlir_source));

  mlir::func::FuncOp main = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_THAT(main, NotNull());

  std::optional<mlir::DictionaryAttr> sdy_meshes =
      xla::sdy::tryGetFrontendAttr<mlir::DictionaryAttr>(
          *module, xla::sdy::kMeshesRoundTripAttr);
  ASSERT_TRUE(sdy_meshes.has_value());

  // Check the sharding on the first argument.
  std::optional<OpSharding> sharding =
      ExtractShardyArgShardingFromFrontendAttrs(main, 0, sdy_meshes);

  TF_ASSERT_OK_AND_ASSIGN(
      xla::HloSharding expected_sharding,
      xla::ParseSharding("{devices=[2,1,2]<=[4] last_tile_dim_replicate}"));
  EXPECT_THAT(sharding, Optional(EqualsProto(expected_sharding.ToProto())));

  // Check the sharding on the second argument.
  sharding = ExtractShardyArgShardingFromFrontendAttrs(main, 1, sdy_meshes);

  TF_ASSERT_OK_AND_ASSIGN(
      expected_sharding,
      xla::ParseSharding(
          "{devices=[1,2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"));
  EXPECT_THAT(sharding, Optional(EqualsProto(expected_sharding.ToProto())));
}

TEST_F(AttributeExporterTest,
       ExtractShardyArgShardingFromFrontendAttrsInlinedMesh) {
  constexpr absl::string_view mlir_source = R"mlir(
    module @test {
      func.func @main(
      %arg0: tensor<8x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<mesh<[\"x\"=2, \"y\"=2]>, [{\"x\"}, {}]>"}},
      %arg1: tensor<8x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<mesh<[\"x\"=2, \"y\"=2]>, [{}, {\"y\"}]>"}}
      ) -> tensor<8x8xf32> {
        %0 = stablehlo.add %arg0, %arg1 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
        return %0 : tensor<8x8xf32>
      }
    }
  )mlir";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModule(mlir_source));

  mlir::func::FuncOp main = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_THAT(main, NotNull());

  std::optional<mlir::DictionaryAttr> sdy_meshes =
      xla::sdy::tryGetFrontendAttr<mlir::DictionaryAttr>(
          *module, xla::sdy::kMeshesRoundTripAttr);

  // Check the sharding on the first argument.
  std::optional<OpSharding> sharding =
      ExtractShardyArgShardingFromFrontendAttrs(main, 0, sdy_meshes);

  TF_ASSERT_OK_AND_ASSIGN(
      xla::HloSharding expected_sharding,
      xla::ParseSharding("{devices=[2,1,2]<=[4] last_tile_dim_replicate}"));
  EXPECT_THAT(sharding, Optional(EqualsProto(expected_sharding.ToProto())));

  // Check the sharding on the second argument.
  sharding = ExtractShardyArgShardingFromFrontendAttrs(main, 1, sdy_meshes);

  TF_ASSERT_OK_AND_ASSIGN(
      expected_sharding,
      xla::ParseSharding(
          "{devices=[1,2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"));
  EXPECT_THAT(sharding, Optional(EqualsProto(expected_sharding.ToProto())));
}

TEST_F(AttributeExporterTest,
       ExtractShardyArgShardingFromFrontendAttrsNoSharding) {
  constexpr absl::string_view mlir_source = R"mlir(
    module @test {
      func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
        %0 = stablehlo.add %arg0, %arg0 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
        return %0 : tensor<8x8xf32>
      }
    }
  )mlir";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModule(mlir_source));

  mlir::func::FuncOp main = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_THAT(main, NotNull());

  std::optional<OpSharding> sharding =
      ExtractShardyArgShardingFromFrontendAttrs(main, 0, std::nullopt);
  EXPECT_FALSE(sharding.has_value());
}

TEST_F(AttributeExporterTest, ExtractShardyResultShardingFromFrontendAttrs) {
  constexpr absl::string_view mlir_source = R"mlir(
    module attributes {mhlo.frontend_attributes = {xla.sdy.meshes =
      "{mesh = #sdy.mesh<[\"x\"=2, \"y\"=4, \"z\"=4]>}"
    }} {
      func.func @main(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
        %0 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%arg0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"x\", \"y\", ?}, {\"z\"}]>]>"}} : (tensor<8x8xf32>) -> tensor<8x8xf32>
        %1 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%arg0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}]>]>"}} : (tensor<8x8xf32>) -> tensor<8x8xf32>
        return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
      }
    }
  )mlir";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModule(mlir_source));

  mlir::func::FuncOp main = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_THAT(main, NotNull());

  std::optional<mlir::DictionaryAttr> sdy_meshes =
      xla::sdy::tryGetFrontendAttr<mlir::DictionaryAttr>(
          *module, xla::sdy::kMeshesRoundTripAttr);
  ASSERT_TRUE(sdy_meshes.has_value());

  // Check the sharding on the first result.
  std::optional<OpSharding> sharding =
      ExtractShardyResultShardingFromFrontendAttrs(main, 0, sdy_meshes);

  TF_ASSERT_OK_AND_ASSIGN(xla::HloSharding expected_sharding,
                          xla::ParseSharding("{devices=[8,4]<=[32]}"));
  EXPECT_THAT(sharding, Optional(EqualsProto(expected_sharding.ToProto())));

  // Check the sharding on the second result.
  sharding = ExtractShardyResultShardingFromFrontendAttrs(main, 1, sdy_meshes);

  TF_ASSERT_OK_AND_ASSIGN(expected_sharding,
                          xla::ParseSharding("{replicated}"));
  EXPECT_THAT(sharding, Optional(EqualsProto(expected_sharding.ToProto())));
}

TEST_F(AttributeExporterTest,
       ExtractShardyResultShardingFromFrontendAttrsInlinedMesh) {
  constexpr absl::string_view mlir_source = R"mlir(
    module @test {
      func.func @main(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
        %0 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%arg0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<mesh<[\"x\"=2, \"y\"=4, \"z\"=4]>, [{\"x\", \"y\", ?}, {\"z\"}]>]>"}} : (tensor<8x8xf32>) -> tensor<8x8xf32>
        %1 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%arg0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<mesh<[\"x\"=2, \"y\"=4, \"z\"=4]>, [{}, {}]>]>"}} : (tensor<8x8xf32>) -> tensor<8x8xf32>
        return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
      }
    }
  )mlir";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModule(mlir_source));

  mlir::func::FuncOp main = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_THAT(main, NotNull());

  std::optional<mlir::DictionaryAttr> sdy_meshes =
      xla::sdy::tryGetFrontendAttr<mlir::DictionaryAttr>(
          *module, xla::sdy::kMeshesRoundTripAttr);

  // Check the sharding on the first result.
  std::optional<OpSharding> sharding =
      ExtractShardyResultShardingFromFrontendAttrs(main, 0, sdy_meshes);

  TF_ASSERT_OK_AND_ASSIGN(xla::HloSharding expected_sharding,
                          xla::ParseSharding("{devices=[8,4]<=[32]}"));
  EXPECT_THAT(sharding, Optional(EqualsProto(expected_sharding.ToProto())));

  // Check the sharding on the second result.
  sharding = ExtractShardyResultShardingFromFrontendAttrs(main, 1, sdy_meshes);

  TF_ASSERT_OK_AND_ASSIGN(expected_sharding,
                          xla::ParseSharding("{replicated}"));
  EXPECT_THAT(sharding, Optional(EqualsProto(expected_sharding.ToProto())));
}

TEST_F(AttributeExporterTest,
       ExtractShardyResultShardingFromFrontendAttrsNoSharding) {
  constexpr absl::string_view mlir_source = R"mlir(
    module @test {
      func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
        %0 = stablehlo.add %arg0, %arg0 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
        return %0 : tensor<8x8xf32>
      }
    }
  )mlir";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModule(mlir_source));

  mlir::func::FuncOp main = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_THAT(main, NotNull());

  std::optional<OpSharding> sharding =
      ExtractShardyResultShardingFromFrontendAttrs(main, 0, std::nullopt);
  EXPECT_FALSE(sharding.has_value());
}

TEST_F(AttributeExporterTest, ConvertReplicaGroups_MeshAxesFromFrontendAttrs) {
  constexpr absl::string_view mlir_source = R"mlir(
    module @jit_f attributes {
      mhlo.frontend_attributes = {
        xla.sdy.meshes = "{mesh = #sdy.mesh<[\"data\"=2, \"seq\"=2]>}"
      }
    } {
      func.func private @mesh()
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %0 = "stablehlo.all_reduce"(%arg0) <{
          channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
          replica_groups = #stablehlo.replica_group_mesh_axes<
            mesh = @mesh,
            axes = [#stablehlo.axis_ref<name = "seq">]
          >,
          use_global_device_ids
        }> ({
        ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
          %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
          "stablehlo.return"(%1) : (tensor<f32>) -> ()
        }) : (tensor<f32>) -> tensor<f32>
        return %0 : tensor<f32>
      }
    }
  )mlir";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModule(mlir_source));

  mlir::func::FuncOp main = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_THAT(main, NotNull());

  mlir::Operation* op = &main.getBody().front().front();
  auto replica_groups = op->getAttr("replica_groups");
  ASSERT_TRUE(replica_groups);

  TF_ASSERT_OK_AND_ASSIGN(auto device_list,
                          xla::ConvertReplicaGroups(replica_groups, op));
  EXPECT_EQ(device_list->version(),
            xla::CollectiveDeviceListVersion::kMeshAxes);
  EXPECT_EQ(device_list->ToString(), "mesh['data'=2,'seq'=2] {'seq'}");
}

TEST_F(AttributeExporterTest,
       ConvertReplicaGroupsStableHloMeshAxesMultipleMeshes) {
  constexpr absl::string_view mlir_source = R"mlir(
    module @jit_f {
      sdy.mesh @mesh = <["replica"=1, "data"=2, "seq"=1, "model"=1]>
      sdy.mesh @mesh_1 = <["replica"=1, "data"=1, "seq"=2, "model"=1]>
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %0 = "stablehlo.all_reduce"(%arg0) <{
          channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
          replica_groups = #stablehlo.replica_group_mesh_axes<
            mesh = @mesh_1,
            axes = [#stablehlo.axis_ref<name = "seq">]
          >,
          use_global_device_ids
        }> ({
        ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
          %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
          "stablehlo.return"(%1) : (tensor<f32>) -> ()
        }) : (tensor<f32>) -> tensor<f32>
        return %0 : tensor<f32>
      }
    }
  )mlir";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModule(mlir_source));

  mlir::func::FuncOp main = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_THAT(main, NotNull());

  mlir::Operation* op = &main.getBody().front().front();
  auto replica_groups = op->getAttr("replica_groups");
  ASSERT_TRUE(replica_groups);

  TF_ASSERT_OK_AND_ASSIGN(auto device_list,
                          xla::ConvertReplicaGroups(replica_groups, op));
  EXPECT_EQ(device_list->version(),
            xla::CollectiveDeviceListVersion::kMeshAxes);
  EXPECT_EQ(device_list->ToString(),
            "mesh['replica'=1,'data'=1,'seq'=2,'model'=1] {'seq'}");
  EXPECT_EQ(device_list->ToString(true), "{{0,1}}");
}

TEST_F(AttributeExporterTest, ConvertReplicaGroupsMhloMeshAxesMultipleMeshes) {
  constexpr absl::string_view mlir_source = R"mlir(
    module @jit_f {
      sdy.mesh @mesh = <["replica"=1, "data"=2, "seq"=1, "model"=1]>
      sdy.mesh @mesh_1 = <["replica"=1, "data"=1, "seq"=2, "model"=1]>

      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %0 = "mhlo.all_reduce"(%arg0) <{
          channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
          replica_groups = #mhlo.replica_group_mesh_axes<
            mesh = @mesh_1,
            axes = [#mhlo.axis_ref<name = "seq">]
          >,
          use_global_device_ids
        }> ({
        ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
          %1 = "mhlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
          "mhlo.return"(%1) : (tensor<f32>) -> ()
        }) : (tensor<f32>) -> tensor<f32>
        return %0 : tensor<f32>
      }
    }
  )mlir";
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModule(mlir_source));

  mlir::func::FuncOp main = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_THAT(main, NotNull());

  mlir::Operation* op = &main.getBody().front().front();
  auto replica_groups = op->getAttr("replica_groups");
  ASSERT_TRUE(replica_groups);

  TF_ASSERT_OK_AND_ASSIGN(auto device_list,
                          xla::ConvertReplicaGroups(replica_groups, op));
  EXPECT_EQ(device_list->version(),
            xla::CollectiveDeviceListVersion::kMeshAxes);
  EXPECT_EQ(device_list->ToString(),
            "mesh['replica'=1,'data'=1,'seq'=2,'model'=1] {'seq'}");
  EXPECT_EQ(device_list->ToString(true), "{{0,1}}");
}

TEST_F(AttributeExporterTest, ConvertOriginalValueNullAttr) {
  mlir::mhlo::OriginalValueAttr attr;
  std::optional<xla::OriginalValueProto> proto = ConvertOriginalValue(attr);
  EXPECT_FALSE(proto.has_value());
}

TEST_F(AttributeExporterTest, ConvertOriginalValueSyntheticCall) {
  mlir::mhlo::OriginalValueAttr attr = mlir::mhlo::OriginalValueAttr::get(
      context_.get(), /*is_synthetic_call=*/true,
      /*elements=*/llvm::ArrayRef<mlir::mhlo::OriginalValueElementAttr>());

  std::optional<xla::OriginalValueProto> proto = ConvertOriginalValue(attr);
  ASSERT_TRUE(proto.has_value());
  EXPECT_TRUE(proto->is_synthetic_call());
  EXPECT_EQ(proto->elements_size(), 0);
}

TEST_F(AttributeExporterTest, ConvertOriginalValueWithElements) {
  mlir::mhlo::OriginalArrayAttr array_attr = mlir::mhlo::OriginalArrayAttr::get(
      context_.get(), mlir::StringAttr::get(context_.get(), "inst1"),
      llvm::ArrayRef<int64_t>({1, 2}));

  mlir::mhlo::OriginalValueElementAttr el_attr1 =
      mlir::mhlo::OriginalValueElementAttr::get(
          context_.get(), llvm::ArrayRef<int64_t>({0}), array_attr);

  mlir::mhlo::OriginalValueElementAttr el_attr2 =
      mlir::mhlo::OriginalValueElementAttr::get(
          context_.get(), llvm::ArrayRef<int64_t>({1}), std::nullopt);

  mlir::mhlo::OriginalValueAttr attr = mlir::mhlo::OriginalValueAttr::get(
      context_.get(), /*is_synthetic_call=*/false,
      llvm::ArrayRef<mlir::mhlo::OriginalValueElementAttr>(
          {el_attr1, el_attr2}));

  std::optional<xla::OriginalValueProto> proto = ConvertOriginalValue(attr);
  ASSERT_TRUE(proto.has_value());
  EXPECT_FALSE(proto->is_synthetic_call());
  ASSERT_EQ(proto->elements_size(), 2);

  const auto& el1 = proto->elements(0);
  EXPECT_THAT(el1.shape_index(), ::testing::ElementsAre(0));
  ASSERT_TRUE(el1.has_original_array());
  EXPECT_EQ(el1.original_array().instruction_name(), "inst1");
  EXPECT_THAT(el1.original_array().shape_index(), ::testing::ElementsAre(1, 2));

  const auto& el2 = proto->elements(1);
  EXPECT_THAT(el2.shape_index(), ::testing::ElementsAre(1));
  EXPECT_FALSE(el2.has_original_array());
}

TEST_F(AttributeExporterTest, ConvertOriginalValueFromWhileOp) {
  constexpr absl::string_view mlir_source = R"mlir(
    module @test {
      func.func @main(%arg0: tensor<i32> {mhlo.original_value = #mhlo.original_value<false, [<[], <"a", []>>]>}, %arg1: tensor<i32>, %arg2: tensor<i32> {mhlo.original_value = #mhlo.original_value<false, [<[], <"c", []>>]>}) -> (tensor<i32>) {
        %c = stablehlo.constant {mhlo.original_value = #mhlo.original_value<false, [<[], <"c", []>>]>} dense<0> : tensor<i32>
        %0:2 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0) : tensor<i32>, tensor<i32> attributes {mhlo.original_value = #mhlo.original_value<false, [<[0], <"while_0", [0]>>, <[1], <"while_1", [1]>>]>}
        cond {
          %total = stablehlo.add %arg1, %arg2 : tensor<i32>
          %1 = stablehlo.compare  LT, %iterArg, %total : (tensor<i32>, tensor<i32>) -> tensor<i1>
          stablehlo.return %1 : tensor<i1>
        } do {
          %1 = stablehlo.add %iterArg, %iterArg_0 : tensor<i32>
          stablehlo.return %1, %1: tensor<i32>, tensor<i32>
        }
        return %0#1 : tensor<i32>
      }
    }
  )mlir";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       ParseMlirModule(mlir_source));

  ASSERT_OK_AND_ASSIGN(auto hlo_module, ConvertMlirHloToHloModule(*module));

  xla::HloInstruction* while_instr = nullptr;
  for (xla::HloInstruction* instr :
       hlo_module->entry_computation()->instructions()) {
    if (instr->opcode() == xla::HloOpcode::kWhile) {
      while_instr = instr;
      break;
    }
  }
  ASSERT_THAT(while_instr, NotNull());

  std::shared_ptr<xla::OriginalValue> original_value =
      while_instr->original_value();
  ASSERT_THAT(original_value, NotNull());
  EXPECT_FALSE(original_value->is_synthetic_call());

  // Element 0: from "while_0"
  auto val0 = original_value->original_array({0});
  ASSERT_TRUE(val0.has_value());
  EXPECT_EQ(val0->instruction_name, "while_0");
  EXPECT_THAT(val0->shape_index, ::testing::ElementsAre(0));

  // Element 1: from "while_1"
  auto val1 = original_value->original_array({1});
  ASSERT_TRUE(val1.has_value());
  EXPECT_EQ(val1->instruction_name, "while_1");
  EXPECT_THAT(val1->shape_index, ::testing::ElementsAre(1));

  // Element 2: from %arg1 (has no original value, so empty placeholder)
  auto val2 = original_value->original_array({2});
  EXPECT_FALSE(val2.has_value());

  // Element 3: from %arg2 (has original value "c" with shape index empty)
  auto val3 = original_value->original_array({3});
  ASSERT_TRUE(val3.has_value());
  EXPECT_EQ(val3->instruction_name, "c");
  EXPECT_TRUE(val3->shape_index.empty());

  xla::HloVerifier verifier(/*layout_sensitive=*/false,
                            /*allow_mixed_precision=*/false);
  EXPECT_OK(verifier.Run(hlo_module.get()).status());
}

TEST_F(AttributeExporterTest, ProjectOriginalValueProtoSingleResult) {
  xla::OriginalValueProto proto;
  proto.set_is_synthetic_call(false);
  auto* el = proto.add_elements();
  el->add_shape_index(0);
  auto* array = el->mutable_original_array();
  array->set_instruction_name("a");
  array->add_shape_index(0);

  // When num_results <= 1, ProjectOriginalValueProto should return the original
  // value proto unchanged if index < num_results.
  std::optional<xla::OriginalValueProto> projected1 =
      ProjectOriginalValueProto(proto, /*index=*/0, /*num_results=*/1);
  ASSERT_TRUE(projected1.has_value());
  EXPECT_FALSE(projected1->is_synthetic_call());
  ASSERT_EQ(projected1->elements_size(), 1);
  EXPECT_THAT(projected1->elements(0).shape_index(), ::testing::ElementsAre(0));
  EXPECT_EQ(projected1->elements(0).original_array().instruction_name(), "a");

  // When index >= num_results, ProjectOriginalValueProto should return
  // std::nullopt.
  std::optional<xla::OriginalValueProto> projected_oob =
      ProjectOriginalValueProto(proto, /*index=*/1, /*num_results=*/1);
  EXPECT_FALSE(projected_oob.has_value());
}

TEST_F(AttributeExporterTest, ProjectOriginalValueProtoMultiResult) {
  xla::OriginalValueProto proto;
  proto.set_is_synthetic_call(false);

  // Element 1: index [0, 1] -> "a" index [0]
  auto* el1 = proto.add_elements();
  el1->add_shape_index(0);
  el1->add_shape_index(1);
  el1->mutable_original_array()->set_instruction_name("a");
  el1->mutable_original_array()->add_shape_index(0);

  // Element 2: index [1, 2] -> "b" index [1]
  auto* el2 = proto.add_elements();
  el2->add_shape_index(1);
  el2->add_shape_index(2);
  el2->mutable_original_array()->set_instruction_name("b");
  el2->mutable_original_array()->add_shape_index(1);

  // Project result index 0 of 2 results:
  std::optional<xla::OriginalValueProto> projected0 =
      ProjectOriginalValueProto(proto, /*index=*/0, /*num_results=*/2);
  ASSERT_TRUE(projected0.has_value());
  ASSERT_EQ(projected0->elements_size(), 1);
  EXPECT_THAT(projected0->elements(0).shape_index(), ::testing::ElementsAre(1));
  EXPECT_EQ(projected0->elements(0).original_array().instruction_name(), "a");

  // Project result index 1 of 2 results:
  std::optional<xla::OriginalValueProto> projected1 =
      ProjectOriginalValueProto(proto, /*index=*/1, /*num_results=*/2);
  ASSERT_TRUE(projected1.has_value());
  ASSERT_EQ(projected1->elements_size(), 1);
  EXPECT_THAT(projected1->elements(0).shape_index(), ::testing::ElementsAre(2));
  EXPECT_EQ(projected1->elements(0).original_array().instruction_name(), "b");

  // When index >= num_results, ProjectOriginalValueProto should return
  // std::nullopt.
  std::optional<xla::OriginalValueProto> projected_oob =
      ProjectOriginalValueProto(proto, /*index=*/2, /*num_results=*/2);
  EXPECT_FALSE(projected_oob.has_value());
}

TEST_F(AttributeExporterTest, ProjectOriginalValueProtoIndexOutOfBounds) {
  xla::OriginalValueProto proto;
  proto.set_is_synthetic_call(true);

  // When index >= num_results, ProjectOriginalValueProto should return
  // std::nullopt.
  std::optional<xla::OriginalValueProto> projected_oob =
      ProjectOriginalValueProto(proto, /*index=*/2, /*num_results=*/2);
  EXPECT_FALSE(projected_oob.has_value());
}

TEST_F(AttributeExporterTest, ProjectOriginalValueProtoNullOpt) {
  std::optional<xla::OriginalValueProto> projected =
      ProjectOriginalValueProto(std::nullopt, /*index=*/0, /*num_results=*/1);
  EXPECT_FALSE(projected.has_value());
}

TEST_F(AttributeExporterTest, ProjectOriginalValueProtoEmptyLeaf) {
  // Test case where the input only has an empty original value element, and its
  // shape index is empty.
  xla::OriginalValueProto empty_proto;
  empty_proto.set_is_synthetic_call(false);
  empty_proto.add_elements();

  std::optional<xla::OriginalValueProto> projected_empty =
      ProjectOriginalValueProto(empty_proto, /*index=*/0, /*num_results=*/1);
  ASSERT_TRUE(projected_empty.has_value());
  EXPECT_FALSE(projected_empty->is_synthetic_call());
  ASSERT_EQ(projected_empty->elements_size(), 1);
  EXPECT_TRUE(projected_empty->elements(0).shape_index().empty());
  EXPECT_FALSE(projected_empty->elements(0).has_original_array());
}

}  // namespace
}  // namespace xla
