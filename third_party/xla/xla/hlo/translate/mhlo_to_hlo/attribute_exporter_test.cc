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

#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/Register.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"
#include "xla/tsl/platform/errors.h"
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
  }

  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseMlirModule(
      absl::string_view mlir_source) {
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(context_.get());
    auto module =
        mlir::parseSourceString<mlir::ModuleOp>(mlir_source, context_.get());
    TF_RETURN_IF_ERROR(diagnostic_handler.ConsumeStatus());
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

TEST_F(AttributeExporterTest, ConvertReplicaGroups_MeshAxes) {
  constexpr absl::string_view mlir_source = R"mlir(
    module @main {
      sdy.mesh @mesh = <["a"=2, "b"=2], device_ids=[0, 1, 2, 3]>
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %0 = "stablehlo.all_reduce"(%arg0) <{
          channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
          replica_groups = #stablehlo.replica_group_mesh_axes<
            mesh_name = @mesh,
            axes = [#stablehlo.axis_ref<name = "a">, #stablehlo.axis_ref<name = "b">]
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
  EXPECT_EQ(device_list->num_replica_groups(), 1);
  EXPECT_EQ(device_list->num_devices_per_group(), 4);
}

}  // namespace
}  // namespace xla
