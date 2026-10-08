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

#include "xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/translate/register.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"

// This file should contain tests for interfaces that can't be tested at the
// MLIR level.

namespace mlir {
namespace {

using testing::_;
using testing::AllOf;
using testing::HasSubstr;

TEST(ConvertMlirHloToHloModuleTest, PropagatesDiagnostics) {
  const std::string mlir_source = R"mlir(
func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<1xindex>, %arg2: tensor<1xindex>, %arg3: tensor<1xindex>) -> tensor<?xf32> {
  %0 = shape.const_shape [14, 1] : tensor<2xindex>
  %1 = "stablehlo.real_dynamic_slice"(%arg0, %arg1, %arg2, %arg3) : (tensor<?xf32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>
}
)mlir";

  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    mlir::BaseScopedDiagnosticHandler handler(&context);
    module = mlir::parseSourceString<mlir::ModuleOp>(mlir_source, &context);
    TF_ASSERT_OK(handler.ConsumeStatus());
  }

  ASSERT_THAT(ConvertMlirHloToHloModule(*module),
              absl_testing::StatusIs(
                  _, AllOf(HasSubstr("Unable to prepare for XLA export"),
                           HasSubstr("real_dynamic_slice"))));
}

TEST(ConvertMlirHloToHloModuleTest, ConvertsDotGeneralPrecisionConfig) {
  const std::string mlir_source = R"mlir(
func.func @main(%arg0: tensor<5x10xbf16>, %arg1: tensor<10x5xbf16>) -> tensor<5x5xbf16> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<5x10xbf16>, tensor<10x5xbf16>) -> tensor<5x5xbf16>
  return %0 : tensor<5x5xbf16>
}
)mlir";

  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    mlir::BaseScopedDiagnosticHandler handler(&context);
    module = mlir::parseSourceString<mlir::ModuleOp>(mlir_source, &context);
    TF_ASSERT_OK(handler.ConsumeStatus());
  }

  TF_ASSERT_OK(ConvertMlirHloToHloModule(*module));
}
TEST(ConvertMlirHloToHloModuleTest, ConvertsConvolutionPrecisionConfig) {
  const std::string mlir_source = R"mlir(
func.func @main(%arg0: tensor<3x3x3x3xf32>, %arg1: tensor<3x3x3x3xf32>) -> tensor<3x3x3x3xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<3x3x3x3xf32>, tensor<3x3x3x3xf32>) -> tensor<3x3x3x3xf32>
  return %0 : tensor<3x3x3x3xf32>
}
)mlir";

  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    mlir::BaseScopedDiagnosticHandler handler(&context);
    module = mlir::parseSourceString<mlir::ModuleOp>(mlir_source, &context);
    TF_ASSERT_OK(handler.ConsumeStatus());
  }

  TF_ASSERT_OK(ConvertMlirHloToHloModule(*module));
}
TEST(ConvertMlirHloToHloModuleTest, ConvertsReplicaGroupMeshAxes) {
  const std::string kMlirModule = R"mlir(
    module @main {
      sdy.mesh @mesh = <["a"=2, "b"=2], device_ids=[0, 1, 2, 3]>
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %0 = "stablehlo.all_reduce"(%arg0) <{
          channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
          replica_groups = #stablehlo.replica_group_mesh_axes<
            mesh = @mesh,
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

  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);

  mlir::BaseScopedDiagnosticHandler handler(&context);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kMlirModule, &context);
  TF_ASSERT_OK(handler.ConsumeStatus());
  ASSERT_TRUE(module);
  auto hlo_module = ConvertMlirHloToHloModule(*module);
  TF_EXPECT_OK(hlo_module.status());
}

TEST(ConvertMlirHloToHloModuleTest, PacksSpmdParametersShardingsForTupleArgs) {
  const std::string kMlirModule = R"mlir(
    module attributes {
      mhlo.spmd_parameters_shardings = [
        "{devices=[1,2]<=[2]}",
        "{{replicated}, {devices=[2,1]<=[2]}}"
      ]
    } {
      func.func @main(
          %arg0: tensor<2x4xf32>,
          %arg1: tuple<tensor<f32>, tensor<2x4xf32>>) -> tensor<2x4xf32> {
        return %arg0 : tensor<2x4xf32>
      }
    }
  )mlir";

  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);

  mlir::BaseScopedDiagnosticHandler handler(&context);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kMlirModule, &context);
  TF_ASSERT_OK(handler.ConsumeStatus());
  ASSERT_TRUE(module);

  MlirToHloConversionOptions options;
  options.use_tuple_args = true;
  auto hlo_module = ConvertMlirHloToHloModule(*module, options);
  TF_ASSERT_OK(hlo_module.status());
  ASSERT_TRUE((*hlo_module)->has_spmd_parameters_shardings());
  ASSERT_EQ((*hlo_module)->spmd_parameters_shardings().size(), 1);
  EXPECT_EQ((*hlo_module)->spmd_parameters_shardings()[0].ToString(),
            "{{devices=[1,2]<=[2]}, {replicated}, {devices=[2,1]<=[2]}}");
}

absl::StatusOr<std::unique_ptr<xla::HloModule>> ParseAndConvert(
    llvm::StringRef mlir_source) {
  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::BaseScopedDiagnosticHandler handler(&context);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_source, &context);
  if (absl::Status status = handler.ConsumeStatus(); !status.ok()) {
    return status;
  }
  return ConvertMlirHloToHloModule(*module);
}

struct ScaledSparseTestCase {
  std::string test_name;
  std::string mlir_source;
  int64_t expected_operand_count;
  // Expected `sparsity_config` / `block_scaling_config` fragment of the
  // converted `dot` instruction, in HLO print order.
  std::string expected_config;
};

class ConvertScaledSparseDotGeneralTest
    : public ::testing::TestWithParam<ScaledSparseTestCase> {};

TEST_P(ConvertScaledSparseDotGeneralTest, ConvertsToDot) {
  const ScaledSparseTestCase& param = GetParam();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                       ParseAndConvert(param.mlir_source));
  const xla::HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  const xla::HloInstruction* dot =
      root->opcode() == xla::HloOpcode::kTuple ? root->operand(0) : root;
  ASSERT_EQ(dot->opcode(), xla::HloOpcode::kDot);
  EXPECT_EQ(dot->operand_count(), param.expected_operand_count);
  EXPECT_THAT(dot->ToString(), HasSubstr(param.expected_config));
}

INSTANTIATE_TEST_SUITE_P(
    ScaledSparseDotGeneralTests, ConvertScaledSparseDotGeneralTest,
    ::testing::Values(
        ScaledSparseTestCase{
            "BlockScaled",
            R"mlir(
func.func @main(%lhs: tensor<64x128xbf16>, %rhs: tensor<128x64xbf16>, %lhs_scale: tensor<64x4xf8E8M0FNU>, %rhs_scale: tensor<4x64xf8E8M0FNU>) -> tensor<64x64xbf16> {
  %0 = stablehlo.dot_general %lhs, %rhs, [%lhs_scale, %rhs_scale], contracting_dims = [1] x [0] {block_scaling_config = #stablehlo.block_scaling_config<lhs = <scale_idx = 2, strides = [1, 32], steps = [1, 1]>, rhs = <scale_idx = 3, strides = [32, 1], steps = [1, 1]>>} : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64x4xf8E8M0FNU>, tensor<4x64xf8E8M0FNU>) -> tensor<64x64xbf16>
  return %0 : tensor<64x64xbf16>
})mlir",
            4,
            "block_scaling_config={lhs={scale_idx=2 "
            "strides=1x32 steps=1x1} rhs={scale_idx=3 "
            "strides=32x1 steps=1x1}}",
        },
        ScaledSparseTestCase{
            "AsymmetricBlockScaledWithZeroPoint",
            R"mlir(
func.func @main(%lhs: tensor<64x128xbf16>, %rhs: tensor<128x64xbf16>, %lhs_scale: tensor<64x4xf8E8M0FNU>, %lhs_zp: tensor<64x4xf8E8M0FNU>, %rhs_scale: tensor<4x64xf8E8M0FNU>, %rhs_zp: tensor<4x64xf8E8M0FNU>) -> tensor<64x64xbf16> {
  %0 = stablehlo.dot_general %lhs, %rhs, [%lhs_scale, %lhs_zp, %rhs_scale, %rhs_zp], contracting_dims = [1] x [0] {block_scaling_config = #stablehlo.block_scaling_config<lhs = <scale_idx = 2, zero_idx = 3, strides = [1, 32], steps = [1, 1]>, rhs = <scale_idx = 4, zero_idx = 5, strides = [32, 1], steps = [1, 1]>>} : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64x4xf8E8M0FNU>, tensor<64x4xf8E8M0FNU>, tensor<4x64xf8E8M0FNU>, tensor<4x64xf8E8M0FNU>) -> tensor<64x64xbf16>
  return %0 : tensor<64x64xbf16>
})mlir",
            6,
            "block_scaling_config={lhs={scale_idx=2 zero_idx=3 "
            "strides=1x32 steps=1x1} rhs={scale_idx=4 zero_idx=5 "
            "strides=32x1 steps=1x1}}",
        },
        ScaledSparseTestCase{
            "RhsStructuredSparse",
            R"mlir(
func.func @main(%lhs: tensor<64x128xbf16>, %rhs: tensor<64x64xbf16>, %rhs_indices: tensor<64x16xi8>) -> tensor<64x64xbf16> {
  %0 = stablehlo.dot_general %lhs, %rhs, [%rhs_indices], contracting_dims = [1] x [0] {sparsity_config = #stablehlo.sparsity_config<rhs = <num_non_zero = 2, block_size = 4, dimension = 0, stride = 1, idx = 2>>} : (tensor<64x128xbf16>, tensor<64x64xbf16>, tensor<64x16xi8>) -> tensor<64x64xbf16>
  return %0 : tensor<64x64xbf16>
})mlir",
            3,
            "sparsity_config={rhs={sparsity=2x4 dimension=0 stride=1 "
            "idx=2}}",
        },
        ScaledSparseTestCase{
            "BatchedScaledSparse",
            R"mlir(
func.func @main(%lhs: tensor<2x64x64xbf16>, %rhs: tensor<2x128x64xbf16>, %lhs_scale: tensor<2x64x2xf8E8M0FNU>, %lhs_indices: tensor<2x64x16xi8>) -> tensor<2x64x64xbf16> {
  %0 = stablehlo.dot_general %lhs, %rhs, [%lhs_scale, %lhs_indices], batching_dims = [0] x [0], contracting_dims = [2] x [1] {block_scaling_config = #stablehlo.block_scaling_config<lhs = <scale_idx = 2, strides = [1, 1, 32], steps = [1, 1, 1]>>, sparsity_config = #stablehlo.sparsity_config<lhs = <num_non_zero = 2, block_size = 4, dimension = 2, stride = 1, idx = 3>>} : (tensor<2x64x64xbf16>, tensor<2x128x64xbf16>, tensor<2x64x2xf8E8M0FNU>, tensor<2x64x16xi8>) -> tensor<2x64x64xbf16>
  return %0 : tensor<2x64x64xbf16>
})mlir",
            4,
            "sparsity_config={lhs={sparsity=2x4 dimension=2 stride=1 "
            "idx=3}}, block_scaling_config={lhs={scale_idx=2 "
            "strides=1x1x32 steps=1x1x1}}",
        }),
    [](const auto& info) { return info.param.test_name; });

}  // namespace
}  // namespace mlir
