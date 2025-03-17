/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/bridge_logger.h"

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Define test modules that are deserialized to module ops.
static const char *const module_with_add =
    R"(module {
func.func @main(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> {
  %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
  func.return %0 : tensor<3x4x5xf32>
}
}
)";

static const char *const module_with_sub =
    R"(module {
func.func @main(%arg0: tensor<7x8x9xi8>, %arg1: tensor<7x8x9xi8>) -> tensor<7x8x9xi8> {
  %0 = "tf.Sub"(%arg0, %arg1) : (tensor<7x8x9xi8>, tensor<7x8x9xi8>) -> tensor<7x8x9xi8>
  func.return %0 : tensor<7x8x9xi8>
}
}
)";

void UnsetEnvironmentVariables() {
  unsetenv(/*name=*/"MLIR_BRIDGE_LOG_PASS_FILTER");
  unsetenv(/*name=*/"MLIR_BRIDGE_LOG_STRING_FILTER");
  unsetenv(/*name=*/"MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES");
}

class BridgeLoggerFilters : public ::testing::Test {
 protected:
  void SetUp() override { UnsetEnvironmentVariables(); }

  mlir::MLIRContext CreateMlirContext() {
    mlir::DialectRegistry mlir_registry;
    mlir::RegisterAllTensorFlowDialects(mlir_registry);
    return mlir::MLIRContext(mlir_registry);
  }

  mlir::func::FuncOp GetFuncOp(mlir::ModuleOp module_op) {
    auto func_ops = module_op.getOps<mlir::func::FuncOp>();
    EXPECT_FALSE(func_ops.empty());
    return *func_ops.begin();
  }
};

// Test pass filter.
TEST_F(BridgeLoggerFilters, TestPassFilter) {
  mlir::MLIRContext mlir_context = CreateMlirContext();
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_add;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_add, &mlir_context,
                                     &mlir_module_with_add));

  std::unique_ptr<mlir::Pass> partitioning_pass =
      mlir::TFTPU::CreateTPUResourceReadsWritesPartitioningPass();
  std::unique_ptr<mlir::Pass> shape_inference_pass =
      mlir::TF::CreateTFShapeInferencePass();
  std::unique_ptr<mlir::Pass> inliner_pass = mlir::createInlinerPass();

  // partitioning_pass and shape_inference_pass should match the filter,
  // inliner_pass should not.
  setenv(/*name=*/"MLIR_BRIDGE_LOG_PASS_FILTER",
         /*value=*/
         "TPUResourceReadsWritesPartitioningPass;TensorFlowShapeInferencePass",
         /*overwrite=*/1);
  BridgeLoggerConfig logger_config;
  EXPECT_TRUE(logger_config.ShouldPrint(partitioning_pass.get(),
                                        mlir_module_with_add.get()));
  EXPECT_TRUE(logger_config.ShouldPrint(shape_inference_pass.get(),
                                        mlir_module_with_add.get()));
  EXPECT_FALSE(logger_config.ShouldPrint(inliner_pass.get(),
                                         mlir_module_with_add.get()));
}

// Test string filter.
TEST_F(BridgeLoggerFilters, TestStringFilter) {
  mlir::MLIRContext mlir_context = CreateMlirContext();
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_add, mlir_module_with_sub;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_add, &mlir_context,
                                     &mlir_module_with_add));
  TF_ASSERT_OK(DeserializeMlirModule(module_with_sub, &mlir_context,
                                     &mlir_module_with_sub));
  // The pass is not relevant for this test since we don't define a pass filter.
  std::unique_ptr<mlir::Pass> dummy_pass =
      mlir::TF::CreateTFShapeInferencePass();

  // One string appears in both modules and the other one not.
  setenv(/*name=*/"MLIR_BRIDGE_LOG_STRING_FILTER",
         /*value=*/"func @main(%arg0: tensor;XXX", /*overwrite=*/1);
  BridgeLoggerConfig logger_config1;
  EXPECT_TRUE(
      logger_config1.ShouldPrint(dummy_pass.get(), mlir_module_with_add.get()));
  EXPECT_TRUE(
      logger_config1.ShouldPrint(dummy_pass.get(), mlir_module_with_sub.get()));

  // Both strings do not appear in any module.
  setenv(/*name=*/"MLIR_BRIDGE_LOG_STRING_FILTER",
         /*value=*/"func @main(%arg0:tensor;XXX", /*overwrite=*/1);
  BridgeLoggerConfig logger_config2;
  EXPECT_FALSE(
      logger_config2.ShouldPrint(dummy_pass.get(), mlir_module_with_add.get()));
  EXPECT_FALSE(
      logger_config2.ShouldPrint(dummy_pass.get(), mlir_module_with_sub.get()));

  // String appears in one module but not in the other.
  setenv(/*name=*/"MLIR_BRIDGE_LOG_STRING_FILTER",
         /*value=*/"\"tf.AddV2\"(%arg0, %arg1) : (tensor<3x4x5xf32>",
         /*overwrite=*/1);
  BridgeLoggerConfig logger_config3;
  EXPECT_TRUE(
      logger_config3.ShouldPrint(dummy_pass.get(), mlir_module_with_add.get()));
  EXPECT_FALSE(
      logger_config3.ShouldPrint(dummy_pass.get(), mlir_module_with_sub.get()));
}

// Test enable only top level passes filter.
TEST_F(BridgeLoggerFilters, TestEnableOnlyTopLevelPassesFilter) {
  mlir::MLIRContext mlir_context = CreateMlirContext();
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_add;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_add, &mlir_context,
                                     &mlir_module_with_add));

  std::unique_ptr<mlir::Pass> shape_inference_pass =
      mlir::TF::CreateTFShapeInferencePass();

  BridgeLoggerConfig logger_config;
  // ShouldPrint returns true for the top-level module operation.
  EXPECT_TRUE(logger_config.ShouldPrint(shape_inference_pass.get(),
                                        mlir_module_with_add.get()));
  // Find the nested function operation within the module.
  mlir::func::FuncOp func_op = GetFuncOp(mlir_module_with_add.get());
  // ShouldPrint returns true for the nested function operation.
  EXPECT_TRUE(logger_config.ShouldPrint(shape_inference_pass.get(), func_op));

  // Set the environment variable to enable only top-level passes.
  setenv(/*name=*/"MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES", /*value=*/"1",
         /*overwrite=*/1);
  BridgeLoggerConfig logger_config_filter;
  // ShouldPrint returns false for the nested function operation.
  EXPECT_FALSE(
      logger_config_filter.ShouldPrint(shape_inference_pass.get(), func_op));
}

// Additional tests for various possible values of
// MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES.
TEST_F(BridgeLoggerFilters, TestEnableOnlyTopLevelPassesEnvVarValues) {
  mlir::MLIRContext mlir_context = CreateMlirContext();
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_add;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_add, &mlir_context,
                                     &mlir_module_with_add));

  std::unique_ptr<mlir::Pass> shape_inference_pass =
      mlir::TF::CreateTFShapeInferencePass();

  mlir::ModuleOp module_op = *mlir_module_with_add;
  // Find the nested function operation within the module.
  mlir::func::FuncOp func_op = GetFuncOp(module_op);

  // Test with environment variable set to "FALSE".
  setenv(/*name=*/"MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES",
         /*value=*/"FALSE", /*overwrite=*/1);
  BridgeLoggerConfig logger_config_false;
  // ShouldPrint should return true for top-level operation.
  EXPECT_TRUE(logger_config_false.ShouldPrint(shape_inference_pass.get(),
                                              mlir_module_with_add.get()));
  // ShouldPrint should return true for nested function.
  EXPECT_TRUE(
      logger_config_false.ShouldPrint(shape_inference_pass.get(), func_op));

  // Test with environment variable unset (default behavior).
  unsetenv(/*name=*/"MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES");
  BridgeLoggerConfig logger_config_default;
  // ShouldPrint should return true for top-level operation.
  EXPECT_TRUE(logger_config_default.ShouldPrint(shape_inference_pass.get(),
                                                mlir_module_with_add.get()));
  // ShouldPrint should return true for nested function since default
  // is disabled.
  EXPECT_TRUE(
      logger_config_default.ShouldPrint(shape_inference_pass.get(), func_op));

  // Test with environment variable set to "1".
  setenv(/*name=*/"MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES", /*value=*/"1",
         /*overwrite=*/1);
  BridgeLoggerConfig logger_config_one;
  // ShouldPrint should return false for nested function since filter
  // is enabled.
  EXPECT_FALSE(
      logger_config_one.ShouldPrint(shape_inference_pass.get(), func_op));
}

// Test combinations of pass filter and string filter.
TEST_F(BridgeLoggerFilters, TestPassFilterAndStringFilter) {
  mlir::MLIRContext mlir_context = CreateMlirContext();
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_add;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_add, &mlir_context,
                                     &mlir_module_with_add));

  std::unique_ptr<mlir::Pass> shape_inference_pass =
      mlir::TF::CreateTFShapeInferencePass();

  // String filter is matched but pass filter is not.
  setenv(/*name=*/"MLIR_BRIDGE_LOG_STRING_FILTER",
         /*value=*/
         "(tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> "
         "tensor<3x4x5xf32>",
         /*overwrite=*/1);
  setenv(/*name=*/"MLIR_BRIDGE_LOG_PASS_FILTER",
         /*value=*/"ensorFlowShapeInferencePass", /*overwrite=*/1);
  BridgeLoggerConfig logger_config1;
  EXPECT_FALSE(logger_config1.ShouldPrint(shape_inference_pass.get(),
                                          mlir_module_with_add.get()));

  // Pass filter is matched but string filter is not.
  setenv(/*name=*/"MLIR_BRIDGE_LOG_STRING_FILTER", /*value=*/"XXX", 1);
  setenv(/*name=*/"MLIR_BRIDGE_LOG_PASS_FILTER",
         /*value=*/"TensorFlowShapeInferencePass", /*overwrite=*/1);
  BridgeLoggerConfig logger_config2;
  EXPECT_FALSE(logger_config2.ShouldPrint(shape_inference_pass.get(),
                                          mlir_module_with_add.get()));

  // Both filters are matched.
  setenv(/*name=*/"MLIR_BRIDGE_LOG_STRING_FILTER",
         /*value=*/
         "(tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> "
         "tensor<3x4x5xf32>",
         /*overwrite=*/1);
  setenv(/*name=*/"MLIR_BRIDGE_LOG_PASS_FILTER",
         /*value=*/"TensorFlowShapeInferencePass", /*overwrite=*/1);
  BridgeLoggerConfig logger_config3;
  EXPECT_TRUE(logger_config3.ShouldPrint(shape_inference_pass.get(),
                                         mlir_module_with_add.get()));
}

// Test combinations of pass filter and enable only top level passes filter.
TEST_F(BridgeLoggerFilters, TestPassFilterAndEnableOnlyTopLevelPassesFilter) {
  mlir::MLIRContext mlir_context = CreateMlirContext();
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_sub;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_sub, &mlir_context,
                                     &mlir_module_with_sub));

  std::unique_ptr<mlir::Pass> shape_inference_pass =
      mlir::TF::CreateTFShapeInferencePass();
  std::unique_ptr<mlir::Pass> inliner_pass = mlir::createInlinerPass();

  // Find the nested function operation within the module.
  mlir::func::FuncOp func_op = GetFuncOp(mlir_module_with_sub.get());

  setenv(/*name=*/"MLIR_BRIDGE_LOG_PASS_FILTER",
         /*value=*/"TensorFlowShapeInferencePass", /*overwrite=*/1);
  BridgeLoggerConfig logger_config;
  // ShouldPrint should return true for top-level operation with matching pass
  // filter.
  EXPECT_TRUE(logger_config.ShouldPrint(shape_inference_pass.get(),
                                        mlir_module_with_sub.get()));
  // ShouldPrint should return true for nested operation when
  // enable_only_top_level_passes_ is false.
  EXPECT_TRUE(logger_config.ShouldPrint(shape_inference_pass.get(), func_op));
  // ShouldPrint should return false for pass not matching the pass filter.
  EXPECT_FALSE(logger_config.ShouldPrint(inliner_pass.get(),
                                         mlir_module_with_sub.get()));

  // Set the environment variable to enable only top-level passes.
  setenv(/*name=*/"MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES", /*value=*/"1",
         /*overwrite=*/1);
  BridgeLoggerConfig logger_config_filter;
  // ShouldPrint should return false for nested operation
  EXPECT_FALSE(
      logger_config_filter.ShouldPrint(shape_inference_pass.get(), func_op));
}

// Test combinations of string filter and enable only top level passes filter.
TEST_F(BridgeLoggerFilters, TestStringFilterAndEnableOnlyTopLevelPassesFilter) {
  mlir::MLIRContext mlir_context = CreateMlirContext();
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_add;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_add, &mlir_context,
                                     &mlir_module_with_add));

  std::unique_ptr<mlir::Pass> shape_inference_pass =
      mlir::TF::CreateTFShapeInferencePass();

  // Find the nested function operation within the module.
  mlir::func::FuncOp func_op = GetFuncOp(mlir_module_with_add.get());

  setenv(/*name=*/"MLIR_BRIDGE_LOG_STRING_FILTER", /*value=*/"tf.AddV2",
         /*overwrite=*/1);
  BridgeLoggerConfig logger_config;
  // ShouldPrint should return true for top-level operation containing
  // "tf.AddV2".
  EXPECT_TRUE(logger_config.ShouldPrint(shape_inference_pass.get(),
                                        mlir_module_with_add.get()));
  // ShouldPrint should return true for nested operation since
  // enable_only_top_level_passes_ is false.
  EXPECT_TRUE(logger_config.ShouldPrint(shape_inference_pass.get(), func_op));

  // Set the environment variable to enable only top-level passes.
  setenv(/*name=*/"MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES", /*value=*/"1",
         /*overwrite=*/1);
  BridgeLoggerConfig logger_config_filter;
  // ShouldPrint should return false for nested operation since string
  // filter matches but enable_only_top_level_passes_ is true.
  EXPECT_FALSE(
      logger_config_filter.ShouldPrint(shape_inference_pass.get(), func_op));

  // Change string filter to not match any operation.
  setenv(/*name=*/"MLIR_BRIDGE_LOG_STRING_FILTER", /*value=*/"NonExistentOp",
         /*overwrite=*/1);
  BridgeLoggerConfig logger_config_no_match;
  // ShouldPrint should return false since string filter does not match.
  EXPECT_FALSE(logger_config_no_match.ShouldPrint(shape_inference_pass.get(),
                                                  mlir_module_with_add.get()));
}

// Test combinations where all filters are set but none match.
TEST_F(BridgeLoggerFilters, TestAllFiltersNoMatch) {
  mlir::MLIRContext mlir_context = CreateMlirContext();
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_sub;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_sub, &mlir_context,
                                     &mlir_module_with_sub));

  std::unique_ptr<mlir::Pass> shape_inference_pass =
      mlir::TF::CreateTFShapeInferencePass();

  // Set pass filter to not match any pass
  setenv(/*name=*/"MLIR_BRIDGE_LOG_PASS_FILTER", /*value=*/"NonExistentPass",
         /*overwrite=*/1);
  // Set string filter to not match any string
  setenv(/*name=*/"MLIR_BRIDGE_LOG_STRING_FILTER", /*value=*/"NonExistentOp",
         /*overwrite=*/1);
  BridgeLoggerConfig logger_config;
  // ShouldPrint should return false since none of the filters match.
  EXPECT_FALSE(logger_config.ShouldPrint(shape_inference_pass.get(),
                                         mlir_module_with_sub.get()));

  // Set the environment variable to enable only top-level passes.
  setenv(/*name=*/"MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES", /*value=*/"1",
         /*overwrite=*/1);
  BridgeLoggerConfig logger_config_filter;
  // ShouldPrint should still return false since pass and string filters do not
  // match.
  EXPECT_FALSE(logger_config_filter.ShouldPrint(shape_inference_pass.get(),
                                                mlir_module_with_sub.get()));
}

// Test combinations of all three filters.
TEST_F(BridgeLoggerFilters, TestAllFiltersCombination) {
  mlir::MLIRContext mlir_context = CreateMlirContext();
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_add;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_add, &mlir_context,
                                     &mlir_module_with_add));

  std::unique_ptr<mlir::Pass> shape_inference_pass =
      mlir::TF::CreateTFShapeInferencePass();
  std::unique_ptr<mlir::Pass> inliner_pass = mlir::createInlinerPass();

  // Find the nested function operation within the module.
  mlir::func::FuncOp func_op = GetFuncOp(mlir_module_with_add.get());

  // Set all three filters.
  setenv(/*name=*/"MLIR_BRIDGE_LOG_PASS_FILTER",
         /*value=*/"TensorFlowShapeInferencePass", /*overwrite=*/1);
  setenv(/*name=*/"MLIR_BRIDGE_LOG_STRING_FILTER", /*value=*/"tf.AddV2",
         /*overwrite=*/1);
  BridgeLoggerConfig logger_config;
  // ShouldPrint should return true if all filters pass and operation is
  // top-level.
  EXPECT_TRUE(logger_config.ShouldPrint(shape_inference_pass.get(),
                                        mlir_module_with_add.get()));

  // ShouldPrint should return false if enable_only_top_level_passes_ is
  // true and operation is nested.
  setenv(/*name=*/"MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES", /*value=*/"1",
         /*overwrite=*/1);
  BridgeLoggerConfig logger_config_filter;
  EXPECT_FALSE(
      logger_config_filter.ShouldPrint(shape_inference_pass.get(), func_op));
  // Change to a pass that does not match the pass filter.
  EXPECT_FALSE(logger_config_filter.ShouldPrint(inliner_pass.get(),
                                                mlir_module_with_add.get()));
  // Set the environment variable to disable only top-level passes.
  unsetenv(/*name=*/"MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES");
  BridgeLoggerConfig logger_config_no_filter;
  // ShouldPrint should return true for nested operation since
  // enable_only_top_level_passes_ is false.
  EXPECT_TRUE(
      logger_config_no_filter.ShouldPrint(shape_inference_pass.get(), func_op));
  // Change to a pass that does not match the pass filter.
  EXPECT_FALSE(logger_config_no_filter.ShouldPrint(inliner_pass.get(),
                                                   mlir_module_with_add.get()));
}
}  // namespace
}  // namespace tensorflow
