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

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
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

// Test pass filter.
TEST(BridgeLoggerFilters, TestPassFilter) {
  mlir::DialectRegistry mlir_registry;
  mlir::RegisterAllTensorFlowDialects(mlir_registry);
  mlir::MLIRContext mlir_context(mlir_registry);
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
  setenv("MLIR_BRIDGE_LOG_PASS_FILTER",
         "TPUResourceReadsWritesPartitioningPass;TensorFlowShapeInferencePass",
         1);
  BridgeLoggerConfig logger_config;
  EXPECT_TRUE(logger_config.ShouldPrint(partitioning_pass.get(),
                                        mlir_module_with_add.get()));
  EXPECT_TRUE(logger_config.ShouldPrint(shape_inference_pass.get(),
                                        mlir_module_with_add.get()));
  EXPECT_FALSE(logger_config.ShouldPrint(inliner_pass.get(),
                                         mlir_module_with_add.get()));
}

// Test string filter.
TEST(BridgeLoggerFilters, TestStringFilter) {
  mlir::DialectRegistry mlir_registry;
  mlir::RegisterAllTensorFlowDialects(mlir_registry);
  mlir::MLIRContext mlir_context(mlir_registry);
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_add, mlir_module_with_sub;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_add, &mlir_context,
                                     &mlir_module_with_add));
  TF_ASSERT_OK(DeserializeMlirModule(module_with_sub, &mlir_context,
                                     &mlir_module_with_sub));
  // The pass is not relevant for this test since we don't define a pass filter.
  std::unique_ptr<mlir::Pass> dummy_pass =
      mlir::TF::CreateTFShapeInferencePass();

  // One string appears in both modules and the other one not.
  setenv("MLIR_BRIDGE_LOG_STRING_FILTER", "func @main(%arg0: tensor;XXX", 1);
  BridgeLoggerConfig logger_config1;
  EXPECT_TRUE(
      logger_config1.ShouldPrint(dummy_pass.get(), mlir_module_with_add.get()));
  EXPECT_TRUE(
      logger_config1.ShouldPrint(dummy_pass.get(), mlir_module_with_sub.get()));

  // Both strings do not appear in any module.
  setenv("MLIR_BRIDGE_LOG_STRING_FILTER", "func @main(%arg0:tensor;XXX", 1);
  BridgeLoggerConfig logger_config2;
  EXPECT_FALSE(
      logger_config2.ShouldPrint(dummy_pass.get(), mlir_module_with_add.get()));
  EXPECT_FALSE(
      logger_config2.ShouldPrint(dummy_pass.get(), mlir_module_with_sub.get()));

  // String appears in one module but not in the other.
  setenv("MLIR_BRIDGE_LOG_STRING_FILTER",
         "\"tf.AddV2\"(%arg0, %arg1) : (tensor<3x4x5xf32>", 1);
  BridgeLoggerConfig logger_config3;
  EXPECT_TRUE(
      logger_config3.ShouldPrint(dummy_pass.get(), mlir_module_with_add.get()));
  EXPECT_FALSE(
      logger_config3.ShouldPrint(dummy_pass.get(), mlir_module_with_sub.get()));
}

// Test both filters together.
TEST(BridgeLoggerFilters, TestBothFilters) {
  mlir::DialectRegistry mlir_registry;
  mlir::RegisterAllTensorFlowDialects(mlir_registry);
  mlir::MLIRContext mlir_context(mlir_registry);
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_add;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_add, &mlir_context,
                                     &mlir_module_with_add));

  std::unique_ptr<mlir::Pass> shape_inference_pass =
      mlir::TF::CreateTFShapeInferencePass();

  // String filter is matched but pass filter is not.
  setenv("MLIR_BRIDGE_LOG_STRING_FILTER",
         "(tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>", 1);
  setenv("MLIR_BRIDGE_LOG_PASS_FILTER", "ensorFlowShapeInferencePass", 1);
  BridgeLoggerConfig logger_config1;
  EXPECT_FALSE(logger_config1.ShouldPrint(shape_inference_pass.get(),
                                          mlir_module_with_add.get()));

  // Pass filter is matched but string filter is not.
  setenv("MLIR_BRIDGE_LOG_STRING_FILTER", "XXX", 1);
  setenv("MLIR_BRIDGE_LOG_PASS_FILTER", "TensorFlowShapeInferencePass", 1);
  BridgeLoggerConfig logger_config2;
  EXPECT_FALSE(logger_config2.ShouldPrint(shape_inference_pass.get(),
                                          mlir_module_with_add.get()));

  // Both filters are matched.
  setenv("MLIR_BRIDGE_LOG_STRING_FILTER",
         "(tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>", 1);
  setenv("MLIR_BRIDGE_LOG_PASS_FILTER", "TensorFlowShapeInferencePass", 1);
  BridgeLoggerConfig logger_config3;
  EXPECT_TRUE(logger_config3.ShouldPrint(shape_inference_pass.get(),
                                         mlir_module_with_add.get()));
}

}  // namespace
}  // namespace tensorflow
