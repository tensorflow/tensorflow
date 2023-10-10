/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/data_dumper_logger_config.h"

#include "llvm/Support/raw_ostream.h"
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

// Test pass filter.
TEST(DataDumperLoggerConfig, TestPassFilter) {
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
  setenv("TF_DUMP_GRAPH_PREFIX", "sponge", 1);

  const string kTestFilename = "test.txt";
  int print_callback_count = 0;
  auto get_filename_fn = [](const string &filename, mlir::Operation *op) {
    return filename;
  };
  auto print_callback = [&](llvm::raw_ostream &out) {
    print_callback_count++;
    return;
  };

  DataDumperLoggerConfig data_dumper_logger_config(get_filename_fn);

  data_dumper_logger_config.printBeforeIfEnabled(
      partitioning_pass.get(), mlir_module_with_add.get(), print_callback);
  EXPECT_EQ(print_callback_count, 1);

  data_dumper_logger_config.printBeforeIfEnabled(
      shape_inference_pass.get(), mlir_module_with_add.get(), print_callback);
  EXPECT_EQ(print_callback_count, 2);

  data_dumper_logger_config.printBeforeIfEnabled(
      inliner_pass.get(), mlir_module_with_add.get(), print_callback);
  EXPECT_EQ(print_callback_count, 2);

  data_dumper_logger_config.printAfterIfEnabled(
      partitioning_pass.get(), mlir_module_with_add.get(), print_callback);
  EXPECT_EQ(print_callback_count, 3);

  data_dumper_logger_config.printAfterIfEnabled(
      shape_inference_pass.get(), mlir_module_with_add.get(), print_callback);
  EXPECT_EQ(print_callback_count, 4);

  data_dumper_logger_config.printAfterIfEnabled(
      inliner_pass.get(), mlir_module_with_add.get(), print_callback);
  EXPECT_EQ(print_callback_count, 4);
}

}  // namespace
}  // namespace tensorflow
