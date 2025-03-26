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
#include <cstdint>
#include <functional>
#include <memory>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"

namespace tensorflow {
namespace {

using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OwningOpRef;
using ::mlir::PassManager;
using ::tensorflow::monitoring::testing::CellReader;

absl::StatusOr<OwningOpRef<ModuleOp>> GetMlirModuleFromString(
    absl::string_view module_string, MLIRContext* context) {
  mlir::DialectRegistry mlir_registry;
  RegisterAllTensorFlowDialects(mlir_registry);
  context->appendDialectRegistry(mlir_registry);

  OwningOpRef<ModuleOp> mlir_module;
  auto status =
      tensorflow::DeserializeMlirModule(module_string, context, &mlir_module);
  if (!status.ok()) {
    return status;
  }
  return mlir_module;
}

bool BuildAndRunPipeline(absl::string_view module_string,
                         const std::function<void(PassManager*)>& passes) {
  mlir::registerPassManagerCLOptions();
  MLIRContext context;

  OwningOpRef<ModuleOp> module =
      GetMlirModuleFromString(module_string, &context).value();

  PassManager pm(&context);

  if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) return false;
  passes(&pm);

  return pm.run(module.get()).succeeded();
}

std::function<void(PassManager*)> legalizeTFPasses() {
  return [](PassManager* pm) {
    pm->addPass(mlir::mhlo::createLegalizeTFPass(
        /* legalize_chlo=*/true, llvm::StringRef("gpu/xpu"),
        /* prefer_tf2xla=*/false));
  };
}

TEST(XlaLegalizeTest, IllegalOp) {
  constexpr char kMlirIllegalOpStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> tensor<1xi32> {
      %0 = "tf.DoesntExist"() : () -> tensor<1xi32>
      func.return %0 : tensor<1xi32>
    }
  })";
  CellReader<int64_t> legalize_failure_count(
      "/tensorflow/core/tf2xla/v1/mlir_failed_xla_legalize_tf_pass_count");

  auto status = BuildAndRunPipeline(kMlirIllegalOpStr, legalizeTFPasses());

  EXPECT_TRUE(status);
  EXPECT_EQ(legalize_failure_count.Read("tf.DoesntExist", "Unknown"), 1);
}

TEST(XlaLegalizeTest, LegalOp) {
  // We expect legalization to fail for legal op with dynamic shapes:
  static constexpr char kMlirLegalOpStr[] = R"(
   func.func @infeed_dequeue_tuple_dynamic_error() -> (tensor<3x3xf32>, tensor<4x?xf32>) {
     %0:2 = "tf.InfeedDequeueTuple"() : () -> (tensor<3x3xf32>, tensor<4x?xf32>) func.return %0#0, %0#1 : tensor<3x3xf32>, tensor<4x?xf32>
   })";
  CellReader<int64_t> legalize_failure_count(
      "/tensorflow/core/tf2xla/v1/mlir_failed_xla_legalize_tf_pass_count");

  auto status = BuildAndRunPipeline(kMlirLegalOpStr, legalizeTFPasses());

  EXPECT_TRUE(status);
  EXPECT_EQ(legalize_failure_count.Read("tf.InfeedDequeueTuple", "Unknown"), 1);
}
}  // namespace

}  // namespace tensorflow
