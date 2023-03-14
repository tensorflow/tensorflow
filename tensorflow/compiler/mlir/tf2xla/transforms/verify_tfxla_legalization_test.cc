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

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"

namespace tensorflow {
namespace {

using ::llvm::StringRef;
using ::mlir::DialectRegistry;
using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OwningOpRef;
using ::tensorflow::monitoring::testing::CellReader;

// Using a string constant here instead of testdata to make this compatible
// with open source.
static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> tensor<1xi32> {
      %0 = "tf.BadValue"() {value = dense<1000> : tensor<1xi32>} : () -> tensor<1xi32>
      func.return %0 : tensor<1xi32>
    }
  })";

static constexpr char kFailedLegalizationStreamz[] =
    "/tensorflow/core/tf2xla/mlir_second_phase_failed_legalization_op_count";

tsl::StatusOr<OwningOpRef<ModuleOp>> GetMlirModuleFromString(
    StringRef string, MLIRContext* context) {
  DialectRegistry mlir_registry;
  RegisterAllTensorFlowDialects(mlir_registry);
  context->appendDialectRegistry(mlir_registry);

  OwningOpRef<ModuleOp> mlir_module;
  auto status =
      tensorflow::DeserializeMlirModule(string, context, &mlir_module);
  if (!status.ok()) {
    return status;
  }
  return mlir_module;
}

TEST(VerifyTfxlaLegalizationTest, RecordsStreamzFailedVerification) {
  MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(OwningOpRef<ModuleOp> module,
                          GetMlirModuleFromString(kMlirModuleStr, &context));

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::CreateVerifyTFXLALegalizationPass(/*legalize_chlo=*/false));

  CellReader<int64_t> error(kFailedLegalizationStreamz);

  EXPECT_TRUE(pm.run(module.get()).failed());
  EXPECT_EQ(error.Delta("tf.BadValue"), 1);
}

TEST(VerifyTfxlaLegalizationTest, RecordsMultipleFailures) {
  // Using a string constant here instead of testdata to make this compatible
  // with open source.
  static constexpr char kMultipleFailures[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> tensor<1xi32> {
      %0 = "tf.BadValue"() {value = dense<1000> : tensor<1xi32>} : () -> tensor<1xi32>
      %1 = "tf.AlsoBad"() {value = dense<10> : tensor<1xi32>} : () -> tensor<1xi32>
      func.return %0 : tensor<1xi32>
    }
  })";

  MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(OwningOpRef<ModuleOp> module,
                          GetMlirModuleFromString(kMultipleFailures, &context));

  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::CreateVerifyTFXLALegalizationPass(/*legalize_chlo=*/false));

  CellReader<int64_t> error(kFailedLegalizationStreamz);

  EXPECT_TRUE(pm.run(module.get()).failed());
  EXPECT_EQ(error.Delta("tf.BadValue"), 1);
  EXPECT_EQ(error.Delta("tf.AlsoBad"), 1);
}

}  // namespace
}  // namespace tensorflow
