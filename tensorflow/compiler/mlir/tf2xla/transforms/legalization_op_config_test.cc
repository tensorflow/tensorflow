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

#include "tensorflow/compiler/mlir/tf2xla/transforms/legalization_op_config.h"

#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/test_utils.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace mlir {
namespace mhlo {

using func::FuncOp;
using mlir::ModuleOp;
using tsl::Status;

static constexpr char kMlirModuleStr[] = R"(
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1442 : i32}} {
  func.func @main(%arg0: tensor<3xi64> {tf._user_specified_name = "resource", tf.aliasing_output = 3 : i64}) -> () attributes {tf.entry_function = {control_outputs = "stateful_normal/RngReadAndSkip,stateful_uniform/RngReadAndSkip,stateful_uniform_full_int/RngReadAndSkip", inputs = "stateful_normal_rngreadandskip_resource", outputs = "identity_RetVal,identity_1_RetVal,identity_2_RetVal"}} {
    %0:3 = "tf.Unpack"(%arg0) {axis = 0 : i64} : (tensor<3xi64>) -> (tensor<i64>, tensor<i64>, tensor<i64>)
    return
  }
})";

class LegalizationOpConfigTest : public ::testing::Test {
 public:
  Status CreateMlirModule(std::string module_string = kMlirModuleStr) {
    TF_ASSIGN_OR_RETURN(
        module_, test::GetMlirModuleFromString(module_string, &context_));

    context_.loadAllAvailableDialects();
    return tsl::OkStatus();
  }

  tsl::StatusOr<FuncOp> GetMain() {
    func::FuncOp main = module_->lookupSymbol<mlir::func::FuncOp>("main");
    if (!main) {
      return absl::NotFoundError("Could not find main function");
    }
    return main;
  }

 protected:
  MLIRContext context_;
  OwningOpRef<ModuleOp> module_;
};

TEST_F(LegalizationOpConfigTest, FailsWithExpectsLegalizationWithMlir) {
  TF_EXPECT_OK(CreateMlirModule());
  EXPECT_FALSE(IsOpLegalizedWithMlir(*module_->getOperation()));
}

TEST_F(LegalizationOpConfigTest, ExpectsFalseForNonMlirOps) {
  TF_EXPECT_OK(CreateMlirModule());
  TF_ASSERT_OK_AND_ASSIGN(FuncOp main, GetMain());

  main.walk([&](Operation* op) { EXPECT_FALSE(IsOpLegalizedWithMlir(*op)); });
}

TEST_F(LegalizationOpConfigTest, ExpectsTrueForMlirTypeID) {
  EXPECT_TRUE(IsTypeLegalizedWithMlir(TypeID::get<TF::ModOp>()));
  EXPECT_FALSE(HasTf2XlaFallback(TypeID::get<TF::ModOp>()));
  EXPECT_FALSE(IsOpAllowedTf2xlaFallback(TypeID::get<TF::ModOp>()));
  EXPECT_FALSE(IsOpAllowedTf2xlaPreferred(TypeID::get<TF::ModOp>()));
}

TEST_F(LegalizationOpConfigTest, ExpectsTrueForTF2XLATypeID) {
  EXPECT_TRUE(HasTf2XlaFallback(TypeID::get<TF::AllOp>()));
  EXPECT_TRUE(IsOpAllowedTf2xlaPreferred(TypeID::get<TF::AllOp>()));
  EXPECT_FALSE(IsTypeLegalizedWithMlir(TypeID::get<TF::AllOp>()));
}

// This test is kind of odd. We go through all the Tensorflow types and check
// whether they are legalized with MLIR, TF2XLA, or both. Ideally the sets are
// disjoint, but until that happens, this tests ensures that the set doesn't
// grow.
TEST_F(LegalizationOpConfigTest, CountLoweringsSet) {
  int mlir_lowering_count = 0;
  int tf2xla_fallback_count = 0;
  int non_categorized_count = 0;

  DialectRegistry dialect_registry;
  dialect_registry.insert<mlir::TF::TensorFlowDialect>();

  MLIRContext context(dialect_registry);
  context.loadAllAvailableDialects();

  for (auto operation : context.getRegisteredOperations()) {
    if (IsTypeLegalizedWithMlir(operation.getTypeID())) {
      mlir_lowering_count++;
    } else if (HasTf2XlaFallback(operation.getTypeID())) {
      tf2xla_fallback_count++;
    } else {
      non_categorized_count++;
    }
  }

  // If an op moves from one lowering implementation to a different one (e.g.
  // from MLIR to TF2XLA), these numbers should change. Or if TF Dialect adds
  // a new op, we should expect these to change too.
  EXPECT_EQ(mlir_lowering_count, 71);
  EXPECT_EQ(tf2xla_fallback_count, 295);
  EXPECT_EQ(non_categorized_count, 418);
}

TEST_F(LegalizationOpConfigTest, CountTypesWhichHaveBothMlirAndTf2xlaFallback) {
  int double_lowering_count = 0;

  DialectRegistry dialect_registry;
  dialect_registry.insert<mlir::TF::TensorFlowDialect>();

  MLIRContext context(dialect_registry);
  context.loadAllAvailableDialects();

  for (auto operation : context.getRegisteredOperations()) {
    if (IsTypeLegalizedWithMlir(operation.getTypeID()) &&
        HasTf2XlaFallback(operation.getTypeID())) {
      double_lowering_count++;
    }
  }

  EXPECT_EQ(double_lowering_count, 3);
}

}  // namespace mhlo
}  // namespace mlir
