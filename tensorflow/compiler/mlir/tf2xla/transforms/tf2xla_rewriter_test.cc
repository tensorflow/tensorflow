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
#include "tensorflow/compiler/mlir/tf2xla/transforms/tf2xla_rewriter.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/test_utils.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace mlir {
namespace mhlo {

using ::mlir::LogicalResult;
using ::mlir::ModuleOp;
using ::tsl::Status;
using ::tsl::StatusOr;
using ::xla::XlaBuilder;
using ::xla::XlaComputation;

static constexpr char kMlirModuleStr[] = R"(
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1442 : i32}} {
  func.func @main(%arg0: tensor<3xi64> {tf._user_specified_name = "resource", tf.aliasing_output = 3 : i64}) -> () attributes {tf.entry_function = {control_outputs = "stateful_normal/RngReadAndSkip,stateful_uniform/RngReadAndSkip,stateful_uniform_full_int/RngReadAndSkip", inputs = "stateful_normal_rngreadandskip_resource", outputs = "identity_RetVal,identity_1_RetVal,identity_2_RetVal"}} {
    %0:3 = "tf.Unpack"(%arg0) {axis = 0 : i64} : (tensor<3xi64>) -> (tensor<i64>, tensor<i64>, tensor<i64>)
    return
  }
})";

class EmptyPatternRewriter : public mlir::PatternRewriter {
 public:
  explicit EmptyPatternRewriter(const OpBuilder& other_builder)
      : mlir::PatternRewriter(other_builder) {}
  ~EmptyPatternRewriter() override = default;
};

class Tf2XlaRewriterTestPeer {
 public:
  explicit Tf2XlaRewriterTestPeer() = delete;
  Tf2XlaRewriterTestPeer(mlir::Operation* op, mlir::PatternRewriter& rewriter)
      : tf2xla_rewriter_(op, rewriter,
                         /*device_type=*/"XLA_CPU_JIT",
                         /*is_module_pass=*/false,
                         /*use_tf2xla_hlo_importer=*/false) {}

  StatusOr<mlir::OwningOpRef<ModuleOp>> GetModuleFromXlaComputation(
      xla::XlaComputation& computation) {
    return tf2xla_rewriter_.CreateModuleFromXlaComputation(computation);
  }

 private:
  Tf2XlaRewriter tf2xla_rewriter_;
};

// This should only have unit tests. End to end tests should be done with
// FILECHECK and MLIR tests.
class Tf2XlaRewriterTest : public ::testing::Test {
 public:
  void SetUp() override {
    tensorflow::XlaOpRegistry::RegisterCompilationKernels();

    StatusOr<OwningOpRef<ModuleOp>> module =
        test::GetMlirModuleFromString(kMlirModuleStr, &context_);
    ASSERT_TRUE(module.ok());

    module_ = std::move(module.value());
    context_.loadAllAvailableDialects();
  }

  Status LegalizeOp() {
    SourceMgrDiagnosticHandler sourceMgrHandler(source_manager_, &context_);

    mlir::func::FuncOp main_func =
        module_->lookupSymbol<mlir::func::FuncOp>("main");
    EXPECT_TRUE(main_func);

    mlir::Operation& first_op = main_func.getBody().front().front();
    mlir::OpBuilder op_builder(&first_op);
    EmptyPatternRewriter pattern_rewriter(op_builder);

    LogicalResult result =
        Tf2XlaRewriter::RewriteOp(&first_op, pattern_rewriter,
                                  /*device_type=*/"XLA_CPU_JIT",
                                  /*is_module_pass=*/false,
                                  /*use_tf2xla_hlo_importer=*/false);
    if (!result.succeeded()) {
      return tsl::errors::Internal("Failed to rewrite op");
    }

    return tsl::OkStatus();
  }

  StatusOr<mlir::OwningOpRef<ModuleOp>> GetModuleFromXlaComputation(
      XlaComputation& computation) {
    SourceMgrDiagnosticHandler sourceMgrHandler(source_manager_, &context_);

    mlir::func::FuncOp main_func =
        module_->lookupSymbol<mlir::func::FuncOp>("main");
    EXPECT_TRUE(main_func);

    mlir::Operation& first_op = main_func.getBody().front().front();
    mlir::OpBuilder op_builder(&first_op);
    EmptyPatternRewriter pattern_rewriter(op_builder);

    Tf2XlaRewriterTestPeer test_peer(&first_op, pattern_rewriter);
    return test_peer.GetModuleFromXlaComputation(computation);
  }

 private:
  MLIRContext context_;
  OwningOpRef<ModuleOp> module_;
  llvm::SourceMgr source_manager_;
};

TEST_F(Tf2XlaRewriterTest, LegalizesOp) { TF_EXPECT_OK(LegalizeOp()); }

TEST_F(Tf2XlaRewriterTest, CreatesModuleFromXla) {
  XlaBuilder xla_builder("test_builder");
  xla::Add(xla::ConstantR0<float>(&xla_builder, 1.0),
           xla::ConstantR0<float>(&xla_builder, 2.0));
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, xla_builder.Build());

  StatusOr<mlir::OwningOpRef<ModuleOp>> generated_module =
      GetModuleFromXlaComputation(computation);

  EXPECT_TRUE(generated_module.ok());
  EXPECT_TRUE(generated_module.value()->lookupSymbol("main"));
}

}  // namespace mhlo
}  // namespace mlir
