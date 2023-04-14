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
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/test_utils.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace mlir {
namespace mhlo {

using ::mlir::LogicalResult;
using ::mlir::ModuleOp;
using ::mlir::OpBuilder;
using ::mlir::Operation;
using ::mlir::func::FuncOp;
using ::tsl::Status;
using ::tsl::StatusOr;
using xla::ReplicaGroup;
using ::xla::ShapeUtil;
using ::xla::XlaBuilder;
using ::xla::XlaComputation;

static constexpr char kMlirModuleStr[] = R"(
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1442 : i32}} {
  func.func @main(%arg0: tensor<3xi64> {tf._user_specified_name = "resource", tf.aliasing_output = 3 : i64}) -> () attributes {tf.entry_function = {control_outputs = "stateful_normal/RngReadAndSkip,stateful_uniform/RngReadAndSkip,stateful_uniform_full_int/RngReadAndSkip", inputs = "stateful_normal_rngreadandskip_resource", outputs = "identity_RetVal,identity_1_RetVal,identity_2_RetVal"}} {
    %0:3 = "tf.Unpack"(%arg0) {axis = 0 : i64} : (tensor<3xi64>) -> (tensor<i64>, tensor<i64>, tensor<i64>)
    return
  }
})";

XlaComputation GetTestXlaComputation() {
  XlaBuilder xla_builder("test");
  xla::Add(xla::ConstantR0<float>(&xla_builder, 1.0),
           xla::ConstantR0<float>(&xla_builder, 2.0));

  return xla_builder.Build().value();
}

class EmptyPatternRewriter : public mlir::PatternRewriter {
 public:
  explicit EmptyPatternRewriter(const OpBuilder& other_builder)
      : mlir::PatternRewriter(other_builder) {}
  ~EmptyPatternRewriter() override = default;
};

class Tf2XlaRewriterTestPeer {
 public:
  explicit Tf2XlaRewriterTestPeer() = delete;
  explicit Tf2XlaRewriterTestPeer(mlir::Operation* op)
      : op_builder_(op),
        empty_rewriter_(op_builder_),
        tf2xla_rewriter_(op, empty_rewriter_,
                         /*device_type=*/"XLA_CPU_JIT",
                         /*is_module_pass=*/false,
                         /*use_tf2xla_hlo_importer=*/true) {}

  StatusOr<FuncOp> ImportXlaComputationIntoModule(XlaComputation& computation) {
    return tf2xla_rewriter_.ImportXlaComputation(computation);
  }

  Status UnpackTupleFromFunction(FuncOp func_op) {
    if (failed(tf2xla_rewriter_.UnpackTupleResults(func_op))) {
      return tsl::errors::Internal("Couldn't unpack tuple results");
    }

    return tsl::OkStatus();
  }

 private:
  OpBuilder op_builder_;
  EmptyPatternRewriter empty_rewriter_;
  Tf2XlaRewriter tf2xla_rewriter_;
};

// This should only have unit tests. End to end tests should be done with
// FILECHECK and MLIR tests.
class Tf2XlaRewriterTest : public ::testing::Test {
 public:
  void SetUp() override {
    tensorflow::XlaOpRegistry::RegisterCompilationKernels();
  }

  Status CreateMlirModule(std::string module_string = kMlirModuleStr) {
    TF_ASSIGN_OR_RETURN(
        module_, test::GetMlirModuleFromString(module_string, &context_));

    context_.loadAllAvailableDialects();
    return tsl::OkStatus();
  }

  Status LegalizeSingleOp(bool use_tf2xla_hlo_importer, Operation& op) {
    SourceMgrDiagnosticHandler sourceMgrHandler(source_manager_, &context_);

    OpBuilder op_builder(&op);
    EmptyPatternRewriter pattern_rewriter(op_builder);

    LogicalResult result = Tf2XlaRewriter::RewriteOp(
        &op, pattern_rewriter,
        /*device_type=*/"XLA_CPU_JIT",
        /*is_module_pass=*/false, use_tf2xla_hlo_importer);
    if (!result.succeeded()) {
      return tsl::errors::Internal("Failed to rewrite op");
    }

    return tsl::OkStatus();
  }

  Status LegalizeModuleWithSingleOp(
      bool use_tf2xla_hlo_importer,
      std::string module_string = kMlirModuleStr) {
    TF_EXPECT_OK(CreateMlirModule(module_string));

    Operation& first_op = GetFirstOpFromMain();
    return LegalizeSingleOp(use_tf2xla_hlo_importer, first_op);
  }

  mlir::func::FuncOp GetMainFunc() {
    func::FuncOp main_func = module_->lookupSymbol<mlir::func::FuncOp>("main");
    EXPECT_TRUE(main_func);

    return main_func;
  }

  mlir::Operation& GetFirstOpFromMain() {
    mlir::func::FuncOp main_func = GetMainFunc();
    return main_func.getBody().front().front();
  }

  StatusOr<FuncOp> ImportXlaComputationIntoModule(XlaComputation& computation) {
    SourceMgrDiagnosticHandler sourceMgrHandler(source_manager_, &context_);

    mlir::Operation& first_op = GetFirstOpFromMain();

    Tf2XlaRewriterTestPeer test_peer(&first_op);
    return test_peer.ImportXlaComputationIntoModule(computation);
  }

 protected:
  MLIRContext context_;
  OwningOpRef<ModuleOp> module_;
  llvm::SourceMgr source_manager_;
};

TEST_F(Tf2XlaRewriterTest, LegalizesOp) {
  TF_EXPECT_OK(LegalizeModuleWithSingleOp(/*use_tf2xla_hlo_importer=*/false));
}

TEST_F(Tf2XlaRewriterTest, LegalizesOpWithTf2xlaHloImporter) {
  TF_EXPECT_OK(LegalizeModuleWithSingleOp(/*use_tf2xla_hlo_importer=*/true));
}

TEST_F(Tf2XlaRewriterTest, ImportsXlaComputationIntoModule) {
  TF_ASSERT_OK(CreateMlirModule());

  XlaComputation computation = GetTestXlaComputation();

  TF_ASSERT_OK_AND_ASSIGN(FuncOp translated_function,
                          ImportXlaComputationIntoModule(computation));

  EXPECT_TRUE(module_->lookupSymbol(computation.name()));
  EXPECT_EQ(translated_function.getName(), computation.name());
}

TEST_F(Tf2XlaRewriterTest, ReturnsMultipleValues) {
  TF_EXPECT_OK(LegalizeModuleWithSingleOp(/*use_tf2xla_hlo_importer=*/true));

  FuncOp expected_generated_function =
      module_->lookupSymbol<mlir::func::FuncOp>(
          "tf2xla_rewriter.tf.Unpack.9.0");

  EXPECT_TRUE(expected_generated_function);
  EXPECT_EQ(expected_generated_function.getNumResults(),
            GetFirstOpFromMain().getNumResults());
}

TEST_F(Tf2XlaRewriterTest, ImportsSingleComputation) {
  XlaBuilder builder("test_builder");
  XlaComputation to_apply;
  {
    auto sub_builder = builder.CreateSubBuilder("add");
    auto arg0 = Parameter(sub_builder.get(), 0,
                          ShapeUtil::MakeScalarShape(xla::F32), "x");
    auto arg1 = Parameter(sub_builder.get(), 1,
                          ShapeUtil::MakeScalarShape(xla::F32), "y");
    Add(arg0, arg1);
    TF_ASSERT_OK_AND_ASSIGN(to_apply, sub_builder->Build());
  }
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(xla::F32, {4, 16}), "x");
  ReplicaGroup group;
  group.add_replica_ids(0);
  group.add_replica_ids(1);
  ReduceScatter(x, to_apply, /*scatter_dimension=*/1, /*shard_count=*/2,
                /*replica_groups=*/{group});

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());
  EXPECT_EQ(computation.proto().computations_size(), 2);

  TF_ASSERT_OK(CreateMlirModule());
  TF_ASSERT_OK_AND_ASSIGN(FuncOp translated_function,
                          ImportXlaComputationIntoModule(computation));
  EXPECT_TRUE(translated_function);

  int num_func_ops = 0;
  module_->walk([&num_func_ops](func::FuncOp func_op) { num_func_ops++; });

  // 2 because we have one from the existing module_ and one that's inserted.
  EXPECT_EQ(num_func_ops, 2);
}

TEST_F(Tf2XlaRewriterTest, InsertsConstantParameters) {
  static constexpr char kModuleWithConstParam[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1442 : i32}} {
    func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
      %0 = "tf.Const"() {value = dense<1.42> : tensor<2xf32>} : () -> tensor<2xf32>
      %1 = "tf.Atan2"(%arg0, %0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
      func.return %0 : tensor<2xf32>
    }
  })";

  TF_ASSERT_OK(CreateMlirModule(kModuleWithConstParam));
  func::FuncOp main_func = GetMainFunc();

  // NOTE: Extra wonkiness here. LLVM returns an iterator and hold references to
  // the op here. However, we're testing a rewriter and replace the original op
  // with a call op. The original op that we have a reference to via the getOps
  // is a use after free. To get around that, we clone the op, lose the ref to
  // it and convert the cloned op instead. For purposes of just this test,
  // it's fine, but this is bad.
  int atan_op_count = 0;
  for (auto atan_op : main_func.getOps<TF::Atan2Op>()) {
    atan_op_count++;
    mlir::Operation* cloned_op = atan_op->clone();

    OpBuilder builder(atan_op);
    builder.insert(cloned_op);

    TF_ASSERT_OK(
        LegalizeSingleOp(/*use_tf2xla_hlo_importer=*/true, *cloned_op));
  }

  EXPECT_EQ(atan_op_count, 1);

  FuncOp expected_generated_function =
      module_->lookupSymbol<mlir::func::FuncOp>("tf2xla_rewriter.tf.Atan2.9.0");

  EXPECT_TRUE(expected_generated_function);
  EXPECT_EQ(expected_generated_function.getNumArguments(), 2);

  EXPECT_TRUE(expected_generated_function.getArgument(1).getUses().empty());
  EXPECT_FALSE(expected_generated_function.getArgument(0).getUses().empty());

  mhlo::ConstantOp const_op = llvm::dyn_cast<mhlo::ConstantOp>(
      expected_generated_function.getBody().front().front());
  EXPECT_TRUE(const_op);
  EXPECT_EQ(const_op.getValue().getSplatValue<float>(), 1.42f);
}

TEST_F(Tf2XlaRewriterTest, ImportsPrivateFunctions) {
  XlaBuilder builder("test_builder");
  XlaComputation to_apply;
  {
    auto sub_builder = builder.CreateSubBuilder("add");
    auto arg0 = Parameter(sub_builder.get(), 0,
                          ShapeUtil::MakeScalarShape(xla::F32), "x");
    auto arg1 = Parameter(sub_builder.get(), 1,
                          ShapeUtil::MakeScalarShape(xla::F32), "y");
    Add(arg0, arg1);
    TF_ASSERT_OK_AND_ASSIGN(to_apply, sub_builder->Build());
  }
  auto a = Parameter(&builder, 0, ShapeUtil::MakeScalarShape(xla::F32), "a");
  auto b = Parameter(&builder, 1, ShapeUtil::MakeScalarShape(xla::F32), "b");
  xla::Call(&builder, to_apply, {a, b});
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());
  EXPECT_EQ(computation.proto().computations_size(), 2);

  TF_ASSERT_OK(CreateMlirModule());
  TF_ASSERT_OK_AND_ASSIGN(FuncOp generated_function,
                          ImportXlaComputationIntoModule(computation));

  EXPECT_TRUE(generated_function);

  int num_func_ops = 0;
  bool has_private_function = false;
  module_->walk([&num_func_ops, &has_private_function](func::FuncOp func_op) {
    num_func_ops++;
    has_private_function &= func_op.isPrivate();
  });

  EXPECT_EQ(num_func_ops, 3);
  EXPECT_FALSE(has_private_function);
}

TEST_F(Tf2XlaRewriterTest, ErasesTupleOpFromMultipleReturnValues) {
  TF_EXPECT_OK(LegalizeModuleWithSingleOp(/*use_tf2xla_hlo_importer=*/true));

  FuncOp expected_generated_function =
      module_->lookupSymbol<mlir::func::FuncOp>(
          "tf2xla_rewriter.tf.Unpack.9.0");
  ASSERT_TRUE(expected_generated_function);

  EXPECT_TRUE(expected_generated_function.getOps<mhlo::TupleOp>().empty());
}

TEST_F(Tf2XlaRewriterTest, FailsUnpackingModuleWithoutTuple) {
  TF_ASSERT_OK(CreateMlirModule());

  FuncOp funcOp = GetFirstOpFromMain().getParentOfType<FuncOp>();
  ASSERT_TRUE(funcOp);

  Tf2XlaRewriterTestPeer test_peer(funcOp.getOperation());
  EXPECT_FALSE(test_peer.UnpackTupleFromFunction(funcOp).ok());
}

TEST(UnpackingModuleTest, FailsUnpackingModuleWithMultipleReturnValues) {
  static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1442 : i32}} {
    func.func @main(%arg0: tensor<3xi64>) -> (tensor<1xi64>, tensor<i64>) {
      %0 = "mhlo.slice"(%arg0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xi64>) -> tensor<1xi64>
      %1 = mhlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
      return %0, %1 : tensor<1xi64>, tensor<i64>
    }
  })";

  MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(
      OwningOpRef<ModuleOp> module,
      test::GetMlirModuleFromString(kMlirModuleStr, &context));

  mlir::func::FuncOp main_func =
      module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);

  Tf2XlaRewriterTestPeer test_peer(main_func.getOperation());
  EXPECT_FALSE(test_peer.UnpackTupleFromFunction(main_func).ok());
}

}  // namespace mhlo
}  // namespace mlir
