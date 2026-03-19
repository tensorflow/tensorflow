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
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/test_utils.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace mlir {
namespace hlo {

using ::mlir::LogicalResult;
using ::mlir::ModuleOp;
using ::mlir::OpBuilder;
using ::mlir::Operation;
using ::mlir::func::FuncOp;
using ::tsl::Status;
using ::tsl::StatusOr;
using ::xla::ReplicaGroup;
using ::xla::ShapeUtil;
using ::xla::XlaBuilder;
using ::xla::XlaComputation;
using ::xla::XlaOp;

static constexpr char kMlirModuleStr[] = R"(
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1442 : i32}} {
  func.func @main(%arg0: tensor<3xi64> {tf._user_specified_name = "resource", tf.aliasing_output = 3 : i64}) -> () attributes {tf.entry_function = {control_outputs = "stateful_normal/RngReadAndSkip,stateful_uniform/RngReadAndSkip,stateful_uniform_full_int/RngReadAndSkip", inputs = "stateful_normal_rngreadandskip_resource", outputs = "identity_RetVal,identity_1_RetVal,identity_2_RetVal"}} {
    %0:3 = "tf.Unpack"(%arg0) {axis = 0 : i64} : (tensor<3xi64>) -> (tensor<i64>, tensor<i64>, tensor<i64>)
    return
  }
})";

XlaComputation GetTestXlaComputation() {
  XlaBuilder xla_builder("test");
  auto param =
      Parameter(&xla_builder, 0, ShapeUtil::MakeScalarShape(xla::F32), "a");

  XlaOp add = xla::Add(param, xla::ConstantR0<float>(&xla_builder, 2.0));

  std::vector<XlaOp> tuple_values;
  tuple_values.push_back(add);

  xla::Tuple(&xla_builder, tuple_values);
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
                         /*device_type=*/"XLA_CPU_JIT") {}

  absl::StatusOr<stablehlo::TupleOp> ImportXlaComputationIntoModule(
      XlaComputation& computation) {
    return tf2xla_rewriter_.ImportXlaComputation(computation);
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
        module_, hlo::test::GetMlirModuleFromString(module_string, &context_));

    context_.loadAllAvailableDialects();
    return absl::OkStatus();
  }

  Status LegalizeSingleOp(Operation& op) {
    SourceMgrDiagnosticHandler sourceMgrHandler(source_manager_, &context_);

    OpBuilder op_builder(&op);
    EmptyPatternRewriter pattern_rewriter(op_builder);

    LogicalResult result =
        Tf2XlaRewriter::RewriteOp(&op, pattern_rewriter,
                                  /*device_type=*/"XLA_CPU_JIT");
    if (!result.succeeded()) {
      return tsl::errors::Internal("Failed to rewrite op");
    }

    return absl::OkStatus();
  }

  Status LegalizeModule(std::string module_string = kMlirModuleStr) {
    TF_EXPECT_OK(CreateMlirModule(module_string));
    FuncOp main = module_->lookupSymbol<mlir::func::FuncOp>("main");
    if (!main) {
      return tsl::errors::InvalidArgument("Could not find a main function");
    }

    WalkResult walk_result = main.walk([&](Operation* op) {
      if (op->getDialect()->getNamespace() !=
          TF::TensorFlowDialect::getDialectNamespace()) {
        return WalkResult::advance();
      }

      if (!LegalizeSingleOp(*op).ok()) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    if (walk_result.wasInterrupted()) {
      return tsl::errors::Internal("Could not legalize all ops");
    }

    return absl::OkStatus();
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

  absl::StatusOr<stablehlo::TupleOp> ImportXlaComputationIntoModule(
      XlaComputation& computation) {
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

TEST_F(Tf2XlaRewriterTest, LegalizesOpWithTf2xlaHloImporter) {
  TF_EXPECT_OK(LegalizeModule());

  int num_tuple_ops = 0;
  module_->walk(
      [&num_tuple_ops](stablehlo::TupleOp tuple_op) { num_tuple_ops += 1; });

  EXPECT_EQ(num_tuple_ops, 0);
}

TEST_F(Tf2XlaRewriterTest, ImportsXlaComputationIntoModule) {
  TF_ASSERT_OK(CreateMlirModule());

  XlaComputation computation = GetTestXlaComputation();

  TF_ASSERT_OK_AND_ASSIGN(stablehlo::TupleOp root_tuple,
                          ImportXlaComputationIntoModule(computation));

  ModuleOp parent_module =
      root_tuple.getOperation()->getParentOfType<ModuleOp>();
  EXPECT_EQ(parent_module, *module_);
}

TEST_F(Tf2XlaRewriterTest, FailsWithoutRootTuple) {
  TF_ASSERT_OK(CreateMlirModule());

  XlaBuilder xla_builder("test_fail");
  xla::Add(xla::ConstantR0<float>(&xla_builder, 1.0),
           xla::ConstantR0<float>(&xla_builder, 2.0));
  XlaComputation bad_computation = xla_builder.Build().value();

  EXPECT_FALSE(ImportXlaComputationIntoModule(bad_computation).ok());
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
  XlaOp reduce_scatter =
      ReduceScatter(x, to_apply, /*scatter_dimension=*/1, /*shard_count=*/2,
                    /*replica_groups=*/{group});

  std::vector<XlaOp> tuple_values;
  tuple_values.push_back(reduce_scatter);
  xla::Tuple(&builder, tuple_values);

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());
  EXPECT_EQ(computation.proto().computations_size(), 2);

  TF_ASSERT_OK(CreateMlirModule());
  TF_ASSERT_OK_AND_ASSIGN(stablehlo::TupleOp root_tuple,
                          ImportXlaComputationIntoModule(computation));
  EXPECT_TRUE(root_tuple);

  int num_func_ops = 0;
  module_->walk([&num_func_ops](func::FuncOp func_op) { num_func_ops++; });

  // Ensure that only a single computation was imported.
  EXPECT_EQ(num_func_ops, 1);
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

  TF_ASSERT_OK(LegalizeModule(kModuleWithConstParam));
}

TEST_F(Tf2XlaRewriterTest, DoesntEnforceCompileTimeConstantCheck) {
  static constexpr char kModuleWithNonConstParam[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1610 : i32}} {
    func.func @main(%arg0: tensor<3x3x10xbf16>, %arg1: tensor<3xi32>) -> tensor<1x?x4xbf16> attributes {allow_soft_placement = false, tf.entry_function = {control_outputs = "", inputs = "_arg0,_arg1,_arg2", outputs = "_retval0"}} {
      %cst = "tf.Const"() {value = dense<[1, -1, 4]> : tensor<3xi32>} : () -> tensor<3xi32>
      %0 = "tf.Slice"(%arg0, %arg1, %cst) {_XlaHasReferenceVars = false, _xla_inferred_shapes = [#tf_type.shape<1x?x4>], device = "/job:localhost/replica:0/task:0/device:TPU:0"} : (tensor<3x3x10xbf16>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x?x4xbf16>
      return %0 : tensor<1x?x4xbf16>
    }
  })";

  TF_ASSERT_OK(LegalizeModule(kModuleWithNonConstParam));
}

TEST_F(Tf2XlaRewriterTest, CreatesDefaultValues) {
  // If a TF op has default value attributes and the mlir is missing them then
  // the LegalizeOp should insert the default values when converting the dialect
  // op to a node def.
  // TF.RandomUniform would fail without the seeds being set if they were not
  // automatically inserted with the default values.
  static constexpr char kModuleWithOpWithoutValuesThatShouldBeDefaulted[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1610 : i32}} {
    func.func @main() -> tensor<1x2x3x4xf32> attributes {allow_soft_placement = false, tf.entry_function = {control_outputs = "", inputs = "_arg0,_arg1,_arg2", outputs = "_retval0"}} {
      %cst = "tf.Const"() {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : () -> tensor<4xi32>
      %0 = "tf.RandomUniform"(%cst) : (tensor<4xi32>) -> tensor<1x2x3x4xf32>
      return %0 : tensor<1x2x3x4xf32>
    }
  })";

  TF_ASSERT_OK(LegalizeModule(kModuleWithOpWithoutValuesThatShouldBeDefaulted));
}

TEST_F(Tf2XlaRewriterTest, OpWithLocationDoesntBreakNodeDefName) {
  // A named location 'Name(Source)' causes the GetNameFromLoc method to append
  // all the other locations to the name with a ';' separator. This test ensures
  // that the name used for the NodeDef does not contain that invalid character.
  static constexpr char kModuleWithOpWithoutValuesThatShouldBeDefaulted[] =
      R"mlir(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1610 : i32}} {
    func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = "tf.Exp"(%arg0) : (tensor<2xf32>) -> tensor<2xf32> loc(fused["exp"("exp"), "exp"])
    func.return %0 : tensor<2xf32>
  }
  })mlir";

  TF_ASSERT_OK(LegalizeModule(kModuleWithOpWithoutValuesThatShouldBeDefaulted));
}

TEST_F(Tf2XlaRewriterTest, ErrorsWithInvalidNumberOfParametersToArgs) {
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
  XlaOp call_op = xla::Call(&builder, to_apply, {a, b});

  std::vector<XlaOp> tuple_values;
  tuple_values.push_back(call_op);
  xla::Tuple(&builder, tuple_values);

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());
  EXPECT_EQ(computation.proto().computations_size(), 2);

  TF_ASSERT_OK(CreateMlirModule());
  absl::StatusOr<stablehlo::TupleOp> status_or_tuple_op =
      ImportXlaComputationIntoModule(computation);
  EXPECT_FALSE(status_or_tuple_op.ok());
}

}  // namespace hlo
}  // namespace mlir
