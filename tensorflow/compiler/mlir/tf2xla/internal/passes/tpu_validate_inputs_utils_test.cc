/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/tpu_validate_inputs_utils.h"

#include <gtest/gtest.h>
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/test_utils.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {
namespace {

using mlir::mhlo::test::GetMlirModuleFromString;

TEST(IsPotentialUnsupportedOp, ClusterOpReturnsFalse) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::tf_device::TensorFlowDeviceDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OpBuilder builder(module_ref->getBodyRegion());

  llvm::SmallVector<mlir::Type, 8> result_types;
  auto cluster = builder.create<mlir::tf_device::ClusterOp>(
      mlir::UnknownLoc::get(&context), result_types);
  cluster->dump();
  EXPECT_FALSE(IsPotentialUnsupportedOp(cluster));
}

TEST(IsPotentialUnsupportedOp, InfeedDequeueTupleOpReturnsTrue) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::TF::TensorFlowDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OpBuilder builder(module_ref->getBodyRegion());

  llvm::SmallVector<mlir::Type, 8> result_types;
  mlir::StringAttr _XlaSharding = mlir::StringAttr::get(&context, "");
  mlir::ArrayAttr layouts = mlir::ArrayAttr::get(&context, {});

  auto infeed_dequeue_tuple = builder.create<InfeedDequeueTupleOp>(
      mlir::UnknownLoc::get(&context), result_types, _XlaSharding, layouts);

  infeed_dequeue_tuple->setAttr(
      kDeviceAttr, mlir::StringAttr::get(&context, kTpuReplicatedCoreZeroAttr));

  EXPECT_TRUE(IsPotentialUnsupportedOp(infeed_dequeue_tuple));
}

TEST(HasV1ControlFlow, ReturnsTrue) {
  static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @graph_contains_v1_control_flow() {
      tf_executor.graph {
        %control = tf_executor.ControlTrigger {}
        tf_executor.fetch
      }
      func.return
    }
  })";
  mlir::MLIRContext context;
  context.loadDialect<mlir::tf_executor::TensorFlowExecutorDialect>();
  auto module = GetMlirModuleFromString(kMlirModuleStr, &context);

  module->get().walk(
      [&](GraphOp graph) { EXPECT_TRUE(HasV1ControlFlow(graph)); });
}

}  // namespace

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
