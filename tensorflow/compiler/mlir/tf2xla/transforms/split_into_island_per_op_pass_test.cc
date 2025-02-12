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
#include "tensorflow/compiler/mlir/tf2xla/transforms/split_into_island_per_op_pass.h"

#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/register_common_dialects.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"

namespace mlir {
namespace TF {

using mlir::DialectRegistry;
using mlir::MLIRContext;
class SplitIntoIslandPerOpPass : public ::testing::Test {
 public:
  SplitIntoIslandPerOpPass() : op_builder_(&context_) {
    mlir::RegisterCommonToolingDialects(registry_);
    context_.appendDialectRegistry(registry_);
    context_.loadAllAvailableDialects();
    context_.allowUnregisteredDialects(true);
  }

  mlir::tf_executor::IslandOp GenerateIslandOp() {
    mlir::OperationState op_state(mlir::UnknownLoc::get(&context_),
                                  StringRef("test_island"));
    llvm::SmallVector<mlir::Type, 1> island_result_types;
    island_result_types.push_back(op_builder_.getF64Type());

    mlir::Operation* yield_op = op_builder_.create<mlir::tf_executor::YieldOp>(
        op_state.location, mlir::ValueRange{});
    mlir::tf_executor::IslandOp island_op =
        op_builder_.create<mlir::tf_executor::IslandOp>(
            op_state.location, island_result_types, mlir::ValueRange{},
            mlir::ArrayRef<mlir::NamedAttribute>{});
    island_op.getBody().push_back(new mlir::Block);
    island_op.getBody().back().push_back(yield_op);
    return island_op;
  }

  mlir::Operation* GenerateOp(StringRef name, bool add_return_type = false) {
    mlir::OperationState state(mlir::UnknownLoc::get(&context_), name);
    if (add_return_type) {
      state.addTypes(ArrayRef<mlir::FloatType>{op_builder_.getF64Type()});
    }
    return mlir::Operation::create(state);
  }

  std::vector<std::string> GetOpNames(mlir::Operation* op) {
    std::vector<std::string> op_names;
    for (auto& op : op->getBlock()->getOperations()) {
      op_names.push_back(op.getName().getStringRef().str());
    }
    return op_names;
  }

  DialectRegistry registry_;
  MLIRContext context_;
  OpBuilder op_builder_;
};

TEST_F(SplitIntoIslandPerOpPass, EmptyIslandPoulatesNoOp) {
  mlir::tf_executor::ControlType control_type;
  mlir::tf_executor::IslandOp islandOp = GenerateIslandOp();

  SplitIsland(islandOp, control_type);

  std::set<std::string> actual_op_names;
  for (auto& op : islandOp.getBody().getOps()) {
    actual_op_names.insert(op.getName().getStringRef().str());
  }
  std::set<std::string> expected_op_names = {"tf_executor.yield", "tf.NoOp"};
  ASSERT_EQ(actual_op_names, expected_op_names);
  islandOp.erase();
}

TEST_F(SplitIntoIslandPerOpPass, IslandOpSingleOpLeftUnchanged) {
  mlir::tf_executor::ControlType control_type;
  mlir::tf_executor::IslandOp islandOp = GenerateIslandOp();
  mlir::Operation* inner_op = GenerateOp("inner_op");
  islandOp.getBody().front().push_front(inner_op);

  SplitIsland(islandOp, control_type);

  std::set<std::string> actual_op_names;
  for (auto& op : islandOp.getBody().getOps()) {
    actual_op_names.insert(op.getName().getStringRef().str());
  }
  std::set<std::string> expected_op_names = {"inner_op", "tf_executor.yield"};
  ASSERT_EQ(actual_op_names, expected_op_names);
  islandOp.erase();
}

TEST_F(SplitIntoIslandPerOpPass, IslandOpTwoOpsSplitsIntoTwoIslands) {
  auto control_type = mlir::tf_executor::ControlType::get(&context_);
  mlir::tf_executor::IslandOp islandOp = GenerateIslandOp();
  mlir::Operation* inner_op_1 = GenerateOp("inner_op_1", true);
  mlir::Operation* inner_op_2 = GenerateOp("inner_op_2", true);
  islandOp.getBody().front().push_front(inner_op_1);
  islandOp.getBody().back().push_front(inner_op_2);
  // Code relies on a parent with a fetch op containing the island op.
  mlir::tf_executor::GraphOp parent_graph_op =
      op_builder_.create<mlir::tf_executor::GraphOp>(
          mlir::UnknownLoc::get(&context_),
          mlir::TypeRange{op_builder_.getF64Type()});
  parent_graph_op.getRegion().push_back(new mlir::Block);
  parent_graph_op.push_back(islandOp);
  mlir::tf_executor::FetchOp fetch_op =
      op_builder_.create<mlir::tf_executor::FetchOp>(parent_graph_op.getLoc());
  parent_graph_op.GetBody().push_back(fetch_op);

  SplitIsland(islandOp, control_type);

  std::vector<std::string> actual_op_names;
  for (auto& op : parent_graph_op.getBody().getOps()) {
    actual_op_names.push_back(op.getName().getStringRef().str());
  }
  std::vector<std::string> expected_op_names = {
      "tf_executor.island", "tf_executor.island", "tf_executor.fetch"};
  ASSERT_EQ(actual_op_names, expected_op_names);
  parent_graph_op.erase();
}

}  // namespace TF
}  // namespace mlir
