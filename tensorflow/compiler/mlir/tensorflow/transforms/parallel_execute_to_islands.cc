/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This pass forms `tf_executor.island` per region of
// `tf_device.parallel_execute`.
//
// For example:
//  %1:2 = tf_executor.island {
//    %2 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
//      tf_executor.yield %2 : tensor<i1>
//  }
//  tf_executor.island() {
//    "tf_device.parallel_execute"() ({
//      %3 = "tf.opB"() : () -> tensor<i1>
//      tf_device.return %3 : tensor<i1>
//    },
//    {
//      %5 = "tf.opC"(%1#0) : (tensor<i1>) -> tensor<i32>
//      tf_device.return
//    }) {} : () -> (tensor<i1>)
//    tf_executor.yield
//  }
//  tf_executor.fetch
//
//  Would become:
//    %1:2 = tf_executor.island {
//      %2 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
//      tf_executor.yield %2 : tensor<i1>
//    }
//
//    // Input barrier sink island that forwards all inputs.
//    %output_0, %control_1 = tf_executor.island {
//      tf_executor.yield %1#0: tensor<i1>
//    }
//
//    // Island for the first region of above parallel_execute.
//    %output_2, %control_3 = tf_executor.island(%control_1) {
//      %3 = "tf.opB"() : () -> tensor<i1>
//      tf_executor.yield %3 : tensor<i1>
//    }
//
//    // Island for the second region of above parallel_execute.
//    %control_5 = tf_executor.island {
//        %5 = "tf.opC"(%output_0) : (tensor<i1>) -> tensor<i32>
//      tf_executor.yield
//    }
//
//    // Output barrier sink island that forwards all outputs.
//    %output_5, %control_6 = tf_executor.island(%control_5) {
//      tf_executor.yield %output_2
//    }
//
//  When tf_device.parallel_execute op is enclosed after tf_device.replicate,
//  then this pass will run following `replicate-to-island` pass and
//  `tf-executor-break-up-islands` pass.

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"

namespace mlir {
namespace TFDevice {
namespace {

struct ParallelExecuteToIslandsPass
    : public PassWrapper<ParallelExecuteToIslandsPass, FunctionPass> {
  void runOnFunction() override;
};

// Convert parallel_execute op to a set of islands where each region of
// parallel_execute op becomes a separate island. This ensures that
// regions of parallel_execute op gets executed concurrently.
LogicalResult ExpandParallelExecuteToIslands(
    tf_executor::IslandOp island_op, tf_executor::IslandOp input_sink_island,
    tf_device::ParallelExecuteOp parallel_execute_op, OpBuilder* builder,
    llvm::SmallVector<tf_executor::IslandOp, 4>* islands) {
  const int num_executions =
      parallel_execute_op.getOperation()->getNumRegions();
  llvm::SmallVector<tf_executor::IslandOp, 4> executions;
  executions.reserve(num_executions);
  builder->setInsertionPoint(island_op);

  auto control_type = tf_executor::ControlType::get(island_op.getContext());
  for (int i : llvm::seq<int>(0, num_executions)) {
    auto execute_region =
        parallel_execute_op.GetRegionBlockWithIndex(i).getParent();

    // If region does not have any inputs, then add explicit control dependency
    // from the input sink island. This guarantees that all inputs of
    // parallel_execute op must be materialized before any of the islands are
    // executed.
    llvm::SetVector<Value> region_inputs;
    getUsedValuesDefinedAbove(*execute_region, region_inputs);
    llvm::SmallVector<Value, 8> execution_control_inputs;
    if (region_inputs.empty())
      execution_control_inputs.emplace_back(input_sink_island.control());

    // Collect result types and operands.
    Operation* terminator = execute_region->front().getTerminator();
    llvm::SmallVector<Type, 8> output_types(terminator->getOperandTypes());

    // Replace terminator with YieldOp as island op always ends with yield op.
    builder->setInsertionPoint(terminator);
    builder->create<tf_executor::YieldOp>(terminator->getLoc(),
                                          terminator->getOperands());
    terminator->erase();

    // Create new island for each region.
    builder->setInsertionPoint(island_op);
    auto execution_island = builder->create<tf_executor::IslandOp>(
        island_op.getLoc(), output_types, control_type,
        execution_control_inputs);

    // Move over tf_device.parallel_execute body region into newly a
    // created island.
    execution_island.body().takeBody(*execute_region);
    islands->push_back(execution_island);
  }

  return success();
}

// Creates an island that works as input sync point for islands. This guarantees
// that all (implicitly captured) inputs of parallel_execute are materialized
// before any of the islands are executed.
tf_executor::IslandOp CreateInputBarrierIsland(
    OpBuilder* builder, tf_executor::IslandOp island_op) {
  builder->setInsertionPoint(island_op);

  llvm::SetVector<Value> island_inputs;
  getUsedValuesDefinedAbove(island_op.body(), island_inputs);

  llvm::SmallVector<Type, 8> input_types;
  input_types.reserve(island_inputs.size());
  for (const auto& input_val : island_inputs)
    input_types.emplace_back(input_val.getType());

  // Create new island for that forwards all inputs.
  auto control_type = tf_executor::ControlType::get(island_op.getContext());
  auto input_sink_island = builder->create<tf_executor::IslandOp>(
      island_op.getLoc(), input_types, control_type, island_op.controlInputs());
  input_sink_island.body().push_back(new Block);

  for (auto input_index_and_value : llvm::enumerate(island_inputs)) {
    int index = input_index_and_value.index();
    Value input_value = input_index_and_value.value();
    replaceAllUsesInRegionWith(input_value, input_sink_island.getResult(index),
                               island_op.body());
  }

  // Create YieldOp for the new input sink island.
  builder->setInsertionPointToEnd(&input_sink_island.GetBody());
  builder->create<tf_executor::YieldOp>(island_op.getLoc(),
                                        llvm::to_vector<8>(island_inputs));
  return input_sink_island;
}

// Creates an islands that works as output sync point. This guarantees that
// execution of all islands must be completed before op following
// parallel_execute runs.
tf_executor::IslandOp CreateOutputBarrierIsland(
    OpBuilder* builder, tf_executor::IslandOp island_op,
    llvm::SmallVectorImpl<tf_executor::IslandOp>* islands) {
  // Add control dependency to island operand if island output has no uses.
  llvm::SmallVector<Value, 8> island_operands;
  for (auto& island : *islands)
    if (island.use_empty()) island_operands.push_back(island.control());

  // Create single island forwarding all island results.
  builder->setInsertionPoint(island_op);
  auto island_output_sink = builder->create<tf_executor::IslandOp>(
      island_op.getLoc(), llvm::to_vector<8>(island_op.getResultTypes()),
      island_operands, llvm::ArrayRef<NamedAttribute>{});
  island_output_sink.body().push_back(new Block);
  return island_output_sink;
}

LogicalResult CreateIslandsFromParallelExecute(
    tf_executor::IslandOp island_op,
    tf_device::ParallelExecuteOp parallel_execute_op) {
  OpBuilder builder(island_op);
  auto input_sink_island = CreateInputBarrierIsland(&builder, island_op);

  // Create N islands where N is the number of regions inside parallel_execute
  // op.
  llvm::SmallVector<tf_executor::IslandOp, 4> islands;
  auto result = ExpandParallelExecuteToIslands(
      island_op, input_sink_island, parallel_execute_op, &builder, &islands);
  if (failed(result)) return result;

  // Remap all results of parallel_execute op with outputs from newly
  // created islands.
  llvm::SmallVector<Value, 8> parallel_execute_outputs;
  parallel_execute_outputs.reserve(
      parallel_execute_op.getOperation()->getNumResults());

  for (auto island : islands)
    for (auto output_value : island.outputs())
      parallel_execute_outputs.emplace_back(output_value);

  parallel_execute_op.getOperation()->replaceAllUsesWith(
      parallel_execute_outputs);

  auto island_output_sink =
      CreateOutputBarrierIsland(&builder, island_op, &islands);

  // Move island YieldOp over to new single island and remap island results.
  island_op.GetYield().getOperation()->moveBefore(
      &island_output_sink.GetBody(), island_output_sink.GetBody().begin());
  island_op.replaceAllUsesWith(island_output_sink);
  island_op.erase();

  return success();
}

// Finds islands with a single `tf_device.parallel_execute` and create
// individual islands per region of parallel_execute.
void LowerSingleIslandParallelExecuteToIslands(
    tf_executor::IslandOp island_op) {
  if (!hasSingleElement(island_op.GetBody().without_terminator())) return;

  if (auto parallel_execute_op = llvm::dyn_cast<tf_device::ParallelExecuteOp>(
          &island_op.GetBody().front()))
    CreateIslandsFromParallelExecute(island_op, parallel_execute_op);
}

void ParallelExecuteToIslandsPass::runOnFunction() {
  getFunction().walk([&](tf_executor::IslandOp island_op) {
    LowerSingleIslandParallelExecuteToIslands(island_op);
  });
}
}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> CreateParallelExecuteToIslandsPass() {
  return std::make_unique<ParallelExecuteToIslandsPass>();
}

static PassRegistration<ParallelExecuteToIslandsPass> pass(
    "tf-parallel-execute-to-islands",
    "Lowers device parallel_execute to executor islands");

}  // namespace TFDevice
}  // namespace mlir
