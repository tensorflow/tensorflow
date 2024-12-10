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
// For example, the following:
//
//  %0 = tf_executor.island {
//    tf_executor.yield
//  }
//  %1:2 = tf_executor.island {
//    %2 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
//      tf_executor.yield %2 : tensor<i1>
//  }
//  %3:2 = tf_executor.island(%0) {
//    %4 = "tf_device.parallel_execute"() ({
//      %5 = "tf.opB"() : () -> tensor<i1>
//      tf_device.return %5 : tensor<i1>
//    }, {
//      %5 = "tf.opC"(%1#0) : (tensor<i1>) -> tensor<i32>
//      tf_device.return
//    }) {} : () -> (tensor<i1>)
//    tf_executor.yield %4 : tensor<i1>
//  }
//  tf_executor.fetch %3#0 : tensor<i1>
//
// gets lowered to:
//
//  %0 = tf_executor.island {
//    tf_executor.yield
//  }
//  %1:2 = tf_executor.island {
//    %2 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
//    tf_executor.yield %2 : tensor<i1>
//  }
//
//  // Island for the first region of above parallel_execute.
//  %3:2 = tf_executor.island(%0) {
//    %4 = "tf.opB"() : () -> tensor<i1>
//    tf_executor.yield %4 : tensor<i1>
//  }
//
//  // Island for the second region of above parallel_execute.
//  %5 = tf_executor.island(%0) {
//    %6 = "tf.opC"(%1#0) : (tensor<i1>) -> tensor<i32>
//    tf_executor.yield
//  }
//
//  tf_executor.fetch %3#0, %5 : tensor<i1>, !tf_executor.control
//
//  When tf_device.parallel_execute op is enclosed after tf_device.replicate,
//  then this pass will run following `replicate-to-island` pass and
//  `tf-executor-break-up-islands` pass.

#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/split_into_island_per_op_pass.h"

namespace mlir {
namespace TFDevice {
namespace {

#define GEN_PASS_DEF_PARALLELEXECUTETOISLANDSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct ParallelExecuteToIslandsPass
    : public impl::ParallelExecuteToIslandsPassBase<
          ParallelExecuteToIslandsPass> {
  explicit ParallelExecuteToIslandsPass(bool legacy_graph_export) {
    legacy_graph_export_ = legacy_graph_export;
  }
  void runOnOperation() override;
};

// Convert parallel_execute op to a set of islands where each region of
// parallel_execute op becomes a separate island. This ensures that the regions
// of the parallel_execute op gets executed concurrently.
void ExpandParallelExecuteToIslands(
    tf_executor::IslandOp island_op,
    tf_device::ParallelExecuteOp parallel_execute_op, OpBuilder* builder,
    llvm::SmallVectorImpl<tf_executor::IslandOp>& executes,
    bool legacy_graph_export, int parallel_group_idx) {
  const int num_regions = parallel_execute_op.getOperation()->getNumRegions();
  executes.reserve(num_regions);

  for (int i : llvm::seq<int>(0, num_regions)) {
    Block& execute_block = parallel_execute_op.GetRegionBlockWithIndex(i);

    // Replace terminator with tf_executor.YieldOp.
    Operation* terminator = execute_block.getTerminator();
    builder->setInsertionPoint(terminator);
    auto yield = builder->create<tf_executor::YieldOp>(
        terminator->getLoc(), terminator->getOperands());
    terminator->erase();

    // Create new island for each region.
    builder->setInsertionPoint(island_op);
    auto execute_island = builder->create<tf_executor::IslandOp>(
        island_op.getLoc(), yield.getOperandTypes(),
        island_op.getControl().getType(), island_op.getControlInputs());

    // Move over tf_device.parallel_execute body region into newly the created
    // island.
    execute_island.getBody().takeBody(*execute_block.getParent());

    // In new graph export pipeline, we will update control dependencies in the
    // end of the pipeline. Mostly, it will rely on side effect analysis by
    // considering accessing resource only. However, for branches under parallel
    // group, there should not be any control deps between them even side effect
    // analysis indicate some control deps. Therefore, we will mark parallel
    // group and branch information here so that `UpdateControlDependenciesPass`
    // can fetch the related information later.
    if (!legacy_graph_export) {
      std::string group_annotation = absl::StrCat(
          "p", std::to_string(parallel_group_idx), ":", std::to_string(i));
      if (auto parallel_group_attr =
              parallel_execute_op->getAttrOfType<StringAttr>(
                  TF::kParallelExecAnnotation)) {
        // Extend the existing attribute so that nested parallel execution
        // structure is supported.
        group_annotation = absl::StrCat(parallel_group_attr.getValue().str(),
                                        ",", group_annotation);
      }
      for (auto& op : execute_island.GetBody()) {
        op.setAttr(TF::kParallelExecAnnotation,
                   builder->getStringAttr(group_annotation));
      }
    }

    executes.push_back(execute_island);
  }
}

void CreateIslandsFromParallelExecute(
    tf_executor::IslandOp island_op,
    tf_device::ParallelExecuteOp parallel_execute_op, bool legacy_graph_export,
    int parallel_group_idx) {
  OpBuilder builder(island_op);

  // Create islands for each region of the parallel_execute op.
  llvm::SmallVector<tf_executor::IslandOp, 4> executes;
  ExpandParallelExecuteToIslands(island_op, parallel_execute_op, &builder,
                                 executes, legacy_graph_export,
                                 parallel_group_idx);

  // Remap all results of parallel_execute op with outputs from newly created
  // islands.
  llvm::SmallVector<Value, 8> parallel_execute_outputs;
  parallel_execute_outputs.reserve(
      parallel_execute_op.getOperation()->getNumResults());

  for (auto& execute : executes)
    parallel_execute_outputs.append(execute.getOutputs().begin(),
                                    execute.getOutputs().end());

  for (auto result :
       llvm::zip(island_op.getOutputs(), parallel_execute_outputs))
    std::get<0>(result).replaceAllUsesWith(std::get<1>(result));

  // Add sink island to pin all islands as a control dependency if there is a
  // control dependency leading from the parallel_execute originally.
  if (!island_op.getControl().use_empty()) {
    llvm::SmallVector<Value, 8> island_operands;
    for (auto& execute : executes)
      island_operands.push_back(execute.getControl());

    builder.setInsertionPoint(island_op);
    auto island_sink = builder.create<tf_executor::IslandOp>(
        island_op.getLoc(), llvm::ArrayRef<Type>{},
        island_op.getControl().getType(), island_operands);
    island_sink.getBody().push_back(new Block);
    builder.setInsertionPointToEnd(&island_sink.GetBody());
    builder.create<tf_executor::YieldOp>(island_op.getLoc(),
                                         llvm::ArrayRef<Value>{});
    island_op.getControl().replaceAllUsesWith(island_sink.getControl());
  }

  if (legacy_graph_export) {
    // Islands with no uses should be pinned to a graph fetch so they still
    // execute.
    llvm::SmallVector<Value, 8> unused_execute_controls;
    for (auto& execute : executes)
      if (execute.use_empty())
        unused_execute_controls.push_back(execute.getControl());

    if (!unused_execute_controls.empty()) {
      auto graph_op = island_op->getParentOfType<tf_executor::GraphOp>();
      tf_executor::FetchOp fetch = graph_op.GetFetch();
      auto fetches = llvm::to_vector<8>(fetch.getOperands());
      fetches.append(unused_execute_controls.begin(),
                     unused_execute_controls.end());
      builder.setInsertionPoint(fetch);
      builder.create<tf_executor::FetchOp>(fetch.getLoc(), fetches);
      fetch.erase();
    }
  } else {
    // Now, finally, we need to maintain the invariant expected to be maintained
    // throughout the graph export pipeline that all islands always perfectly
    // wrap a single op. So we'll split all islands which wrap multiple ops.
    auto control_type = tf_executor::ControlType::get(island_op.getContext());
    for (auto& execute : executes) {
      if (execute.GetBody().getOperations().size() > 1) {
        mlir::TF::SplitIsland(execute, control_type);
      }
    }
  }

  island_op.erase();
}

void ParallelExecuteToIslandsPass::runOnOperation() {
  // Find islands with a single `tf_device.parallel_execute` and create
  // individual islands per execute region of the parallel_execute.
  llvm::SmallVector<tf_executor::IslandOp, 4> parallel_execute_op_islands;
  getOperation().walk([&](tf_executor::GraphOp graph_op) {
    for (auto island_op : graph_op.getOps<tf_executor::IslandOp>()) {
      if (!island_op.WrapsSingleOp()) {
        island_op.emitError(
            "tf_executor.island must perfectly wrap a single op");
        signalPassFailure();
      }

      if (isa<tf_device::ParallelExecuteOp>(&island_op.GetBody().front()))
        parallel_execute_op_islands.push_back(island_op);
    }
  });

  // This number is unique within each function which is sufficient for
  // `UpdateControlDependenciesPass` which consumes the related attributes.
  // However, this assumes that we don't inline functions between this pass
  // and `UpdateControlDependenciesPass`.
  // If we need globally unique parallel group IDs in the future,
  // we can either make this pass a module pass (using a global counter)
  // or use an atomic counter.
  int parallel_group_idx = 0;
  for (tf_executor::IslandOp island_op : parallel_execute_op_islands) {
    auto parallel_execute_op =
        cast<tf_device::ParallelExecuteOp>(island_op.GetBody().front());
    CreateIslandsFromParallelExecute(island_op, parallel_execute_op,
                                     legacy_graph_export_, parallel_group_idx);
  }
}
}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateParallelExecuteToIslandsPass(
    bool legacy_graph_export) {
  return std::make_unique<ParallelExecuteToIslandsPass>(legacy_graph_export);
}

}  // namespace TFDevice
}  // namespace mlir
