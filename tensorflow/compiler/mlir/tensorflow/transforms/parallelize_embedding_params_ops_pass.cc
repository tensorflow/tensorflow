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

// This transformation parallelizes TPU embedding params assigned to different
// shards using the parallel execute op. This is useful to avoid introducing
// control dependency between these ops that are known to be independent.

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.h"

namespace mlir {
namespace TFDevice {

namespace {

struct ParallelizeEmbeddingParamsOpsPass
    : public PassWrapper<ParallelizeEmbeddingParamsOpsPass, FunctionPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<tf_device::TensorFlowDeviceDialect>();
  }

  void runOnFunction() override;
};

bool IsLoadTPUEmbeddingParmasOp(Operation& op) {
  static const auto* algorithms = []() {
    auto* algorithms = new llvm::SmallSet<std::string, 16>();
    for (tensorflow::tpu::OptimizationAlgorithm alg :
         tensorflow::tpu::GetOptimizationAlgorithms()) {
      const auto alg_name = tensorflow::tpu::GetOptimizationAlgorithmName(alg);
      algorithms->insert(alg_name);
    }
    return algorithms;
  }();
  StringRef op_name = op.getName().getStringRef();
  return op_name.consume_front("tf.LoadTPUEmbedding") &&
         op_name.consume_back("Parameters") &&
         algorithms->contains(op_name.str());
}

static LogicalResult RunOnIsland(tf_executor::IslandOp island) {
  Block* block = island.getBody();

  // Map from op to the id of the shard it is assigned for ops that can execute
  // in parallel across shards.
  llvm::SmallMapVector<Operation*, int64_t, 4> assigned_shard;
  llvm::SmallVector<Value, 8> resources;
  llvm::SmallSet<int64_t, 16> shard_ids;
  for (Operation& op : llvm::reverse(*block)) {
    int64_t shard = -1;
    if (IsLoadTPUEmbeddingParmasOp(op)) {
      auto shard_id = op.getAttrOfType<mlir::IntegerAttr>("shard_id");
      if (!shard_id) {
        return op.emitOpError("requires 'shard_id' integer attribute");
      }
      shard = shard_id.getInt();
      shard_ids.insert(shard);
    } else if (auto read_op = llvm::dyn_cast<TF::ReadVariableOp>(op)) {
      if (assigned_shard.empty()) continue;

      for (Operation* user : op.getUsers()) {
        auto iter = assigned_shard.find(user);
        if (iter == assigned_shard.end() ||
            (shard != -1 && shard != iter->second)) {
          shard = -1;
          break;
        }
        shard = iter->second;
      }
      if (shard != -1) resources.push_back(read_op.resource());
    }

    if (shard != -1) assigned_shard.insert(std::make_pair(&op, shard));
  }

  // No transformations are required.
  int num_shards = shard_ids.size();
  if (num_shards <= 1) return success();

  // If the resources are used for ops other than read variable op, then moving
  // read variable ops to the parallel_execute may not preserve the semantics.
  for (Value resource : resources) {
    for (Operation* user : resource.getUsers())
      if (!llvm::isa<TF::ReadVariableOp>(*user)) return success();
  }

  // Create parallel_execute op at the end of the block and move operations
  // to their corresponding shard.
  auto builder = OpBuilder::atBlockTerminator(block);
  auto parallel_execute_op = builder.create<tf_device::ParallelExecuteOp>(
      island.getLoc(), num_shards, llvm::ArrayRef<Type>());
  for (int shard_id = 0; shard_id < num_shards; ++shard_id) {
    mlir::Block& b = parallel_execute_op.GetRegionBlockWithIndex(shard_id);
    builder.setInsertionPointToStart(&b);
    builder.create<tf_device::ReturnOp>(island.getLoc());
  }

  for (auto op_shard : assigned_shard) {
    int64_t shard = op_shard.second;
    if (shard >= num_shards) {
      return island.emitOpError(
          "load tpu embedding ops require continuous range of shards");
    }
    mlir::Block& b = parallel_execute_op.GetRegionBlockWithIndex(shard);
    op_shard.first->moveBefore(&b, b.begin());
  }
  return success();
}

void ParallelizeEmbeddingParamsOpsPass::runOnFunction() {
  getFunction().walk([&](tf_executor::IslandOp island) {
    if (failed(RunOnIsland(island))) {
      signalPassFailure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateParallelizeEmbeddingParamsOpsPass() {
  return std::make_unique<ParallelizeEmbeddingParamsOpsPass>();
}
}  // namespace TFDevice
}  // namespace mlir

static mlir::PassRegistration<mlir::TFDevice::ParallelizeEmbeddingParamsOpsPass>
    pass("tf-parallize-embedding-params-ops",
         "Parallelizes TPU embedding params assigned to different shards using "
         "the parallel_execte op");
