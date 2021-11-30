/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for lowering TensorFlow dialect's collective
// ops (TF/XLA) to the HLO dialect.

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_structs.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/convert_op_folder.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/hlo_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace mlir {
namespace mhlo {

namespace {

class LegalizeTFCollective
    : public PassWrapper<LegalizeTFCollective, OperationPass<ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mhlo::MhloDialect>();
  }
  StringRef getArgument() const final { return "xla-legalize-tf-collective"; }
  StringRef getDescription() const final {
    return "Legalize TF/XLA collective ops (TensorFlow dialect) to the HLO "
           "dialect";
  }
  void runOnOperation() override;
};

bool IsTfXlaCollectiveOp(Operation* op) { return isa<TF::XlaAllReduceOp>(op); }

LogicalResult ConvertReplicaGroups(OpBuilder& builder, Operation* op,
                                   Value group_assignment_value,
                                   DenseIntElementsAttr& replica_groups) {
  DenseIntElementsAttr group_assignment;
  if (!matchPattern(group_assignment_value, m_Constant(&group_assignment))) {
    return op->emitOpError() << "expects constant group_assignment";
  }
  replica_groups =
      hlo::ConvertElementsAttr(group_assignment, builder.getIntegerType(64))
          .cast<DenseIntElementsAttr>();
  if (replica_groups.getType().getRank() != 2) {
    return op->emitOpError() << "group_assignment should have rank 2, got "
                             << replica_groups.getType().getRank();
  }
  return success();
}

ChannelHandle ConvertChannel(OpBuilder& builder, int64_t channel_id,
                             StringRef mode) {
  if (mode == "CrossReplica") {
    return ChannelHandle();
  }
  return ChannelHandle::get(
      /*handle=*/builder.getI64IntegerAttr(channel_id),
      /*type=*/
      builder.getI64IntegerAttr(xla::ChannelHandle::DEVICE_TO_DEVICE),
      builder.getContext());
}

LogicalResult ConvertAllReduce(OpBuilder& builder, int64_t channel_id,
                               TF::XlaAllReduceOp op) {
  builder.setInsertionPoint(op);
  DenseIntElementsAttr replica_groups;
  if (failed(ConvertReplicaGroups(builder, op, op.group_assignment(),
                                  replica_groups)))
    return failure();
  ChannelHandle channel_handle = ConvertChannel(builder, channel_id, op.mode());
  Location loc = op.getLoc();
  Type element_type = getElementTypeOrSelf(op.input().getType());
  auto all_reduce = builder.create<AllReduceOp>(loc, op.getType(), op.input(),
                                                replica_groups, channel_handle);
  StringRef reduce_op = op.reduce_op();
  if (reduce_op == "Add") {
    BuildReduceBody<AddOp>(element_type, &all_reduce.computation(), &builder);
  } else if (reduce_op == "Mul") {
    BuildReduceBody<MulOp>(element_type, &all_reduce.computation(), &builder);
  } else if (reduce_op == "Min") {
    BuildReduceBody<MinOp>(element_type, &all_reduce.computation(), &builder);
  } else if (reduce_op == "Max") {
    BuildReduceBody<MaxOp>(element_type, &all_reduce.computation(), &builder);
  } else if (reduce_op == "Mean") {
    // For mean, add replicas in the same group. Then divide the sum by the
    // number of replicas in each group below.
    BuildReduceBody<AddOp>(element_type, &all_reduce.computation(), &builder);
  } else {
    return op.emitOpError() << "invalid reduce_op " << reduce_op
                            << ", want one of [Add, Mul, Min, Max, Mean]";
  }
  Value result = all_reduce.getResult();

  // For mean, divide the merge result by group size.
  if (reduce_op == "Mean") {
    int64_t replica_group_size = replica_groups.getType().getDimSize(1);
    auto divisor =
        GetScalarConstOfType(element_type, loc, replica_group_size, &builder);
    auto broadcast_dims = GetI64ElementsAttr({}, &builder);
    result = builder.create<chlo::BroadcastDivOp>(
        loc, result, divisor.getResult(), broadcast_dims);
  }

  op.replaceAllUsesWith({result});
  op.erase();
  return success();
}

LogicalResult ConvertTfXlaCollective(OpBuilder& builder, int64_t channel_id,
                                     Operation* op) {
  if (auto all_reduce = dyn_cast<TF::XlaAllReduceOp>(op)) {
    return ConvertAllReduce(builder, channel_id, all_reduce);
  }
  return failure();
}

void LegalizeTFCollective::runOnOperation() {
  int64_t channel_id = 0;
  OpBuilder builder(&getContext());
  auto result = getOperation().walk([&](Operation* op) {
    if (IsTfXlaCollectiveOp(op)) {
      ++channel_id;
      if (failed(ConvertTfXlaCollective(builder, channel_id, op))) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) signalPassFailure();
}

static PassRegistration<LegalizeTFCollective> pass;
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeTFCollectivePass() {
  return std::make_unique<LegalizeTFCollective>();
}

}  // namespace mhlo
}  // namespace mlir
