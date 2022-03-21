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

#include <string>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_structs.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/convert_op_folder.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/hlo_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/utils.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf_passes_detail.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace mlir {
namespace mhlo {

namespace {

constexpr absl::string_view kGroupSizeAttrName =
    "tf2xla.collective_info.group_size";
constexpr absl::string_view kGroupKeyAttrName =
    "tf2xla.collective_info.group_key";

class LegalizeTFCollective
    : public LegalizeTFCollectiveBase<LegalizeTFCollective> {
 public:
  void runOnOperation() override;
};

LogicalResult SetOnceModuleAttribute(StringRef attr_name,
                                     IntegerAttr attr_value, Operation* op,
                                     ModuleOp& module) {
  const auto ex_attr_value = module->getAttrOfType<IntegerAttr>(attr_name);
  if (ex_attr_value == nullptr) {
    module->setAttr(attr_name, attr_value);
    return success();
  }
  if (ex_attr_value == attr_value) {
    return success();
  }
  return op->emitOpError() << "module already contains an attribute "
                           << attr_name << "=" << ex_attr_value.getInt()
                           << ", overwritting to a new value "
                           << attr_value.getInt() << " is not allowed.";
}

LogicalResult SetCollectiveInfo(IntegerAttr group_size, IntegerAttr group_key,
                                Operation* op, ModuleOp& module) {
  // The StringRef cast is necessary before cxx14.
  if (failed(SetOnceModuleAttribute(
          StringRef(kGroupSizeAttrName.data(), kGroupSizeAttrName.size()),
          group_size, op, module))) {
    return failure();
  }
  if (failed(SetOnceModuleAttribute(
          StringRef(kGroupKeyAttrName.data(), kGroupKeyAttrName.size()),
          group_key, op, module))) {
    return failure();
  }
  return success();
}

LogicalResult ConvertReplicaGroups(OpBuilder& builder,
                                   Value group_assignment_value,
                                   DenseIntElementsAttr& replica_groups,
                                   Operation* op) {
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
                               TensorType result_type,
                               DenseIntElementsAttr replica_groups,
                               StringRef mode, Value input, StringRef reduce_op,
                               Operation* op) {
  builder.setInsertionPoint(op);
  ChannelHandle channel_handle = ConvertChannel(builder, channel_id, mode);
  Location loc = op->getLoc();
  Type element_type = getElementTypeOrSelf(input.getType());
  auto all_reduce = builder.create<AllReduceOp>(loc, result_type, input,
                                                replica_groups, channel_handle);
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
    return op->emitOpError() << "invalid reduce_op " << reduce_op
                             << ", want one of [Add, Mul, Min, Max, Mean]";
  }

  Operation* result = all_reduce;
  // For mean, divide the merge result by group size.
  if (reduce_op == "Mean") {
    int64_t replica_group_size = replica_groups.getType().getDimSize(1);
    auto divisor =
        GetScalarConstOfType(element_type, loc, replica_group_size, &builder);
    auto broadcast_dims = GetI64ElementsAttr({}, &builder);
    result = builder.create<chlo::BroadcastDivOp>(
        loc, all_reduce.getResult(), divisor.getResult(), broadcast_dims);
  }
  op->replaceAllUsesWith(result);

  op->erase();
  return success();
}

LogicalResult ConvertTfXlaCollective(OpBuilder& builder, int64_t channel_id,
                                     TF::XlaAllReduceOp& all_reduce,
                                     ModuleOp& module) {
  DenseIntElementsAttr replica_groups;
  if (failed(ConvertReplicaGroups(builder, all_reduce.group_assignment(),
                                  replica_groups, all_reduce)))
    return failure();
  IntegerAttr group_size = builder.getI32IntegerAttr(replica_groups.size());
  IntegerAttr group_key = builder.getI32IntegerAttr(0);
  if (failed(SetCollectiveInfo(group_size, group_key, all_reduce, module))) {
    return failure();
  }
  return ConvertAllReduce(builder, channel_id, all_reduce.getType(),
                          replica_groups, all_reduce.mode(), all_reduce.input(),
                          all_reduce.reduce_op(), all_reduce);
}

LogicalResult ConvertTfCollectiveReduceV2(OpBuilder& builder,
                                          int64_t channel_id,
                                          TF::CollectiveReduceV2Op& all_reduce,
                                          ModuleOp& module) {
  DenseIntElementsAttr group_size_attr;
  if (!matchPattern(all_reduce.group_size(), m_Constant(&group_size_attr))) {
    return all_reduce.emitOpError()
           << "group_size must be a compile time constant";
  }
  if (!group_size_attr.isSplat() || group_size_attr.size() != 1) {
    return all_reduce.emitOpError() << "group_size must be a scalar";
  }
  DenseIntElementsAttr group_key_attr;
  if (!matchPattern(all_reduce.group_key(), m_Constant(&group_key_attr))) {
    return all_reduce.emitOpError()
           << "group_key must be a compile time constant";
  }
  if (!group_key_attr.isSplat() || group_key_attr.size() != 1) {
    return all_reduce.emitOpError() << "group_key must be a scalar";
  }
  const auto group_size = group_size_attr.getSplatValue<IntegerAttr>();
  const auto group_key = group_key_attr.getSplatValue<IntegerAttr>();

  // Create an empty group assignment.
  auto replica_groups = mlir::DenseIntElementsAttr::get<int64_t>(
      mlir::RankedTensorType::get({0, 0}, builder.getI64Type()), {});

  if (failed(SetCollectiveInfo(group_size, group_key, all_reduce, module))) {
    return failure();
  }

  // CrossReplicaAndPartition:
  // Even though TF2XLA will setup the device assignment to include
  // devices in this group as replicas before launching this module,
  // "CrossReplica" mode (no channel) produces a deadlock when
  // not using XLA SPMD expansion.
  return ConvertAllReduce(
      builder, channel_id, all_reduce.getType(), replica_groups,
      /* mode= */ "CrossReplicaAndPartition", all_reduce.input(),
      all_reduce.merge_op(), all_reduce);
}

#include "tensorflow/compiler/mlir/xla/transforms/generated_legalize_tf_collective.inc"

LogicalResult ConvertTfCollective(Operation* op) {
  MLIRContext* context = op->getContext();
  RewritePatternSet patterns(context);
  patterns.insert<RewriteCollectiveAssignGroupV2CollectiveReduceV2>(context);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
    return failure();
  }
  return success();
}

void LegalizeTFCollective::runOnOperation() {
  if (failed(ConvertTfCollective(getOperation()))) {
    signalPassFailure();
    return;
  }
  int64_t channel_id = 0;
  OpBuilder builder(&getContext());

  auto module = getOperation();
  auto result = module.walk([&](Operation* op) {
    if (auto all_reduce = dyn_cast<TF::XlaAllReduceOp>(op)) {
      ++channel_id;
      if (failed(ConvertTfXlaCollective(builder, channel_id, all_reduce,
                                        module))) {
        return WalkResult::interrupt();
      }
    } else if (auto all_reduce = dyn_cast<TF::CollectiveReduceV2Op>(op)) {
      ++channel_id;
      if (failed(ConvertTfCollectiveReduceV2(builder, channel_id, all_reduce,
                                             module))) {
        return WalkResult::interrupt();
      }
    } else if (isa<TF::CollectiveAssignGroupV2Op>(op)) {
      if (op->use_empty()) {
        op->erase();
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) signalPassFailure();
}
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeTFCollectivePass() {
  return std::make_unique<LegalizeTFCollective>();
}

}  // namespace mhlo
}  // namespace mlir
