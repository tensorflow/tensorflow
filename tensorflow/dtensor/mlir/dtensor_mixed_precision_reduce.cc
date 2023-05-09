/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/strings/string_view.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/cc/dtensor_utils.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORMIXEDPRECISIONREDUCE
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

// Extracts the reduction group size from the group_assignment operand of the
// reduce op. group_assignment is a 2-dimensional array where each element is
// the list of devices that are a part of the same reduction group.
template <class ReduceOpType>
mlir::LogicalResult GetAllReduceGroupSize(ReduceOpType reduce_op,
                                          int32* group_size) {
  mlir::DenseIntElementsAttr group_assignment_attr;
  if (!matchPattern(reduce_op.getGroupAssignment(),
                    m_Constant(&group_assignment_attr)))
    return mlir::emitError(reduce_op.getLoc(),
                           "group_assigment must be a constant.");
  if (group_assignment_attr.getType().getRank() != 2)
    return mlir::emitError(reduce_op.getLoc(),
                           "group_assignment should have two dimensions.");

  *group_size = group_assignment_attr.getType().getShape()[1];
  return mlir::success();
}

// For large enough reduction groups, we compute reductions in a higher
// precision type to ensure accuracy is not lost with sequential addition
// of large numbers in a lower precision type. If the given reduce op meets the
// following criteria:
//   - the tensors being reduced are of type bfloat16,
//   - the reduction group is at least as large as the configurable env var
//     DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE,
// then the tensors are upcasted to float32 for the reduction before being
// downcasted again.
template <class ReduceOpType>
mlir::LogicalResult MaybeUpcastForReduction(ReduceOpType reduce_op,
                                            bool* changed) {
  const mlir::RankedTensorType& input_type =
      reduce_op.getInput()
          .getType()
          .template dyn_cast<mlir::RankedTensorType>();
  if (!input_type.getElementType().isBF16()) {
    // Upcast only applies for bfloat16 input.
    return mlir::success();
  }

  mlir::OpBuilder builder(reduce_op);
  const mlir::Location loc = reduce_op.getLoc();

  int32 group_size;
  if (mlir::failed(GetAllReduceGroupSize(reduce_op, &group_size)))
    return mlir::failure();
  if (group_size <= ReduceInBfloat16MaxGroupSize())
    // Reduce group size is not sufficient, so we do not modify the ops.
    return mlir::success();

  const auto reduce_layout = ExtractRequiredSingleLayoutFromOp(reduce_op);
  if (!reduce_layout.ok())
    return reduce_op.emitOpError(llvm::formatv(
        "Malformed layout specification for DTensor reduce op found: {0}",
        reduce_layout.status().message()));

  // The original output tensor type that would have been used by all users of
  // the reduce op.
  const mlir::RankedTensorType& output_type =
      reduce_op.getOutput()
          .getType()
          .template dyn_cast<mlir::RankedTensorType>();

  mlir::TF::CastOp upcast = builder.create<mlir::TF::CastOp>(
      loc,
      mlir::RankedTensorType::get(input_type.getShape(), builder.getF32Type()),
      reduce_op.getInput());
  reduce_op->setOperand(0, upcast.getY());
  reduce_op.getOutput().setType(upcast.getY().getType());

  builder.setInsertionPointAfter(reduce_op);
  mlir::TF::CastOp downcast = builder.create<mlir::TF::CastOp>(
      loc,
      mlir::RankedTensorType::get(output_type.getShape(),
                                  output_type.getElementType()),
      reduce_op);
  // Match the layout of the downcast with the reduce op, this is required for
  // the later passes.
  SetSingleLayoutOnOp(downcast, *reduce_layout);
  reduce_op.getOutput().replaceAllUsesExcept(downcast.getY(), downcast);

  *changed = true;
  return mlir::success();
}

template <class ReduceOpType>
mlir::LogicalResult TryMixedPrecisionReduce(mlir::func::FuncOp function,
                                            absl::string_view opName) {
  int32_t reduceOpsCounter = 0;
  int32_t changedReduceOpsCounter = 0;

  mlir::WalkResult walk_result = function.walk([&](ReduceOpType reduce_op) {
    if (reduce_op.getReduceOp().str() == kReduceOpAdd) {
      reduceOpsCounter += 1;
      bool changed = false;
      if (mlir::failed(MaybeUpcastForReduction(reduce_op, &changed)))
        return mlir::WalkResult::interrupt();
      if (changed) changedReduceOpsCounter += 1;
    }
    return mlir::WalkResult::advance();
  });
  if (walk_result.wasInterrupted()) return mlir::failure();

  VLOG(2) << "Applied mixed precision to " << changedReduceOpsCounter << " of "
          << reduceOpsCounter << " Add " << opName << " ops.";

  return mlir::success();
}

// MLIR pass that enables tensor upcasting within mixed-precision reduction.
struct DTensorMixedPrecisionReducePass
    : public impl::DTensorMixedPrecisionReduceBase<
          DTensorMixedPrecisionReducePass> {
  void runOnOperation() override {
    mlir::func::FuncOp function = getOperation();

    if (mlir::failed(TryMixedPrecisionReduce<mlir::TF::DTensorAllReduceOp>(
            function, "DTensorAllReduce")))
      return signalPassFailure();
    if (mlir::failed(TryMixedPrecisionReduce<mlir::TF::DTensorReduceScatterOp>(
            function, "DTensorReduceScatter")))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorMixedPrecisionReducePass() {
  return std::make_unique<DTensorMixedPrecisionReducePass>();
}

}  // namespace dtensor
}  // namespace tensorflow
