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

#include <cstdint>
#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_REWRITETPUEMBEDDINGOPSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// Rewrites RecvTPUEmbeddingActivationsOp and SendTPUEmbeddingGradients ops to
// internal variants by introducing XlaRecvTPUEmbeddingDeduplicationData op.
struct RewriteTPUEmbeddingOps
    : public impl::RewriteTPUEmbeddingOpsPassBase<RewriteTPUEmbeddingOps> {
  void runOnOperation() override;
};

// Rewrites the given op to `OpT` op after adding the given operand at the end.
template <typename OpT>
OpT AddOperandAndRewriteAs(Operation* op, Value operand, NamedAttrList attr,
                           OpBuilder* builder) {
  builder->setInsertionPoint(op);
  auto operands = llvm::to_vector<4>(op->getOperands());
  operands.push_back(operand);
  auto new_op = builder->create<OpT>(op->getLoc(), op->getResultTypes(),
                                     operands, attr.getAttrs());
  op->replaceAllUsesWith(new_op.getOperation()->getResults());
  op->erase();
  return new_op;
}

// Returns success if the function has at most one op of the template type and
// assigns it to `result`, if present. If there are multiple such ops, returns
// failure.
template <typename OpT>
LogicalResult GetOp(Region* region, OpT* result) {
  *result = {};
  for (auto op : region->getOps<OpT>()) {
    if (*result) return op.emitError("should be unique within a function");
    *result = op;
  }
  return success();
}

LogicalResult RunOnRegion(Region* region) {
  RecvTPUEmbeddingActivationsOp recv_op;
  if (failed(GetOp(region, &recv_op))) return failure();

  SendTPUEmbeddingGradientsOp send_op;
  if (failed(GetOp(region, &send_op))) return failure();

  // No TPU embedding ops.
  if (!recv_op && !send_op) return success();

  Location loc = recv_op ? recv_op.getLoc() : send_op.getLoc();
  StringRef config = recv_op ? recv_op.getConfig() : send_op.getConfig();

  // Create XlaRecvTPUEmbeddingDeduplicationData op.
  OpBuilder builder(region);
  auto output_ty =
      RankedTensorType::get({}, VariantType::get(region->getContext()));
  auto dedup_op = builder.create<XlaRecvTPUEmbeddingDeduplicationDataOp>(
      loc, output_ty, config);

  // Rewrite RecvTPUEmbeddingActivations op to the corresponding internal op.
  if (recv_op)
    AddOperandAndRewriteAs<XlaRecvTPUEmbeddingActivationsOp>(
        recv_op, dedup_op, recv_op->getAttrs(), &builder);

  // Rewrite SendTPUEmbeddingGradients op to the corresponding internal op and
  // then update the OperandSegmentSize attribute.
  if (send_op) {
    int32_t operand_sizes[] = {static_cast<int32_t>(send_op.getN()),
                               static_cast<int32_t>(send_op.getNN()), 1};
    auto operand_size_attr = builder.getDenseI32ArrayAttr(operand_sizes);
    NamedAttrList attrs(send_op->getAttrs());
    attrs.set(send_op.getOperandSegmentSizeAttr(), operand_size_attr);

    AddOperandAndRewriteAs<XlaSendTPUEmbeddingGradientsOp>(send_op, dedup_op,
                                                           attrs, &builder);
  }
  return success();
}

void RewriteTPUEmbeddingOps::runOnOperation() {
  func::FuncOp func = getOperation();
  if (failed(RunOnRegion(&func.getBody()))) return signalPassFailure();

  func.walk([&](Operation* op) {
    for (Region& region : op->getRegions()) {
      if (failed(RunOnRegion(&region))) return signalPassFailure();
    }
  });
}

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateRewriteTPUEmbeddingOpsPass() {
  return std::make_unique<RewriteTPUEmbeddingOps>();
}

}  // namespace TF
}  // namespace mlir
