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

#include <memory>
#include <string>
#include <utility>

#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Rewrite/PatternApplicator.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/core/lib/monitoring/gauge.h"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";
constexpr char kAllowSoftPlacementAttr[] = "allow_soft_placement";

auto* auto_outside_compilation_gauge =
    tensorflow::monitoring::Gauge<bool, 0>::New(
        "/tensorflow/core/use_auto_outside_compilation",
        "Tracks if auto outside compilation is enabled");

// This pass marks unsupported ops in a device cluster with
// `_xla_outside_compilation` attribute so the operations will run on the host
// instead of the device.  Unsupported ops are ops that can not be code
// generated to run on the device for the cluster.
struct MarkOpsForOutsideCompilation
    : public PassWrapper<MarkOpsForOutsideCompilation,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

// Adds any canonicalization patterns to list of supported `patterns`.
// TODO(b/161726307): Move or import the relevant patterns to LowerTF pass and
// remove this.
void AddCanonicalizationPatterns(MLIRContext* context,
                                 OwningRewritePatternList* patterns) {
  for (auto* op : context->getRegisteredOperations())
    op->getCanonicalizationPatterns(*patterns, context);
}

// TODO(b/159128666): Check the control flow legalization passes instead once
// added.
void AddSupportedControlFlowOps(MLIRContext* context,
                                llvm::DenseSet<OperationName>* supported_ops) {
  supported_ops->insert(
      OperationName(TF::IfRegionOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::WhileRegionOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::YieldOp::getOperationName(), context));
}

// These embedding ops are rewritten when running TPUCompileOp.
void AddRewrittenEmbeddingOps(MLIRContext* context,
                              llvm::DenseSet<OperationName>* supported_ops) {
  supported_ops->insert(OperationName(
      TF::RecvTPUEmbeddingActivationsOp::getOperationName(), context));
  supported_ops->insert(OperationName(
      TF::SendTPUEmbeddingGradientsOp::getOperationName(), context));
}

// Stack, TensorList and TensorArray ops are rewritten during the second phase
// of the bridge (compilation of TPUCompile op). They would not match any
// legalization/canonicalization pattern and have to be manually added to the
// list of supported ops.
void AddRewrittenCompositeOps(MLIRContext* context,
                              llvm::DenseSet<OperationName>* supported_ops) {
#define GET_OPERATION_NAME(op) OperationName(op::getOperationName(), context)
  llvm::SmallDenseSet<OperationName, 32> allowlist_ops = {
      // Stack ops.
      GET_OPERATION_NAME(TF::StackV2Op),
      GET_OPERATION_NAME(TF::StackPushV2Op),
      GET_OPERATION_NAME(TF::StackPopV2Op),
      // Tensor Array ops.
      GET_OPERATION_NAME(TF::TensorArrayV3Op),
      GET_OPERATION_NAME(TF::TensorArrayReadV3Op),
      GET_OPERATION_NAME(TF::TensorArrayWriteV3Op),
      GET_OPERATION_NAME(TF::TensorArrayConcatV3Op),
      GET_OPERATION_NAME(TF::TensorArraySplitV3Op),
      GET_OPERATION_NAME(TF::TensorArraySizeV3Op),
      GET_OPERATION_NAME(TF::TensorArrayGradV3Op),
      GET_OPERATION_NAME(TF::TensorArrayGatherV3Op),
      GET_OPERATION_NAME(TF::TensorArrayScatterV3Op),
      // Tensor List Ops.
      GET_OPERATION_NAME(TF::EmptyTensorListOp),
      GET_OPERATION_NAME(TF::TensorListReserveOp),
      GET_OPERATION_NAME(TF::TensorListFromTensorOp),
      GET_OPERATION_NAME(TF::TensorListPushBackOp),
      GET_OPERATION_NAME(TF::TensorListPopBackOp),
      GET_OPERATION_NAME(TF::TensorListGetItemOp),
      GET_OPERATION_NAME(TF::TensorListSetItemOp),
      GET_OPERATION_NAME(TF::TensorListLengthOp),
      GET_OPERATION_NAME(TF::TensorListElementShapeOp),
      GET_OPERATION_NAME(TF::TensorListGatherOp),
      GET_OPERATION_NAME(TF::TensorListScatterIntoExistingListOp),
      GET_OPERATION_NAME(TF::TensorListStackOp),
  };
#undef GET_OPERATION_NAME

  supported_ops->insert(allowlist_ops.begin(), allowlist_ops.end());
}

bool IsStringType(Type type) {
  if (type.isa<TF::StringType>()) return true;

  auto sub_type = type.dyn_cast<TF::TensorFlowTypeWithSubtype>();
  if (!sub_type) return false;

  bool has_string = llvm::any_of(sub_type.GetSubtypes(), [](TensorType type) {
    return type.getElementType().isa<TF::StringType>();
  });
  return has_string;
}

bool HasStringOperand(Operation& op) {
  for (auto operand : op.getOperands()) {
    auto operand_type = getElementTypeOrSelf(operand);
    if (IsStringType(operand_type)) return true;
  }
  return false;
}

bool HasStringResult(Operation& op) {
  for (auto result : op.getResults()) {
    auto result_type = getElementTypeOrSelf(result);
    if (IsStringType(result_type)) return true;
  }
  return false;
}

bool MatchesPattern(Operation& op,
                    const llvm::DenseSet<OperationName>& supported_ops) {
  return (supported_ops.contains(op.getName()));
}

// Checks if the op is supported inside of a device cluster.  Ops not
// in `tf_dialect` are considered supported.
bool IsSupportedOp(Operation& op,
                   const llvm::DenseSet<OperationName>& supported_ops,
                   const Dialect* tf_dialect) {
  if (op.getDialect() != tf_dialect)
    return true;
  // Assert has a legalization that later removes it so we don't want to outside
  // compile it ever for performance reasons.
  if (llvm::isa<TF::AssertOp>(op)) return true;
  return !HasStringOperand(op) && !HasStringResult(op) &&
         (MatchesPattern(op, supported_ops) ||
          mhlo::IsOpAllowedTf2XlaFallback(&op));
}

// Checks all regions of `op` for captured string operands.
bool HasCapturedStringOperand(Operation* op) {
  bool string_operand = false;
  for (auto& region : op->getRegions()) {
    mlir::visitUsedValuesDefinedAbove(
        region, region, [&](mlir::OpOperand* operand) {
          if (getElementTypeOrSelf(operand->get()).isa<TF::StringType>())
            string_operand = true;
        });
    if (string_operand) return string_operand;
  }
  return string_operand;
}

// Marks uncompilable ops that are in `tf_dialect` for outside compilation.
LogicalResult MarkUncompilableOps(
    const Dialect* tf_dialect, Block* block,
    llvm::DenseSet<OperationName>& supported_ops) {
  // Automatically marked ops for outside compilation have
  // `_xla_outside_compilation` attribute value of "auto" plus
  // an increasing counter.  Manually marked ops for outside compilation only
  // have an increasing counteri for the attribute value.  Therefore there is no
  // collision in
  // `_xla_outside_compilation` attribute between automatically and manually
  // marking ops.
  int outside_compiled_cluster_counter = 0;
  block->walk([&](Operation* op) {
    if (!IsSupportedOp(*op, supported_ops, tf_dialect)) {
      op->setAttr(
          kXlaOutsideCompilationAttr,
          StringAttr::get(
              llvm::formatv("auto{0}", outside_compiled_cluster_counter).str(),
              op->getContext()));
      outside_compiled_cluster_counter++;
    }
  });
  if (outside_compiled_cluster_counter > 0) {
    auto_outside_compilation_gauge->GetCell()->Set(true);
  }
  return success();
}

// Unmarks outside compilation for any op that has parents already
// marked for outside compilation since the child will be extracted
// anyways.
void UnmarkChildren(Block* block) {
  block->walk([&](Operation* op) {
    if (!op->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) return;
    Operation* iter_op = op;
    bool remove_attr = false;
    while (auto* parent_op = iter_op->getParentOp()) {
      if (parent_op->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) {
        remove_attr = true;
        break;
      }
      iter_op = parent_op;
    }
    if (remove_attr) op->removeAttr(kXlaOutsideCompilationAttr);
  });
}

void MarkOpsForOutsideCompilation::runOnOperation() {
  auto module = getOperation();
  const Dialect* tf_dialect = getContext().getLoadedDialect("tf");
  if (!tf_dialect) {
    getOperation().emitError() << "'tf' dialect is not registered";
    return signalPassFailure();
  }
  OwningRewritePatternList patterns;
  mhlo::PopulateLegalizeTfPatterns(module.getContext(), &patterns);
  TF::PopulateLoweringTFPatterns(module.getContext(), &patterns);
  AddCanonicalizationPatterns(module.getContext(), &patterns);

  // `supported_ops` contains the name of all of the ops that can potentially be
  // lowered into HLO on the device. This doesn't always mean that the op can
  // be lowered in the future passes but if the op is not in this set, it can't
  // be lowered in a subsequent pass.
  llvm::DenseSet<OperationName> supported_ops;
  PatternApplicator(std::move(patterns))
      .walkAllPatterns([&](const Pattern& pattern) {
        Optional<OperationName> root_kind = pattern.getRootKind();
        if (root_kind.hasValue()) supported_ops.insert(root_kind.getValue());
      });
  AddSupportedControlFlowOps(module.getContext(), &supported_ops);
  AddRewrittenEmbeddingOps(module.getContext(), &supported_ops);
  AddRewrittenCompositeOps(module.getContext(), &supported_ops);

  auto result = module.walk([&](tf_device::ClusterOp cluster) {
    // Only if `allow_soft_placement` attribute is true should we mark ops
    // for outside compilation.
    auto soft_placement_attr =
        cluster.getAttrOfType<BoolAttr>(kAllowSoftPlacementAttr);
    if (!(soft_placement_attr && soft_placement_attr.getValue())) {
      return WalkResult::advance();
    }
    if (failed(
            MarkUncompilableOps(tf_dialect, &cluster.GetBody(), supported_ops)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (result.wasInterrupted()) return signalPassFailure();

  module.walk([&](tf_device::ClusterOp cluster) {
    // Only if `allow_soft_placement` attribute is true should we unmark ops
    // for outside compilation.
    auto soft_placement_attr =
        cluster.getAttrOfType<BoolAttr>(kAllowSoftPlacementAttr);
    if (!(soft_placement_attr && soft_placement_attr.getValue())) {
      return;
    }
    UnmarkChildren(&cluster.GetBody());
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateMarkOpsForOutsideCompilationPass() {
  return std::make_unique<MarkOpsForOutsideCompilation>();
}

static PassRegistration<MarkOpsForOutsideCompilation> pass(
    "tf-mark-ops-for-outside-compilation",
    "Marks unsupported ops a device cluster for outside compilation.");

}  // namespace TFDevice
}  // namespace mlir
