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
#include <queue>
#include <string>
#include <utility>

#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Rewrite/PatternApplicator.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
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

struct MarkOpsForOutsideCompilation
    : public TF::MarkOpsForOutsideCompilationPassBase<
          MarkOpsForOutsideCompilation> {
  void runOnOperation() override;
};

// Adds any canonicalization patterns to list of supported `patterns`.
// TODO(b/161726307): Move or import the relevant patterns to LowerTF pass and
// remove this.
void AddCanonicalizationPatterns(MLIRContext* context,
                                 RewritePatternSet* patterns) {
  for (auto op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(*patterns, context);
}

// Adds the list of ops that are supported on TPU through constant folding which
// may depend on the inputs shapes not known at this point. Such ops may not
// have any legalization or canonicalization patterns but shouldn't be marked
// for outside compilation.
//
// TODO(b/177523289): Remove manual handling once we support constant folding
// and shape inference through the computation on the host side.
void AddSupportedOpsUsingFolding(MLIRContext* context,
                                 llvm::DenseSet<OperationName>* supported_ops) {
  llvm::SmallDenseSet<OperationName, 8> allowlist_ops = {
      OperationName(TF::BroadcastArgsOp::getOperationName(), context),
      OperationName(TF::BroadcastGradientArgsOp::getOperationName(), context),
      OperationName(TF::ConcatOffsetOp::getOperationName(), context),
      OperationName(TF::EmptyOp::getOperationName(), context),
      OperationName(TF::ListDiffOp::getOperationName(), context),
      OperationName(TF::RankOp::getOperationName(), context),
      OperationName(TF::RangeOp::getOperationName(), context),
      OperationName(TF::ShapeOp::getOperationName(), context),
      OperationName(TF::ShapeNOp::getOperationName(), context),
      OperationName(TF::SizeOp::getOperationName(), context),
  };

  supported_ops->insert(allowlist_ops.begin(), allowlist_ops.end());
}

// Adds the list of ops that are supported through dynamic padder using op by op
// fallback to the TF2XLA bridge.
// TODO(b/168036682): Remove this once ops are supported using dynamic padder
// on MLIR bridge.
void AddSupportedOpsUsingDynamicPadder(
    MLIRContext* context, llvm::DenseSet<OperationName>* supported_ops) {
  llvm::SmallDenseSet<OperationName, 8> allowlist_ops = {
      OperationName(TF::WhereOp::getOperationName(), context),
      OperationName(TF::UniqueOp::getOperationName(), context),
      OperationName(TF::XlaSetDynamicDimensionSizeOp::getOperationName(),
                    context),
  };

  supported_ops->insert(allowlist_ops.begin(), allowlist_ops.end());
}

// TODO(b/159128666): Check the control flow legalization passes instead once
// added.
void AddSupportedFunctionalOps(MLIRContext* context,
                               llvm::DenseSet<OperationName>* supported_ops) {
  supported_ops->insert(
      OperationName(TF::CaseRegionOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::IfRegionOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::InplaceAddOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::WhileRegionOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaReduceOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaReduceWindowOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaRngBitGeneratorOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaScatterOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaSelectAndScatterOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::SymbolicGradientOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaVariadicReduceOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaVariadicReduceV2Op::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaVariadicSortOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(TF::XlaReplicaIdOp::getOperationName(), context));
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

bool IsVariant(Value value) {
  return getElementTypeOrSelf(value.getType()).isa<TF::VariantType>();
}

bool HasOutsideCompiledAncestor(Operation* op) {
  Operation* parent = op->getParentOp();
  while (parent) {
    if (parent->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr))
      return true;
    parent = parent->getParentOp();
  }
  return false;
}

// If any tf.variants are inputs/outputs to the another outside compiled
// Operation, `op`, mark  them for outside compilation unless they are already
// marks with outside compilation attribute.
void MarkVariantInputsOutputs(tf_device::ClusterOp tpu_cluster) {
  std::queue<Operation*> outside_compiled_ops;
  tpu_cluster.walk([&](Operation* op) {
    if (op->hasAttrOfType<StringAttr>(kXlaOutsideCompilationAttr))
      outside_compiled_ops.push(op);
  });

  while (!outside_compiled_ops.empty()) {
    Operation* host_op = outside_compiled_ops.front();
    outside_compiled_ops.pop();
    host_op->walk([&](Operation* op) {
      // Add any operations that provide variant inputs to the cluster.
      for (auto value : op->getOperands()) {
        Operation* input_defining_op = value.getDefiningOp();
        if (IsVariant(value) && input_defining_op &&
            !HasOutsideCompiledAncestor(input_defining_op) &&
            !input_defining_op->hasAttrOfType<StringAttr>(
                kXlaOutsideCompilationAttr)) {
          input_defining_op->setAttr(
              kXlaOutsideCompilationAttr,
              StringAttr::get(input_defining_op->getContext(), "auto"));
          outside_compiled_ops.push(input_defining_op);
        }
      }
      // Mark for outside compilation any operations that consume variant
      // outputs from an outside compiled operation.
      for (auto value : op->getResults()) {
        if (IsVariant(value)) {
          for (auto user : value.getUsers()) {
            if (!user->hasTrait<OpTrait::IsTerminator>() &&
                !HasOutsideCompiledAncestor(user) &&
                !user->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) {
              user->setAttr(kXlaOutsideCompilationAttr,
                            StringAttr::get(user->getContext(), "auto"));
              outside_compiled_ops.push(user);
            }
          }
        }
      }
    });
  }
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
      VLOG(3) << "Cloud TPU: Op " << op->getName().getStringRef().str()
              << " isn't compilable, adding outside_compilation attr. "
                 "This op will automatically be placed on CPU.";
      op->setAttr(kXlaOutsideCompilationAttr,
                  StringAttr::get(
                      op->getContext(),
                      llvm::formatv("auto{0}", outside_compiled_cluster_counter)
                          .str()));
      outside_compiled_cluster_counter++;
    }
  });
  if (outside_compiled_cluster_counter > 0) {
    auto_outside_compilation_gauge->GetCell()->Set(true);
  }
  return success();
}

// Check for uncompilable ops that are in `tf_dialect` and are not already
// marked for outside compilation.
bool ContainsUncompilableOps(const Dialect* tf_dialect, Block* block,
                             llvm::DenseSet<OperationName>& supported_ops) {
  int uncompilable_op_count = 0;
  // Check if op or any parent is already marked for outside compilation.
  block->walk([&](Operation* op) {
    Operation* iter_op = op;
    while (iter_op && !llvm::isa<tf_device::ClusterOp>(iter_op)) {
      if (iter_op->hasAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) {
        return;
      }
      iter_op = iter_op->getParentOp();
    }

    if (!IsSupportedOp(*op, supported_ops, tf_dialect)) {
      op->emitOpError() << "isn't compilable for TPU device. enable "
                           "soft_device_placement option to run on CPU";
      ++uncompilable_op_count;
    }
  });
  return uncompilable_op_count > 0;
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
  RewritePatternSet patterns(&getContext());
  mhlo::PopulateLegalizeTfPatterns(module.getContext(), &patterns);
  TF::PopulateTFLoweringBeforeHLOPatterns(module.getContext(), &patterns);
  TF::PopulateLoweringQuantizedPatterns(module.getContext(), &patterns);
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
  AddSupportedFunctionalOps(module.getContext(), &supported_ops);
  AddSupportedOpsUsingFolding(module.getContext(), &supported_ops);
  AddSupportedOpsUsingDynamicPadder(module.getContext(), &supported_ops);
  AddRewrittenEmbeddingOps(module.getContext(), &supported_ops);
  AddRewrittenCompositeOps(module.getContext(), &supported_ops);

  auto result = module.walk([&](tf_device::ClusterOp cluster) {
    // Only if `allow_soft_placement` attribute is true should we mark ops
    // for outside compilation.
    auto soft_placement_attr =
        cluster->getAttrOfType<BoolAttr>(kAllowSoftPlacementAttr);
    if ((soft_placement_attr && soft_placement_attr.getValue())) {
      if (failed(MarkUncompilableOps(tf_dialect, &cluster.GetBody(),
                                     supported_ops)))
        return WalkResult::interrupt();
    } else {
      if (ContainsUncompilableOps(tf_dialect, &cluster.GetBody(),
                                  supported_ops))
        return WalkResult::interrupt();
    }
    MarkVariantInputsOutputs(cluster);

    return WalkResult::advance();
  });

  if (result.wasInterrupted()) return signalPassFailure();

  module.walk([&](tf_device::ClusterOp cluster) {
    // Only if `allow_soft_placement` attribute is true should we unmark ops
    // for outside compilation.
    auto soft_placement_attr =
        cluster->getAttrOfType<BoolAttr>(kAllowSoftPlacementAttr);
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

}  // namespace TFDevice
}  // namespace mlir
