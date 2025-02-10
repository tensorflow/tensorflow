/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <optional>
#include <queue>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Rewrite/PatternApplicator.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/string_util.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/legalization_op_config.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/core/lib/monitoring/gauge.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

namespace {

using mlir::Block;
using mlir::BoolAttr;
using mlir::Dialect;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::OperationName;
using mlir::OperationPass;
using mlir::Pattern;
using mlir::PatternApplicator;
using mlir::RewritePatternSet;
using mlir::StringAttr;
using mlir::TensorType;
using mlir::Type;
using mlir::Value;
using mlir::WalkResult;

auto* auto_outside_compilation_gauge =
    tensorflow::monitoring::Gauge<bool, 0>::New(
        "/tensorflow/core/use_auto_outside_compilation",
        "Tracks if auto outside compilation is enabled");

#define GEN_PASS_DEF_MARKOPSFOROUTSIDECOMPILATIONPASS
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h.inc"

struct MarkOpsForOutsideCompilation
    : public impl::MarkOpsForOutsideCompilationPassBase<
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
      OperationName(mlir::TF::BroadcastArgsOp::getOperationName(), context),
      OperationName(mlir::TF::BroadcastGradientArgsOp::getOperationName(),
                    context),
      OperationName(mlir::TF::ConcatOffsetOp::getOperationName(), context),
      OperationName(mlir::TF::EmptyOp::getOperationName(), context),
      OperationName(mlir::TF::ListDiffOp::getOperationName(), context),
      OperationName(mlir::TF::RankOp::getOperationName(), context),
      OperationName(mlir::TF::RangeOp::getOperationName(), context),
      OperationName(mlir::TF::ShapeOp::getOperationName(), context),
      OperationName(mlir::TF::ShapeNOp::getOperationName(), context),
      OperationName(mlir::TF::SizeOp::getOperationName(), context),
  };

  supported_ops->insert(allowlist_ops.begin(), allowlist_ops.end());
}

// Adds the list of ops that are only supported in the old bridge.
// TODO(b/168036682): Remove bounded dynamism ops now that MLIR bridge supports
// bounded dynamism.
// TODO(b/257574556): Remove the need for this manual list by making use of old
// bridge phase 2 op list.
void AddOldBridgeOnlyOps(MLIRContext* context,
                         llvm::DenseSet<OperationName>* supported_ops) {
  llvm::SmallDenseSet<OperationName, 8> allowlist_ops = {
      OperationName(mlir::TF::DynamicPartitionOp::getOperationName(), context),
      OperationName(mlir::TF::OutfeedEnqueueOp::getOperationName(), context),
      OperationName(mlir::TF::WhereOp::getOperationName(), context),
      OperationName(mlir::TF::UniqueOp::getOperationName(), context),
      OperationName(mlir::TF::XlaSetDynamicDimensionSizeOp::getOperationName(),
                    context),
      OperationName(mlir::TF::XlaSpmdFullToShardShapeOp::getOperationName(),
                    context),
      OperationName(mlir::TF::XlaSpmdShardToFullShapeOp::getOperationName(),
                    context),
  };

  supported_ops->insert(allowlist_ops.begin(), allowlist_ops.end());
}

// TODO(b/159128666): Check the control flow legalization passes instead once
// added.
void AddSupportedFunctionalOps(MLIRContext* context,
                               llvm::DenseSet<OperationName>* supported_ops) {
  supported_ops->insert(
      OperationName(mlir::TF::CaseRegionOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(mlir::TF::IfRegionOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(mlir::TF::InplaceAddOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(mlir::TF::WhileRegionOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(mlir::TF::XlaCallModuleOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(mlir::TF::XlaHostComputeOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(mlir::TF::XlaReduceOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(mlir::TF::XlaReduceWindowOp::getOperationName(), context));
  supported_ops->insert(OperationName(
      mlir::TF::XlaRngBitGeneratorOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(mlir::TF::XlaScatterOp::getOperationName(), context));
  supported_ops->insert(OperationName(
      mlir::TF::XlaSelectAndScatterOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(mlir::TF::SymbolicGradientOp::getOperationName(), context));
  supported_ops->insert(OperationName(
      mlir::TF::XlaVariadicReduceOp::getOperationName(), context));
  supported_ops->insert(OperationName(
      mlir::TF::XlaVariadicReduceV2Op::getOperationName(), context));
  supported_ops->insert(
      OperationName(mlir::TF::XlaVariadicSortOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(mlir::TF::XlaReplicaIdOp::getOperationName(), context));
  supported_ops->insert(
      OperationName(mlir::TF::YieldOp::getOperationName(), context));
}

// These embedding ops are rewritten when running TPUCompileOp.
void AddRewrittenEmbeddingOps(MLIRContext* context,
                              llvm::DenseSet<OperationName>* supported_ops) {
  supported_ops->insert(OperationName(
      mlir::TF::RecvTPUEmbeddingActivationsOp::getOperationName(), context));
  supported_ops->insert(OperationName(
      mlir::TF::SendTPUEmbeddingGradientsOp::getOperationName(), context));
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
      GET_OPERATION_NAME(mlir::TF::StackV2Op),
      GET_OPERATION_NAME(mlir::TF::StackPushV2Op),
      GET_OPERATION_NAME(mlir::TF::StackPopV2Op),
      // Tensor Array ops.
      GET_OPERATION_NAME(mlir::TF::TensorArrayV3Op),
      GET_OPERATION_NAME(mlir::TF::TensorArrayReadV3Op),
      GET_OPERATION_NAME(mlir::TF::TensorArrayWriteV3Op),
      GET_OPERATION_NAME(mlir::TF::TensorArrayConcatV3Op),
      GET_OPERATION_NAME(mlir::TF::TensorArraySplitV3Op),
      GET_OPERATION_NAME(mlir::TF::TensorArraySizeV3Op),
      GET_OPERATION_NAME(mlir::TF::TensorArrayGradV3Op),
      GET_OPERATION_NAME(mlir::TF::TensorArrayGatherV3Op),
      GET_OPERATION_NAME(mlir::TF::TensorArrayScatterV3Op),
      // Tensor List Ops.
      GET_OPERATION_NAME(mlir::TF::EmptyTensorListOp),
      GET_OPERATION_NAME(mlir::TF::TensorListReserveOp),
      GET_OPERATION_NAME(mlir::TF::TensorListFromTensorOp),
      GET_OPERATION_NAME(mlir::TF::TensorListPushBackOp),
      GET_OPERATION_NAME(mlir::TF::TensorListPopBackOp),
      GET_OPERATION_NAME(mlir::TF::TensorListGetItemOp),
      GET_OPERATION_NAME(mlir::TF::TensorListSetItemOp),
      GET_OPERATION_NAME(mlir::TF::TensorListLengthOp),
      GET_OPERATION_NAME(mlir::TF::TensorListElementShapeOp),
      GET_OPERATION_NAME(mlir::TF::TensorListGatherOp),
      GET_OPERATION_NAME(mlir::TF::TensorListScatterIntoExistingListOp),
      GET_OPERATION_NAME(mlir::TF::TensorListStackOp),
  };
#undef GET_OPERATION_NAME

  supported_ops->insert(allowlist_ops.begin(), allowlist_ops.end());
}

bool IsStringType(Type type) {
  if (mlir::isa<mlir::TF::StringType>(type)) return true;

  auto sub_type = mlir::dyn_cast<mlir::TF::TensorFlowTypeWithSubtype>(type);
  if (!sub_type) return false;

  bool has_string = llvm::any_of(sub_type.GetSubtypes(), [](TensorType type) {
    return mlir::isa<mlir::TF::StringType>(type.getElementType());
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
  if (op.getDialect() != tf_dialect) return true;
  // Assert has a legalization that later removes it so we don't want to outside
  // compile it ever for performance reasons.
  if (llvm::isa<mlir::TF::AssertOp>(op)) return true;

  if (HasStringOperand(op)) return false;
  if (HasStringResult(op)) return false;
  if (MatchesPattern(op, supported_ops)) return true;

  auto abstractOp = op.getRegisteredInfo();
  if (!abstractOp) return false;
  return mlir::mhlo::HasTf2XlaFallback(abstractOp->getTypeID());
}

bool IsVariant(Value value) {
  return mlir::isa<mlir::TF::VariantType>(
      getElementTypeOrSelf(value.getType()));
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
void MarkVariantInputsOutputs(mlir::tf_device::ClusterOp tpu_cluster) {
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
            if (!user->hasTrait<mlir::OpTrait::IsTerminator>() &&
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
  return mlir::success();
}

// Check for uncompilable ops that are in `tf_dialect` and are not already
// marked for outside compilation.
bool ContainsUncompilableOps(const Dialect* tf_dialect, Block* block,
                             llvm::DenseSet<OperationName>& supported_ops) {
  int uncompilable_op_count = 0;
  // Check if op or any parent is already marked for outside compilation.
  block->walk([&](Operation* op) {
    Operation* iter_op = op;
    while (iter_op && !llvm::isa<mlir::tf_device::ClusterOp>(iter_op)) {
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
void UnmarkChildren(ModuleOp module) {
  module->walk([&](Operation* op) {
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

constexpr int kTooManyOutsideCompileRegionThreshold = 32;
constexpr int kOpDetailCount = 8;

void WarnOnExcessOutsideCompilationOps(ModuleOp module) {
  // Count the number of outside compilation ops. If it exceeds the reporting
  // threshold, warn the user that their model may run slowly.
  llvm::SmallVector<Operation*, 8> outside_compile_ops;
  module->walk([&](Operation* op) {
    if (op->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr)) {
      outside_compile_ops.push_back(op);
    }
  });

  if (outside_compile_ops.size() > kTooManyOutsideCompileRegionThreshold) {
    llvm::SmallVector<std::string, kOpDetailCount> op_info;
    for (int i = 0; i < kOpDetailCount; ++i) {
      auto& op = outside_compile_ops[i];
      op_info.push_back(tensorflow::OpAsString(*op));
    }

    LOG(WARNING) << outside_compile_ops.size() << " outside compilation "
                 << "regions found while processing "
                 << module->getName().getStringRef().str()
                 << ". This may result in excessively slow model execution. "
                 << "First " << op_info.size()
                 << " ops: " << absl::StrJoin(op_info, "\n");
  } else {
    LOG(INFO) << "Found " << outside_compile_ops.size()
              << " outside compilation regions.";
  }
}

void MarkOpsForOutsideCompilation::runOnOperation() {
  auto module = getOperation();
  const Dialect* tf_dialect = getContext().getLoadedDialect("tf");
  if (!tf_dialect) {
    getOperation().emitError() << "'tf' dialect is not registered";
    return signalPassFailure();
  }
  RewritePatternSet patterns(&getContext());
  mlir::mhlo::PopulateLegalizeTfPatterns(module.getContext(), &patterns);
  mlir::TF::PopulateTFLoweringBeforeHLOPatterns(module.getContext(), &patterns);
  mlir::TF::PopulateLoweringQuantizedPatterns(module.getContext(), &patterns);
  AddCanonicalizationPatterns(module.getContext(), &patterns);

  // `supported_ops` contains the name of all of the ops that can potentially be
  // lowered into HLO on the device. This doesn't always mean that the op can
  // be lowered in the future passes but if the op is not in this set, it can't
  // be lowered in a subsequent pass.
  llvm::DenseSet<OperationName> supported_ops;
  PatternApplicator(std::move(patterns))
      .walkAllPatterns([&](const Pattern& pattern) {
        std::optional<OperationName> root_kind = pattern.getRootKind();
        if (root_kind.has_value()) supported_ops.insert(root_kind.value());
      });
  AddSupportedFunctionalOps(module.getContext(), &supported_ops);
  AddSupportedOpsUsingFolding(module.getContext(), &supported_ops);
  AddOldBridgeOnlyOps(module.getContext(), &supported_ops);
  AddRewrittenEmbeddingOps(module.getContext(), &supported_ops);
  AddRewrittenCompositeOps(module.getContext(), &supported_ops);

  auto result = module.walk([&](mlir::tf_device::ClusterOp cluster) {
    // Only if `allow_soft_placement` attribute is true should we mark ops
    // for outside compilation.
    auto soft_placement_attr =
        cluster->getAttrOfType<BoolAttr>(mlir::TF::kAllowSoftPlacementAttr);
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

  UnmarkChildren(module);

  WarnOnExcessOutsideCompilationOps(module);
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateMarkOpsForOutsideCompilationPass() {
  return std::make_unique<MarkOpsForOutsideCompilation>();
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
