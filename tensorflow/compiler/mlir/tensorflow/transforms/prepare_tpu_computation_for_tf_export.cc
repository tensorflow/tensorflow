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

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_remaining_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"

namespace mlir {
namespace TF {
namespace {

// Returns true if the given op is TF/XLA communication op in the old bridge.
bool IsCommunicationOp(Operation* op) {
  return isa<TF::XlaHostComputeOp, TF::XlaSendToHostOp, TF::XlaRecvFromHostOp>(
      op);
}

// Returns true if the given op is one of ops supported to have communication
// subcomputation in the TF/XLA bridge.
bool SupportsCommunicationComputation(Operation* op) {
  return isa<TF::IfRegionOp, TF::WhileRegionOp, TF::CaseRegionOp,
             TF::StatefulPartitionedCallOp, TF::PartitionedCallOp,
             TF::LegacyCallOp>(op);
}

class PrepareTpuComputationForTfExportPass
    : public PrepareTpuComputationForTfExportPassBase<
          PrepareTpuComputationForTfExportPass> {
  void runOnOperation() override;
};

class RewriteXlaHostComputeMlir
    : public OpRewritePattern<TF::_XlaHostComputeMlirOp> {
 public:
  using OpRewritePattern<TF::_XlaHostComputeMlirOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::_XlaHostComputeMlirOp op,
                                PatternRewriter& rewriter) const override {
    llvm::SmallVector<Attribute> shape_attrs;
    shape_attrs.reserve(op.getNumResults());
    for (Type ty : op.getResultTypes()) {
      shape_attrs.push_back(
          TF::ShapeAttr::get(rewriter.getContext(), ty.cast<ShapedType>()));
    }

    // Clone the `host_func` in the `host_mlir_module` attribute if it exists
    // and use it for `shape_inference_graph` attribute on XlaHostCompute.
    FuncOp cloned_func;
    SymbolTable manager(op->getParentOfType<ModuleOp>());
    StringRef host_module = op.host_mlir_module();
    if (!host_module.empty()) {
      mlir::OwningModuleRef module_for_func;

      FuncOp func = op.GetHostFunc(&module_for_func);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(op->getParentOfType<FuncOp>());
      cloned_func =
          llvm::dyn_cast_or_null<FuncOp>(rewriter.clone(*func.getOperation()));
      manager.insert(cloned_func);
      rewriter.setInsertionPointToStart(&cloned_func.body().front());
      auto result_type =
          RankedTensorType::get({3}, rewriter.getType<TF::StringType>());
      auto dynamic_key =
          rewriter.create<TF::_TPUCompileMlirPlaceholderProgramKeyOp>(
              func.getLoc(), /*program=*/result_type, llvm::ArrayRef<Value>{});

      auto recv_at_host = rewriter.create<TF::_XlaRecvAtHostOp>(
          func.getLoc(), op.getOperandTypes(), /*dynamic_key=*/dynamic_key,
          op.send_keyAttr(),
          /*device_ordinal=*/rewriter.getI64IntegerAttr(0));
      for (auto result :
           llvm::zip(cloned_func.getArguments(), recv_at_host->getResults())) {
        std::get<0>(result).replaceAllUsesWith(std::get<1>(result));
      }

      rewriter.setInsertionPoint(cloned_func.body().front().getTerminator());
      rewriter.create<TF::_XlaSendFromHostOp>(
          func.getLoc(),
          cloned_func.body().front().getTerminator()->getOperands(),
          /*dynamic_key=*/dynamic_key, op.recv_keyAttr(),
          /*device_ordinal=*/rewriter.getI64IntegerAttr(0));
    }

    constexpr int64_t kDefaultCostEstimate = 1000000;
    rewriter.replaceOpWithNewOp<TF::XlaHostComputeOp>(
        op, op.getResultTypes(), op.inputs(),
        /*ancestors=*/rewriter.getArrayAttr({}),
        rewriter.getArrayAttr(shape_attrs),
        /*shape_inference_graph=*/
        cloned_func ? rewriter.getSymbolRefAttr(cloned_func) : SymbolRefAttr(),
        /*key=*/rewriter.getStringAttr(""), op.send_keyAttr(),
        op.recv_keyAttr(),
        /*cost_estimate_ns=*/rewriter.getI64IntegerAttr(kDefaultCostEstimate),
        op.tpu_coreAttr());
    return success();
  }
};

void UpdateArgAttributes(mlir::FuncOp func) {
  OpBuilder builder(func.getBody());
  for (int i = 0; i < func.getNumArguments(); ++i) {
    constexpr char kShardingAttr[] = "mhlo.sharding";
    if (auto sharding =
            func.getArgAttrOfType<mlir::StringAttr>(i, kShardingAttr)) {
      if (!sharding.getValue().empty()) {
        BlockArgument arg = func.getArgument(i);
        // TODO(hinsu): Instead of setting both 'sharding' and '_XlaSharding'
        // attributes, only set the 'sharding' attribute. Both attributes are
        // currently required as the XlaSharding xla op kernel doesn't use the
        // 'sharding' attribute.
        auto updated_arg = builder.create<TF::XlaShardingOp>(
            func.getLoc(), arg.getType(), arg, sharding, sharding);
        func.getArgument(i).replaceAllUsesExcept(
            updated_arg, llvm::SmallPtrSet<Operation*, 1>({updated_arg}));
      }

      func.removeArgAttr(i, builder.getIdentifier(kShardingAttr));
    }
  }
}

LogicalResult RewriteCommunicationOps(ModuleOp module) {
  MLIRContext* ctx = module.getContext();
  mlir::OwningRewritePatternList patterns(ctx);
  patterns.insert<RewriteXlaHostComputeMlir>(ctx);
  if (failed(mlir::applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
    return module.emitError("failed to apply tf export preparation patterns");
  }

  // TODO(hinsu): Investigate if the semantics of keys for these communication
  // ops between the old bridge and new bridge can be reconciled.
  module.walk([&](Operation* op) {
    if (isa<TF::XlaSendToHostOp>(op)) {
      StringRef old_key = op->getAttrOfType<StringAttr>("key").getValue();
      auto new_key = StringAttr::get(ctx, old_key.str() + "_dtoh_0");
      op->setAttr("key", new_key);
    } else if (isa<TF::XlaRecvFromHostOp>(op)) {
      StringRef old_key = op->getAttrOfType<StringAttr>("key").getValue();
      auto new_key = StringAttr::get(ctx, old_key.str() + "_htod_0");
      op->setAttr("key", new_key);
    }
  });
  return success();
}

// Sets token input node names attribute and their corresponding original node
// names for tf/xla communication related ops. These attributes are used to
// order operations on device. First op in the region should have a special
// argument token and then remaining operations should have node name of the
// previous communication ops.
LogicalResult SetTokenInputAttrs(ModuleOp module) {
  // Collect all the ops that needs to have token input names attributes. These
  // ops are communication ops and all their parent ops via nesting or function
  // calls. For example, IfRegion op and PartitionedCall op.
  std::vector<Operation*> worklist;
  absl::flat_hash_set<Operation*> ops_with_tokens;
  module.walk([&](Operation* op) {
    if (IsCommunicationOp(op)) {
      ops_with_tokens.insert(op);
      worklist.push_back(op);
    }
  });

  SymbolTableCollection table;
  SymbolUserMap symbol_map(table, module);

  // Regions that contains ops requiring token input attributes.
  absl::flat_hash_set<Region*> regions_with_token;
  while (!worklist.empty()) {
    Operation* op = worklist.back();
    worklist.pop_back();

    Region* region = op->getParentRegion();
    regions_with_token.insert(region);

    // If the parent is not a FuncOp, then add the parent op containing a region
    // to worklist.
    Operation* parent = region->getParentOp();
    if (!isa<FuncOp>(parent)) {
      if (ops_with_tokens.insert(parent).second) {
        worklist.push_back(parent);
      }
      continue;
    }

    // For functions, get all the users and add them to the worklist.
    for (auto& user : symbol_map.getUsers(parent)) {
      if (ops_with_tokens.insert(user).second) {
        worklist.push_back(user);
      }
    }
  }

  // Use name mapper to uniquely name all ops in the module as export to
  // TensorFlow graph may change node names. These op names here doesn't need to
  // match the actual names in the graph as this sets original node name
  // attribute for all the relevant nodes.
  tensorflow::OpOrArgLocNameMapper name_mapper;
  MLIRContext* ctx = module.getContext();
  for (Region* region : regions_with_token) {
    // Initialize the token with the special argument token. This gets mapped to
    // input token in the parent op or a new token for the entry computation.
    auto token = StringAttr::get(ctx, tensorflow::kXlaTokenArgNodeName);
    for (Operation& op : region->getOps()) {
      // Only communication related ops that needs to have token should have the
      // extra attribute.
      if (!ops_with_tokens.contains(&op)) continue;

      if (!IsCommunicationOp(&op) && !SupportsCommunicationComputation(&op)) {
        return op.emitOpError(
            "does not support subcomputations with tf/xla communication ops");
      }

      op.setAttr(tensorflow::kXlaTokenInputNodesAttrName,
                 ArrayAttr::get(ctx, {token}));

      auto node_name = StringAttr::get(ctx, name_mapper.GetUniqueName(&op));
      op.setAttr(tensorflow::kXlaOriginalOutsideCompilationNodeName, node_name);
      token = node_name;
    }
  }
  return success();
}

void PrepareTpuComputationForTfExportPass::runOnOperation() {
  ModuleOp module = getOperation();

  for (FuncOp func : module.getOps<FuncOp>()) {
    UpdateArgAttributes(func);
  }

  // First rewrite communication ops used in the new bridge to match old bridge
  // semantics and then set token input node names attributes on the supported
  // ops.
  if (failed(RewriteCommunicationOps(module)) ||
      failed(SetTokenInputAttrs(module))) {
    signalPassFailure();
    return;
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreatePrepareTpuComputationForTfExportPass() {
  return std::make_unique<PrepareTpuComputationForTfExportPass>();
}

}  // namespace TF
}  // namespace mlir
