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

#include <optional>
#include <utility>

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_LEGALIZEHASHTABLESPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// This file has Legalize hash tables pass which is responsible for:
// - Converting static hash table ops to the TFLite equivalent ops.
//
// There are needs to fall back to Flex for the following cases:
// - Mutable hash table cases
// - Other resource operators consuming a hash table resource tensor

class LegalizeHashTableOpPattern : public OpRewritePattern<TF::HashTableV2Op> {
 public:
  using OpRewritePattern<TF::HashTableV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::HashTableV2Op hashtable_op,
                                PatternRewriter& rewriter) const override {
    auto output_type = RankedTensorType::get(
        {1}, TF::ResourceType::get(rewriter.getContext()));

    // Hash the shared name to generate integer hash table id. The TFLite
    // native resource design is based on integer keys to identify the
    // corresponding resource objects.
    auto table_id =
        static_cast<int32_t>(::llvm::hash_value(hashtable_op.getSharedName()));
    auto key_dtype = hashtable_op.getKeyDtype();
    auto value_dtype = hashtable_op.getValueDtype();

    rewriter.replaceOpWithNewOp<TFL::HashtableOp>(
        hashtable_op, output_type, table_id, key_dtype, value_dtype);
    return success();
  }
};

class LegalizeHashTableFindOpPattern
    : public OpRewritePattern<TF::LookupTableFindV2Op> {
 public:
  using OpRewritePattern<TF::LookupTableFindV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::LookupTableFindV2Op find_op,
                                PatternRewriter& rewriter) const override {
    auto handle_op = find_op.getTableHandle().getDefiningOp();
    if (handle_op == nullptr) return failure();
    auto hashtable_op = llvm::dyn_cast<TFL::HashtableOp>(handle_op);
    if (hashtable_op == nullptr) return failure();
    rewriter.replaceOpWithNewOp<TFL::HashtableFindOp>(
        find_op, find_op->getResultTypes(), find_op.getTableHandle(),
        find_op.getKeys(), find_op.getDefaultValue());
    return success();
  }
};

class LegalizeHashTableImportOpPattern
    : public OpRewritePattern<TF::LookupTableImportV2Op> {
 public:
  using OpRewritePattern<TF::LookupTableImportV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::LookupTableImportV2Op import_op,
                                PatternRewriter& rewriter) const override {
    auto handle_op = import_op.getTableHandle().getDefiningOp();
    if (handle_op == nullptr) return failure();
    auto hashtable_op = llvm::dyn_cast<TFL::HashtableOp>(handle_op);
    if (hashtable_op == nullptr) return failure();
    rewriter.replaceOpWithNewOp<TFL::HashtableImportOp>(
        import_op, import_op->getResultTypes(), import_op.getTableHandle(),
        import_op.getKeys(), import_op.getValues());
    return success();
  }
};

class LegalizeHashTableSizeOpPattern
    : public OpRewritePattern<TF::LookupTableSizeV2Op> {
 public:
  using OpRewritePattern<TF::LookupTableSizeV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::LookupTableSizeV2Op size_op,
                                PatternRewriter& rewriter) const override {
    auto handle_op = size_op.getTableHandle().getDefiningOp();
    if (handle_op == nullptr) return failure();
    auto hashtable_op = llvm::dyn_cast<TFL::HashtableOp>(handle_op);
    if (hashtable_op == nullptr) return failure();
    rewriter.replaceOpWithNewOp<TFL::HashtableSizeOp>(
        size_op, size_op->getResultTypes(), size_op.getTableHandle());
    return success();
  }
};

template <typename T>
std::vector<T> GetAllOps(mlir::ModuleOp* module) {
  std::vector<T> ops;
  module->walk([&](T op) { ops.emplace_back(op); });
  return ops;
}

bool checkWhetherGraphHasValidStaticLookupTables(ModuleOp module) {
  auto hashtables = GetAllOps<TF::HashTableV2Op>(&module);
  // No needs to run the legalization patterns.
  if (hashtables.empty()) {
    return false;
  }

  for (auto hashtable : hashtables) {
    auto key_dtype = hashtable.getKeyDtype();
    auto value_dtype = hashtable.getValueDtype();

    // Only allow string -> int64 and int64 -> string mappings due to kernel
    // capability.
    if (!((mlir::isa<TF::StringType>(key_dtype) &&
           mlir::isa<IntegerType>(value_dtype) &&
           mlir::cast<IntegerType>(value_dtype).getWidth() == 64) ||
          (mlir::isa<TF::StringType>(value_dtype) &&
           mlir::isa<IntegerType>(key_dtype) &&
           mlir::cast<IntegerType>(key_dtype).getWidth() == 64))) {
      return false;
    }

    for (auto& use : hashtable->getUses()) {
      Operation* user = use.getOwner();

      // Allow consuming hash table ops that can be covered by TensorFlow Lite
      // hash table kernels.
      if (auto find_op = llvm::dyn_cast<TF::LookupTableFindV2Op>(user))
        continue;
      if (auto import_op = llvm::dyn_cast<TF::LookupTableImportV2Op>(user))
        continue;
      if (auto size_op = llvm::dyn_cast<TF::LookupTableSizeV2Op>(user))
        continue;

      return false;
    }
  }
  return true;
}

// Pass which legalizes TF hash tables only when they are covered by the
// TensorFlow Lite hash table kernels.
class LegalizeHashTablesPass
    : public impl::LegalizeHashTablesPassBase<LegalizeHashTablesPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeHashTablesPass)

  void runOnOperation() override {
    auto module = getOperation();

    if (!checkWhetherGraphHasValidStaticLookupTables(module)) {
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns
        .add<LegalizeHashTableOpPattern, LegalizeHashTableFindOpPattern,
             LegalizeHashTableImportOpPattern, LegalizeHashTableSizeOpPattern>(
            &getContext());
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeHashTablesPass() {
  return std::make_unique<LegalizeHashTablesPass>();
}

}  // namespace TFL
}  // namespace mlir
