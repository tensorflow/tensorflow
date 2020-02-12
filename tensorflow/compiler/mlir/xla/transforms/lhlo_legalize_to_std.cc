/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for lowering LHLO dialect to Affine dialect.

#include "absl/memory/memory.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"  // TF:llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"

namespace mlir {
namespace xla_lhlo {
namespace {

FlatSymbolRefAttr CallExternalFunc(
    PatternRewriter &rewriter,
    ModuleOp module,
    Location loc,
    const std::string& func_name,
    std::vector<Type> &input_types,
    std::vector<Type> &result_types,
    ArrayRef<NamedAttribute> attrs) {

  // TODO: FIXME, how about the same function name, by differernt params ?
  // External function symbol is existed
  FlatSymbolRefAttr func_name_attr = rewriter.getSymbolRefAttr(func_name);
  if (module.lookupSymbol(func_name)) {
    return func_name_attr;
  }

  auto context = rewriter.getContext();
  // function type
  auto func_type = FunctionType::get(input_types, result_types, context);

  // create func op
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<mlir::FuncOp>(loc, func_name_attr.getValue(),
                                func_type, attrs);

  return func_name_attr;
}


struct UniqueCountConverter : public OpRewritePattern<UniqueCountOp> {
  using OpRewritePattern<UniqueCountOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(UniqueCountOp op,
                                     PatternRewriter& rewriter) const override {


    auto context = rewriter.getContext();
    auto loc = op.getLoc();
    std::vector<Type> input_types;
    std::vector<Type> result_types;
    input_types.push_back(op.input().getType());
    input_types.push_back(rewriter.getIndexType());
    result_types.push_back(op.input().getType().dyn_cast<MemRefType>().getElementType());
    FlatSymbolRefAttr output_func_ref = CallExternalFunc(
        rewriter, op.getParentOfType<ModuleOp>(), loc,
        "_global_get_unique_ids_count",
        input_types, result_types,
        ArrayRef<NamedAttribute>{});

    // Create a CallOp to call `_global_get_unique_ids_count`
    auto runtime_ids_count = rewriter.create<mlir::DimOp>(op.getLoc(), op.input(), 0);
    SmallVector<Value, 4> unique_ids_count_func_param;
    unique_ids_count_func_param.push_back(op.input());
    unique_ids_count_func_param.push_back(runtime_ids_count);
    // get unique ids count, params: ids memref + runtime ids count

    auto get_unique_count_op = rewriter.create<mlir::CallOp>(
        loc, output_func_ref.getValue(),
        result_types,
        unique_ids_count_func_param);

    rewriter.create<StoreOp>(op.getLoc(), get_unique_count_op.getResult(0), op.output());
    rewriter.eraseOp(op);
    return this->matchSuccess();
  }
 
};

struct UniqueConverter : public OpRewritePattern<UniqueOp> {
  using OpRewritePattern<UniqueOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(
      UniqueOp op,
      PatternRewriter& rewriter) const override {
    auto context = rewriter.getContext();
    auto loc = op.getLoc();

    std::vector<Type> input_types;
    std::vector<Type> result_types;
    input_types.push_back(op.x().getType());
    input_types.push_back(op.y().getType());
    input_types.push_back(op.idx().getType());
 
    auto tensor_type = op.y().getType().dyn_cast<MemRefType>().getElementType();
    auto idx_type = op.idx().getType().dyn_cast<MemRefType>().getElementType();
    std::string func_name("_global_unique");
    if (tensor_type.isInteger(64)) {
      func_name += "_i64";
    } else if (tensor_type.isInteger(32)) {
      func_name += "_i32";
    } else {
      llvm::errs() << "Now only support int32 and int64 type.\n";
      return this->matchFailure();
    }
    if (idx_type.isInteger(64)) {
      func_name += "_i64";
    } else if (idx_type.isInteger(32)) {
      func_name += "_i32";
    } else {
      llvm::errs() << "Unique only support int32 and int64 index type.\n";
      return this->matchFailure();
    }

    FlatSymbolRefAttr output_func_ref = CallExternalFunc(
        rewriter, op.getParentOfType<ModuleOp>(), loc,
        func_name, input_types, result_types,
        ArrayRef<NamedAttribute>{});

    // Create a CallOp to call `_global_unique_i64_i32`
    SmallVector<Value, 4> func_param;
    func_param.push_back(op.x());
    func_param.push_back(op.y());
    func_param.push_back(op.idx());

    rewriter.create<mlir::CallOp>(
        loc, output_func_ref.getValue(),
        result_types,
        func_param);

    rewriter.eraseOp(op);

    return this->matchSuccess();
  }
};

void populateLHLOToStdConversionPattern(MLIRContext* context,
                                           OwningRewritePatternList* patterns) {
  // clang-format off
  patterns->insert<
      UniqueCountConverter,
      UniqueConverter
    >(context);
  // clang-format on
}

struct LhloLegalizeToStd: public FunctionPass<LhloLegalizeToStd> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect>();
    // NOTE(jiankeng.pt): Advanced skills: prevent the error of UniqueOp lowering process.
    // Cause the `func` op here has no one legalize pattern.
    // If you don't want the Op with concrete information which you specify
    // in the anonymous function be lowered at the pass, please do this.
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return
        op.getName() == "_global_unique_i64_i64" ||
        op.getName() == "_global_unique_i64_i32" ||
        op.getName() == "_global_unique_i32_i64" ||
        op.getName() == "_global_unique_i32_i32" ||
        op.getName() == "_global_unique_index32" ||
        op.getName() == "_global_unique_index64" ||
        op.getName() == "_global_get_unique_ids_count" ||
        op.getName() == "_global_unique_ids";}); 
    auto func = getFunction();
    populateLHLOToStdConversionPattern(func.getContext(), &patterns);
    if (failed(applyPartialConversion(func, target, patterns, nullptr))) {
      signalPassFailure();
    }
    // applyPatternsGreedily(func, patterns);
  }
};

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> createLegalizeToStdPass() {
  return absl::make_unique<LhloLegalizeToStd>();
}

static PassRegistration<LhloLegalizeToStd> legalize_pass(
    "lhlo-legalize-to-std", "Legalize from LHLO dialect to stdandard dialect");

}  // namespace xla_lhlo
}  // namespace mlir
