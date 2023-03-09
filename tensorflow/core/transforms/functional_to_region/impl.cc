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

#include "tensorflow/core/transforms/functional_to_region/impl.h"

#include <algorithm>
#include <tuple>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/ir/utility.h"
#include "tensorflow/core/transforms/utils/utils.h"

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// Pattern Definitions
//===----------------------------------------------------------------------===//

namespace {
// Base class for patterns that convert functional ops to region-based ops. This
// class contains common utility functions and class members.
class BasePattern {
 public:
  BasePattern(SymbolTable &table, TFGraphDialect &dialect)
      : table_(table), dialect_(dialect) {}

 protected:
  // Lookup, using the symbol table, a graph function.
  GraphFuncOp LookupFunc(FuncAttr func_ref) const {
    return table_.lookup<GraphFuncOp>(func_ref.getName().getLeafReference());
  }

  // Split a range of non-control and control operands.
  std::pair<ValueRange, ValueRange> SplitControl(ValueRange values) const {
    return SplitDataAndControlValues(values, dialect_.getControlType());
  }

  // Convert the terminator of a region from `return` to `yield`.
  YieldOp ReplaceReturnWithYield(Block &block, TypeRange types,
                                 PatternRewriter &rewriter) const;

  // Copy a region from a function body to a loop body, reordering the arguments
  // from function order (pairs of data and control values) to loop order (all
  // data values followed by all control values).
  void CloneAndReorderArgs(TypeRange types, Region &from, Region &to,
                           PatternRewriter &rewriter) const;

  // Clone ops from one region to another with a given value mapping. Rename
  // clone ops with unique names.
  void CloneAndRename(Region &from, Region &to, IRMapping &bv) const;

 protected:
  // Symbol table for looking up branch/loop functions.
  SymbolTable &table_;
  // Dialect reference for getting cached values.
  TFGraphDialect &dialect_;
};

// Base class for converting a functional control-flow `SourceOp` to a
// region-based `DestOp`.
template <typename SourceOp, typename DestOp>
class ConvertFunctionalToRegionPattern : public OpRewritePattern<SourceOp>,
                                         public BasePattern {
 public:
  explicit ConvertFunctionalToRegionPattern(MLIRContext *context,
                                            SymbolTable &table,
                                            TFGraphDialect &dialect)
      : OpRewritePattern<SourceOp>(context, /*benefit=*/1,
                                   {DestOp::getOperationName()}),
        BasePattern(table, dialect) {}
};

// Base class for patterns to convert an if-like TFG op to region form.
template <typename IfLikeOp, typename IfLikeRegionOp>
struct ConvertIfLikeOp
    : public ConvertFunctionalToRegionPattern<IfLikeOp, IfLikeRegionOp> {
  using ConvertFunctionalToRegionPattern<
      IfLikeOp, IfLikeRegionOp>::ConvertFunctionalToRegionPattern;

  LogicalResult matchAndRewrite(IfLikeOp op,
                                PatternRewriter &rewriter) const override;
};

using ConvertIfOp = ConvertIfLikeOp<IfOp, IfRegionOp>;
using ConvertStatelessIfOp =
    ConvertIfLikeOp<StatelessIfOp, StatelessIfRegionOp>;
using ConvertStatefulIfOp = ConvertIfLikeOp<StatefulIfOp, StatefulIfRegionOp>;

// Base class for patterns to convert a case-like TFG op to region form.
template <typename CaseLikeOp, typename CaseLikeRegionOp>
struct ConvertCaseLikeOp
    : public ConvertFunctionalToRegionPattern<CaseLikeOp, CaseLikeRegionOp> {
  using ConvertFunctionalToRegionPattern<
      CaseLikeOp, CaseLikeRegionOp>::ConvertFunctionalToRegionPattern;

  LogicalResult matchAndRewrite(CaseLikeOp op,
                                PatternRewriter &rewriter) const override;
};

using ConvertCaseOp = ConvertCaseLikeOp<CaseOp, CaseRegionOp>;
using ConvertStatelessCaseOp =
    ConvertCaseLikeOp<StatelessCaseOp, StatelessCaseRegionOp>;
using ConvertStatefulCaseOp =
    ConvertCaseLikeOp<StatefulCaseOp, StatefulCaseRegionOp>;

// Base class for patterns to convert a while-like TFG op to region form.
template <typename WhileLikeOp, typename WhileLikeRegionOp>
struct ConvertWhileLikeOp
    : public ConvertFunctionalToRegionPattern<WhileLikeOp, WhileLikeRegionOp> {
  using ConvertFunctionalToRegionPattern<
      WhileLikeOp, WhileLikeRegionOp>::ConvertFunctionalToRegionPattern;

  LogicalResult matchAndRewrite(WhileLikeOp op,
                                PatternRewriter &rewriter) const override;
};

using ConvertWhileOp = ConvertWhileLikeOp<WhileOp, WhileRegionOp>;
using ConvertStatelessWhileOp =
    ConvertWhileLikeOp<StatelessWhileOp, StatelessWhileRegionOp>;
using ConvertStatefulWhileOp =
    ConvertWhileLikeOp<StatefulWhileOp, StatefulWhileRegionOp>;

// Convert a functional for-loop to a region-based for-loop.
struct ConvertForOp
    : public ConvertFunctionalToRegionPattern<ForOp, ForRegionOp> {
  using ConvertFunctionalToRegionPattern<
      ForOp, ForRegionOp>::ConvertFunctionalToRegionPattern;

  LogicalResult matchAndRewrite(tfg::ForOp op,
                                PatternRewriter &rewriter) const override;
};

}  // namespace

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

// We cannot inline or modify a function if it does not exist, if it is generic,
// if it has a computed gradient, or if it is marked for compilation (e.g. by
// XLA).
static bool CannotInline(GraphFuncOp func) {
  return !func || func.getGeneric() || func.getGradient() ||
         func.isMarkedForCompilation();
}

// Determine which optional attributes of a non-generic function to preserve.
// Preserved attributes:
// - `description`
// - `is_stateful`
// - `resource_arg_unique_ids_keys`
// - `resource_arg_unique_ids_values`
//
// The attributes of a non-generic function to preserve:
// - Intrinsic `tfg.*` attributes are preserved.
// - Non-intrinsic `tf.*` attributes are preserved.
//
// The result attributes of a non-generic function to preserve:
// - Intrinsic `tfg.*` attributes are preserved.
static DictionaryAttr PreserveFunctionAttributes(GraphFuncOp func) {
  NamedAttrList preserved_attrs;
  const auto preserve = [&](StringAttr name) {
    if (Attribute attr = func->getAttr(name))
      preserved_attrs.append(name, attr);
  };
  preserve(func.getDescriptionAttrName());
  preserve(func.getIsStatefulAttrName());
  preserve(func.getResourceArgUniqueIdsKeysAttrName());
  preserve(func.getResourceArgUniqueIdsValuesAttrName());
  // Propagate tf.* attributes.
  // TODO(jeffniu): `tf` dialect is not loaded.
  for (const NamedAttribute &attr : func->getAttrs())
    if (attr.getName().getValue().startswith("tf."))
      preserved_attrs.append(attr);

  // Certain pipelines (Brella) will split a graph into subgraphs before merging
  // them back together. If the subgraphs pass through conversion to and from
  // region form, the previously unique branch/loop body function names become
  // not unique, which prevents the graphs from being correctly merged back
  // together. Also, if an op is referenced in two different subgraphs, if
  // Grappler changes the function name, the reference will only be valid in the
  // first subgraph, leading to a function-not-found error. Preserve the
  // original function name.
  preserve(func.getSymNameAttrName());

  return preserved_attrs.getDictionary(func.getContext());
}

// Given the function, argument, and result attributes to be preserved,
// determine if they are empty and can be dropped.
static bool ArePreservedAttrsEmpty(DictionaryAttr func_attrs,
                                   ArrayAttr arg_attrs, ArrayAttr res_attrs) {
  const auto is_empty = [](DictionaryAttr dict) { return dict.empty(); };
  return func_attrs.empty() &&
         llvm::all_of(arg_attrs.getAsRange<DictionaryAttr>(), is_empty) &&
         llvm::all_of(res_attrs.getAsRange<DictionaryAttr>(), is_empty);
}

// Determine if the region attributes are empty.
static bool AreRegionAttrsEmpty(RegionAttr attrs) {
  return ArePreservedAttrsEmpty(attrs.getAttrs(), attrs.getArgAttrs(),
                                attrs.getResAttrs());
}

// Preserve certain attributes of a function so that they can be used later if
// the region op is converted back to functional form. When `If` and `Case` are
// converted, all arguments attributes are dropped because the arguments are
// converted to implicit captures. For `While` and `For`, no arguments are
// removed.
//
// If `drop_args` is set, then all argument attributes are dropped, regardless
// of the number of arguments in the function.
//
// If `allow_empty` is set, then this function will always return a non-null
// attribute, even if the region attributes are empty.
static RegionAttr PreserveAttributes(GraphFuncOp func, bool drop_args = false,
                                     bool allow_empty = false) {
  DictionaryAttr func_attrs = PreserveFunctionAttributes(func);
  // Since all argument and result attributes are preserved, just propagate the
  // array attributes. Remove the control argument attributes from the argument
  // attributes.
  const auto every_other = [](ArrayAttr attrs) {
    SmallVector<Attribute> others;
    for (unsigned i = 0; i < attrs.size(); i += 2) others.push_back(attrs[i]);
    return ArrayAttr::get(attrs.getContext(), others);
  };

  ArrayAttr arg_attrs = drop_args || !func.getArgAttrs()
                            ? ArrayAttr::get(func.getContext(), {})
                            : every_other(*func.getArgAttrs());
  ArrayAttr res_attrs = func.getResAttrs()
                            ? *func.getResAttrs()
                            : ArrayAttr::get(func.getContext(), {});

  if (!allow_empty && ArePreservedAttrsEmpty(func_attrs, arg_attrs, res_attrs))
    return nullptr;
  return RegionAttr::get(func_attrs, arg_attrs, res_attrs);
}

YieldOp BasePattern::ReplaceReturnWithYield(Block &block, TypeRange types,
                                            PatternRewriter &rewriter) const {
  auto op = cast<ReturnOp>(block.getTerminator());
  rewriter.setInsertionPoint(op);
  ValueRange args, ctls;
  std::tie(args, ctls) = SplitControl(op.getOperands());
  return rewriter.replaceOpWithNewOp<YieldOp>(op, args, ctls);
}

void BasePattern::CloneAndReorderArgs(TypeRange types, Region &from, Region &to,
                                      PatternRewriter &rewriter) const {
  ControlType control_ty = dialect_.getControlType();
  IRMapping bv;
  CloneAndRename(from, to, bv);
  SmallVector<Location> arg_locs(types.size(), from.getLoc());
  for (auto &it :
       llvm::enumerate(llvm::to_vector(to.addArguments(types, arg_locs)))) {
    BlockArgument arg = to.getArgument(it.index() * 2);
    BlockArgument ctl = to.getArgument(arg.getArgNumber() + 1);
    arg.replaceAllUsesWith(it.value());
    ctl.replaceAllUsesWith(to.addArgument(control_ty, arg.getLoc()));
  }
  llvm::BitVector erase_indices(to.getNumArguments());
  erase_indices.set(0, types.size() * 2);
  to.front().eraseArguments(erase_indices);
}

void BasePattern::CloneAndRename(Region &from, Region &to,
                                 IRMapping &bv) const {
  from.cloneInto(&to, bv);
  StringAttr name_id = dialect_.getNameAttrIdentifier();
  auto op_name = to.getParentOp()->getAttrOfType<StringAttr>(name_id);
  if (!op_name) return;
  for (Operation &op : to.getOps()) {
    if (auto name = op.getAttrOfType<StringAttr>(name_id)) {
      auto new_name =
          StringAttr::get(op.getContext(), name.getValue() + "_tfg_inlined_" +
                                               op_name.getValue() + "_" +
                                               Twine(to.getRegionNumber()));
      op.setAttr(name_id, new_name);
    }
  }
}

//===----------------------------------------------------------------------===//
// ConvertIfLikeOp
//===----------------------------------------------------------------------===//

template <typename IfLikeOp, typename IfLikeRegionOp>
LogicalResult ConvertIfLikeOp<IfLikeOp, IfLikeRegionOp>::matchAndRewrite(
    IfLikeOp op, PatternRewriter &rewriter) const {
  GraphFuncOp then_func = this->LookupFunc(op.getThenBranch());
  GraphFuncOp else_func = this->LookupFunc(op.getElseBranch());
  if (CannotInline(then_func) || CannotInline(else_func)) return failure();

  // Create the region-based op, passing in the required attributes.
  ValueRange args, ctls;
  std::tie(args, ctls) = this->SplitControl(op.getArgs());
  auto region_op = rewriter.create<IfLikeRegionOp>(
      op.getLoc(), op.getResultTypes(), op.getCond(), ctls,
      op.getThenBranch().getAttrs(), op.getElseBranch().getAttrs(),
      PreserveAttributes(then_func, /*drop_args=*/true),
      PreserveAttributes(else_func, /*drop_args=*/true));
  util::ForwardNonIntrinsicAttributes(op, region_op);

  // Move the regions over and replace the block arguments.
  ControlType control_ty = this->dialect_.getControlType();
  IRMapping then_bv, else_bv;
  auto func_args =
      llvm::zip(then_func.getArguments(), else_func.getArguments()).begin();
  rewriter.setInsertionPoint(region_op);
  Value then_arg, else_arg, then_ctl, else_ctl;
  for (Value arg : args) {
    std::tie(then_arg, else_arg) = *func_args;
    ++func_args;
    std::tie(then_ctl, else_ctl) = *func_args;
    ++func_args;
    Value ctl = LookupControlDependency(arg);
    then_bv.map(then_arg, arg);
    else_bv.map(else_arg, arg);
    then_bv.map(then_ctl, ctl);
    else_bv.map(else_ctl, ctl);
  }
  this->CloneAndRename(then_func.getBody(), region_op.getThenRegion(), then_bv);
  this->CloneAndRename(else_func.getBody(), region_op.getElseRegion(), else_bv);

  // Replace the terminators `return` with `yield`.
  TypeRange ret_types = region_op.getOuts().getTypes();
  this->ReplaceReturnWithYield(region_op.getThenBlock(), ret_types, rewriter);
  this->ReplaceReturnWithYield(region_op.getElseBlock(), ret_types, rewriter);
  rewriter.replaceOp(op, region_op.getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// ConvertCaseLikeOp
//===----------------------------------------------------------------------===//

template <typename CaseLikeOp, typename CaseLikeRegionOp>
LogicalResult ConvertCaseLikeOp<CaseLikeOp, CaseLikeRegionOp>::matchAndRewrite(
    CaseLikeOp op, PatternRewriter &rewriter) const {
  // Lookup all the branch functions and save their attributes.
  SmallVector<GraphFuncOp> branch_funcs;
  SmallVector<Attribute> branch_attrs;
  branch_funcs.reserve(op.getBranches().size());
  for (auto attr : op.getBranches().template getAsRange<FuncAttr>()) {
    GraphFuncOp branch_func = this->LookupFunc(attr);
    if (CannotInline(branch_func)) return failure();
    branch_funcs.push_back(branch_func);
    branch_attrs.push_back(attr.getAttrs());
  }

  SmallVector<Attribute> preserved_attrs;
  for (GraphFuncOp func : branch_funcs) {
    preserved_attrs.push_back(
        PreserveAttributes(func, /*drop_args=*/true, /*allow_empty=*/true));
  }
  ArrayAttr region_attrs = nullptr;
  if (!llvm::all_of(preserved_attrs, [](Attribute attr) {
        return AreRegionAttrsEmpty(attr.cast<RegionAttr>());
      }))
    region_attrs = rewriter.getArrayAttr(preserved_attrs);

  // Create the region-based op, passing in the required attributes.
  ValueRange args, ctls;
  std::tie(args, ctls) = this->SplitControl(op.getArgs());
  auto region_op = rewriter.create<CaseLikeRegionOp>(
      op.getLoc(), op.getResultTypes(), op.getBranchIndex(), ctls,
      rewriter.getArrayAttr(branch_attrs), region_attrs,
      op.getBranches().size());
  util::ForwardNonIntrinsicAttributes(op, region_op);

  // Move the regions over and replace the block arguments.
  ControlType control_ty = this->dialect_.getControlType();
  SmallVector<IRMapping> bvs(branch_funcs.size(), {});
  rewriter.setInsertionPoint(region_op);
  for (auto &arg : llvm::enumerate(args)) {
    for (auto it : llvm::zip(branch_funcs, bvs)) {
      BlockArgument branch_arg =
          GraphFuncOp::getDataValue(std::get<0>(it).getBody(), arg.index());
      IRMapping &bv = std::get<1>(it);
      bv.map(branch_arg, arg.value());
      bv.map(GraphFuncOp::getControlTokenOf(branch_arg),
             LookupControlDependency(arg.value()));
    }
  }
  for (auto it : llvm::zip(branch_funcs, region_op.getBranches(), bvs)) {
    this->CloneAndRename(std::get<0>(it).getBody(), std::get<1>(it),
                         std::get<2>(it));
  }

  // Replace the terminators `return` with `yield`.
  TypeRange ret_types = region_op.getOuts().getTypes();
  for (Region &branch : region_op.getBranches())
    this->ReplaceReturnWithYield(branch.front(), ret_types, rewriter);
  rewriter.replaceOp(op, region_op.getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// ConvertWhileLikeOp
//===----------------------------------------------------------------------===//

template <typename WhileLikeOp, typename WhileLikeRegionOp>
LogicalResult
ConvertWhileLikeOp<WhileLikeOp, WhileLikeRegionOp>::matchAndRewrite(
    WhileLikeOp op, PatternRewriter &rewriter) const {
  GraphFuncOp cond_func = this->LookupFunc(op.getCond());
  GraphFuncOp body_func = this->LookupFunc(op.getBody());
  if (CannotInline(cond_func) || CannotInline(body_func)) return failure();

  // Note that `tfg.While` may not have the same input and output types. We will
  // need to insert casts.
  // TODO(jeffniu): Change this to call the infer return types builder.
  ValueRange init, ctls;
  std::tie(init, ctls) = this->SplitControl(op.getArgs());
  auto region_op = rewriter.create<WhileLikeRegionOp>(
      op.getLoc(), op.getResultTypes(), init, ctls,
      op.getParallelIterationsAttr(), op.getCond().getAttrs(),
      op.getBody().getAttrs(), PreserveAttributes(cond_func),
      PreserveAttributes(body_func));
  util::ForwardNonIntrinsicAttributes(op, region_op);

  // Just copy the function bodies into the regions. `RegionBranchOpInterface`
  // requires that we re-order the block arguments such that the control tokens
  // all come after the data arguments.
  this->CloneAndReorderArgs(init.getTypes(), cond_func.getBody(),
                            region_op.getCondRegion(), rewriter);
  this->CloneAndReorderArgs(init.getTypes(), body_func.getBody(),
                            region_op.getBodyRegion(), rewriter);
  this->ReplaceReturnWithYield(region_op.getBodyBlock(), init.getTypes(),
                               rewriter);

  // Replace `return(tensor<*xi1>)` with `condition`.
  auto ret_op = cast<ReturnOp>(region_op.getCondBlock().getTerminator());
  ValueRange ret_args, ret_ctls;
  std::tie(ret_args, ret_ctls) = this->SplitControl(ret_op.getOperands());
  rewriter.setInsertionPoint(ret_op);
  rewriter.replaceOpWithNewOp<ConditionOp>(
      ret_op, ret_args.front(),
      GetLoopRegionDataArgs(region_op.getCondRegion()), ret_ctls);
  rewriter.replaceOp(op, region_op->getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// ConvertForOp
//===----------------------------------------------------------------------===//

LogicalResult ConvertForOp::matchAndRewrite(tfg::ForOp op,
                                            PatternRewriter &rewriter) const {
  GraphFuncOp body_func = LookupFunc(op.getBody());
  if (CannotInline(body_func)) return failure();

  // Note that `For` may not have the same input and output typse, although
  // `ForRegion` does. We will need to insert casts.
  ValueRange init, ctls;
  std::tie(init, ctls) = SplitControl(op.getArgs());
  auto region_op = rewriter.create<ForRegionOp>(
      op.getLoc(), op.getResultTypes(), op.getStart(), op.getLimit(),
      op.getDelta(), init, ctls, op.getBody().getAttrs(),
      PreserveAttributes(body_func));
  util::ForwardNonIntrinsicAttributes(op, region_op);

  // Copy the function body into the region. One index type must be added.
  OperandRange args = op.getOperands().drop_front(2).drop_back(ctls.size());
  CloneAndReorderArgs(args.getTypes(), body_func.getBody(),
                      region_op.getBodyRegion(), rewriter);
  ReplaceReturnWithYield(region_op.getBodyBlock(), init.getTypes(), rewriter);
  rewriter.replaceOp(op, region_op->getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// Populate Patterns
//===----------------------------------------------------------------------===//

void PopulateFunctionalToRegionPatterns(RewritePatternSet &patterns,
                                        SymbolTable &table) {
  patterns.insert<ConvertIfOp, ConvertStatelessIfOp, ConvertStatefulIfOp,
                  ConvertWhileOp, ConvertStatelessWhileOp,
                  ConvertStatefulWhileOp, ConvertCaseOp, ConvertStatelessCaseOp,
                  ConvertStatefulCaseOp, ConvertForOp>(
      patterns.getContext(), table,
      *patterns.getContext()->getOrLoadDialect<TFGraphDialect>());
}

}  // namespace tfg
}  // namespace mlir
