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

#include "tensorflow/core/transforms/region_to_functional/impl.h"

#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
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
// Cached attribute name identifiers shared by all patterns.
struct CachedIdentifiers {
  explicit CachedIdentifiers(TFGraphDialect *dialect)
      : tfg_name(dialect->getTfgNameAttrIdentifier()),
        tfg_regenerate_output_shapes(StringAttr::get(
            dialect->getContext(), "tfg.regenerate_output_shapes")) {}

  // Cached identifier for "tfg.name".
  StringAttr tfg_name;
  // Cached identifier for "tfg.regenerate_output_shapes".
  StringAttr tfg_regenerate_output_shapes;
};

// A helper for uniqueing argument, result, and control result names, which must
// be unique for a function.
class NameUniquer {
 public:
  explicit NameUniquer(MLIRContext *ctx) : ctx_(ctx) {}

  // Unique a name. If the name is unused, returns the name. Otherwise,
  // allocates a new name.
  StringAttr GetUniqued(StringAttr name) {
    auto it = unique_names_.insert(name);
    if (it.second) return name;
    unsigned suffix = 0;
    StringAttr next_name;
    do {
      next_name =
          StringAttr::get(ctx_, name.getValue() + "_" + Twine(suffix++));
      it = unique_names_.insert(next_name);
    } while (!it.second);
    return next_name;
  }

 private:
  // The MLIR context.
  MLIRContext *ctx_;
  // This set contains the occupied names.
  DenseSet<StringAttr> unique_names_;
};

// Base class for patterns used to convert region control-flow ops to functional
// control-flow ops. This class contains common utility functions and cached
// attribute identifiers.
class BasePattern {
 public:
  BasePattern(TFGraphDialect &dialect, SymbolTable &table,
              bool force_control_capture, CachedIdentifiers ids)
      : ctx_(dialect.getContext()),
        dialect_(dialect),
        table_(table),
        force_control_capture_(force_control_capture),
        ids_(ids) {}

 protected:
  // Collect all values used in the region that are defined above the region.
  // If a control token is encountered, collect its associated data value. If it
  // doesn't have one, add it to `ctls`.
  void CollectValuesDefinedAbove(Region &region, SetVector<Value> &datas,
                                 SetVector<Value> &ctls) const;
  // Collect data values used in any of the given regions that are defined above
  // the regions. These are the values that will be converted to explicit
  // capture. If a control token with no associated data value is encountered
  // and `force_control_capture_` is not set, then this function returns
  // failure. Otherwise, it inserts chain constants and rewrites uses of the
  // token to use the control outputs of the constants.
  FailureOr<std::vector<Value>> CollectValuesDefinedAboveAll(
      RegionRange regions, PatternRewriter &rewriter) const;
  // Rewrite the regions to be isolated from above by replacing uses of the
  // given data values with block arguments. Use the same set of values for each
  // region so that their block arguments are the same.
  void IsolateRegions(RegionRange regions, MutableArrayRef<Value> datas) const;
  // Create a chain `Const` operation. The op's data result is unused; its
  // only purpose is to convert a control edge into a data edge.
  Operation *MakeChainConstant(Operation *parent, Value ctl, unsigned idx,
                               PatternRewriter &rewriter) const;

  // Infer or propagate function attributes. Use a name uniquer to unique names
  // across function arguments, results, and control results.
  NamedAttrList BuildAttributes(RegionAttr preserved, ValueRange arguments,
                                ValueRange results,
                                NameUniquer *name_uniquer) const;

  // Try to find a name for a data or control value. For op results, check the
  // op for a name. Otherwise, check the enclosing function's arg attributes.
  StringAttr TryFindName(Value value, std::optional<ValueRange> args) const;

  // Get the `control_ret_attrs` attributes for control returns. Use a name
  // uniquer to unique names across function arguments, results, and control
  // results,
  ArrayAttr GetControlRetAttrs(ValueRange ctls, ValueRange args,
                               NameUniquer *name_uniquer) const;

  // Create a function with the given name and attributes. Use the types of the
  // block arguments and the given results types. Take the body of the region.
  GraphFuncOp CreateFunc(Location loc, const Twine &sym_name, Region &region,
                         TypeRange res_types, NamedAttrList attrs) const;

  // Convert a (yield-terminated) region to a function and return a reference.
  FuncAttr Outline(Operation *op, PatternRewriter &rewriter, ValueRange args,
                   Region &region, RegionAttr preserved, DictionaryAttr attrs,
                   const Twine &func_name) const;

  // A region function to outline.
  struct RegionFunction {
    // The function body.
    Region &region;
    // Potentially null preserved function attributes.
    RegionAttr preserved_attrs;
    // The function call attributes.
    DictionaryAttr call_attrs;
    // The function name to use.
    std::string func_name;
  };
  // Outline a list of (yield-terminated) region functions, but if any function
  // could not be re-used, then new functions are created for all of them.
  template <typename FuncAttrT>
  void ReuseAllOrOutline(Operation *op, PatternRewriter &rewriter,
                         ValueRange args, ArrayRef<RegionFunction> regions,
                         SmallVectorImpl<FuncAttrT> &functions) const;

  // Try to find a "reusable" function that has the same body as the provided
  // region. A function is "reusable" if its body has the same topology as the
  // provided region, corresponding operands have the same attributes, except
  // for node name, and value types are compatible.
  FuncAttr FindReusableFunc(Region &region, RegionAttr preserved,
                            DictionaryAttr attrs) const;

  // If a function exists and has nested regions, return false. Otherwise,
  // return true.
  bool FuncHasNestedRegions(RegionAttr preserved) const;

 protected:
  // Reference to the context.
  MLIRContext *ctx_;
  // Dialect reference for getting cached values;
  TFGraphDialect &dialect_;
  // Symbol table to use to look up existing functions.
  SymbolTable &table_;
  // Whether control tokens with no data values should be forcefully captured by
  // inserting a chain `Const` op.
  bool force_control_capture_;
  // Cached attribute identifiers.
  CachedIdentifiers ids_;
};

//===----------------------------------------------------------------------===//
// ConvertToExplicitCapture

template <typename OpT>
struct ConvertToExplicitCapture : public BasePattern {
  using BasePattern::BasePattern;

  virtual ~ConvertToExplicitCapture() = default;

  // Convert the regions of the operation to explicit capture. Returns the
  // newly captured values and an updated op.
  FailureOr<std::pair<OpT, std::vector<Value>>> Run(OpT op,
                                                    PatternRewriter &rewriter);

  // Rebuild the regions of the operation with added values.
  virtual OpT RebuildWith(OpT op, ValueRange added,
                          PatternRewriter &rewriter) const = 0;
};

template <typename IfLikeRegionOp>
struct ConvertIfLikeRegionOpToExplicitCapture
    : public ConvertToExplicitCapture<IfLikeRegionOp> {
  using ConvertToExplicitCapture<IfLikeRegionOp>::ConvertToExplicitCapture;

  IfLikeRegionOp RebuildWith(IfLikeRegionOp op, ValueRange added,
                             PatternRewriter &rewriter) const override {
    return rewriter.create<IfLikeRegionOp>(
        op.getLoc(), op.getResultTypes(), op.getCond(), op.getCtls(),
        op.getThenAttrsAttr(), op.getElseAttrsAttr(),
        op.getThenRegionAttrsAttr(), op.getElseRegionAttrsAttr());
  }
};

template <typename CaseLikeRegionOp>
struct ConvertCaseLikeRegionOpToExplicitCapture
    : public ConvertToExplicitCapture<CaseLikeRegionOp> {
  using ConvertToExplicitCapture<CaseLikeRegionOp>::ConvertToExplicitCapture;

  CaseLikeRegionOp RebuildWith(CaseLikeRegionOp op, ValueRange added,
                               PatternRewriter &rewriter) const override {
    return rewriter.create<CaseLikeRegionOp>(
        op.getLoc(), op.getResultTypes(), op.getBranchIndex(), op.getCtls(),
        op.getBranchAttrsAttr(), op.getRegionAttrsAttr(),
        op.getBranches().size());
  }
};

// Get the block arguments that correspond to the passthrough iteration
// arguments created from converting implicit captures. Append them to the
// previous region results `prev`.
static SmallVector<Value> GetForwardedValues(ValueRange added,
                                             Block::BlockArgListType block_args,
                                             ValueRange prev) {
  SmallVector<Value> args(prev.begin(), prev.end());
  llvm::append_range(args, block_args.slice(prev.size(), added.size()));
  return args;
}

template <typename WhileLikeRegionOp>
struct ConvertWhileLikeRegionOpToExplicitCapture
    : public ConvertToExplicitCapture<WhileLikeRegionOp> {
  using ConvertToExplicitCapture<WhileLikeRegionOp>::ConvertToExplicitCapture;

  WhileLikeRegionOp RebuildWith(WhileLikeRegionOp op, ValueRange added,
                                PatternRewriter &rewriter) const override {
    ConditionOp cond_op = op.getCondCondition();
    rewriter.setInsertionPoint(cond_op);
    rewriter.replaceOpWithNewOp<ConditionOp>(
        cond_op, cond_op.getCond(),
        GetForwardedValues(added, op.getCondRegion().getArguments(),
                           cond_op.getArgs()),
        cond_op.getCtls());

    YieldOp yield_op = op.getBodyYield();
    rewriter.setInsertionPoint(yield_op);
    rewriter.replaceOpWithNewOp<YieldOp>(
        yield_op,
        GetForwardedValues(added, op.getBodyRegion().getArguments(),
                           yield_op.getArgs()),
        yield_op.getCtls());

    SmallVector<Value> operands = llvm::to_vector(op.getInit());
    llvm::append_range(operands, added);
    SmallVector<Type> results = llvm::to_vector(op.getOuts().getTypes());
    llvm::append_range(results, added.getTypes());
    util::LoopRegionResultAdded(op.getBodyRegion(), added.size());

    rewriter.setInsertionPoint(op);
    return rewriter.create<WhileLikeRegionOp>(
        op.getLoc(), results, op.getCtl().getType(), operands, op.getCtls(),
        op.getParallelIterationsAttr(), op.getCondAttrsAttr(),
        op.getBodyAttrsAttr(), op.getCondRegionAttrsAttr(),
        op.getBodyRegionAttrsAttr());
  }
};

struct ConvertForRegionOpToExplicitCapture
    : public ConvertToExplicitCapture<ForRegionOp> {
  using ConvertToExplicitCapture<ForRegionOp>::ConvertToExplicitCapture;

  ForRegionOp RebuildWith(ForRegionOp op, ValueRange added,
                          PatternRewriter &rewriter) const override {
    YieldOp yield_op = op.getBodyYield();
    rewriter.setInsertionPoint(yield_op);
    // Get the iteration arguments excluding the for loop index argument.
    auto iter_args = GetLoopRegionDataArgs(op.getBodyRegion()).slice(1);
    rewriter.replaceOpWithNewOp<YieldOp>(
        yield_op, GetForwardedValues(added, iter_args, yield_op.getArgs()),
        yield_op.getCtls());

    SmallVector<Value> operands = llvm::to_vector(op.getInit());
    llvm::append_range(operands, added);
    SmallVector<Type> results = llvm::to_vector(op.getOuts().getTypes());
    llvm::append_range(results, added.getTypes());
    util::LoopRegionResultAdded(op.getBodyRegion(), added.size());

    rewriter.setInsertionPoint(op);
    return rewriter.create<ForRegionOp>(
        op.getLoc(), results, op.getCtl().getType(), op.getStart(),
        op.getLimit(), op.getDelta(), operands, op.getCtls(),
        op.getBodyAttrsAttr(), op.getRegionAttrsAttr());
  }
};

//===----------------------------------------------------------------------===//
// ConvertRegionToFunctional

template <typename SourceOp, typename DestOp>
struct ConvertRegionToFunctionalPattern : public OpRewritePattern<SourceOp>,
                                          public BasePattern {
  explicit ConvertRegionToFunctionalPattern(MLIRContext *context,
                                            TFGraphDialect &dialect,
                                            SymbolTable &table,
                                            bool force_control_capture,
                                            CachedIdentifiers ids)
      : OpRewritePattern<SourceOp>(context, /*benefit=*/1,
                                   {DestOp::getOperationName()}),
        BasePattern(dialect, table, force_control_capture, ids) {}
};

// Base class for patterns to convert an if-like TFG region op to
// functional form.
template <typename IfLikeRegionOp, typename IfLikeOp>
struct ConvertIfLikeOp
    : public ConvertRegionToFunctionalPattern<IfLikeRegionOp, IfLikeOp> {
  using ConvertRegionToFunctionalPattern<
      IfLikeRegionOp, IfLikeOp>::ConvertRegionToFunctionalPattern;

  LogicalResult matchAndRewrite(IfLikeRegionOp op,
                                PatternRewriter &rewriter) const override;
};

using ConvertIfOp = ConvertIfLikeOp<IfRegionOp, IfOp>;
using ConvertStatelessIfOp =
    ConvertIfLikeOp<StatelessIfRegionOp, StatelessIfOp>;
using ConvertStatefulIfOp = ConvertIfLikeOp<StatefulIfRegionOp, StatefulIfOp>;

// Base class for patterns to convert an case-like TFG region op to
// functional form.
template <typename CaseLikeRegionOp, typename CaseLikeOp>
struct ConvertCaseLikeOp
    : public ConvertRegionToFunctionalPattern<CaseLikeRegionOp, CaseLikeOp> {
  using ConvertRegionToFunctionalPattern<
      CaseLikeRegionOp, CaseLikeOp>::ConvertRegionToFunctionalPattern;

  LogicalResult matchAndRewrite(CaseLikeRegionOp op,
                                PatternRewriter &rewriter) const override;
};

using ConvertCaseOp = ConvertCaseLikeOp<CaseRegionOp, CaseOp>;
using ConvertStatelessCaseOp =
    ConvertCaseLikeOp<StatelessCaseRegionOp, StatelessCaseOp>;
using ConvertStatefulCaseOp =
    ConvertCaseLikeOp<StatefulCaseRegionOp, StatefulCaseOp>;

// Base class for patterns to convert a while-like TFG region op to functional
// form.
template <typename WhileLikeRegionOp, typename WhileLikeOp>
struct ConvertWhileLikeOp
    : public ConvertRegionToFunctionalPattern<WhileLikeRegionOp, WhileLikeOp> {
  using ConvertRegionToFunctionalPattern<
      WhileLikeRegionOp, WhileLikeOp>::ConvertRegionToFunctionalPattern;

  LogicalResult matchAndRewrite(WhileLikeRegionOp op,
                                PatternRewriter &rewriter) const override;
};

using ConvertWhileOp = ConvertWhileLikeOp<WhileRegionOp, WhileOp>;
using ConvertStatelessWhileOp =
    ConvertWhileLikeOp<StatelessWhileRegionOp, StatelessWhileOp>;
using ConvertStatefulWhileOp =
    ConvertWhileLikeOp<StatefulWhileRegionOp, StatefulWhileOp>;

// Convert a region-based for-loop to a functional for-loop.
struct ConvertForOp
    : public ConvertRegionToFunctionalPattern<ForRegionOp, ForOp> {
  using ConvertRegionToFunctionalPattern<
      ForRegionOp, ForOp>::ConvertRegionToFunctionalPattern;

  LogicalResult matchAndRewrite(ForRegionOp op,
                                PatternRewriter &rewriter) const override;
};

}  // namespace

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

void BasePattern::CollectValuesDefinedAbove(Region &region,
                                            SetVector<Value> &datas,
                                            SetVector<Value> &ctls) const {
  ControlType control_ty = dialect_.getControlType();
  visitUsedValuesDefinedAbove(region, [&](OpOperand *operand) {
    Value value = operand->get();
    if (value.getType() != control_ty) {
      datas.insert(value);
    } else if (std::optional<Value> data = LookupDataValue(value)) {
      datas.insert(*data);
    } else {
      ctls.insert(value);
    }
  });
}

Operation *BasePattern::MakeChainConstant(Operation *parent, Value ctl,
                                          unsigned idx,
                                          PatternRewriter &rewriter) const {
  OperationName name("tfg.Const", ctl.getContext());
  OperationState state(ctl.getLoc(), name);
  IntegerType i32 = rewriter.getI32Type();
  ShapedType tensor_type = RankedTensorType::get({}, i32);
  state.addOperands(ctl);
  state.addAttribute("value", DenseElementsAttr::get(tensor_type, 0));
  state.addAttribute("dtype", TypeAttr::get(i32));
  state.addTypes({tensor_type, ctl.getType()});

  // Inherit `tfg.tpu_replicate`, `assigned_device`, and `device`.
  for (StringAttr attr_name : {StringAttr::get(ctx_, "_tpu_replicate"),
                               dialect_.getAssignedDeviceAttrIdentifier(),
                               dialect_.getDeviceAttrIdentifier()}) {
    if (Attribute attr = parent->getAttr(attr_name))
      state.addAttribute(attr_name, attr);
  }

  // Inherit a name based on the parent name.
  StringAttr name_id = dialect_.getNameAttrIdentifier();
  if (StringAttr name = parent->getAttrOfType<StringAttr>(name_id)) {
    auto const_name = rewriter.getStringAttr(
        name.getValue() + "_mlir_const_capture_" + Twine(idx));
    state.addAttribute(name_id, const_name);
  }

  return rewriter.create(state);
}

FailureOr<std::vector<Value>> BasePattern::CollectValuesDefinedAboveAll(
    RegionRange regions, PatternRewriter &rewriter) const {
  SetVector<Value> data_set, ctl_only;
  for (Region &region : llvm::make_pointee_range(regions))
    CollectValuesDefinedAbove(region, data_set, ctl_only);
  llvm::SmallVector<Value, 0> data_sv = data_set.takeVector();
  std::vector<Value> datas(data_sv.begin(), data_sv.end());
  // If in any of the regions we found a use of a control token defined above
  // the regions with no associated data value, then it cannot be converted to
  // explicit capture unless we insert chain constants. If this option was not
  // set, return failure because the region op cannot be converted.
  if (!force_control_capture_ && !ctl_only.empty()) return failure();

  Operation *parent = regions.front()->getParentOp();
  for (const auto &ctl : llvm::enumerate(ctl_only.takeVector())) {
    Operation *const_op =
        MakeChainConstant(parent, ctl.value(), ctl.index(), rewriter);
    for (Region *region : regions)
      replaceAllUsesInRegionWith(ctl.value(), const_op->getResult(1), *region);
    datas.push_back(const_op->getResult(0));
  }

  return datas;
}

void BasePattern::IsolateRegions(RegionRange regions,
                                 MutableArrayRef<Value> datas) const {
  ValueControlRetRange ctls(datas);
  Value data, ctl;
  for (Region &region : llvm::make_pointee_range(regions)) {
    for (auto it : llvm::zip(datas, ctls)) {
      std::tie(data, ctl) = it;
      util::LoopRegionArgumentUpdate result =
          util::LoopRegionAddArgument(region, data.getType());
      replaceAllUsesInRegionWith(data, result.data, region);
      replaceAllUsesInRegionWith(ctl, result.ctl, region);
    }
  }
}

NamedAttrList BasePattern::BuildAttributes(RegionAttr preserved,
                                           ValueRange arguments,
                                           ValueRange results,
                                           NameUniquer *name_uniquer) const {
  NamedAttrList attrs(preserved ? preserved.getAttrs() : DictionaryAttr());
  // The original function name is preserved in the region attributes, but don't
  // re-use it when creating a new function.
  attrs.erase(SymbolTable::getSymbolAttrName());

  SmallVector<Attribute> arg_attrs, res_attrs;
  ArrayAttr preserved_arg_attrs =
      preserved ? preserved.getArgAttrs() : ArrayAttr();
  ArrayAttr preserved_res_attrs =
      preserved ? preserved.getResAttrs() : ArrayAttr();

  // For each argument and result, lookup a name and regenerate output shapes.
  const auto build_attrs = [&](ArrayAttr attr, auto &it,
                               std::optional<ValueRange> args) {
    NamedAttrList attrs(attr ? attr[it.index()].template cast<DictionaryAttr>()
                             : DictionaryAttr());
    // If no name was preserved, try to find one.
    if (!attrs.get(ids_.tfg_name)) {
      if (StringAttr name = TryFindName(it.value(), args))
        attrs.set(ids_.tfg_name, name_uniquer->GetUniqued(name));
    }
    attrs.set(ids_.tfg_regenerate_output_shapes, UnitAttr::get(ctx_));
    return attrs.getDictionary(ctx_);
  };

  for (const auto &it : llvm::enumerate(arguments)) {
    arg_attrs.append({build_attrs(preserved_arg_attrs, it, {}),
                      DictionaryAttr::get(ctx_, {})});
  }
  for (const auto &it : llvm::enumerate(results))
    res_attrs.push_back(build_attrs(preserved_res_attrs, it, arguments));

  std::optional<RegisteredOperationName> name =
      RegisteredOperationName::lookup(GraphFuncOp::getOperationName(), ctx_);
  attrs.append(GraphFuncOp::getArgAttrsAttrName(*name),
               ArrayAttr::get(ctx_, arg_attrs));
  attrs.append(GraphFuncOp::getResAttrsAttrName(*name),
               ArrayAttr::get(ctx_, res_attrs));
  return attrs;
}

StringAttr BasePattern::TryFindName(Value value,
                                    std::optional<ValueRange> args) const {
  // If this is an op result, return the op's name.
  if (auto result = value.dyn_cast<OpResult>()) {
    Operation *op = result.getOwner();
    if (auto name =
            op->getAttrOfType<StringAttr>(dialect_.getNameAttrIdentifier())) {
      return StringAttr::get(ctx_, name.getValue() + "_tfg_result_" +
                                       Twine(result.getResultNumber()));
    }
    return {};
  }

  auto arg = value.cast<BlockArgument>();
  Operation *parent = arg.getOwner()->getParentOp();
  auto iface = dyn_cast<ControlArgumentInterface>(parent);
  if (!iface) return {};
  // If we were given a control token, lookup a name using the data value.
  if (arg.getType() == dialect_.getControlType())
    arg = iface.getDataValueOf(arg);
  // If the parent is a function, try to find a `tfg.name`.
  if (auto func = dyn_cast<GraphFuncOp>(*iface))
    return func.getArgAttrOfType<StringAttr>(arg.getArgNumber(), ids_.tfg_name);
  // Otherwise, "see through" to the corresponding operand.
  if (args) {
    assert(arg.getArgNumber() < args->size());
    return TryFindName((*args)[arg.getArgNumber()], {});
  }
  if (auto for_op = dyn_cast<ForRegionOp>(parent)) {
    unsigned arg_idx = arg.getArgNumber();
    if (arg_idx == 0) return TryFindName(for_op.getStart(), {});
    return TryFindName(for_op.getInit()[arg_idx - 1], {});
  }
  auto branch = cast<RegionBranchOpInterface>(parent);
  ValueRange inputs = branch.getEntrySuccessorOperands(arg.getParentRegion());
  return TryFindName(inputs[arg.getArgNumber()], {});
}

ArrayAttr BasePattern::GetControlRetAttrs(ValueRange ctls, ValueRange args,
                                          NameUniquer *name_uniquer) const {
  SmallVector<Attribute> ctl_ret_attrs;
  for (Value ctl : ctls) {
    NamedAttrList ctl_attrs;
    if (StringAttr name = TryFindName(ctl, args)) {
      ctl_attrs.set(dialect_.getTfgNameAttrIdentifier(),
                    name_uniquer->GetUniqued(name));
    }
    ctl_ret_attrs.push_back(ctl_attrs.getDictionary(ctx_));
  }
  return ArrayAttr::get(ctx_, ctl_ret_attrs);
}

GraphFuncOp BasePattern::CreateFunc(Location loc, const Twine &sym_name,
                                    Region &region, TypeRange res_types,
                                    NamedAttrList attrs) const {
  SmallVector<Type> arg_types;
  for (BlockArgument operand : GetLoopRegionDataArgs(region))
    arg_types.append({operand.getType(), dialect_.getControlType()});
  auto func_type = FunctionType::get(ctx_, arg_types, res_types);
  auto func = OpBuilder(ctx_).create<GraphFuncOp>(loc, sym_name, func_type,
                                                  /*generic=*/false);

  attrs.append(func->getAttrs());
  func->setAttrs(attrs.getDictionary(ctx_));

  SmallVector<BlockArgument> args =
      llvm::to_vector(GetLoopRegionDataArgs(region));
  SmallVector<BlockArgument> ctls =
      llvm::to_vector(GetLoopRegionControlTokens(region));
  // TODO(jeffniu): Change GraphFuncOp to use the same argument order as region
  // loop ops.
  for (auto it : llvm::zip(args, ctls)) {
    BlockArgument arg, ctl;
    std::tie(arg, ctl) = it;
    arg.replaceAllUsesWith(region.addArgument(arg.getType(), arg.getLoc()));
    ctl.replaceAllUsesWith(region.addArgument(ctl.getType(), ctl.getLoc()));
  }
  llvm::BitVector indices(region.getNumArguments());
  indices.set(0, args.size() * 2);
  region.front().eraseArguments(indices);

  func.getBody().takeBody(region);
  return func;
}

// Check the region attributes for a preserved function name
// TODO(jeffniu): RegionAttr should have an optional parameter for the function
// name, since it is treated differently from the other attributes.
static StringAttr GetFunctionName(RegionAttr preserved) {
  if (!preserved) return {};
  return preserved.getAttrs().getAs<StringAttr>(
      SymbolTable::getSymbolAttrName());
}

FuncAttr BasePattern::Outline(Operation *op, PatternRewriter &rewriter,
                              ValueRange args, Region &region,
                              RegionAttr preserved, DictionaryAttr attrs,
                              const Twine &func_name) const {
  // Create a name scope for the function.
  NameUniquer name_uniquer(ctx_);

  NamedAttrList func_attrs = BuildAttributes(
      preserved, args, cast<YieldOp>(region.front().getTerminator()).getArgs(),
      &name_uniquer);

  auto yield = cast<YieldOp>(region.front().getTerminator());
  SmallVector<Value> yieldArgs(yield.getArgs());
  rewriter.setInsertionPoint(yield);
  auto ret_op = rewriter.replaceOpWithNewOp<ReturnOp>(
      yield, yield.getOperands(),
      GetControlRetAttrs(yield.getCtls(), args, &name_uniquer));

  // Derive a function name. Use a default name. If a previous name exists,
  // use it. If the op also has a name, derive a name based on that.
  std::string new_func_name = func_name.str();
  if (StringAttr existing_name = GetFunctionName(preserved)) {
    new_func_name = existing_name.getValue().str();
    if (auto op_name =
            op->getAttrOfType<StringAttr>(dialect_.getNameAttrIdentifier())) {
      llvm::raw_string_ostream os(new_func_name);
      os << "_tfg_region_specialized_";
      for (char c : llvm::map_range(
               op_name.getValue(), [](char c) { return isalnum(c) ? c : '_'; }))
        os << c;
      os << '_' << llvm::to_string(region.getRegionNumber());
      os.flush();
    }
  }

  // Create the function.
  GraphFuncOp func = CreateFunc(op->getLoc(), new_func_name, region,
                                TFOp(ret_op).getNonControlOperands().getTypes(),
                                std::move(func_attrs));
  return FuncAttr::get(ctx_, table_.insert(func),
                       attrs ? attrs : DictionaryAttr::get(ctx_, {}));
}

template <typename FuncAttrT>
void BasePattern::ReuseAllOrOutline(
    Operation *op, PatternRewriter &rewriter, ValueRange args,
    ArrayRef<RegionFunction> regions,
    SmallVectorImpl<FuncAttrT> &functions) const {
  // Try to find reusable functions for all regions.
  const auto get_reusable_func = [this,
                                  &functions](const RegionFunction &func) {
    FuncAttr ref =
        FindReusableFunc(func.region, func.preserved_attrs, func.call_attrs);
    functions.push_back(ref);
    return ref;
  };
  if (llvm::all_of(regions, get_reusable_func)) return;

  // At least one region needs to be outlined.
  functions.clear();
  for (const RegionFunction &func : regions) {
    functions.push_back(Outline(op, rewriter, args, func.region,
                                func.preserved_attrs, func.call_attrs,
                                func.func_name));
  }
}

// Returns true if the region has any nested regions.
static bool HasNestedRegions(Region &region) {
  return llvm::any_of(region.getOps(),
                      [](Operation &op) { return op.getNumRegions(); });
}

// Check if the region is "equivalent" to the body of the given function, and so
// the function can be re-used when outlining the region. This compares
// (topologically) the arguments, results, and ops, ignoring the op names and
// checking for compatible types.
static bool RegionEqualTo(Region &region, GraphFuncOp func) {
  assert(!HasNestedRegions(region));
  assert(!HasNestedRegions(func.getBody()));

  // Outlining is performed "bottom-up". I.e. regions with no nested regions are
  // outlined first, which means that we will not have to worry about comparing
  // `While` to `WhileRegion`. Also, it means that we can directly compare the
  // operations.
  DenseMap<Value, Value> value_map;
  auto map_value = [&](Value lhs, Value rhs) {
    if (!tf_type::HasCompatibleElementTypes(lhs.getType(), rhs.getType()))
      return false;
    return value_map.insert({lhs, rhs}).first->second == rhs;
  };

  // Compare the non-control block arguments.
  if (region.getNumArguments() != func.getNumArguments()) return false;
  for (const auto &it : llvm::enumerate(GetLoopRegionDataArgs(region))) {
    Value rhs = GraphFuncOp::getDataValue(func.getBody(), it.index());
    if (!map_value(it.value(), rhs)) return false;
  }

  // Compare the bodies except the terminators. We can't use
  // OperationEquivalence due to relaxed type equality.
  auto map_value_range = [](ValueRange lhs_range, ValueRange rhs_range,
                            auto map_value) {
    if (lhs_range.size() != rhs_range.size()) return false;
    for (auto it : llvm::zip(lhs_range, rhs_range))
      if (!map_value(std::get<0>(it), std::get<1>(it))) return false;
    return true;
  };

  StringAttr name_id =
      cast<TFGraphDialect>(func->getDialect())->getNameAttrIdentifier();

  auto compare_ops = [&](Operation &lhs, Operation &rhs) {
    if (lhs.getName() != rhs.getName()) return false;

    DictionaryAttr lhs_attrs = lhs.getAttrDictionary();
    DictionaryAttr rhs_attrs = rhs.getAttrDictionary();
    if (lhs_attrs.size() != rhs_attrs.size()) return false;
    for (auto it : llvm::zip(lhs_attrs, rhs_attrs)) {
      NamedAttribute lhs_attr = std::get<0>(it);
      NamedAttribute rhs_attr = std::get<1>(it);
      if (lhs_attr.getName() != rhs_attr.getName()) return false;
      if (lhs_attr.getName() == name_id) continue;
      if (lhs_attr.getValue() != rhs_attr.getValue()) return false;
    }
    if (!map_value_range(lhs.getOperands(), rhs.getOperands(), map_value))
      return false;
    if (!map_value_range(lhs.getResults(), rhs.getResults(), map_value))
      return false;
    assert(!lhs.getNumRegions() && !rhs.getNumRegions());
    return true;
  };
  if (!llvm::all_of_zip(region.front().without_terminator(),
                        func.getBody().front().without_terminator(),
                        compare_ops))
    return false;

  // Compare just the operands of the terminators.
  auto return_op = cast<ReturnOp>(func.getBody().front().getTerminator());
  Operation *terminator = region.front().getTerminator();
  if (auto yield = dyn_cast<YieldOp>(terminator)) {
    return map_value_range(yield->getOperands(), return_op->getOperands(),
                           map_value);
  } else {
    auto cond = cast<ConditionOp>(terminator);
    return map_value(cond.getCond(), return_op->getOperand(0)) &&
           map_value_range(
               cond.getCtls(),
               return_op->getOperands().slice(1, cond.getCtls().size()),
               map_value);
  }
}

FuncAttr BasePattern::FindReusableFunc(Region &region, RegionAttr preserved,
                                       DictionaryAttr attrs) const {
  StringAttr name = GetFunctionName(preserved);
  if (!name) return {};
  auto func = table_.lookup<GraphFuncOp>(name);
  if (!func) return {};
  if (!RegionEqualTo(region, func)) return {};
  return FuncAttr::get(region.getContext(), name.getValue(),
                       attrs ? attrs : DictionaryAttr::get(ctx_, {}));
}

bool BasePattern::FuncHasNestedRegions(RegionAttr preserved) const {
  StringAttr name = GetFunctionName(preserved);
  if (!name) return false;
  auto func = table_.lookup<GraphFuncOp>(name);
  return func && HasNestedRegions(func.getBody());
}

//===----------------------------------------------------------------------===//
// ConvertToExplicitCapture
//===----------------------------------------------------------------------===//

template <typename OpT>
FailureOr<std::pair<OpT, std::vector<Value>>>
ConvertToExplicitCapture<OpT>::Run(OpT op, PatternRewriter &rewriter) {
  FailureOr<std::vector<Value>> operands =
      this->CollectValuesDefinedAboveAll(op->getRegions(), rewriter);
  if (failed(operands)) return failure();
  this->IsolateRegions(op->getRegions(), *operands);
  OpT new_op = RebuildWith(op, *operands, rewriter);
  util::ForwardNonIntrinsicAttributes(op, new_op);
  for (auto it : llvm::zip(op->getRegions(), new_op->getRegions()))
    std::get<1>(it).takeBody(std::get<0>(it));
  rewriter.replaceOp(op, new_op->getResults().slice(0, op->getNumResults()));
  return std::make_pair(new_op, std::move(*operands));
}

//===----------------------------------------------------------------------===//
// ConvertIfLikeOp
//===----------------------------------------------------------------------===//

template <typename IfLikeRegionOp, typename IfLikeOp>
LogicalResult ConvertIfLikeOp<IfLikeRegionOp, IfLikeOp>::matchAndRewrite(
    IfLikeRegionOp op, PatternRewriter &rewriter) const {
  if (HasNestedRegions(op.getThenRegion()) ||
      HasNestedRegions(op.getElseRegion()))
    return failure();
  if (this->FuncHasNestedRegions(op.getThenRegionAttrsAttr()) ||
      this->FuncHasNestedRegions(op.getElseRegionAttrsAttr()))
    return failure();

  // Convert the op to explicit capture.
  ConvertIfLikeRegionOpToExplicitCapture<IfLikeRegionOp> converter(
      this->dialect_, this->table_, this->force_control_capture_, this->ids_);
  auto result = converter.Run(op, rewriter);
  if (failed(result)) return failure();
  std::vector<Value> args;
  std::tie(op, args) = std::move(*result);

  // Outline the regions.
  SmallVector<FuncAttr, 2> branches;
  this->ReuseAllOrOutline(op, rewriter, args,
                          {{op.getThenRegion(), op.getThenRegionAttrsAttr(),
                            op.getThenAttrsAttr(), "if_then_function"},
                           {op.getElseRegion(), op.getElseRegionAttrsAttr(),
                            op.getElseAttrsAttr(), "if_else_function"}},
                          branches);

  // Build the functional if-like op.
  SmallVector<Value> operands = llvm::to_vector(args);
  llvm::append_range(operands, op.getCtls());

  rewriter.setInsertionPoint(op);
  auto func_op =
      rewriter.create<IfLikeOp>(op.getLoc(), op.getResultTypes(), op.getCond(),
                                operands, branches[0], branches[1]);
  util::ForwardNonIntrinsicAttributes(op, func_op);
  rewriter.replaceOp(op, func_op.getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// ConvertCaseLikeOp
//===----------------------------------------------------------------------===//

template <typename CaseLikeRegionOp, typename CaseLikeOp>
LogicalResult ConvertCaseLikeOp<CaseLikeRegionOp, CaseLikeOp>::matchAndRewrite(
    CaseLikeRegionOp op, PatternRewriter &rewriter) const {
  if (llvm::any_of(op.getBranches(), HasNestedRegions)) return failure();
  if (ArrayAttr preserved = op.getRegionAttrsAttr()) {
    if (llvm::any_of(preserved.getAsRange<RegionAttr>(), [&](auto preserved) {
          return this->FuncHasNestedRegions(preserved);
        }))
      return failure();
  }

  // Convert the op to explicit capture.
  ConvertCaseLikeRegionOpToExplicitCapture<CaseLikeRegionOp> converter(
      this->dialect_, this->table_, this->force_control_capture_, this->ids_);
  auto result = converter.Run(op, rewriter);
  if (failed(result)) return failure();
  std::vector<Value> args;
  std::tie(op, args) = std::move(*result);

  // Outline the regions.
  ArrayAttr branch_func_attrs = op.getBranchAttrsAttr();
  SmallVector<BasePattern::RegionFunction> branch_regions;
  for (const auto &it : llvm::enumerate(op.getBranches())) {
    unsigned idx = it.index();
    // Get the preserved attributes, if there are any.
    RegionAttr preserved =
        op.getRegionAttrs()
            ? op.getRegionAttrsAttr()[idx].template cast<RegionAttr>()
            : nullptr;
    DictionaryAttr attrs =
        branch_func_attrs
            ? branch_func_attrs[idx].template cast<DictionaryAttr>()
            : nullptr;
    branch_regions.push_back(BasePattern::RegionFunction{
        it.value(), preserved, attrs, ("case_function_" + Twine(idx)).str()});
  }
  SmallVector<Attribute> branches;
  this->ReuseAllOrOutline(op, rewriter, args, branch_regions, branches);

  // Build the functional case-like op.
  SmallVector<Value> operands = llvm::to_vector(args);
  llvm::append_range(operands, op.getCtls());

  rewriter.setInsertionPoint(op);
  auto func_op = rewriter.create<CaseLikeOp>(op.getLoc(), op.getResultTypes(),
                                             op.getBranchIndex(), operands,
                                             rewriter.getArrayAttr(branches));
  util::ForwardNonIntrinsicAttributes(op, func_op);
  rewriter.replaceOp(op, func_op.getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// ConvertWhileLikeOp
//===----------------------------------------------------------------------===//

template <typename WhileLikeRegionOp, typename WhileLikeOp>
LogicalResult
ConvertWhileLikeOp<WhileLikeRegionOp, WhileLikeOp>::matchAndRewrite(
    WhileLikeRegionOp op, PatternRewriter &rewriter) const {
  if (HasNestedRegions(op.getCondRegion()) ||
      HasNestedRegions(op.getBodyRegion()))
    return failure();
  if (this->FuncHasNestedRegions(op.getCondRegionAttrsAttr()) ||
      this->FuncHasNestedRegions(op.getBodyRegionAttrsAttr()))
    return failure();

  // Convert the op to explicit capture.
  ConvertWhileLikeRegionOpToExplicitCapture<WhileLikeRegionOp> converter(
      this->dialect_, this->table_, this->force_control_capture_, this->ids_);
  auto result = converter.Run(op, rewriter);
  if (failed(result)) return failure();
  op = result->first;

  // Try to find re-usable functions for both the condition and body regions.
  FuncAttr body_ref = this->FindReusableFunc(
      op.getBodyRegion(), op.getBodyRegionAttrsAttr(), op.getBodyAttrsAttr());
  FuncAttr cond_ref = this->FindReusableFunc(
      op.getCondRegion(), op.getCondRegionAttrsAttr(), op.getCondAttrsAttr());

  // If a function for either region could not be re-used, outline them out.
  if (!body_ref || !cond_ref) {
    // Handle the condition region. Unlike other regions, the terminator is
    // special and the function only has one result.
    ConditionOp cond_op = op.getCondCondition();
    // Create a name scope for the condition function.
    NameUniquer name_uniquer(this->ctx_);
    // Create the function.

    NamedAttrList cond_attrs =
        this->BuildAttributes(op.getCondRegionAttrsAttr(), op.getInit(),
                              cond_op.getCond(), &name_uniquer);
    GraphFuncOp cond_func =
        this->CreateFunc(op.getLoc(), "while_cond_function", op.getCondRegion(),
                         cond_op.getCond().getType(), std::move(cond_attrs));

    // Replace the condition terminator.
    rewriter.setInsertionPoint(cond_op);
    SmallVector<Value> cond_rets = {cond_op.getCond()};
    llvm::append_range(cond_rets, cond_op.getCtls());
    rewriter.replaceOpWithNewOp<ReturnOp>(
        cond_op, cond_rets,
        this->GetControlRetAttrs(cond_op.getCtls(), op.getInit(),
                                 &name_uniquer));
    // Insert the function and grab a reference.
    cond_ref = FuncAttr::get(
        op.getContext(), this->table_.insert(cond_func),
        op.getCondAttrs().value_or(rewriter.getDictionaryAttr({})));

    // Outline the body.
    body_ref = this->Outline(op, rewriter, op.getInit(), op.getBodyRegion(),
                             op.getBodyRegionAttrsAttr(), op.getBodyAttrsAttr(),
                             "while_body_function");
  }

  // Create the functional op.
  SmallVector<Value> operands = llvm::to_vector(op.getInit());
  llvm::append_range(operands, op.getCtls());

  rewriter.setInsertionPoint(op);
  auto func_op = rewriter.create<WhileLikeOp>(op.getLoc(), op.getResultTypes(),
                                              operands, cond_ref, body_ref,
                                              op.getParallelIterationsAttr());
  util::ForwardNonIntrinsicAttributes(op, func_op);
  rewriter.replaceOp(op, func_op.getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// ConvertForOp
//===----------------------------------------------------------------------===//

LogicalResult ConvertForOp::matchAndRewrite(ForRegionOp op,
                                            PatternRewriter &rewriter) const {
  if (HasNestedRegions(op.getBodyRegion())) return failure();
  if (this->FuncHasNestedRegions(op.getRegionAttrsAttr())) return failure();

  // Convert the op to explicit capture.
  ConvertForRegionOpToExplicitCapture converter(dialect_, table_,
                                                force_control_capture_, ids_);
  auto result = converter.Run(op, rewriter);
  if (failed(result)) return failure();
  op = result->first;

  // Outline to body.
  SmallVector<Value> func_args(/*Size=*/1, op.getStart());
  llvm::append_range(func_args, op.getInit());
  SmallVector<FuncAttr, 1> body_ref;
  ReuseAllOrOutline(op, rewriter, func_args,
                    {{op.getBodyRegion(), op.getRegionAttrsAttr(),
                      op.getBodyAttrsAttr(), "for_body_function"}},
                    body_ref);

  // Create the functional op.
  SmallVector<Value> operands = llvm::to_vector(op.getInit());
  llvm::append_range(operands, op.getCtls());

  rewriter.setInsertionPoint(op);
  auto func_op = rewriter.create<tfg::ForOp>(
      op.getLoc(), op.getResultTypes(), op.getStart(), op.getLimit(),
      op.getDelta(), operands, body_ref[0]);
  util::ForwardNonIntrinsicAttributes(op, func_op);
  rewriter.replaceOp(op, func_op.getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void PopulateRegionToFunctionalPatterns(RewritePatternSet &patterns,
                                        SymbolTable &table,
                                        bool force_control_capture) {
  auto *dialect = patterns.getContext()->getOrLoadDialect<TFGraphDialect>();
  patterns.insert<ConvertIfOp, ConvertStatelessIfOp, ConvertStatefulIfOp,
                  ConvertCaseOp, ConvertStatelessCaseOp, ConvertStatefulCaseOp,
                  ConvertWhileOp, ConvertStatelessWhileOp,
                  ConvertStatefulWhileOp, ConvertForOp>(
      patterns.getContext(), *dialect, table, force_control_capture,
      CachedIdentifiers(dialect));
}

}  // namespace tfg
}  // namespace mlir
