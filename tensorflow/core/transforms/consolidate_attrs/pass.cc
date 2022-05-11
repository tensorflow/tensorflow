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

#include "tensorflow/core/transforms/consolidate_attrs/pass.h"

#include <memory>
#include <utility>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/ir/tf_op_wrapper.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/ir/utility.h"
#include "tensorflow/core/transforms/pass_detail.h"

namespace mlir {
namespace tfg {

static const char *kRegenerateOutputShapes = "tfg.regenerate_output_shapes";

// Returns true if an attribute is an array of shapes;
static bool IsArrayOfShapes(ArrayAttr array) {
  return llvm::all_of(array,
                      [](Attribute attr) { return attr.isa<ShapeAttr>(); });
}

// Given a tensor type and shape information, try to refine the type.
static Type GetReifiedType(Type orig, ShapeAttr shape) {
  Type element_type = orig.cast<ShapedType>().getElementType();
  TensorType inferred;
  if (shape.hasRank()) {
    // Replace dimensions less than -1 with ?
    SmallVector<int64_t> dims = llvm::to_vector(shape.getShape());
    for (int64_t &dim : dims)
      if (dim < -1) dim = -1;
    inferred = RankedTensorType::get(dims, element_type);
  } else {
    inferred = UnrankedTensorType::get(element_type);
  }
  Type reified_type = tf_type::GetCastCompatibleType(inferred, orig);
  // If the types are not compatible, return the original type.
  return reified_type ? reified_type : orig;
}

namespace {
// CRTP base class for consolidate attribute passes. This base class defines
// cached identifiers for the attributes.
template <typename PassT>
class AttributesPassBase : public PassWrapper<PassT, OperationPass<>> {
 public:
  LogicalResult initialize(MLIRContext *context) override {
    input_shapes_id_ = StringAttr::get(context, "tf._input_shapes");
    regenerate_input_shapes_id_ =
        StringAttr::get(context, "tfg.regenerate_input_shapes");
    output_shapes_id_ = StringAttr::get(context, "tf._output_shapes");
    regenerate_output_shapes_id_ =
        StringAttr::get(context, "tfg.regenerate_output_shapes");
    handle_data_id_ = StringAttr::get(context, "tfg.handle_data");
    dtype_id_ = StringAttr::get(context, "tfg.dtype");
    is_ref_id_ = StringAttr::get(context, "tfg.is_ref");
    control_type_ = ControlType::get(context);
    return success();
  }

 protected:
  // Identifier for `tf._input_shapes`.
  StringAttr input_shapes_id_;
  // Identifier for `tf._regenerate_input_shapes`.
  StringAttr regenerate_input_shapes_id_;
  // Identifier for `tf._output_shapes`.
  StringAttr output_shapes_id_;
  // Identifier for `tf._regenerate_output_shapes`.
  StringAttr regenerate_output_shapes_id_;
  // Identifier for `tfg.handle_data`.
  StringAttr handle_data_id_;
  // Identifier for `tfg.dtype`.
  StringAttr dtype_id_;
  // Identifier for `tfg.is_ref`.
  StringAttr is_ref_id_;
  // Cacched control type.
  ControlType control_type_;
};

class ConsolidateAttributesPassImpl
    : public AttributesPassBase<ConsolidateAttributesPassImpl> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConsolidateAttributesPassImpl)

  void runOnOperation() override;

 private:
  // Reify `tf._input_shapes`, `tf._output_shapes` and `tfg.handle_data` into
  // the types of the function arguments. Drop the attributes `tfg.dtype` and
  // `tfg.is_ref`. Return the new argument attributes.
  ArrayAttr reifyAndDropFunctionArgumentAttributes(GraphFuncOp func);
  // Reify `tf._output_shapes` and `tfg.handle_data` into the types of the
  // function results. Drop the attribute `tfg.dtype`. Return the new result
  // attributes.
  ArrayAttr reifyAndDropFunctionResultAttributes(GraphFuncOp func);

  // Refine a type with `tf._output_shapes`.
  Type refineTypeWithOutputShapes(Type type, NamedAttrList &attrs);
  // Refine a type with `tfg.handle_data`.
  Type refineTypeWithHandleData(Type type, Attribute handle_data);
};
}  // namespace

Type ConsolidateAttributesPassImpl::refineTypeWithOutputShapes(
    Type type, NamedAttrList &attrs) {
  // Get the output shapes attribute. If the attribute is not an array of
  // exactly one shape, ignore it.
  if (auto output_shapes =
          attrs.get(output_shapes_id_).dyn_cast_or_null<ArrayAttr>()) {
    if (output_shapes.size() == 1 && IsArrayOfShapes(output_shapes)) {
      attrs.erase(output_shapes_id_);
      attrs.set(regenerate_output_shapes_id_, UnitAttr::get(&getContext()));
      return GetReifiedType(type, output_shapes[0].cast<ShapeAttr>());
    }
  }
  return type;
}

Type ConsolidateAttributesPassImpl::refineTypeWithHandleData(
    Type type, Attribute handle_data) {
  if (!handle_data) return type;
  SmallVector<TensorType> subtypes;
  // Because `tfg.handle_data` is a TFG internal attribute, it will be
  // well-formed.
  for (Type type : handle_data.cast<ArrayAttr>().getAsValueRange<TypeAttr>())
    subtypes.push_back(type.cast<TensorType>());
  auto resource =
      UnrankedTensorType::get(ResourceType::get(subtypes, &getContext()));
  Type reified = tf_type::GetCastCompatibleType(resource, type);
  return reified ? reified : type;
}

ArrayAttr ConsolidateAttributesPassImpl::reifyAndDropFunctionArgumentAttributes(
    GraphFuncOp func) {
  // Get the input shapes attribute. If it is a UnitAttr, then it is empty and
  // we will ignore it. If it isn't an array of shapes or has an inconsistent
  // number of shapes, ignore it.
  ArrayAttr input_shapes =
      func->getAttr(input_shapes_id_).dyn_cast_or_null<ArrayAttr>();
  unsigned num_args = func.getNumArguments() / 2;
  if (input_shapes) {
    if (input_shapes.size() != num_args || !IsArrayOfShapes(input_shapes)) {
      input_shapes = {};
    } else {
      func->removeAttr(input_shapes_id_);
      func->setAttr(regenerate_input_shapes_id_, UnitAttr::get(&getContext()));
    }
  }

  SmallVector<Attribute> arg_attrs;
  auto empty_dict = DictionaryAttr::get(&getContext());
  for (auto i : llvm::seq<unsigned>(0, num_args)) {
    BlockArgument arg = GraphFuncOp::getDataValue(func.body(), i);
    NamedAttrList attrs(func.getArgAttrs(arg.getArgNumber()));
    Type arg_type = arg.getType();
    arg_type = refineTypeWithOutputShapes(arg_type, attrs);
    arg_type = refineTypeWithHandleData(arg_type, attrs.erase(handle_data_id_));
    if (input_shapes)
      arg_type = GetReifiedType(arg_type, input_shapes[i].cast<ShapeAttr>());
    arg.setType(arg_type);
    attrs.erase(dtype_id_);
    attrs.erase(is_ref_id_);
    arg_attrs.append({attrs.getDictionary(&getContext()), empty_dict});
  }
  return ArrayAttr::get(&getContext(), arg_attrs);
}

ArrayAttr ConsolidateAttributesPassImpl::reifyAndDropFunctionResultAttributes(
    GraphFuncOp func) {
  SmallVector<Attribute> ret_attrs;
  // The result types are propagated to the data operands to `return`.
  auto ret_op = cast<ReturnOp>(func.body().front().getTerminator());
  for (auto &it :
       llvm::enumerate(func.getAllResultAttrs().getAsRange<DictionaryAttr>())) {
    NamedAttrList attrs(it.value());
    Value ret = ret_op.getOperand(it.index());
    Type ret_type = ret.getType();
    ret_type = refineTypeWithOutputShapes(ret_type, attrs);
    ret_type = refineTypeWithHandleData(ret_type, attrs.erase(handle_data_id_));
    ret.setType(ret_type);
    attrs.erase(dtype_id_);
    ret_attrs.push_back(attrs.getDictionary(&getContext()));
  }
  return ArrayAttr::get(&getContext(), ret_attrs);
}

namespace {
// This pattern reifies an op's result shape info into the result types and
// drops the output shapes attributes.
class ReifyOperationOutputShapes : public RewritePattern {
 public:
  ReifyOperationOutputShapes(MLIRContext *context, PatternBenefit benefit,
                             StringRef attr_name)
      : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context),
        output_shapes_id_(StringAttr::get(context, attr_name)) {}

  // Returns true if this instance of the pattern should match the op.
  virtual bool shouldMatch(Operation *op) const = 0;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!shouldMatch(op)) return failure();

    ResultRange results = TFOp(op).getNonControlResults();

    // Get the output shapes attribute. Ignore it if it is not an array
    // attribute, if it has an inconsistent number of shapes, or if it is not
    // an array of shapes.
    ArrayAttr output_shapes =
        op->getAttr(output_shapes_id_).dyn_cast_or_null<ArrayAttr>();
    if (!output_shapes || results.size() != output_shapes.size() ||
        !IsArrayOfShapes(output_shapes))
      return failure();

    rewriter.updateRootInPlace(op, [&] {
      op->removeAttr(output_shapes_id_);
      assert(output_shapes.size() == results.size());
      for (auto it :
           llvm::zip(results, output_shapes.getAsRange<ShapeAttr>())) {
        Value result = std::get<0>(it);
        result.setType(GetReifiedType(result.getType(), std::get<1>(it)));
      }
      rewriteImpl(op, rewriter);
    });
    return success();
  }

  virtual void rewriteImpl(Operation *op, PatternRewriter &rewriter) const {}

 private:
  // Identifier for `_output_shapes`.
  StringAttr output_shapes_id_;
};

// This pattern matches and TFG op and reifies `_output_shapes`. The pattern
// leaves behind an attribute `_regenerate_output_shapes` that is used by the
// converse pattern to detect whether the attribute should be materialized.
class ReifyTFGOpOutputShapes : public ReifyOperationOutputShapes {
 public:
  explicit ReifyTFGOpOutputShapes(MLIRContext *context)
      : ReifyOperationOutputShapes(context, /*benefit=*/1, "_output_shapes"),
        dialect_(context->getOrLoadDialect<TFGraphDialect>()),
        regenerate_output_shapes_id_(
            StringAttr::get(context, kRegenerateOutputShapes)) {}

  bool shouldMatch(Operation *op) const override {
    return op->getDialect() == dialect_ && op->getNumResults();
  }

  void rewriteImpl(Operation *op, PatternRewriter &rewriter) const override {
    op->setAttr(regenerate_output_shapes_id_, rewriter.getUnitAttr());
  }

 private:
  // Cached TFG dialect instance.
  TFGraphDialect *dialect_;
  // Identifier to `_regenerate_output_shapes`.
  StringAttr regenerate_output_shapes_id_;
};

// This pattern matches `If`, `Case`, and `While` and reifies their
// `output_shapes` attribute.
struct ReifyCFOpOutputShapes : public ReifyOperationOutputShapes {
  // Set a higher benefit to ensure that "output_shapes" is reified before
  // "_output_shapes".
  explicit ReifyCFOpOutputShapes(MLIRContext *context)
      : ReifyOperationOutputShapes(context, /*benefit=*/2, "output_shapes") {}

  bool shouldMatch(Operation *op) const override {
    return isa<IfOp, StatelessIfOp, StatefulIfOp, CaseOp, StatelessCaseOp,
               StatefulCaseOp, WhileOp, StatelessWhileOp, StatefulWhileOp>(op);
  }
};

// This pattern removes a list of attributes from the given op types.
template <typename... OpTs>
class DropAttributes : public RewritePattern {
 public:
  // Create the pattern. Specify which attributes to remove.
  DropAttributes(MLIRContext *context, ArrayRef<StringRef> attr_names)
      : RewritePattern(Pattern::MatchAnyOpTypeTag(), /*benefit=*/1, context) {
    for (StringRef attr_name : attr_names)
      attr_ids_.push_back(StringAttr::get(context, attr_name));
  }

  // Remove the specified attributes from the op. Fail if none of the attributes
  // were present.
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isa<OpTs...>(op)) return failure();
    rewriter.startRootUpdate(op);
    if (!llvm::count_if(attr_ids_, [&](StringAttr attr_id) {
          return op->removeAttr(attr_id);
        })) {
      rewriter.cancelRootUpdate(op);
      return failure();
    }
    rewriter.finalizeRootUpdate(op);
    return success();
  }

 private:
  // The identifiers of the attributes to remove.
  SmallVector<StringAttr> attr_ids_;
};
}  // namespace

template <typename... OpTs>
static std::unique_ptr<RewritePattern> RemoveAttributes(
    MLIRContext *context, ArrayRef<StringRef> attr_names) {
  return std::make_unique<DropAttributes<OpTs...>>(context, attr_names);
}

void ConsolidateAttributesPassImpl::runOnOperation() {
  // Skip this pass on generic functions. Generic functions contain only opaque
  // tensor types, into which shape and data type info cannot be reified.
  auto func = dyn_cast<GraphFuncOp>(getOperation());
  if (func && func.generic()) return;

  // Reify operation attributes.
  RewritePatternSet patterns(&getContext());
  patterns.insert<ReifyTFGOpOutputShapes, ReifyCFOpOutputShapes>(&getContext());
  patterns.add(RemoveAttributes<IfOp, StatelessIfOp, StatefulIfOp>(
      &getContext(), {"Tcond", "Tin", "Tout"}));
  patterns.add(RemoveAttributes<CaseOp, StatelessCaseOp, StatefulCaseOp>(
      &getContext(), {"Tin", "Tout"}));
  patterns.add(
      RemoveAttributes<WhileOp, StatelessWhileOp, StatefulWhileOp, ForOp>(
          &getContext(), {"T"}));
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    getOperation()->emitError(getArgument() + " pass failed");
    signalPassFailure();
    return;
  }

  // If the pass was run on a function, reify its attributes and then rebuild
  // the signature. Because the attributes may have conflicting type info, the
  // order in which we visit the attributes is the priority.
  if (!func) return;
  ArrayAttr arg_attrs = reifyAndDropFunctionArgumentAttributes(func);
  ArrayAttr res_attrs = reifyAndDropFunctionResultAttributes(func);
  Block &body = func.body().front();
  auto type = FunctionType::get(
      &getContext(), body.getArgumentTypes(),
      TFOp(body.getTerminator()).getNonControlOperands().getTypes());
  NamedAttrList attrs(func->getAttrDictionary());
  attrs.set(func.function_typeAttrName(), TypeAttr::get(type));
  attrs.set(func.arg_attrsAttrName(), arg_attrs);
  attrs.set(func.res_attrsAttrName(), res_attrs);
  func->setAttrs(attrs.getDictionary(&getContext()));
}

namespace {
class PrepareAttributesForExportPassImpl
    : public AttributesPassBase<PrepareAttributesForExportPassImpl> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      PrepareAttributesForExportPassImpl)

  void runOnOperation() override;

 private:
  // Materialize required `tfg.` attributes for export. Also, adds
  // `tf._input_shapes` to the function attributes. And `tf._output_shapes` and
  // `tf._handle_data` to the argument and result attributes.
  void prepareFunctionAttributes(GraphFuncOp func);

  // Prepare attributes for a single type.
  DictionaryAttr prepareAttributesFor(Type type, DictionaryAttr attr_dict);
};
}  // namespace

void PrepareAttributesForExportPassImpl::prepareFunctionAttributes(
    GraphFuncOp func) {
  NamedAttrList attrs(func->getAttrDictionary());
  SmallVector<Attribute> input_shapes, arg_attrs, res_attrs;
  for (auto it :
       llvm::zip(func.getArgumentTypes(),
                 func.getAllArgAttrs().getAsRange<DictionaryAttr>())) {
    Type type = std::get<0>(it);
    DictionaryAttr attrs = std::get<1>(it);
    if (type == control_type_) {
      arg_attrs.push_back(attrs);
      continue;
    }
    arg_attrs.push_back(prepareAttributesFor(type, attrs));
    if (auto ranked = type.dyn_cast<RankedTensorType>()) {
      input_shapes.push_back(ShapeAttr::get(&getContext(), ranked.getShape()));
    } else {
      input_shapes.push_back(ShapeAttr::get(&getContext(), llvm::None));
    }
  }
  for (auto it :
       llvm::zip(func.getResultTypes(),
                 func.getAllResultAttrs().getAsRange<DictionaryAttr>()))
    res_attrs.push_back(prepareAttributesFor(std::get<0>(it), std::get<1>(it)));

  // Add input shapes only if its regeneration is required.
  if (attrs.erase(regenerate_input_shapes_id_))
    attrs.set(input_shapes_id_, ArrayAttr::get(&getContext(), input_shapes));
  attrs.set(func.arg_attrsAttrName(), ArrayAttr::get(&getContext(), arg_attrs));
  attrs.set(func.res_attrsAttrName(), ArrayAttr::get(&getContext(), res_attrs));
  func->setAttrs(attrs.getDictionary(&getContext()));
}

DictionaryAttr PrepareAttributesForExportPassImpl::prepareAttributesFor(
    Type type, DictionaryAttr attr_dict) {
  NamedAttrList attrs(attr_dict);
  // Add shape data if requested.
  if (attrs.erase(regenerate_output_shapes_id_)) {
    auto shape = ShapeAttr::get(&getContext(),
                                type.isa<RankedTensorType>()
                                    ? type.cast<RankedTensorType>().getShape()
                                    : Optional<ArrayRef<int64_t>>());
    attrs.set(output_shapes_id_, ArrayAttr::get(&getContext(), {shape}));
  }
  auto element_type = type.cast<TensorType>().getElementType();
  if (auto resource = element_type.dyn_cast<ResourceType>()) {
    SmallVector<Attribute> handle_data;
    for (TensorType subtype : resource.getSubtypes())
      handle_data.push_back(TypeAttr::get(subtype));
    // Only bother adding handle data if there are subtypes.
    if (!handle_data.empty())
      attrs.set(handle_data_id_, ArrayAttr::get(&getContext(), handle_data));
  }
  if (element_type.isa<tf_type::TensorFlowRefType>())
    attrs.set(is_ref_id_, UnitAttr::get(&getContext()));
  return attrs.getDictionary(&getContext());
}

// Get the element types of the values as an array attributes.
static ArrayAttr GetElementTypesAttr(PatternRewriter &rewriter,
                                     ValueRange values) {
  SmallVector<Attribute> types;
  for (Value value : values) {
    types.push_back(
        TypeAttr::get(value.getType().cast<TensorType>().getElementType()));
  }
  return rewriter.getArrayAttr(types);
}

namespace {
// Base class for patterns that materialize control-flow op attributes. This
// patterns contains a cached control type.
template <typename OpT>
class MaterializeAttrsPattern : public OpRewritePattern<OpT> {
 public:
  // Create the pattern with a cached control type instance.
  explicit MaterializeAttrsPattern(ControlType control_type)
      : OpRewritePattern<OpT>(control_type.getContext()),
        control_type_(control_type) {}

  // Get an array of the element types of the data arguments of the op. The
  // arguments exclude "op-specific" operands such as if condition, case branch
  // index, and for loop indices.
  ArrayAttr getArgumentElementTypesAttr(PatternRewriter &rewriter,
                                        OpT op) const {
    return GetElementTypesAttr(
        rewriter, SplitDataAndControlValues(op.args(), control_type_).first);
  }

 private:
  // The cached control type.
  ControlType control_type_;
};

template <typename IfLikeOp>
struct MaterializeIfAttrs : public MaterializeAttrsPattern<IfLikeOp> {
  using MaterializeAttrsPattern<IfLikeOp>::MaterializeAttrsPattern;

  // Materialize `Tcond`, `Tin`, and `Tout`.
  LogicalResult matchAndRewrite(IfLikeOp op,
                                PatternRewriter &rewriter) const override {
    if (op.Tcond() && op.Tin() && op.Tout()) return failure();
    NamedAttrList attrs(op->getAttrDictionary());
    attrs.set(
        op.TcondAttrName(),
        TypeAttr::get(
            op.cond().getType().template cast<TensorType>().getElementType()));
    attrs.set(op.TinAttrName(),
              this->getArgumentElementTypesAttr(rewriter, op));
    attrs.set(op.ToutAttrName(), GetElementTypesAttr(rewriter, op.outs()));
    rewriter.updateRootInPlace(
        op, [&] { op->setAttrs(attrs.getDictionary(op->getContext())); });
    return success();
  }
};

template <typename CaseLikeOp>
struct MaterializeCaseAttrs : public MaterializeAttrsPattern<CaseLikeOp> {
  using MaterializeAttrsPattern<CaseLikeOp>::MaterializeAttrsPattern;

  // Materialize `Tin` and `Tout`.
  LogicalResult matchAndRewrite(CaseLikeOp op,
                                PatternRewriter &rewriter) const override {
    if (op.Tin() && op.Tout()) return failure();
    NamedAttrList attrs(op->getAttrDictionary());
    attrs.set(op.TinAttrName(),
              this->getArgumentElementTypesAttr(rewriter, op));
    attrs.set(op.ToutAttrName(), GetElementTypesAttr(rewriter, op.outs()));
    rewriter.updateRootInPlace(
        op, [&] { op->setAttrs(attrs.getDictionary(op->getContext())); });
    return success();
  }
};

template <typename WhileOrForLikeOp>
struct MaterializeTAttr : public MaterializeAttrsPattern<WhileOrForLikeOp> {
  using MaterializeAttrsPattern<WhileOrForLikeOp>::MaterializeAttrsPattern;

  // Materialize `T`.
  LogicalResult matchAndRewrite(WhileOrForLikeOp op,
                                PatternRewriter &rewriter) const override {
    if (op.T()) return failure();
    rewriter.updateRootInPlace(
        op, [&] { op.TAttr(this->getArgumentElementTypesAttr(rewriter, op)); });
    return success();
  }
};

// Base class for a pattern that
class MaterializeOutputShapesBase : public RewritePattern {
 public:
  explicit MaterializeOutputShapesBase(MLIRContext *context,
                                       StringRef attr_name)
      : RewritePattern(Pattern::MatchAnyOpTypeTag(), /*benefit=*/1, context),
        attr_id_(StringAttr::get(context, attr_name)) {}

  virtual bool shouldMatch(Operation *op) const = 0;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Exclude internal TFG ops.
    if (isa<ReturnOp>(op)) return failure();
    if (!shouldMatch(op) || op->hasAttr(attr_id_)) return failure();
    ResultRange results = TFOp(op).getNonControlResults();

    SmallVector<Attribute> shapes;
    for (Value result : results) {
      if (auto ranked = result.getType().dyn_cast<RankedTensorType>()) {
        shapes.push_back(ShapeAttr::get(op->getContext(), ranked.getShape()));
      } else {
        shapes.push_back(ShapeAttr::get(op->getContext(), llvm::None));
      }
    }
    rewriter.updateRootInPlace(op, [&] {
      op->setAttr(attr_id_, rewriter.getArrayAttr(shapes));
      rewriteImpl(op, rewriter);
    });
    return success();
  }

  virtual void rewriteImpl(Operation *op, PatternRewriter &rewriter) const {}

 private:
  // Cached identifier for the output shapes attribute.
  StringAttr attr_id_;
};

// Materialize `_output_shapes` for any TFG op.
class MaterializeTFGOpOutputShapes : public MaterializeOutputShapesBase {
 public:
  explicit MaterializeTFGOpOutputShapes(MLIRContext *context)
      : MaterializeOutputShapesBase(context, "_output_shapes"),
        dialect_(context->getOrLoadDialect<TFGraphDialect>()),
        regenerate_output_shapes_id_(
            StringAttr::get(context, kRegenerateOutputShapes)) {}

  bool shouldMatch(Operation *op) const override {
    return op->getDialect() == dialect_ &&
           op->getAttrOfType<UnitAttr>(regenerate_output_shapes_id_);
  }

  void rewriteImpl(Operation *op, PatternRewriter &rewriter) const override {
    op->removeAttr(regenerate_output_shapes_id_);
  }

 private:
  // Cached TFG dialect instance.
  TFGraphDialect *dialect_;
  // Identifier to `_regenerate_output_shapes`.
  StringAttr regenerate_output_shapes_id_;
};

// Materialize `output_shapes` for `If`, `Case`, and `While` ops.
struct MaterializeCFOpOutputShapes : public MaterializeOutputShapesBase {
  explicit MaterializeCFOpOutputShapes(MLIRContext *context)
      : MaterializeOutputShapesBase(context, "output_shapes") {}

  bool shouldMatch(Operation *op) const override {
    return isa<IfOp, StatelessIfOp, StatefulIfOp, CaseOp, StatelessCaseOp,
               StatefulCaseOp, WhileOp, StatelessWhileOp, StatefulWhileOp>(op);
  }
};
}  // namespace

template <template <typename OpT> class PatternT, typename... OpTs,
          typename... Args>
static void InsertPatterns(RewritePatternSet &patterns, Args &&...args) {
  patterns.insert<PatternT<OpTs>...>(std::forward<Args>(args)...);
}

void PrepareAttributesForExportPassImpl::runOnOperation() {
  // Skip this pass on generic functions. Generic functions contain only opaque
  // tensor types, into which shape and data type info cannot be reified.
  auto func = dyn_cast<GraphFuncOp>(getOperation());
  if (func && func.generic()) return;

  RewritePatternSet patterns(&getContext());
  ControlType control_type = ControlType::get(&getContext());
  InsertPatterns<MaterializeIfAttrs, IfOp, StatelessIfOp, StatefulIfOp>(
      patterns, control_type);
  InsertPatterns<MaterializeCaseAttrs, CaseOp, StatelessCaseOp, StatefulCaseOp>(
      patterns, control_type);
  InsertPatterns<MaterializeTAttr, WhileOp, StatelessWhileOp, StatefulWhileOp,
                 ForOp>(patterns, control_type);
  patterns.insert<MaterializeTFGOpOutputShapes, MaterializeCFOpOutputShapes>(
      &getContext());
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    getOperation()->emitError(getArgument() + " pass failed");
    signalPassFailure();
    return;
  }

  // If the pass was run on a function, materialize function, argument, and
  // result attributes with type info.
  if (func) prepareFunctionAttributes(func);
}

namespace {
struct ConsolidateAttributesPass
    : public ConsolidateAttributesBase<ConsolidateAttributesPass> {
  void runOnOperation() override {
    // Run the sub-pass on both `tfg.graph` and `tfg.func`.
    PassManager mgr(&getContext());
    mgr.addNestedPass<GraphOp>(
        std::make_unique<ConsolidateAttributesPassImpl>());
    mgr.addNestedPass<GraphFuncOp>(
        std::make_unique<ConsolidateAttributesPassImpl>());
    if (failed(runPipeline(mgr, getOperation()))) signalPassFailure();
  }
};

struct PrepareAttributesForExportPass
    : public PrepareAttributesForExportBase<PrepareAttributesForExportPass> {
  void runOnOperation() override {
    // Run the sub-pass on both `tfg.graph` and `tfg.func`.
    PassManager mgr(&getContext());
    mgr.addNestedPass<GraphOp>(
        std::make_unique<PrepareAttributesForExportPassImpl>());
    mgr.addNestedPass<GraphFuncOp>(
        std::make_unique<PrepareAttributesForExportPassImpl>());
    if (failed(runPipeline(mgr, getOperation()))) signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<Pass> CreateConsolidateAttributesPass() {
  return std::make_unique<ConsolidateAttributesPass>();
}

std::unique_ptr<Pass> CreatePrepareAttributesForExportPass() {
  return std::make_unique<PrepareAttributesForExportPass>();
}

}  // namespace tfg
}  // namespace mlir
