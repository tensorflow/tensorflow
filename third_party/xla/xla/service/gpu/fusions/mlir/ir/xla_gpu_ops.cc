/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_dialect.cc.inc"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {
namespace {

using llvm::ArrayRef;
using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::failure;
using mlir::getAffineConstantExpr;
using mlir::getAffineDimExpr;
using mlir::getAffineSymbolExpr;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::OperationState;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::Region;
using mlir::SmallVector;
using mlir::success;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;

namespace arith = mlir::arith;

struct XlaGpuInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  // Returns true if the given operation 'callable', that implements the
  // 'CallableOpInterface', can be inlined into the position given call
  // operation 'call', that is registered to the current dialect and implements
  // the `CallOpInterface`. 'wouldBeCloned' is set to true if the region of the
  // given 'callable' is set to be cloned during the inlining process, or false
  // if the region is set to be moved in-place (i.e. no duplicates would be
  // created).
  bool isLegalToInline(mlir::Operation* call, mlir::Operation* callable,
                       bool wouldBeCloned) const final {
    if (!wouldBeCloned) {
      // If no duplicate would be created, 'call' is likely the only caller of
      // 'callable'.
      return true;
    }
    // Otherwise, inline only if the called function is small. We could
    // theoretically also inline if there is no other caller in the function
    // that contains the callee that has a call path to the callable, but that
    // is more expensive to check.
    auto func_op = mlir::dyn_cast<mlir::func::FuncOp>(callable);
    if (!func_op) {
      return false;
    }
    auto region = func_op.getCallableRegion();
    if (!region) {
      return false;
    }

    // If callee and caller call the same third function, inline. We have no
    // guarantee that the indices are the same, but there is a good chance they
    // are (or if the callee gets inlined as well, there will be CSE
    // opportunities).
    // This is duct tape to work around the limitations of our partitioner.
    // Ideally, the partitioner would be aware of the actual indexing and create
    // the partitions based on it (i.e., the case where the indices are the same
    // would never happen).
    llvm::SmallDenseSet<llvm::StringRef> callee_calls;
    for (auto call : region->getOps<PureCallOp>()) {
      callee_calls.insert(call.getCallee());
    }
    for (auto call : call->getParentRegion()->getOps<PureCallOp>()) {
      if (callee_calls.contains(call.getCallee())) {
        return true;
      }
    }

    constexpr int kMaxOperationsToInline = 8;
    int num_ops = 0;
    region->front().walk([&](mlir::Operation* op) { ++num_ops; });

    // Don't inline functions that are called more than once and contain more
    // than one call themselves.
    return num_ops <= kMaxOperationsToInline;
  }
  // Returns true if the given operation 'op', that is registered to this
  // dialect, can be inlined into the given region, false otherwise.
  // 'wouldBeCloned' is set to true if the given 'op' is set to be cloned
  // during the inlining process, or false if the operation is set to be moved
  // in-place(i.e. no duplicates would be created). 'valueMapping' contains any
  // remapped values from within the 'src' region. This can be used to examine
  // what values may potentially replace the operands to 'op'.
  bool isLegalToInline(mlir::Operation* op, mlir::Region* dest,
                       bool wouldBeCloned,
                       mlir::IRMapping& valueMapping) const final {
    // We allow any op from the xla_gpu dialect to be inlined.
    return true;
  }
};

}  // namespace

void XlaGpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.cc.inc"
#undef GET_OP_LIST
      >();
  addInterfaces<XlaGpuInlinerInterface>();
}

LogicalResult PureCallOp::verifySymbolUses(
    mlir::SymbolTableCollection& symbolTable) {
  auto callee = getCalleeAttr();
  auto function =
      symbolTable.lookupNearestSymbolFrom<mlir::func::FuncOp>(*this, callee);
  if (!function) {
    return emitError("'f' attribute refers to an undefined function: ")
           << callee;
  }

  int func_arg_count = function.getFunctionType().getNumInputs();
  int arg_count = getOperands().size();

  if (arg_count != func_arg_count) {
    return emitError() << "argument count mismatch: 'operands' has "
                       << arg_count << " arguments, but '" << callee
                       << "' expects " << func_arg_count;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllocateSharedOp
//===----------------------------------------------------------------------===//

void AllocateSharedOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, mlir::StringRef)> setNameFn) {
  setNameFn(getResult(), "shmem");
}

//===----------------------------------------------------------------------===//
// ApplyIndexingOp
//===----------------------------------------------------------------------===//

void ApplyIndexingOp::build(OpBuilder& builder, OperationState& result,
                            ValueRange dims, ValueRange symbols,
                            const IndexingMap& indexing_map) {
  SmallVector<Value, 4> operands;
  operands.reserve(dims.size() + symbols.size());
  operands.append(dims.begin(), dims.end());
  operands.append(symbols.begin(), symbols.end());
  build(builder, result, operands, indexing_map);
}

void ApplyIndexingOp::build(OpBuilder& builder, OperationState& result,
                            ValueRange operands,
                            const IndexingMap& indexing_map) {
  build(builder, result, operands, indexing_map.GetAffineMap(),
        indexing_map.GetDimVars(), indexing_map.GetRangeVars());
}

void ApplyIndexingOp::build(OpBuilder& builder, OperationState& result,
                            ValueRange operands, AffineMap affine_map,
                            ArrayRef<DimVar> dim_vars,
                            ArrayRef<RangeVar> range_vars) {
  SmallVector<int64_t, 4> lower_bounds, upper_bounds;
  for (const DimVar& dim_var : dim_vars) {
    lower_bounds.push_back(dim_var.bounds.lower);
    upper_bounds.push_back(dim_var.bounds.upper);
  }
  for (const RangeVar& range_var : range_vars) {
    lower_bounds.push_back(range_var.range.lower);
    upper_bounds.push_back(range_var.range.upper);
  }
  build(builder, result, operands, affine_map, lower_bounds, upper_bounds);
}

void ApplyIndexingOp::build(OpBuilder& builder, OperationState& result,
                            ValueRange operands, AffineMap affine_map,
                            ArrayRef<int64_t> lower_bounds,
                            ArrayRef<int64_t> upper_bounds) {
  SmallVector<Type, 2> result_types(affine_map.getNumResults(),
                                    builder.getIndexType());
  build(builder, result, result_types, operands, affine_map, lower_bounds,
        upper_bounds);
}

// Parser a comma-separated list of type %operand in [lower_bound, upper_bound].
// Adds the parsed elements to the provided containers.
mlir::ParseResult parseOperandsWithBoundsList(
    mlir::OpAsmParser& parser,
    SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4>* operands,
    SmallVector<int64_t, 4>* lower_bounds,
    SmallVector<int64_t, 4>* upper_bounds) {
  int64_t lower_bound, upper_bound;
  mlir::OpAsmParser::UnresolvedOperand operand;
  if (parser.parseCommaSeparatedList([&]() {
        if (parser.parseOperand(operand) || parser.parseKeyword("in") ||
            parser.parseLSquare() || parser.parseInteger(lower_bound) ||
            parser.parseComma() || parser.parseInteger(upper_bound) ||
            parser.parseRSquare()) {
          return failure();
        }
        operands->push_back(operand);
        lower_bounds->push_back(lower_bound);
        upper_bounds->push_back(upper_bound);
        return success();
      })) {
    return failure();
  }
  return success();
}

mlir::ParseResult ApplyIndexingOp::parse(mlir::OpAsmParser& parser,
                                         OperationState& result) {
  mlir::Builder& builder = parser.getBuilder();
  auto index_type = builder.getIndexType();

  mlir::AffineMapAttr affine_map_attr;
  if (parser.parseAttribute(affine_map_attr, "map", result.attributes)) {
    return failure();
  }

  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> operands;
  SmallVector<int64_t, 4> lower_bounds, upper_bounds;
  if (succeeded(parser.parseOptionalLParen())) {
    if (parseOperandsWithBoundsList(parser, &operands, &lower_bounds,
                                    &upper_bounds) ||
        parser.parseRParen()) {
      return failure();
    }
  }
  if (succeeded(parser.parseOptionalLSquare())) {
    if (parseOperandsWithBoundsList(parser, &operands, &lower_bounds,
                                    &upper_bounds) ||
        parser.parseRSquare()) {
      return failure();
    }
  }
  if (parser.resolveOperands(operands, index_type, result.operands) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  result.addAttribute("lower_bounds",
                      builder.getDenseI64ArrayAttr(lower_bounds));
  result.addAttribute("upper_bounds",
                      builder.getDenseI64ArrayAttr(upper_bounds));

  auto map = affine_map_attr.getAffineMap();
  result.addTypes(SmallVector<Type, 2>(map.getNumResults(), index_type));
  return success();
}

void ApplyIndexingOp::print(mlir::OpAsmPrinter& p) {
  mlir::AffineMapAttr affine_map_attr = getMapAttr();
  AffineMap affine_map = affine_map_attr.getAffineMap();
  p << " " << affine_map_attr;

  auto lower_bounds = getLowerBounds();
  auto upper_bounds = getUpperBounds();
  auto operands = getOperands();
  unsigned num_dimensions = affine_map.getNumDims();
  if (num_dimensions > 0) {
    p << '(';
    for (int dim_id = 0; dim_id < num_dimensions; ++dim_id) {
      p << operands[dim_id] << " in " << '[' << lower_bounds[dim_id] << ", "
        << upper_bounds[dim_id] << ']';
      if (dim_id != num_dimensions - 1) {
        p << ", ";
      }
    }
    p << ')';
  }
  unsigned num_symbols = affine_map.getNumSymbols();
  if (num_symbols > 0) {
    p << '[';
    for (int symbol_id = 0; symbol_id < num_symbols; ++symbol_id) {
      unsigned operand_id = num_dimensions + symbol_id;
      p << operands[operand_id] << " in " << '[' << lower_bounds[operand_id]
        << ", " << upper_bounds[operand_id] << ']';
      if (symbol_id != num_symbols - 1) {
        p << ", ";
      }
    }
    p << ']';
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{
                              "map", "lower_bounds", "upper_bounds"});
}

LogicalResult ApplyIndexingOp::verify() {
  auto affine_map = getMapAttr().getAffineMap();
  unsigned num_variables = affine_map.getNumDims() + affine_map.getNumSymbols();
  if (getOperands().size() != num_variables ||
      getLowerBounds().size() != num_variables ||
      getUpperBounds().size() != num_variables) {
    return emitOpError(
        "operand, lower_bounds, upper_bounds count and affine map dimension "
        "and symbol count must match");
  }
  return success();
}

IndexingMap ApplyIndexingOp::getIndexingMap() {
  auto lower_bounds = getLowerBounds();
  auto upper_bounds = getUpperBounds();

  AffineMap affine_map = getAffineMap();
  unsigned num_dimensions = affine_map.getNumDims();
  std::vector<DimVar> dim_vars;
  dim_vars.reserve(num_dimensions);
  for (unsigned id = 0; id < num_dimensions; ++id) {
    dim_vars.push_back(DimVar{Interval{lower_bounds[id], upper_bounds[id]}});
  }
  unsigned num_symbols = affine_map.getNumSymbols();
  std::vector<RangeVar> range_vars;
  range_vars.reserve(num_symbols);
  for (unsigned id = num_dimensions; id < num_symbols + num_dimensions; ++id) {
    range_vars.push_back(
        RangeVar{Interval{lower_bounds[id], upper_bounds[id]}});
  }
  return IndexingMap(affine_map, std::move(dim_vars), std::move(range_vars),
                     /*rt_vars=*/{});
}

namespace {

// Simplifies the indexing map, removes unused variables.
struct SimplifyIndexingMap : public mlir::OpRewritePattern<ApplyIndexingOp> {
  using OpRewritePattern<ApplyIndexingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ApplyIndexingOp indexing_op,
                                PatternRewriter& rewriter) const override {
    IndexingMap indexing_map = indexing_op.getIndexingMap();
    bool is_simplified = indexing_map.Simplify();

    // Remove unused symbols.
    auto unused_symbols_bit_vector = indexing_map.RemoveUnusedVars();
    bool symbols_removed = unused_symbols_bit_vector.count() != 0;

    if (!is_simplified && !symbols_removed) {
      return rewriter.notifyMatchFailure(indexing_op,
                                         "IndexingMap stayed unchanged");
    }
    if (!unused_symbols_bit_vector.empty()) {
      SmallVector<Value, 4> operands;
      operands.reserve(unused_symbols_bit_vector.count());
      for (int i = 0; i < unused_symbols_bit_vector.size(); ++i) {
        if (!unused_symbols_bit_vector[i]) {
          operands.push_back(indexing_op.getOperand(i));
        }
      }
      rewriter.replaceOpWithNewOp<ApplyIndexingOp>(indexing_op, operands,
                                                   indexing_map);
    } else {
      rewriter.replaceOpWithNewOp<ApplyIndexingOp>(
          indexing_op, indexing_op.getOperands(), indexing_map);
    }
    return success();
  }
};

struct FoldApplyIndexingSequence
    : public mlir::OpRewritePattern<ApplyIndexingOp> {
  using OpRewritePattern<ApplyIndexingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ApplyIndexingOp indexing_op,
                                PatternRewriter& rewriter) const override {
    MLIRContext* ctx = indexing_op.getContext();
    int num_dims = indexing_op.getAffineMap().getNumDims();
    int num_syms = indexing_op.getAffineMap().getNumSymbols();
    mlir::DenseMap<Value, AffineExpr> operand_exprs;
    for (auto& operand : indexing_op->getOpOperands()) {
      int operand_number = operand.getOperandNumber();
      operand_exprs[operand.get()] =
          operand_number < num_dims
              ? getAffineDimExpr(operand_number, ctx)
              : getAffineSymbolExpr(operand_number - num_dims, ctx);
    }

    auto this_map = indexing_op.getIndexingMap();

    SmallVector<Value> added_dim_args;
    SmallVector<Value> added_sym_args;
    auto new_dim_vars = this_map.GetDimVars();
    auto new_sym_vars = this_map.GetRangeVars();

    mlir::DenseMap<AffineExpr, AffineExpr> replacements;
    for (auto& operand : indexing_op->getOpOperands()) {
      if (auto producer = operand.get().getDefiningOp<ApplyIndexingOp>()) {
        auto producer_map = producer.getIndexingMap();
        int producer_result_id =
            mlir::cast<mlir::OpResult>(operand.get()).getResultNumber();
        int num_producer_dims = producer.getAffineMap().getNumDims();
        SmallVector<AffineExpr> producer_dim_replacements;
        SmallVector<AffineExpr> producer_sym_replacements;
        for (auto& producer_operand : producer->getOpOperands()) {
          int producer_operand_number = producer_operand.getOperandNumber();
          bool is_dim = producer_operand_number < num_producer_dims;
          auto& replacement_expr = operand_exprs[producer_operand.get()];
          if (!replacement_expr) {
            if (is_dim) {
              int dim_num = producer_operand_number;
              replacement_expr =
                  getAffineDimExpr(num_dims + added_dim_args.size(), ctx);
              added_dim_args.push_back(producer_operand.get());
              new_dim_vars.push_back(producer_map.GetDimVars(dim_num));
            } else {
              int sym_num = producer_operand_number -
                            producer.getAffineMap().getNumDims();
              replacement_expr =
                  getAffineSymbolExpr(num_syms + added_sym_args.size(), ctx);
              added_sym_args.push_back(producer_operand.get());
              new_sym_vars.push_back(producer_map.GetRangeVar(sym_num));
            }
          }

          if (is_dim) {
            producer_dim_replacements.push_back(replacement_expr);
          } else {
            producer_sym_replacements.push_back(replacement_expr);
          }
        }

        replacements[operand_exprs[operand.get()]] =
            producer.getAffineMap()
                .getResult(producer_result_id)
                .replaceDimsAndSymbols(producer_dim_replacements,
                                       producer_sym_replacements);
      }
    }

    if (replacements.empty()) {
      return rewriter.notifyMatchFailure(indexing_op,
                                         "No apply_indexing sequences found");
    }

    int new_num_operands = indexing_op->getNumOperands() +
                           added_dim_args.size() + added_sym_args.size();
    auto new_affine_map = indexing_op.getAffineMap().replace(
        replacements, num_dims + added_dim_args.size(),
        num_syms + added_sym_args.size());
    IndexingMap new_indexing_map(new_affine_map, new_dim_vars, new_sym_vars,
                                 /*rt_vars=*/{});
    if (!new_indexing_map.Simplify()) {
      return rewriter.notifyMatchFailure(
          indexing_op, "Folded indexing map was not simplified");
    }
    SmallVector<Value> new_operands;
    new_operands.reserve(new_num_operands);

    auto begin = indexing_op.getOperands().begin();
    new_operands.append(begin, begin + num_dims);
    new_operands.append(added_dim_args);
    new_operands.append(begin + num_dims, begin + num_dims + num_syms);
    new_operands.append(added_sym_args);

    rewriter.replaceOpWithNewOp<ApplyIndexingOp>(indexing_op, new_operands,
                                                 new_indexing_map);
    return success();
  }
};

// Folds constants into the indexing map.
struct FoldApplyIndexingOperands
    : public mlir::OpRewritePattern<ApplyIndexingOp> {
  using OpRewritePattern<ApplyIndexingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ApplyIndexingOp indexing_op,
                                PatternRewriter& rewriter) const override {
    AffineMap affine_map = indexing_op.getAffineMap();

    MLIRContext* ctx = affine_map.getContext();
    unsigned num_operands = indexing_op->getNumOperands();
    unsigned num_dims = affine_map.getNumDims();
    unsigned num_symbols = affine_map.getNumSymbols();

    SmallVector<std::optional<int64_t>> constant_values(num_operands,
                                                        std::nullopt);
    int num_constants = 0;
    SmallVector<int64_t> dim_id_map(num_dims, -1);
    SmallVector<int64_t> symbol_id_map(num_symbols, -1);
    for (auto& operand : indexing_op->getOpOperands()) {
      if (auto constant =
              operand.get().getDefiningOp<arith::ConstantIndexOp>()) {
        constant_values[operand.getOperandNumber()] = constant.value();
        ++num_constants;
      }
    }
    if (num_constants == 0) {
      return rewriter.notifyMatchFailure(indexing_op,
                                         "No constant operands found");
    }
    SmallVector<AffineExpr, 2> dim_replacements, symbol_replacements;
    dim_replacements.reserve(num_dims);
    symbol_replacements.reserve(num_symbols);

    unsigned new_num_operands = indexing_op->getNumOperands() - num_constants;
    SmallVector<Value, 4> new_operands;
    new_operands.reserve(new_num_operands);
    SmallVector<int64_t, 4> new_lbs, new_ubs;
    new_lbs.reserve(new_num_operands);
    new_ubs.reserve(new_num_operands);

    unsigned new_num_dims = 0;
    unsigned new_num_symbols = 0;
    for (auto [operand, constant_value, lb, ub] : llvm::zip(
             indexing_op->getOpOperands(), constant_values,
             indexing_op.getLowerBounds(), indexing_op.getUpperBounds())) {
      unsigned operand_id = operand.getOperandNumber();
      if (constant_value.has_value()) {
        if (operand_id < num_dims) {
          dim_replacements.push_back(
              getAffineConstantExpr(*constant_value, ctx));
        } else {
          symbol_replacements.push_back(
              getAffineConstantExpr(*constant_value, ctx));
        }
      } else {
        if (operand_id < num_dims) {
          dim_replacements.push_back(getAffineDimExpr(new_num_dims++, ctx));
        } else {
          symbol_replacements.push_back(
              getAffineSymbolExpr(new_num_symbols++, ctx));
        }
        new_operands.push_back(operand.get());
        new_lbs.push_back(lb);
        new_ubs.push_back(ub);
      }
    }
    rewriter.replaceOpWithNewOp<ApplyIndexingOp>(
        indexing_op, new_operands,
        affine_map.replaceDimsAndSymbols(dim_replacements, symbol_replacements,
                                         new_num_dims, new_num_symbols),
        new_lbs, new_ubs);
    return success();
  }
};

// Folds constant and dim/symbol expression results.
struct FoldApplyIndexingResults
    : public mlir::OpRewritePattern<ApplyIndexingOp> {
  using OpRewritePattern<ApplyIndexingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ApplyIndexingOp indexing_op,
                                PatternRewriter& rewriter) const override {
    mlir::Location loc = indexing_op.getLoc();
    IndexingMap indexing_map = indexing_op.getIndexingMap();
    if (indexing_map.IsKnownEmpty()) {
      return rewriter.notifyMatchFailure(indexing_op,
                                         "Domain of the indexing map is empty");
    }
    AffineMap* affine_map = &indexing_map.GetMutableAffineMap();
    unsigned num_results = affine_map->getNumResults();
    SmallVector<AffineExpr, 4> new_exprs;
    new_exprs.reserve(num_results);
    SmallVector<Value, 4> new_values;
    new_values.reserve(num_results);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    for (mlir::OpResult opresult : indexing_op->getOpResults()) {
      if (opresult.use_empty()) {
        new_values.push_back(zero);
        continue;
      }

      unsigned id = opresult.getResultNumber();
      AffineExpr result_expr = affine_map->getResult(id);
      if (auto const_expr =
              mlir::dyn_cast<mlir::AffineConstantExpr>(result_expr)) {
        new_values.push_back(rewriter.create<arith::ConstantIndexOp>(
            loc, const_expr.getValue()));
        continue;
      }
      if (auto dim_expr = mlir::dyn_cast<mlir::AffineDimExpr>(result_expr)) {
        new_values.push_back(indexing_op.getOperand(dim_expr.getPosition()));
        continue;
      }
      if (auto symbol_expr =
              mlir::dyn_cast<mlir::AffineSymbolExpr>(result_expr)) {
        new_values.push_back(indexing_op.getOperand(
            indexing_map.GetDimVarsCount() + symbol_expr.getPosition()));
        continue;
      }
      new_exprs.push_back(result_expr);
      new_values.push_back(Value{});
    }
    if (new_exprs.size() == num_results) {
      return rewriter.notifyMatchFailure(
          indexing_op, "No constant or dim/symbol expression found");
    }
    *affine_map =
        AffineMap::get(affine_map->getNumDims(), affine_map->getNumSymbols(),
                       new_exprs, affine_map->getContext());
    auto new_indexing_op = rewriter.create<ApplyIndexingOp>(
        loc, indexing_op.getOperands(), indexing_map);
    for (int new_result_id = 0, new_indexing_op_result_id = 0;
         new_result_id < new_values.size(); ++new_result_id) {
      auto& new_value = new_values[new_result_id];
      if (new_value) continue;
      new_value = new_indexing_op.getResult(new_indexing_op_result_id++);
    }
    rewriter.replaceOp(indexing_op, new_values);
    return success();
  }
};

}  // namespace

void ApplyIndexingOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet& results, MLIRContext* context) {
  results.add<FoldApplyIndexingOperands, FoldApplyIndexingResults,
              SimplifyIndexingMap, FoldApplyIndexingSequence>(context);
}

mlir::LogicalResult ApplyIndexingOp::fold(
    FoldAdaptor adaptor, llvm::SmallVectorImpl<mlir::OpFoldResult>& results) {
  auto map = getAffineMap();
  for (auto expr : map.getResults()) {
    if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
      results.push_back(getOperand(dim.getPosition()));
    } else if (auto sym = mlir::dyn_cast<mlir::AffineSymbolExpr>(expr)) {
      results.push_back(getOperand(map.getNumDims() + sym.getPosition()));
    } else {
      results.clear();
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AtomicRMWOp
//===----------------------------------------------------------------------===//

void AtomicRMWOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, mlir::StringRef)> setNameFn) {
  setNameFn(getResult(), "atomic_rmw");
}

void AtomicRMWOp::build(OpBuilder& builder, OperationState& result,
                        Value tensor, ValueRange ivs) {
  OpBuilder::InsertionGuard g(builder);
  result.addOperands(tensor);
  result.addOperands(ivs);
  result.addTypes(tensor.getType());

  auto tensor_type = llvm::cast<RankedTensorType>(tensor.getType());
  Region* body = result.addRegion();
  builder.createBlock(body);
  body->addArgument(tensor_type.getElementType(), tensor.getLoc());
}

//===----------------------------------------------------------------------===//
// PureCallOp
//===----------------------------------------------------------------------===//

void PureCallOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, mlir::StringRef)> setNameFn) {
  for (auto result : getResults()) {
    setNameFn(result, "pure_call");
  }
}

//===----------------------------------------------------------------------===//
// SyncThreadsOp
//===----------------------------------------------------------------------===//

void SyncThreadsOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, mlir::StringRef)> setNameFn) {
  for (auto result : getResults()) {
    setNameFn(result, "synced_tensor");
  }
}

}  // namespace gpu
}  // namespace xla

#define GET_OP_CLASSES
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.cc.inc"
