/* Copyright 2022 The OpenXLA Authors.

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

// This file supports the lowering of CHLO/HLO/LHLO dialect to Linalg dialect.

#ifndef MLIR_HLO_DIALECT_MHLO_TRANSFORMS_LEGALIZE_TO_LINALG_UTILS_H_
#define MLIR_HLO_DIALECT_MHLO_TRANSFORMS_LEGALIZE_TO_LINALG_UTILS_H_

#include <algorithm>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the last `nReduction` elements, where are "reduction"
/// attributes.
SmallVector<utils::IteratorType, 3> getParallelAndReductionIterators(
    unsigned nLoops, unsigned nReduction);

/// Returns an ArrayAttr that contains `nParallelLoops` "parallel" attributes.
SmallVector<utils::IteratorType, 3> getNParallelLoopsAttrs(
    unsigned nParallelLoops);

/// Generates an init sparse tensor.
Value getEmptySparseTensor(OpBuilder& b, Location loc, ShapedType type,
                           ArrayRef<Value> dynSizes);

/// Generates a tensor.empty op.
Value getEmptyTensor(OpBuilder& b, Location loc, ShapedType type,
                     ArrayRef<Value> dynSizes);

/// Generates an empty tensor for the result of the operation, which could be a
/// dense tensor or a sparse tensor.
Value getEmptyTensorFor(OpBuilder& b, Location loc, ShapedType resultType,
                        Operation* op, ValueRange operands);

/// Sparsifies a (block of) operation(s) that cannot be handled directly
/// by the sparse compiler but has well-known semi-ring semantics.
///
/// This yields something of the following form:
///
///   %result = sparse_tensor.unary %values[0]
///     present={
///       ^bb1(%val):
///         ... codegen proceeds here using %val ....
///         sparse_tensor.yield
///     }
///     absent={}
///   linalg.yield %result
Value preSparsify(Operation* op, llvm::SmallVector<Value, 2>& values, Type rtp,
                  OpBuilder* b);

/// Finalizes sparse semi-ring construction.
Value postSparsify(Operation* op, Value semiring, Value result, OpBuilder* b);

/// Returns true if all operands are tensors with rank 0.
bool allOperandsAreScalarTensors(Operation* op);

/// Returns true if parent op is linalg.
bool isInBodyOfLinalgOps(Operation* op);

/// Converts a HLO operation to a linalg.generic op that contains the
/// corresponding scalar operations.
template <typename OpTy>
class PointwiseToLinalgConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    // Find maximum rank / number of loops.
    auto getRank = [](Value v) {
      return mlir::cast<ShapedType>(v.getType()).getRank();
    };
    auto isScalar = [&](Value v) { return getRank(v) == 0; };
    auto it = llvm::find_if_not(adaptor.getOperands(), isScalar);
    Value maxRankArg =
        it != adaptor.getOperands().end() ? *it : adaptor.getOperands().front();
    int64_t nloops = getRank(maxRankArg);

    // Apply only if all operands are scalar or have the same rank. Some ops,
    // like `mhlo.select`, support implicit broadcasting of scalars.
    if (!llvm::all_of(adaptor.getOperands(), [&](Value v) {
          int64_t r = getRank(v);
          return r == 0 || r == nloops;
        })) {
      return rewriter.notifyMatchFailure(
          op, "Operands must be os same rank or scalar.");
    }

    // Find result type, if on tensors.
    std::optional<ShapedType> resultTy;
    resultTy = mlir::dyn_cast<ShapedType>(
        this->typeConverter->convertType(op->getResultTypes().front()));

    // Check result type compatibility.
    if (!resultTy || !resultTy->hasRank() || resultTy->getRank() != nloops ||
        !(resultTy->getElementType().isSignlessIntOrFloat() ||
          isa<ComplexType>(resultTy->getElementType()))) {
      return rewriter.notifyMatchFailure(
          op, "mismatched operand/result types or iterator count");
    }

    if (allOperandsAreScalarTensors(op) && isInBodyOfLinalgOps(op))
      return failure();

    // Find input/output values and types.
    ValueRange inputs = adaptor.getOperands();
    Value output =
        getEmptyTensorFor(rewriter, loc, *resultTy, op, adaptor.getOperands());

    // Create indexing maps.
    AffineMap scalarMap = AffineMap::get(nloops, 0, rewriter.getContext());
    AffineMap idMap = rewriter.getMultiDimIdentityMap(nloops);
    SmallVector<AffineMap, 4> maps;
    for (Value v : inputs) maps.push_back(isScalar(v) ? scalarMap : idMap);
    maps.push_back(idMap);

    // Build `linalg.generic` op.
    bool failed = false;
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, resultTy ? *resultTy : TypeRange{}, inputs, output, maps,
        getNParallelLoopsAttrs(nloops),
        [&](OpBuilder& nestedBuilder, Location /*nested_loc*/,
            ValueRange args) {
          Type innerResultTy = getElementTypeOrSelf(output);
          auto argvec = llvm::to_vector<2>(args.take_front(inputs.size()));
          auto semiring = preSparsify(op, argvec, innerResultTy, &rewriter);
          Value innerResult = mhlo::MhloOpToStdScalarOp::mapOp(
              op, innerResultTy, argvec, /*attributes=*/std::nullopt,
              &rewriter);
          if (innerResult == nullptr) {
            failed = true;
          } else {
            innerResult = postSparsify(op, semiring, innerResult, &rewriter);
            nestedBuilder.create<linalg::YieldOp>(loc, innerResult);
          }
        },
        linalg::getPrunedAttributeList(op));
    if (failed) return failure();

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

}  // namespace mhlo

}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_MHLO_TRANSFORMS_LEGALIZE_TO_LINALG_UTILS_H_
