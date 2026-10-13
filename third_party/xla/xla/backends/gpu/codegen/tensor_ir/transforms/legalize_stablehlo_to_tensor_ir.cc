/* Copyright 2026 The OpenXLA Authors.

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

#include <cstdint>
#include <optional>
#include <utility>

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Dialect/TensorIRAttrs.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/Arith/IR/ArithDialect.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/gpu/codegen/tensor_ir/transforms/type_conversion.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::nv_tensor_ir::xla {
#define GEN_PASS_DEF_LEGALIZESTABLEHLOTOTENSORIRPASS
#include "xla/backends/gpu/codegen/tensor_ir/transforms/passes.h.inc"

namespace {
template <typename SourceOp, typename TargetOp>
struct ElementwiseUnaryOpConversion
    : public mlir::OpConversionPattern<SourceOp> {
  using mlir::OpConversionPattern<SourceOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      SourceOp op, typename SourceOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type result_type =
        this->getTypeConverter()->convertType(op.getType());
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    rewriter.replaceOpWithNewOp<TargetOp>(op, result_type,
                                          adaptor.getOperand());
    return mlir::success();
  }
};

template <typename SourceOp, typename TargetOp>
struct ElementwiseBinaryOpConversion
    : public mlir::OpConversionPattern<SourceOp> {
  using mlir::OpConversionPattern<SourceOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      SourceOp op, typename SourceOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type result_type =
        this->getTypeConverter()->convertType(op.getType());
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    rewriter.replaceOpWithNewOp<TargetOp>(op, result_type, adaptor.getLhs(),
                                          adaptor.getRhs());
    return mlir::success();
  }
};

struct SelectOpConversion
    : public mlir::OpConversionPattern<mlir::stablehlo::SelectOp> {
  using mlir::OpConversionPattern<
      mlir::stablehlo::SelectOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::SelectOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type result_type =
        this->getTypeConverter()->convertType(op.getType());
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    rewriter.replaceOpWithNewOp<mlir::nv_tensor_ir::BinarySelectOp>(
        op, result_type, adaptor.getPred(), adaptor.getOnTrue(),
        adaptor.getOnFalse());
    return mlir::success();
  }
};

struct CompareOpConversion
    : public mlir::OpConversionPattern<mlir::stablehlo::CompareOp> {
  using mlir::OpConversionPattern<
      mlir::stablehlo::CompareOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::CompareOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (op.getCompareType().has_value() &&
        *op.getCompareType() == mlir::stablehlo::ComparisonType::TOTALORDER) {
      return rewriter.notifyMatchFailure(
          op, "TOTALORDER comparison is not supported in nv_tensor_ir");
    }

    mlir::Type converted_res_type =
        this->getTypeConverter()->convertType(op.getType());
    if (!converted_res_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    bool is_float = false;
    if (op.getCompareType().has_value() &&
        *op.getCompareType() != mlir::stablehlo::ComparisonType::NOTYPE) {
      is_float =
          (*op.getCompareType() == mlir::stablehlo::ComparisonType::FLOAT);
    } else {
      mlir::Type elem_type = op.getLhs().getType().getElementType();
      if (mlir::isa<mlir::FloatType>(elem_type)) {
        is_float = true;
      } else if (mlir::isa<mlir::IntegerType>(elem_type)) {
        is_float = false;
      } else {
        return rewriter.notifyMatchFailure(
            op, "unsupported operand element type for comparison");
      }
    }

    mlir::nv_tensor_ir::Comparator comparator;
    if (is_float) {
      // Every float direction maps to its ordered comparator except NE, which
      // maps to the *unordered* `une`. This asymmetry is deliberate and is not
      // a typo: IEEE 754 defines `NaN != x` as true, so not-equal is the only
      // direction that must hold when either operand is NaN. Ordered `one`
      // returns false in that case and would silently change the result of any
      // fusion that can produce a NaN.
      switch (op.getComparisonDirection()) {
        case mlir::stablehlo::ComparisonDirection::EQ:
          comparator = mlir::nv_tensor_ir::Comparator::oeq;
          break;
        case mlir::stablehlo::ComparisonDirection::NE:
          comparator = mlir::nv_tensor_ir::Comparator::une;
          break;
        case mlir::stablehlo::ComparisonDirection::GE:
          comparator = mlir::nv_tensor_ir::Comparator::oge;
          break;
        case mlir::stablehlo::ComparisonDirection::GT:
          comparator = mlir::nv_tensor_ir::Comparator::ogt;
          break;
        case mlir::stablehlo::ComparisonDirection::LE:
          comparator = mlir::nv_tensor_ir::Comparator::ole;
          break;
        case mlir::stablehlo::ComparisonDirection::LT:
          comparator = mlir::nv_tensor_ir::Comparator::olt;
          break;
      }
    } else {
      switch (op.getComparisonDirection()) {
        case mlir::stablehlo::ComparisonDirection::EQ:
          comparator = mlir::nv_tensor_ir::Comparator::eq;
          break;
        case mlir::stablehlo::ComparisonDirection::NE:
          comparator = mlir::nv_tensor_ir::Comparator::neq;
          break;
        case mlir::stablehlo::ComparisonDirection::GE:
          comparator = mlir::nv_tensor_ir::Comparator::ge;
          break;
        case mlir::stablehlo::ComparisonDirection::GT:
          comparator = mlir::nv_tensor_ir::Comparator::gt;
          break;
        case mlir::stablehlo::ComparisonDirection::LE:
          comparator = mlir::nv_tensor_ir::Comparator::le;
          break;
        case mlir::stablehlo::ComparisonDirection::LT:
          comparator = mlir::nv_tensor_ir::Comparator::lt;
          break;
      }
    }

    rewriter.replaceOpWithNewOp<mlir::nv_tensor_ir::CmpOp>(
        op, converted_res_type, comparator, adaptor.getLhs(), adaptor.getRhs());
    return mlir::success();
  }
};

struct ConvertOpConversion
    : public mlir::OpConversionPattern<mlir::stablehlo::ConvertOp> {
  using mlir::OpConversionPattern<
      mlir::stablehlo::ConvertOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ConvertOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type result_type =
        this->getTypeConverter()->convertType(op.getType());
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    rewriter.replaceOpWithNewOp<mlir::nv_tensor_ir::ConvertOp>(
        op, result_type, adaptor.getOperand());
    return mlir::success();
  }
};

// stablehlo.clamp(min, operand, max) has no nv_tensor_ir counterpart and is
// expressed as min(max_value, max(min_value, operand)). The nesting order and
// the operand order within each of the two operations are chosen to match the
// string-building converter this pass replaces, so that the two produce
// byte-identical output for the same fusion.
struct ClampOpConversion
    : public mlir::OpConversionPattern<mlir::stablehlo::ClampOp> {
  using mlir::OpConversionPattern<
      mlir::stablehlo::ClampOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ClampOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type result_type =
        this->getTypeConverter()->convertType(op.getType());
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    // StableHLO additionally allows rank-0 `min`/`max` operands that are
    // broadcast against `operand`. nv_tensor_ir.max and nv_tensor_ir.min
    // require all operands to have the same shape, and silently pairing a
    // scalar with a shaped operand would not verify, so reject that form
    // explicitly rather than emitting something that cannot be built. XLA's
    // HLO importer never produces it, because HLO clamp is already
    // shape-matched by the time a fusion is formed.
    if (adaptor.getMin().getType() != result_type ||
        adaptor.getMax().getType() != result_type) {
      return rewriter.notifyMatchFailure(
          op,
          "scalar clamp bounds are not supported; min and max must have "
          "the same shape as the operand");
    }
    mlir::Value lower_bounded = mlir::nv_tensor_ir::MaxOp::create(
        rewriter, op.getLoc(), result_type, adaptor.getMin(),
        adaptor.getOperand());
    rewriter.replaceOpWithNewOp<mlir::nv_tensor_ir::MinOp>(
        op, result_type, adaptor.getMax(), lower_bounded);
    return mlir::success();
  }
};

// Returns a splat nv_tensor_ir.constant of the integral value `value` over
// `type`, which must have a floating-point element type.
mlir::Value CreateFloatSplat(mlir::ConversionPatternRewriter& rewriter,
                             mlir::Location loc, mlir::RankedTensorType type,
                             int64_t value) {
  auto element_type = mlir::cast<mlir::FloatType>(type.getElementType());
  llvm::APFloat splat(element_type.getFloatSemantics(), value);
  return mlir::nv_tensor_ir::ConstantOp::create(
      rewriter, loc, mlir::DenseElementsAttr::get(type, splat));
}

// NOTE: Decomposing Expm1(x) to Exp(x) - 1.0 can result in a severe loss of
// precision for values of x close to 0. nv_tensor_ir has no expm1, so this is
// the only available lowering; the caveat is reproduced from the converter
// this pass replaces because it is invisible in the emitted IR.
struct Expm1OpConversion
    : public mlir::OpConversionPattern<mlir::stablehlo::Expm1Op> {
  using mlir::OpConversionPattern<
      mlir::stablehlo::Expm1Op>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::Expm1Op op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto result_type = mlir::dyn_cast_or_null<mlir::RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    if (!result_type ||
        !mlir::isa<mlir::FloatType>(result_type.getElementType())) {
      return rewriter.notifyMatchFailure(
          op, "expected a ranked floating-point result");
    }
    mlir::Location loc = op.getLoc();
    mlir::Value one = CreateFloatSplat(rewriter, loc, result_type, /*value=*/1);
    mlir::Value exp = mlir::nv_tensor_ir::ExpOp::create(
        rewriter, loc, result_type, adaptor.getOperand());
    rewriter.replaceOpWithNewOp<mlir::nv_tensor_ir::SubOp>(op, result_type, exp,
                                                           one);
    return mlir::success();
  }
};

// NOTE: Decomposing Log1p(x) to Log(x + 1.0) can result in a severe loss of
// precision for values of x close to 0. See the note on Expm1OpConversion.
struct Log1pOpConversion
    : public mlir::OpConversionPattern<mlir::stablehlo::Log1pOp> {
  using mlir::OpConversionPattern<
      mlir::stablehlo::Log1pOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::Log1pOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto result_type = mlir::dyn_cast_or_null<mlir::RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    if (!result_type ||
        !mlir::isa<mlir::FloatType>(result_type.getElementType())) {
      return rewriter.notifyMatchFailure(
          op, "expected a ranked floating-point result");
    }
    mlir::Location loc = op.getLoc();
    mlir::Value one = CreateFloatSplat(rewriter, loc, result_type, /*value=*/1);
    mlir::Value sum = mlir::nv_tensor_ir::AddOp::create(
        rewriter, loc, result_type, adaptor.getOperand(), one);
    rewriter.replaceOpWithNewOp<mlir::nv_tensor_ir::LogOp>(op, result_type,
                                                           sum);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Structural operations.
//===----------------------------------------------------------------------===//

struct ReshapeOpConversion
    : public mlir::OpConversionPattern<mlir::stablehlo::ReshapeOp> {
  using mlir::OpConversionPattern<
      mlir::stablehlo::ReshapeOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReshapeOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type result_type =
        this->getTypeConverter()->convertType(op.getType());
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    rewriter.replaceOpWithNewOp<mlir::nv_tensor_ir::ReshapeOp>(
        op, result_type, adaptor.getOperand(),
        /*dynamic_sizes=*/mlir::ValueRange{});
    return mlir::success();
  }
};

struct TransposeOpConversion
    : public mlir::OpConversionPattern<mlir::stablehlo::TransposeOp> {
  using mlir::OpConversionPattern<
      mlir::stablehlo::TransposeOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::TransposeOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type result_type =
        this->getTypeConverter()->convertType(op.getType());
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    // Both dialects define the permutation as output[i] = input[perm[i]], so
    // it can be forwarded as is.
    rewriter.replaceOpWithNewOp<mlir::nv_tensor_ir::TransposeOp>(
        op, result_type, adaptor.getOperand(), op.getPermutationAttr());
    return mlir::success();
  }
};

struct SliceOpConversion
    : public mlir::OpConversionPattern<mlir::stablehlo::SliceOp> {
  using mlir::OpConversionPattern<
      mlir::stablehlo::SliceOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::SliceOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type result_type =
        this->getTypeConverter()->convertType(op.getType());
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    rewriter.replaceOpWithNewOp<mlir::nv_tensor_ir::SliceOp>(
        op, result_type, adaptor.getOperand(), op.getStartIndicesAttr(),
        op.getLimitIndicesAttr(), op.getStridesAttr());
    return mlir::success();
  }
};

struct ConcatenateOpConversion
    : public mlir::OpConversionPattern<mlir::stablehlo::ConcatenateOp> {
  using mlir::OpConversionPattern<
      mlir::stablehlo::ConcatenateOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ConcatenateOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type result_type =
        this->getTypeConverter()->convertType(op.getType());
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    rewriter.replaceOpWithNewOp<mlir::nv_tensor_ir::ConcatenateOp>(
        op, result_type, adaptor.getInputs(), op.getDimension());
    return mlir::success();
  }
};

struct IotaOpConversion
    : public mlir::OpConversionPattern<mlir::stablehlo::IotaOp> {
  using mlir::OpConversionPattern<mlir::stablehlo::IotaOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::IotaOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type result_type =
        this->getTypeConverter()->convertType(op.getType());
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    rewriter.replaceOpWithNewOp<mlir::nv_tensor_ir::IotaOp>(
        op, result_type, op.getIotaDimension(),
        /*dynamic_sizes=*/mlir::ValueRange{});
    return mlir::success();
  }
};

struct ConstantOpConversion
    : public mlir::OpConversionPattern<mlir::stablehlo::ConstantOp> {
  using mlir::OpConversionPattern<
      mlir::stablehlo::ConstantOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ConstantOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto result_type = mlir::dyn_cast_or_null<mlir::RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    auto value = mlir::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
    if (!value) {
      return rewriter.notifyMatchFailure(op,
                                         "expected a dense elements attribute");
    }
    // nv_tensor_ir.constant infers its result type from the value attribute
    // (InferTypeOpAdaptor), so the attribute itself has to be rebuilt over the
    // converted element type. The raw storage is bit-identical because the
    // type conversion only turns signless integers into signed integers of the
    // same width.
    if (value.getType() != result_type) {
      if (!mlir::DenseElementsAttr::isValidRawBuffer(result_type,
                                                     value.getRawData())) {
        return rewriter.notifyMatchFailure(
            op,
            "constant value cannot be reinterpreted over the converted type");
      }
      value = mlir::DenseElementsAttr::getFromRawBuffer(result_type,
                                                        value.getRawData());
    }
    rewriter.replaceOpWithNewOp<mlir::nv_tensor_ir::ConstantOp>(
        op, mlir::cast<mlir::TypedAttr>(value));
    return mlir::success();
  }
};

struct BroadcastInDimOpConversion
    : public mlir::OpConversionPattern<mlir::stablehlo::BroadcastInDimOp> {
  using mlir::OpConversionPattern<
      mlir::stablehlo::BroadcastInDimOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::BroadcastInDimOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto result_type = mlir::dyn_cast_or_null<mlir::RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    auto input_type =
        mlir::dyn_cast<mlir::RankedTensorType>(adaptor.getOperand().getType());
    if (!input_type) {
      return rewriter.notifyMatchFailure(op, "expected a ranked tensor input");
    }
    llvm::ArrayRef<int64_t> broadcast_dimensions = op.getBroadcastDimensions();
    if (static_cast<int64_t>(broadcast_dimensions.size()) !=
        input_type.getRank()) {
      return rewriter.notifyMatchFailure(
          op, "expected one broadcast dimension per input dimension");
    }

    mlir::Type element_type = result_type.getElementType();
    mlir::Location loc = op.getLoc();
    mlir::Value value = adaptor.getOperand();

    // nv_tensor_ir.broadcast preserves the rank and can only expand literal
    // unit dimensions, so the input is first reshaped to the output rank. That
    // reshape relinearizes row-major, which only produces the intended result
    // if the input dimensions already appear in output order. If they do not,
    // sort them with a transpose first.
    if (!llvm::is_sorted(broadcast_dimensions)) {
      llvm::SmallVector<int64_t> permutation =
          llvm::to_vector(llvm::seq<int64_t>(0, input_type.getRank()));
      llvm::sort(permutation, [&](int64_t lhs, int64_t rhs) {
        return broadcast_dimensions[lhs] < broadcast_dimensions[rhs];
      });
      llvm::SmallVector<int64_t> transposed_shape;
      transposed_shape.reserve(permutation.size());
      for (int64_t dimension : permutation) {
        transposed_shape.push_back(input_type.getDimSize(dimension));
      }
      value = mlir::nv_tensor_ir::TransposeOp::create(
          rewriter, loc,
          mlir::RankedTensorType::get(transposed_shape, element_type), value,
          rewriter.getDenseI64ArrayAttr(permutation));
    }

    llvm::SmallVector<int64_t> expanded_shape(result_type.getRank(), 1);
    for (auto [dimension, output_dimension] :
         llvm::enumerate(broadcast_dimensions)) {
      if (output_dimension < 0 || output_dimension >= result_type.getRank()) {
        return rewriter.notifyMatchFailure(
            op, "broadcast dimension is out of bounds");
      }
      expanded_shape[output_dimension] = input_type.getDimSize(dimension);
    }
    auto expanded_type =
        mlir::RankedTensorType::get(expanded_shape, element_type);
    if (expanded_type != value.getType()) {
      value = mlir::nv_tensor_ir::ReshapeOp::create(
          rewriter, loc, expanded_type, value,
          /*dynamic_sizes=*/mlir::ValueRange{});
    }
    // A broadcast that changes no dimension is rejected by the verifier.
    if (expanded_type != result_type) {
      value = mlir::nv_tensor_ir::BroadcastOp::create(
          rewriter, loc, result_type, value,
          /*dynamic_sizes=*/mlir::ValueRange{});
    }
    rewriter.replaceOp(op, value);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Bitcast.
//===----------------------------------------------------------------------===//

// Returns true if `minor_to_major` is the default descending layout, i.e. the
// tensor is already stored in row-major order.
bool IsDefaultLayout(llvm::ArrayRef<int64_t> minor_to_major) {
  int64_t rank = minor_to_major.size();
  for (int64_t dimension : llvm::seq<int64_t>(0, rank)) {
    if (minor_to_major[dimension] != rank - 1 - dimension) {
      return false;
    }
  }
  return true;
}

// Reads a minor-to-major layout that HloFunctionImporter attached to `op` as a
// discardable `dense<[...]> : tensor<Nxindex>` attribute. Returns std::nullopt
// if the attribute is absent or is not a permutation of [0, rank).
std::optional<llvm::SmallVector<int64_t>> GetLayoutAttribute(
    mlir::Operation* op, llvm::StringRef name, int64_t rank) {
  auto attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(name);
  if (!attr || attr.getNumElements() != rank) {
    return std::nullopt;
  }
  llvm::SmallVector<int64_t> minor_to_major;
  minor_to_major.reserve(rank);
  for (const llvm::APInt& value : attr.getValues<llvm::APInt>()) {
    minor_to_major.push_back(value.getSExtValue());
  }
  llvm::SmallVector<bool> seen(rank, false);
  for (int64_t dimension : minor_to_major) {
    if (dimension < 0 || dimension >= rank || seen[dimension]) {
      return std::nullopt;
    }
    seen[dimension] = true;
  }
  return minor_to_major;
}

// Returns the dimensions of a tensor of shape `shape` and layout
// `minor_to_major`, reordered from most-major to most-minor. This is the shape
// the same buffer would have if it were relabelled to carry the default
// descending layout, i.e. xla::ShapeUtil's
// MakeShapeWithDescendingLayoutAndSamePhysicalLayout.
llvm::SmallVector<int64_t> GetPhysicalShape(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> minor_to_major) {
  llvm::SmallVector<int64_t> physical_shape;
  physical_shape.reserve(shape.size());
  for (int64_t dimension : llvm::reverse(minor_to_major)) {
    physical_shape.push_back(shape[dimension]);
  }
  return physical_shape;
}

// Lowers mhlo.bitcast, which reinterprets a buffer under a new shape and
// layout without moving any element.
//
// nv_tensor_ir tensors, like StableHLO tensors, have no layout: every tensor
// is row-major. A bitcast between two non-row-major shapes therefore cannot be
// a bare reshape, because a reshape relinearizes in row-major order. The
// lowering normalizes both sides to row-major and relinearizes in between:
//
//   1. transpose the operand from its logical order into physical
//      (major-to-minor) order, if its layout is not already the default;
//   2. reshape, if the two physical shapes differ;
//   3. transpose from the result's physical order back into its logical
//      order, if the result layout is not the default.
//
// Each step is skipped when it would be the identity, so a bitcast between two
// default-layout shapes lowers to a single reshape.
//
// The layouts are read from the `source_layout` and `result_layout` attributes
// that HloFunctionImporter attaches to every mhlo.bitcast it creates. They are
// discardable (mhlo.bitcast declares no attributes in ODS), so they are the
// only record of the layout anywhere in the imported module -- the MLIR tensor
// type cannot carry it. Without them the physical arrangement is unknowable,
// so their absence is a match failure rather than an assumption that the
// layout is the default one: guessing would silently miscompile exactly the
// non-default-layout bitcasts this code exists to handle.
struct BitcastOpConversion
    : public mlir::OpConversionPattern<mlir::mhlo::BitcastOp> {
  using mlir::OpConversionPattern<mlir::mhlo::BitcastOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::mhlo::BitcastOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto result_type = mlir::dyn_cast_or_null<mlir::RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    auto input_type =
        mlir::dyn_cast<mlir::RankedTensorType>(adaptor.getOperand().getType());
    if (!input_type) {
      return rewriter.notifyMatchFailure(op, "expected a ranked tensor input");
    }
    // A bitcast reinterprets the buffer, it does not convert elements.
    if (input_type.getElementType() != result_type.getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "expected the input and result element types to match");
    }

    std::optional<llvm::SmallVector<int64_t>> source_layout =
        GetLayoutAttribute(op, "source_layout", input_type.getRank());
    std::optional<llvm::SmallVector<int64_t>> result_layout =
        GetLayoutAttribute(op, "result_layout", result_type.getRank());
    if (!source_layout.has_value() || !result_layout.has_value()) {
      return rewriter.notifyMatchFailure(
          op,
          "mhlo.bitcast is missing the source_layout or result_layout "
          "attribute, so the physical element order is unknown");
    }

    mlir::Location loc = op.getLoc();
    mlir::Type element_type = result_type.getElementType();
    mlir::Value value = adaptor.getOperand();

    // 1. Logical -> physical order for the operand.
    if (!IsDefaultLayout(*source_layout)) {
      llvm::SmallVector<int64_t> permutation =
          llvm::to_vector(llvm::reverse(*source_layout));
      value = mlir::nv_tensor_ir::TransposeOp::create(
          rewriter, loc,
          mlir::RankedTensorType::get(
              GetPhysicalShape(input_type.getShape(), *source_layout),
              element_type),
          value, rewriter.getDenseI64ArrayAttr(permutation));
    }

    // 2. Relinearize, both sides now being row-major.
    auto physical_result_type = mlir::RankedTensorType::get(
        GetPhysicalShape(result_type.getShape(), *result_layout), element_type);
    if (physical_result_type != value.getType()) {
      value = mlir::nv_tensor_ir::ReshapeOp::create(
          rewriter, loc, physical_result_type, value,
          /*dynamic_sizes=*/mlir::ValueRange{});
    }

    // 3. Physical -> logical order for the result. The permutation is the
    // inverse of the one used in step 1, because it undoes the reordering
    // rather than applying it.
    if (!IsDefaultLayout(*result_layout)) {
      llvm::SmallVector<int64_t> permutation(result_type.getRank());
      for (auto [physical, logical] :
           llvm::enumerate(llvm::reverse(*result_layout))) {
        permutation[logical] = physical;
      }
      value = mlir::nv_tensor_ir::TransposeOp::create(
          rewriter, loc, result_type, value,
          rewriter.getDenseI64ArrayAttr(permutation));
    }

    rewriter.replaceOp(op, value);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Reductions.
//===----------------------------------------------------------------------===//

// Returns the built-in reduction mode implemented by the combiner `op`, or
// std::nullopt if `op` is not a recognized combiner. The nv_tensor_ir variants
// are accepted as well because the elementwise patterns may already have
// rewritten the combiner inside the reduction body.
std::optional<mlir::nv_tensor_ir::ReductionMode> GetReductionMode(
    mlir::Operation& op) {
  if (mlir::isa<mlir::stablehlo::AddOp, mlir::nv_tensor_ir::AddOp>(op)) {
    return mlir::nv_tensor_ir::ReductionMode::add;
  }
  if (mlir::isa<mlir::stablehlo::MulOp, mlir::nv_tensor_ir::MulOp>(op)) {
    return mlir::nv_tensor_ir::ReductionMode::mul;
  }
  if (mlir::isa<mlir::stablehlo::MaxOp, mlir::nv_tensor_ir::MaxOp>(op)) {
    return mlir::nv_tensor_ir::ReductionMode::max;
  }
  if (mlir::isa<mlir::stablehlo::MinOp, mlir::nv_tensor_ir::MinOp>(op)) {
    return mlir::nv_tensor_ir::ReductionMode::min;
  }
  return std::nullopt;
}

// Returns true if `value` is the identity element of `mode`.
// nv_tensor_ir.reduce has no init operand and implicitly uses the identity of
// its reduction mode, so a stablehlo.reduce with any other init value must not
// be lowered onto it.
bool IsReductionIdentity(mlir::DenseElementsAttr value,
                         mlir::nv_tensor_ir::ReductionMode mode) {
  if (!value.isSplat()) {
    return false;
  }
  mlir::Type element_type = value.getElementType();
  if (mlir::isa<mlir::FloatType>(element_type)) {
    llvm::APFloat splat = value.getSplatValue<llvm::APFloat>();
    switch (mode) {
      case mlir::nv_tensor_ir::ReductionMode::add:
        return splat.isZero();
      case mlir::nv_tensor_ir::ReductionMode::mul:
        return splat == llvm::APFloat(splat.getSemantics(), 1);
      case mlir::nv_tensor_ir::ReductionMode::max:
        return splat.isInfinity() && splat.isNegative();
      case mlir::nv_tensor_ir::ReductionMode::min:
        return splat.isInfinity() && !splat.isNegative();
      default:
        return false;
    }
  }
  auto integer_type = mlir::dyn_cast<mlir::IntegerType>(element_type);
  if (!integer_type) {
    return false;
  }
  llvm::APInt splat = value.getSplatValue<llvm::APInt>();
  unsigned width = integer_type.getWidth();
  bool is_unsigned = integer_type.isUnsigned();
  switch (mode) {
    case mlir::nv_tensor_ir::ReductionMode::add:
      return splat.isZero();
    case mlir::nv_tensor_ir::ReductionMode::mul:
      return splat == llvm::APInt(width, 1);
    case mlir::nv_tensor_ir::ReductionMode::max:
      return splat == (is_unsigned ? llvm::APInt::getMinValue(width)
                                   : llvm::APInt::getSignedMinValue(width));
    case mlir::nv_tensor_ir::ReductionMode::min:
      return splat == (is_unsigned ? llvm::APInt::getMaxValue(width)
                                   : llvm::APInt::getSignedMaxValue(width));
    default:
      return false;
  }
}

// Returns the signless equivalent of `type` if it is a signed or unsigned
// integer, and `type` itself otherwise. The reduce_ud region verifier
// (TensorOps.cpp, ReduceUDOp::verifyRegions) compares the block argument and
// yield types against the *signless* form of the identity's type, so the whole
// region is built over signless integers even though the identity and the
// tensor operands keep their signedness. This deliberately bypasses
// TensorIrTypeConverter, which maps signless integers to signed.
mlir::Type ToSignlessType(mlir::Type type) {
  if (auto integer_type = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (!integer_type.isSignless()) {
      return mlir::IntegerType::get(type.getContext(), integer_type.getWidth());
    }
  }
  return type;
}

// Returns a scalar TypedAttr holding the single value of the splat `value`,
// retyped to `element_type`. Used both for the reduce_ud identity, which keeps
// the signed/unsigned element type, and for constants in the combiner region,
// which are signless.
mlir::TypedAttr GetScalarAttribute(mlir::DenseElementsAttr value,
                                   mlir::Type element_type) {
  if (mlir::isa<mlir::FloatType>(element_type)) {
    return mlir::FloatAttr::get(element_type,
                                value.getSplatValue<llvm::APFloat>());
  }
  return mlir::IntegerAttr::get(element_type,
                                value.getSplatValue<llvm::APInt>());
}

// Translates one operation of a stablehlo.reduce body into the arith
// operations that implement it on signless scalars, and appends them at the
// builder's insertion point. `operands` are the already-translated scalar
// operands. Returns a null Value if the operation has no translation.
//
// The body of stablehlo.reduce operates on rank-0 tensors, whereas the
// reduce_ud region operates on plain scalars, so this is a translation and not
// a clone. Signedness is read from the *stablehlo* operand type, which still
// carries it (`ui32` for unsigned, signless `i32` for signed), because the
// scalars themselves are uniformly signless.
mlir::Value ConvertReductionBodyOp(mlir::Operation& op,
                                   mlir::ValueRange operands,
                                   mlir::OpBuilder& builder) {
  mlir::Location loc = op.getLoc();

  if (auto constant = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
    auto value = mlir::dyn_cast<mlir::DenseElementsAttr>(constant.getValue());
    if (!value || !value.isSplat()) {
      return nullptr;
    }
    return mlir::arith::ConstantOp::create(
        builder, loc,
        GetScalarAttribute(value, ToSignlessType(value.getElementType())));
  }

  // Everything below is an elementwise operation, so the element type of the
  // first operand decides which arith variant applies.
  auto operand_type = mlir::dyn_cast<mlir::RankedTensorType>(
      op.getNumOperands() > 0 ? op.getOperand(0).getType()
                              : op.getResult(0).getType());
  if (!operand_type) {
    return nullptr;
  }
  mlir::Type element_type = operand_type.getElementType();
  bool is_float = mlir::isa<mlir::FloatType>(element_type);
  // StableHLO spells unsigned integers `ui<N>` and signed ones as signless
  // `i<N>`. `i1` is XLA's PRED, which XLA treats as unsigned.
  bool is_unsigned =
      element_type.isUnsignedInteger() || element_type.isInteger(/*width=*/1);

  if (mlir::isa<mlir::stablehlo::AddOp>(op)) {
    if (is_float) {
      return mlir::arith::AddFOp::create(builder, loc, operands[0],
                                         operands[1]);
    }
    return mlir::arith::AddIOp::create(builder, loc, operands[0], operands[1]);
  }
  if (mlir::isa<mlir::stablehlo::MulOp>(op)) {
    if (is_float) {
      return mlir::arith::MulFOp::create(builder, loc, operands[0],
                                         operands[1]);
    }
    return mlir::arith::MulIOp::create(builder, loc, operands[0], operands[1]);
  }
  if (mlir::isa<mlir::stablehlo::MaxOp>(op)) {
    if (is_float) {
      return mlir::arith::MaximumFOp::create(builder, loc, operands[0],
                                             operands[1]);
    }
    if (is_unsigned) {
      return mlir::arith::MaxUIOp::create(builder, loc, operands[0],
                                          operands[1]);
    }
    return mlir::arith::MaxSIOp::create(builder, loc, operands[0], operands[1]);
  }
  if (mlir::isa<mlir::stablehlo::MinOp>(op)) {
    if (is_float) {
      return mlir::arith::MinimumFOp::create(builder, loc, operands[0],
                                             operands[1]);
    }
    if (is_unsigned) {
      return mlir::arith::MinUIOp::create(builder, loc, operands[0],
                                          operands[1]);
    }
    return mlir::arith::MinSIOp::create(builder, loc, operands[0], operands[1]);
  }
  if (mlir::isa<mlir::stablehlo::AndOp>(op)) {
    return mlir::arith::AndIOp::create(builder, loc, operands[0], operands[1]);
  }
  if (mlir::isa<mlir::stablehlo::OrOp>(op)) {
    return mlir::arith::OrIOp::create(builder, loc, operands[0], operands[1]);
  }
  if (mlir::isa<mlir::stablehlo::XorOp>(op)) {
    return mlir::arith::XOrIOp::create(builder, loc, operands[0], operands[1]);
  }
  if (mlir::isa<mlir::stablehlo::SelectOp>(op)) {
    return mlir::arith::SelectOp::create(builder, loc, operands[0], operands[1],
                                         operands[2]);
  }
  if (mlir::isa<mlir::stablehlo::ClampOp>(op)) {
    // clamp(min, x, max) == min(max, max(min, x)); see ClampOpConversion.
    if (is_float) {
      mlir::Value lower = mlir::arith::MaximumFOp::create(
          builder, loc, operands[0], operands[1]);
      return mlir::arith::MinimumFOp::create(builder, loc, operands[2], lower);
    }
    if (is_unsigned) {
      mlir::Value lower =
          mlir::arith::MaxUIOp::create(builder, loc, operands[0], operands[1]);
      return mlir::arith::MinUIOp::create(builder, loc, operands[2], lower);
    }
    mlir::Value lower =
        mlir::arith::MaxSIOp::create(builder, loc, operands[0], operands[1]);
    return mlir::arith::MinSIOp::create(builder, loc, operands[2], lower);
  }
  if (auto compare = mlir::dyn_cast<mlir::stablehlo::CompareOp>(op)) {
    // arith has no total-order comparison; see CompareOpConversion.
    if (compare.getCompareType().has_value() &&
        *compare.getCompareType() ==
            mlir::stablehlo::ComparisonType::TOTALORDER) {
      return nullptr;
    }
    if (is_float) {
      mlir::arith::CmpFPredicate predicate;
      switch (compare.getComparisonDirection()) {
        case mlir::stablehlo::ComparisonDirection::EQ:
          predicate = mlir::arith::CmpFPredicate::OEQ;
          break;
        // See CompareOpConversion for why not-equal alone is unordered.
        case mlir::stablehlo::ComparisonDirection::NE:
          predicate = mlir::arith::CmpFPredicate::UNE;
          break;
        case mlir::stablehlo::ComparisonDirection::GE:
          predicate = mlir::arith::CmpFPredicate::OGE;
          break;
        case mlir::stablehlo::ComparisonDirection::GT:
          predicate = mlir::arith::CmpFPredicate::OGT;
          break;
        case mlir::stablehlo::ComparisonDirection::LE:
          predicate = mlir::arith::CmpFPredicate::OLE;
          break;
        case mlir::stablehlo::ComparisonDirection::LT:
          predicate = mlir::arith::CmpFPredicate::OLT;
          break;
      }
      return mlir::arith::CmpFOp::create(builder, loc, predicate, operands[0],
                                         operands[1]);
    }
    mlir::arith::CmpIPredicate predicate;
    switch (compare.getComparisonDirection()) {
      case mlir::stablehlo::ComparisonDirection::EQ:
        predicate = mlir::arith::CmpIPredicate::eq;
        break;
      case mlir::stablehlo::ComparisonDirection::NE:
        predicate = mlir::arith::CmpIPredicate::ne;
        break;
      case mlir::stablehlo::ComparisonDirection::GE:
        predicate = is_unsigned ? mlir::arith::CmpIPredicate::uge
                                : mlir::arith::CmpIPredicate::sge;
        break;
      case mlir::stablehlo::ComparisonDirection::GT:
        predicate = is_unsigned ? mlir::arith::CmpIPredicate::ugt
                                : mlir::arith::CmpIPredicate::sgt;
        break;
      case mlir::stablehlo::ComparisonDirection::LE:
        predicate = is_unsigned ? mlir::arith::CmpIPredicate::ule
                                : mlir::arith::CmpIPredicate::sle;
        break;
      case mlir::stablehlo::ComparisonDirection::LT:
        predicate = is_unsigned ? mlir::arith::CmpIPredicate::ult
                                : mlir::arith::CmpIPredicate::slt;
        break;
    }
    return mlir::arith::CmpIOp::create(builder, loc, predicate, operands[0],
                                       operands[1]);
  }

  return nullptr;
}

struct ReduceOpConversion
    : public mlir::OpConversionPattern<mlir::stablehlo::ReduceOp> {
  // Runs before the elementwise patterns so that the combiner in the reduction
  // body has not been rewritten yet when the body is inspected.
  ReduceOpConversion(const mlir::TypeConverter& type_converter,
                     mlir::MLIRContext* context)
      : mlir::OpConversionPattern<mlir::stablehlo::ReduceOp>(
            type_converter, context, /*benefit=*/2) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReduceOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (adaptor.getInputs().size() != 1 || op->getNumResults() != 1) {
      // nv_tensor_ir.reduce_ud is variadic and could express a multi-output
      // reduction, but nothing upstream produces one yet, so the extra
      // bookkeeping is not carried here.
      return rewriter.notifyMatchFailure(
          op, "only single-input, single-result reductions are supported");
    }
    auto result_type = mlir::dyn_cast_or_null<mlir::RankedTensorType>(
        this->getTypeConverter()->convertType(op->getResult(0).getType()));
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    auto input_type = mlir::dyn_cast<mlir::RankedTensorType>(
        adaptor.getInputs()[0].getType());
    if (!input_type) {
      return rewriter.notifyMatchFailure(op, "expected a ranked tensor input");
    }
    if (input_type.getElementType() != result_type.getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "expected the input and result element types to match");
    }

    mlir::DenseElementsAttr init_value;
    if (!mlir::matchPattern(adaptor.getInitValues()[0],
                            mlir::m_Constant(&init_value)) ||
        !init_value.isSplat()) {
      return rewriter.notifyMatchFailure(
          op, "expected a splat constant reduction init value");
    }

    llvm::SmallVector<int32_t> dimensions;
    dimensions.reserve(op.getDimensions().size());
    for (int64_t dimension : op.getDimensions()) {
      // nv_tensor_ir.reduce takes an i32 dense array. Bounding the dimension
      // by the tensor rank also makes the narrowing below lossless.
      if (dimension < 0 || dimension >= input_type.getRank()) {
        return rewriter.notifyMatchFailure(
            op, "reduction dimension is out of range");
      }
      dimensions.push_back(static_cast<int32_t>(dimension));
    }
    if (dimensions.empty()) {
      return rewriter.notifyMatchFailure(
          op, "expected at least one reduction dimension");
    }

    // Both nv_tensor_ir.reduce and nv_tensor_ir.reduce_ud preserve the rank
    // and contract the reduced dimensions to 1, while stablehlo.reduce drops
    // them, so a reshape is needed to recover the stablehlo result shape.
    llvm::SmallVector<int64_t> reduced_shape(input_type.getShape());
    for (int32_t dimension : dimensions) {
      reduced_shape[dimension] = 1;
    }
    auto reduced_type =
        mlir::RankedTensorType::get(reduced_shape, input_type.getElementType());

    mlir::Value reduced = BuildReduce(op, adaptor, reduced_type, dimensions,
                                      init_value, rewriter);
    if (!reduced) {
      return rewriter.notifyMatchFailure(
          op, "reduction body has no nv_tensor_ir equivalent");
    }
    if (reduced_type != result_type) {
      reduced = mlir::nv_tensor_ir::ReshapeOp::create(
          rewriter, op.getLoc(), result_type, reduced,
          /*dynamic_sizes=*/mlir::ValueRange{});
    }
    rewriter.replaceOp(op, reduced);
    return mlir::success();
  }

 private:
  // Emits the reduction itself, preferring the built-in nv_tensor_ir.reduce
  // over nv_tensor_ir.reduce_ud because the tiling pipeline understands the
  // built-in modes best. Returns a null Value if neither applies.
  static mlir::Value BuildReduce(mlir::stablehlo::ReduceOp op,
                                 OpAdaptor adaptor,
                                 mlir::RankedTensorType reduced_type,
                                 llvm::ArrayRef<int32_t> dimensions,
                                 mlir::DenseElementsAttr init_value,
                                 mlir::ConversionPatternRewriter& rewriter) {
    if (std::optional<mlir::nv_tensor_ir::ReductionMode> mode =
            GetBuiltinReductionMode(op, init_value)) {
      return mlir::nv_tensor_ir::ReduceOp::create(
          rewriter, op.getLoc(), reduced_type, adaptor.getInputs()[0],
          rewriter.getDenseI32ArrayAttr(dimensions), *mode);
    }
    return BuildUserDefinedReduce(op, adaptor, reduced_type, dimensions,
                                  init_value, rewriter);
  }

  // Returns the built-in reduction mode this reduction can use, if any. The
  // combiner has to be a single recognized operation over the two block
  // arguments, and the init value has to be that mode's implicit identity,
  // because nv_tensor_ir.reduce has no init operand.
  static std::optional<mlir::nv_tensor_ir::ReductionMode>
  GetBuiltinReductionMode(mlir::stablehlo::ReduceOp op,
                          mlir::DenseElementsAttr init_value) {
    mlir::Block& body = op.getBody().front();
    if (!llvm::hasNItems(body.without_terminator(), 1)) {
      return std::nullopt;
    }
    mlir::Operation& combiner = *body.begin();
    mlir::Operation* terminator = body.getTerminator();
    if (terminator->getNumOperands() != 1 || combiner.getNumResults() != 1 ||
        terminator->getOperand(0) != combiner.getResult(0)) {
      return std::nullopt;
    }
    // The combiner has to consume both block arguments and nothing else.
    // add/mul/max/min are all commutative, so the order does not matter, but
    // a body such as `add(%arg0, %arg0)` is not a plain reduction and must not
    // be mistaken for one.
    if (body.getNumArguments() != 2 || combiner.getNumOperands() != 2) {
      return std::nullopt;
    }
    mlir::Value lhs = combiner.getOperand(0);
    mlir::Value rhs = combiner.getOperand(1);
    mlir::Value arg0 = body.getArgument(0);
    mlir::Value arg1 = body.getArgument(1);
    if (!((lhs == arg0 && rhs == arg1) || (lhs == arg1 && rhs == arg0))) {
      return std::nullopt;
    }
    std::optional<mlir::nv_tensor_ir::ReductionMode> mode =
        GetReductionMode(combiner);
    if (!mode.has_value() || !IsReductionIdentity(init_value, *mode)) {
      return std::nullopt;
    }
    return mode;
  }

  // Emits nv_tensor_ir.reduce_ud, whose explicit identity and combiner region
  // can express any reduction body, and translates the stablehlo body into
  // that region. Returns a null Value if the body cannot be translated.
  static mlir::Value BuildUserDefinedReduce(
      mlir::stablehlo::ReduceOp op, OpAdaptor adaptor,
      mlir::RankedTensorType reduced_type, llvm::ArrayRef<int32_t> dimensions,
      mlir::DenseElementsAttr init_value,
      mlir::ConversionPatternRewriter& rewriter) {
    mlir::Location loc = op.getLoc();
    // The identity keeps the converted (signed or unsigned) element type: the
    // op verifier compares it against the result element type as is.
    mlir::TypedAttr identity =
        GetScalarAttribute(init_value, reduced_type.getElementType());
    auto reduce_op = mlir::nv_tensor_ir::ReduceUDOp::create(
        rewriter, loc, mlir::TypeRange{reduced_type}, adaptor.getInputs(),
        rewriter.getDenseI32ArrayAttr(dimensions),
        rewriter.getArrayAttr({identity}));

    // The region takes (prev_result, curr_operand) over *signless* scalars.
    mlir::Type scalar_type = ToSignlessType(reduced_type.getElementType());
    mlir::Block* body = rewriter.createBlock(&reduce_op.getBody());
    body->addArgument(scalar_type, loc);
    body->addArgument(scalar_type, loc);

    mlir::Block& source_body = op.getBody().front();
    if (source_body.getNumArguments() != body->getNumArguments()) {
      rewriter.eraseOp(reduce_op);
      return nullptr;
    }
    mlir::IRMapping mapping;
    mapping.map(source_body.getArguments(), body->getArguments());
    for (mlir::Operation& source_op : source_body.without_terminator()) {
      llvm::SmallVector<mlir::Value> operands;
      operands.reserve(source_op.getNumOperands());
      for (mlir::Value operand : source_op.getOperands()) {
        operands.push_back(mapping.lookupOrNull(operand));
      }
      if (llvm::is_contained(operands, mlir::Value())) {
        rewriter.eraseOp(reduce_op);
        return nullptr;
      }
      mlir::Value result =
          ConvertReductionBodyOp(source_op, operands, rewriter);
      if (!result || source_op.getNumResults() != 1) {
        rewriter.eraseOp(reduce_op);
        return nullptr;
      }
      mapping.map(source_op.getResult(0), result);
    }

    llvm::SmallVector<mlir::Value> yielded;
    for (mlir::Value operand : source_body.getTerminator()->getOperands()) {
      yielded.push_back(mapping.lookupOrNull(operand));
    }
    if (yielded.size() != 1 || llvm::is_contained(yielded, mlir::Value())) {
      rewriter.eraseOp(reduce_op);
      return nullptr;
    }
    mlir::nv_tensor_ir::YieldOp::create(rewriter, loc, yielded);

    rewriter.setInsertionPointAfter(reduce_op);
    return reduce_op.getResult(0);
  }
};

//===----------------------------------------------------------------------===//
// Matmul.
//===----------------------------------------------------------------------===//

// Returns the dimensions of a rank-`rank` operand that are neither batch nor
// contracting dimensions ("free" dimensions), in increasing order. Keeping
// them in the operand's original relative order is what makes the final
// reshape back to the stablehlo result shape a pure relinearization.
llvm::SmallVector<int64_t> GetFreeDimensions(
    int64_t rank, llvm::ArrayRef<int64_t> batch_dimensions,
    llvm::ArrayRef<int64_t> contracting_dimensions) {
  llvm::SmallVector<int64_t> free_dimensions;
  for (int64_t dimension : llvm::seq<int64_t>(0, rank)) {
    if (llvm::is_contained(batch_dimensions, dimension) ||
        llvm::is_contained(contracting_dimensions, dimension)) {
      continue;
    }
    free_dimensions.push_back(dimension);
  }
  return free_dimensions;
}

// Returns the product of the extents of `dimensions` in `type`, i.e. the size
// of the single dimension they collapse into. The product of no dimensions is
// 1, which is the right neutral extent for a missing batch or free group.
int64_t GetDimensionProduct(mlir::RankedTensorType type,
                            llvm::ArrayRef<int64_t> dimensions) {
  int64_t product = 1;
  for (int64_t dimension : dimensions) {
    product *= type.getDimSize(dimension);
  }
  return product;
}

// Returns `value` permuted by `permutation`, or `value` unchanged if the
// permutation is the identity. Both dialects define the permutation as
// output[i] = input[permutation[i]].
mlir::Value MaybeTranspose(mlir::ConversionPatternRewriter& rewriter,
                           mlir::Location loc, mlir::Value value,
                           llvm::ArrayRef<int64_t> permutation) {
  // A permutation of [0, rank) is the identity exactly when it is sorted.
  if (llvm::is_sorted(permutation)) {
    return value;
  }
  auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
  llvm::SmallVector<int64_t> shape;
  shape.reserve(permutation.size());
  for (int64_t dimension : permutation) {
    shape.push_back(type.getDimSize(dimension));
  }
  return mlir::nv_tensor_ir::TransposeOp::create(
      rewriter, loc, type.clone(shape), value,
      rewriter.getDenseI64ArrayAttr(permutation));
}

// Returns `value` reshaped to `shape`, or `value` unchanged if it already has
// exactly that type. Comparing types rather than shapes keeps an identity
// reshape from showing up in the output.
mlir::Value MaybeReshape(mlir::ConversionPatternRewriter& rewriter,
                         mlir::Location loc, mlir::Value value,
                         llvm::ArrayRef<int64_t> shape) {
  auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
  mlir::RankedTensorType reshaped_type = type.clone(shape);
  if (reshaped_type == type) {
    return value;
  }
  return mlir::nv_tensor_ir::ReshapeOp::create(
      rewriter, loc, reshaped_type, value,
      /*dynamic_sizes=*/mlir::ValueRange{});
}

// Returns failure if `precision_config` requests anything other than DEFAULT.
//
// nv_tensor_ir.matmul carries neither a precision nor an algorithm attribute,
// so precision_config and algorithm have nowhere to go. Rather than silently
// dropping a request that changes the numerics, reject it. An all-DEFAULT
// precision_config asks for nothing in particular and is accepted (and
// dropped), which is what almost every dot carries.
mlir::LogicalResult CheckDefaultPrecision(
    mlir::Operation* op, std::optional<mlir::ArrayAttr> precision_config,
    mlir::ConversionPatternRewriter& rewriter) {
  if (!precision_config.has_value()) {
    return mlir::success();
  }
  for (mlir::Attribute precision : *precision_config) {
    auto precision_attr =
        mlir::dyn_cast<mlir::stablehlo::PrecisionAttr>(precision);
    if (!precision_attr ||
        precision_attr.getValue() != mlir::stablehlo::Precision::DEFAULT) {
      return rewriter.notifyMatchFailure(
          op, "nv_tensor_ir.matmul cannot express a non-default precision");
    }
  }
  return mlir::success();
}

// Lowers a dot onto nv_tensor_ir.matmul by reducing it to a canonical rank-3
// (batched) or rank-2 (unbatched) matmul:
//
//   1. lhs: transpose to [batch..., free..., contracting...], then reshape to
//      [B, M, K] with B = prod(batch), M = prod(free), K = prod(contracting).
//   2. rhs: transpose to [batch..., contracting..., free...], then reshape to
//      [B, K, N].
//   3. matmul -> [B, M, N].
//   4. reshape to the stablehlo result shape, which is
//      [batch..., lhs_free..., rhs_free...] in exactly that order.
//
// Each transpose and reshape is only emitted when it is not the identity, so
// the common already-canonical dot lowers to a bare matmul. On success `op` is
// replaced.
mlir::LogicalResult LowerDotToMatmul(
    mlir::Operation* op, mlir::Value lhs_value, mlir::Value rhs_value,
    llvm::ArrayRef<int64_t> lhs_batch, llvm::ArrayRef<int64_t> rhs_batch,
    llvm::ArrayRef<int64_t> lhs_contract, llvm::ArrayRef<int64_t> rhs_contract,
    mlir::RankedTensorType result_type,
    mlir::ConversionPatternRewriter& rewriter) {
  auto lhs_type = mlir::dyn_cast<mlir::RankedTensorType>(lhs_value.getType());
  auto rhs_type = mlir::dyn_cast<mlir::RankedTensorType>(rhs_value.getType());
  if (!lhs_type || !rhs_type) {
    return rewriter.notifyMatchFailure(op, "expected ranked tensor operands");
  }

  if (lhs_batch.size() != rhs_batch.size()) {
    return rewriter.notifyMatchFailure(
        op, "expected the same number of batch dimensions on both operands");
  }
  if (lhs_contract.size() != rhs_contract.size()) {
    return rewriter.notifyMatchFailure(
        op,
        "expected the same number of contracting dimensions on both "
        "operands");
  }
  auto in_range = [](llvm::ArrayRef<int64_t> dimensions, int64_t rank) {
    return llvm::all_of(dimensions, [rank](int64_t dimension) {
      return dimension >= 0 && dimension < rank;
    });
  };
  if (!in_range(lhs_batch, lhs_type.getRank()) ||
      !in_range(lhs_contract, lhs_type.getRank()) ||
      !in_range(rhs_batch, rhs_type.getRank()) ||
      !in_range(rhs_contract, rhs_type.getRank())) {
    return rewriter.notifyMatchFailure(op, "dot dimension is out of range");
  }
  // The batch and contracting lists pair up positionally between the two
  // operands, so the paired extents have to agree for the collapsed B and K
  // to be the same on both sides.
  for (auto [lhs_dimension, rhs_dimension] : llvm::zip(lhs_batch, rhs_batch)) {
    if (lhs_type.getDimSize(lhs_dimension) !=
        rhs_type.getDimSize(rhs_dimension)) {
      return rewriter.notifyMatchFailure(
          op, "paired batch dimensions have different extents");
    }
  }
  for (auto [lhs_dimension, rhs_dimension] :
       llvm::zip(lhs_contract, rhs_contract)) {
    if (lhs_type.getDimSize(lhs_dimension) !=
        rhs_type.getDimSize(rhs_dimension)) {
      return rewriter.notifyMatchFailure(
          op, "paired contracting dimensions have different extents");
    }
  }

  llvm::SmallVector<int64_t> lhs_free =
      GetFreeDimensions(lhs_type.getRank(), lhs_batch, lhs_contract);
  llvm::SmallVector<int64_t> rhs_free =
      GetFreeDimensions(rhs_type.getRank(), rhs_batch, rhs_contract);

  int64_t batch_size = GetDimensionProduct(lhs_type, lhs_batch);
  int64_t contracting_size = GetDimensionProduct(lhs_type, lhs_contract);
  int64_t lhs_free_size = GetDimensionProduct(lhs_type, lhs_free);
  int64_t rhs_free_size = GetDimensionProduct(rhs_type, rhs_free);
  // Without batch dimensions the canonical form is the plain rank-2 matmul;
  // introducing a size-1 batch dimension would only add two reshapes.
  bool has_batch = !lhs_batch.empty();

  mlir::Location loc = op->getLoc();

  llvm::SmallVector<int64_t> lhs_permutation = llvm::to_vector(lhs_batch);
  llvm::append_range(lhs_permutation, lhs_free);
  llvm::append_range(lhs_permutation, lhs_contract);
  llvm::SmallVector<int64_t> lhs_shape;
  if (has_batch) {
    lhs_shape.push_back(batch_size);
  }
  lhs_shape.push_back(lhs_free_size);
  lhs_shape.push_back(contracting_size);
  mlir::Value lhs = MaybeTranspose(rewriter, loc, lhs_value, lhs_permutation);
  lhs = MaybeReshape(rewriter, loc, lhs, lhs_shape);

  llvm::SmallVector<int64_t> rhs_permutation = llvm::to_vector(rhs_batch);
  llvm::append_range(rhs_permutation, rhs_contract);
  llvm::append_range(rhs_permutation, rhs_free);
  llvm::SmallVector<int64_t> rhs_shape;
  if (has_batch) {
    rhs_shape.push_back(batch_size);
  }
  rhs_shape.push_back(contracting_size);
  rhs_shape.push_back(rhs_free_size);
  mlir::Value rhs = MaybeTranspose(rewriter, loc, rhs_value, rhs_permutation);
  rhs = MaybeReshape(rewriter, loc, rhs, rhs_shape);

  llvm::SmallVector<int64_t> matmul_shape;
  if (has_batch) {
    matmul_shape.push_back(batch_size);
  }
  matmul_shape.push_back(lhs_free_size);
  matmul_shape.push_back(rhs_free_size);
  mlir::Value matmul = mlir::nv_tensor_ir::MatmulOp::create(
      rewriter, loc, result_type.clone(matmul_shape), lhs, rhs);

  rewriter.replaceOp(
      op, MaybeReshape(rewriter, loc, matmul, result_type.getShape()));
  return mlir::success();
}

struct DotGeneralOpConversion
    : public mlir::OpConversionPattern<mlir::stablehlo::DotGeneralOp> {
  using mlir::OpConversionPattern<
      mlir::stablehlo::DotGeneralOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::DotGeneralOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto result_type = mlir::dyn_cast_or_null<mlir::RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    if (op.getAlgorithm().has_value()) {
      return rewriter.notifyMatchFailure(
          op, "nv_tensor_ir.matmul cannot express a dot algorithm");
    }
    if (mlir::failed(
            CheckDefaultPrecision(op, op.getPrecisionConfig(), rewriter))) {
      return mlir::failure();
    }

    mlir::stablehlo::DotDimensionNumbersAttr dimension_numbers =
        op.getDotDimensionNumbers();
    return LowerDotToMatmul(op, adaptor.getLhs(), adaptor.getRhs(),
                            dimension_numbers.getLhsBatchingDimensions(),
                            dimension_numbers.getRhsBatchingDimensions(),
                            dimension_numbers.getLhsContractingDimensions(),
                            dimension_numbers.getRhsContractingDimensions(),
                            result_type, rewriter);
  }
};

// stablehlo.dot is what the HLO importer emits for a plain two-operand dot.
// It has no dimension numbers: it always contracts the last dimension of the
// lhs with the first dimension of the rhs and has no batch dimensions, so it
// is just a dot_general with those numbers filled in.
struct DotOpConversion
    : public mlir::OpConversionPattern<mlir::stablehlo::DotOp> {
  using mlir::OpConversionPattern<mlir::stablehlo::DotOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::DotOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto result_type = mlir::dyn_cast_or_null<mlir::RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    if (mlir::failed(
            CheckDefaultPrecision(op, op.getPrecisionConfig(), rewriter))) {
      return mlir::failure();
    }
    auto lhs_type =
        mlir::dyn_cast<mlir::RankedTensorType>(adaptor.getLhs().getType());
    if (!lhs_type || lhs_type.getRank() == 0) {
      return rewriter.notifyMatchFailure(
          op, "expected a ranked lhs of rank at least one");
    }
    int64_t lhs_contract = lhs_type.getRank() - 1;
    int64_t rhs_contract = 0;
    return LowerDotToMatmul(op, adaptor.getLhs(), adaptor.getRhs(),
                            /*lhs_batch=*/{}, /*rhs_batch=*/{},
                            llvm::ArrayRef<int64_t>(lhs_contract),
                            llvm::ArrayRef<int64_t>(rhs_contract), result_type,
                            rewriter);
  }
};

struct ResultsOpConversion
    : public mlir::OpConversionPattern<mlir::nv_tensor_ir::ResultsOp> {
  using mlir::OpConversionPattern<
      mlir::nv_tensor_ir::ResultsOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::nv_tensor_ir::ResultsOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::nv_tensor_ir::ResultsOp>(
        op, adaptor.getOperands());
    return mlir::success();
  }
};

void populateLegalizeStablehloToTensorIrPatterns(
    mlir::RewritePatternSet& patterns, mlir::TypeConverter& type_converter,
    mlir::MLIRContext* context) {
  // Unary elementwise operations.
  patterns.add<ElementwiseUnaryOpConversion<mlir::stablehlo::AbsOp,
                                            mlir::nv_tensor_ir::AbsOp>,
               ElementwiseUnaryOpConversion<mlir::stablehlo::CeilOp,
                                            mlir::nv_tensor_ir::CeilOp>,
               ElementwiseUnaryOpConversion<mlir::stablehlo::FloorOp,
                                            mlir::nv_tensor_ir::FloorOp>,
               ElementwiseUnaryOpConversion<mlir::stablehlo::NegOp,
                                            mlir::nv_tensor_ir::NegOp>,
               ElementwiseUnaryOpConversion<mlir::stablehlo::CosineOp,
                                            mlir::nv_tensor_ir::CosOp>,
               ElementwiseUnaryOpConversion<mlir::stablehlo::SineOp,
                                            mlir::nv_tensor_ir::SinOp>,
               ElementwiseUnaryOpConversion<mlir::stablehlo::TanOp,
                                            mlir::nv_tensor_ir::TanOp>,
               ElementwiseUnaryOpConversion<mlir::stablehlo::ExpOp,
                                            mlir::nv_tensor_ir::ExpOp>,
               ElementwiseUnaryOpConversion<mlir::stablehlo::LogOp,
                                            mlir::nv_tensor_ir::LogOp>,
               ElementwiseUnaryOpConversion<mlir::stablehlo::RsqrtOp,
                                            mlir::nv_tensor_ir::RsqrtOp>,
               ElementwiseUnaryOpConversion<mlir::stablehlo::SqrtOp,
                                            mlir::nv_tensor_ir::SqrtOp>,
               ElementwiseUnaryOpConversion<mlir::stablehlo::TanhOp,
                                            mlir::nv_tensor_ir::TanhFwdOp>,
               ElementwiseUnaryOpConversion<mlir::stablehlo::LogisticOp,
                                            mlir::nv_tensor_ir::SigmoidFwdOp>,
               ElementwiseUnaryOpConversion<mlir::stablehlo::NotOp,
                                            mlir::nv_tensor_ir::LogicalNotOp>,
               // The HLO importer has no StableHLO op to emit for Erf -- there
               // is none -- so it emits mhlo.erf.
               ElementwiseUnaryOpConversion<mlir::mhlo::ErfOp,
                                            mlir::nv_tensor_ir::ErfOp>>(
      type_converter, context);

  // Binary elementwise operations.
  // Note: stablehlo.remainder follows the sign of the dividend (truncated
  // remainder), matching nv_tensor_ir.rem, NOT nv_tensor_ir.mod which follows
  // floored-remainder semantics.
  patterns.add<ElementwiseBinaryOpConversion<mlir::stablehlo::AddOp,
                                             mlir::nv_tensor_ir::AddOp>,
               ElementwiseBinaryOpConversion<mlir::stablehlo::SubtractOp,
                                             mlir::nv_tensor_ir::SubOp>,
               ElementwiseBinaryOpConversion<mlir::stablehlo::MulOp,
                                             mlir::nv_tensor_ir::MulOp>,
               ElementwiseBinaryOpConversion<mlir::stablehlo::DivOp,
                                             mlir::nv_tensor_ir::DivOp>,
               ElementwiseBinaryOpConversion<mlir::stablehlo::MaxOp,
                                             mlir::nv_tensor_ir::MaxOp>,
               ElementwiseBinaryOpConversion<mlir::stablehlo::MinOp,
                                             mlir::nv_tensor_ir::MinOp>,
               ElementwiseBinaryOpConversion<mlir::stablehlo::PowOp,
                                             mlir::nv_tensor_ir::PowOp>,
               ElementwiseBinaryOpConversion<mlir::stablehlo::RemOp,
                                             mlir::nv_tensor_ir::RemOp>,
               ElementwiseBinaryOpConversion<mlir::stablehlo::Atan2Op,
                                             mlir::nv_tensor_ir::Atan2Op>,
               ElementwiseBinaryOpConversion<mlir::stablehlo::AndOp,
                                             mlir::nv_tensor_ir::LogicalAndOp>,
               ElementwiseBinaryOpConversion<mlir::stablehlo::OrOp,
                                             mlir::nv_tensor_ir::LogicalOrOp>>(
      type_converter, context);

  // Select, Compare, Convert, and Results.
  // Note: stablehlo.xor has no nv_tensor_ir equivalent at tensor granularity
  // and is intentionally left unhandled so that the conversion target and
  // final check reject it. Inside a reduction body it is expressible, and
  // ConvertReductionBodyOp does handle it.
  patterns.add<SelectOpConversion, CompareOpConversion, ConvertOpConversion,
               ResultsOpConversion>(type_converter, context);

  // Operations without a direct nv_tensor_ir counterpart, expressed in terms
  // of the operations that do exist.
  patterns.add<ClampOpConversion, Expm1OpConversion, Log1pOpConversion>(
      type_converter, context);

  // Structural operations, constants and matmul.
  // Note: pad, gather, scatter, dynamic_slice, sort, reverse, convolution and
  // the control flow operations have no nv_tensor_ir equivalent and are
  // intentionally left unhandled.
  patterns.add<ReshapeOpConversion, TransposeOpConversion, SliceOpConversion,
               ConcatenateOpConversion, IotaOpConversion, ConstantOpConversion,
               BroadcastInDimOpConversion, BitcastOpConversion,
               DotGeneralOpConversion, DotOpConversion>(type_converter,
                                                        context);

  // Reductions, registered with a higher benefit than the elementwise
  // patterns so that the combiner in the reduction body can be inspected.
  patterns.add<ReduceOpConversion>(type_converter, context);
}

// Rewrites the rank-0 tensors CudaTile cannot handle to rank 1, reshaping on
// the inside so that the rest of the body is unchanged.
//
// CudaTile's layout propagation rejects a rank-0 tensor in three places, each
// with an unhelpful message: a rank-0 graph argument and a rank-0 constant
// both fail with "failed to compute layout", and a rank-0 graph result with
// "failed to normalize layout". A rank-0 value produced by a `reshape` is
// accepted, which is what makes this rewrite possible at all. XLA, meanwhile,
// produces all three routinely: a scalar fusion parameter, a scalar constant
// feeding a broadcast, and a full reduce.
//
// An XLA buffer for a rank-0 shape and for the corresponding `[1]` shape have
// the same size and alignment, and both have a (trivially) default layout. So
// declaring the boundary as `[1]` leaves the kernel ABI untouched, and leaves
// `AttachLayoutStrides`, which only emits an attribute for a non-default
// layout, with nothing to do in either case.
void PromoteRank0Tensors(mlir::nv_tensor_ir::GraphOp graph) {
  static constexpr int64_t kRank1Shape[] = {1};

  mlir::Block& body = graph.getGraphBody().front();
  auto results_op =
      mlir::dyn_cast<mlir::nv_tensor_ir::ResultsOp>(body.getTerminator());
  if (results_op == nullptr) {
    return;
  }

  mlir::OpBuilder builder(graph.getContext());

  // `create` inserts before the insertion point without moving it, so
  // successive reshapes land in argument order rather than reversed.
  builder.setInsertionPointToStart(&body);
  llvm::SmallVector<mlir::Type> input_types;
  input_types.reserve(body.getNumArguments());
  for (mlir::BlockArgument arg : body.getArguments()) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(arg.getType());
    if (type == nullptr || type.getRank() != 0) {
      input_types.push_back(arg.getType());
      continue;
    }
    arg.setType(type.clone(kRank1Shape));
    auto reshape = mlir::nv_tensor_ir::ReshapeOp::create(
        builder, arg.getLoc(), type, arg,
        /*dynamic_sizes=*/mlir::ValueRange{});
    arg.replaceAllUsesExcept(reshape.getResult(), reshape);
    input_types.push_back(arg.getType());
  }

  builder.setInsertionPoint(results_op);
  llvm::SmallVector<mlir::Type> result_types;
  result_types.reserve(results_op->getNumOperands());
  for (mlir::OpOperand& operand : results_op->getOpOperands()) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(operand.get().getType());
    if (type == nullptr || type.getRank() != 0) {
      result_types.push_back(operand.get().getType());
      continue;
    }
    mlir::RankedTensorType promoted = type.clone(kRank1Shape);
    auto reshape = mlir::nv_tensor_ir::ReshapeOp::create(
        builder, results_op.getLoc(), promoted, operand.get(),
        /*dynamic_sizes=*/mlir::ValueRange{});
    operand.set(reshape.getResult());
    result_types.push_back(promoted);
  }

  graph.setFunctionType(
      mlir::FunctionType::get(graph.getContext(), input_types, result_types));

  // A rank-0 constant is rejected even though it is not on the boundary, so
  // rebuild it at rank 1 and reshape back down.
  llvm::SmallVector<mlir::nv_tensor_ir::ConstantOp> rank0_constants;
  graph.walk([&](mlir::nv_tensor_ir::ConstantOp constant) {
    auto type =
        mlir::dyn_cast<mlir::RankedTensorType>(constant.getResult().getType());
    if (type != nullptr && type.getRank() == 0) {
      rank0_constants.push_back(constant);
    }
  });
  for (mlir::nv_tensor_ir::ConstantOp constant : rank0_constants) {
    auto value = mlir::dyn_cast<mlir::DenseElementsAttr>(constant.getValue());
    if (value == nullptr) {
      continue;
    }
    auto type =
        mlir::cast<mlir::RankedTensorType>(constant.getResult().getType());
    builder.setInsertionPoint(constant);
    auto promoted = mlir::nv_tensor_ir::ConstantOp::create(
        builder, constant.getLoc(),
        mlir::cast<mlir::TypedAttr>(value.reshape(type.clone(kRank1Shape))));
    auto reshape = mlir::nv_tensor_ir::ReshapeOp::create(
        builder, constant.getLoc(), type, promoted.getResult(),
        /*dynamic_sizes=*/mlir::ValueRange{});
    constant.getResult().replaceAllUsesWith(reshape.getResult());
    constant.erase();
  }

  // A reshape is a pure shape reinterpretation, so reshape(reshape(x)) is just
  // reshape(x), and where the two cancel out it is x itself. Promotion creates
  // such chains wherever it meets a reshape the importer already emitted: a
  // full reduce, for example, is `reduce -> tensor<1xT>` followed by
  // `reshape -> tensor<T>`, and promoting the result reshapes it straight back
  // up. Collapse them rather than leaving them in the graph.
  //
  // `walk` is post-order, so the inner reshape of a chain is always rewritten
  // before the outer one and one pass suffices. Erasing is left to a second
  // pass, running outermost first, because erasing during the first would
  // leave dangling entries in `reshapes`.
  llvm::SmallVector<mlir::nv_tensor_ir::ReshapeOp> reshapes;
  graph.walk([&](mlir::nv_tensor_ir::ReshapeOp reshape) {
    // More than one operand means dynamic result sizes, which XLA never
    // produces and which would not be safe to re-point.
    if (reshape->getNumOperands() == 1) {
      reshapes.push_back(reshape);
    }
  });
  for (mlir::nv_tensor_ir::ReshapeOp reshape : reshapes) {
    auto inner =
        reshape.getOperand(0).getDefiningOp<mlir::nv_tensor_ir::ReshapeOp>();
    if (inner != nullptr && inner->getNumOperands() == 1) {
      reshape.setOperand(0, inner.getOperand(0));
    }
  }
  for (mlir::nv_tensor_ir::ReshapeOp reshape : llvm::reverse(reshapes)) {
    if (reshape.getOperand(0).getType() == reshape.getResult().getType()) {
      reshape.getResult().replaceAllUsesWith(reshape.getOperand(0));
    }
    if (reshape.use_empty()) {
      reshape.erase();
    }
  }
}

class LegalizeStablehloToTensorIrPass
    : public impl::LegalizeStablehloToTensorIrPassBase<
          LegalizeStablehloToTensorIrPass> {
 public:
  using LegalizeStablehloToTensorIrPassBase::
      LegalizeStablehloToTensorIrPassBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext* context = &getContext();

    TensorIrTypeConverter type_converter;

    // Collect func.func operations to process.
    llvm::SmallVector<mlir::func::FuncOp> func_ops;
    module.walk(
        [&](mlir::func::FuncOp func_op) { func_ops.push_back(func_op); });

    for (mlir::func::FuncOp func_op : func_ops) {
      mlir::OpBuilder builder(func_op);

      // Convert function signature types.
      auto func_type = func_op.getFunctionType();
      llvm::SmallVector<mlir::Type> new_inputs;
      llvm::SmallVector<mlir::Type> new_results;
      if (failed(
              type_converter.convertTypes(func_type.getInputs(), new_inputs)) ||
          failed(type_converter.convertTypes(func_type.getResults(),
                                             new_results))) {
        func_op.emitOpError("failed to convert function signature types");
        signalPassFailure();
        return;
      }
      auto converted_func_type =
          builder.getFunctionType(new_inputs, new_results);

      auto graph_op = mlir::nv_tensor_ir::GraphOp::create(
          builder, func_op.getLoc(), func_op.getName(),
          /*sym_visibility=*/nullptr, converted_func_type,
          func_op.getArgAttrsAttr(), func_op.getResAttrsAttr());

      // Move the function's body region into the graph's graphBody region.
      graph_op.getGraphBody().takeBody(func_op.getBody());

      // Convert block argument types.
      mlir::Block& entry_block = graph_op.getGraphBody().front();
      for (auto [arg, new_type] :
           llvm::zip(entry_block.getArguments(), new_inputs)) {
        arg.setType(new_type);
      }

      // Replace the terminating mlir::func::ReturnOp with
      // mlir::nv_tensor_ir::ResultsOp carrying the same operands.
      if (auto return_op = mlir::dyn_cast<mlir::func::ReturnOp>(
              entry_block.getTerminator())) {
        builder.setInsertionPoint(return_op);
        mlir::nv_tensor_ir::ResultsOp::create(builder, return_op.getLoc(),
                                              return_op.getOperands());
        return_op.erase();
      }

      func_op.erase();
    }

    mlir::ConversionTarget target(*context);
    // Illegal by default. Only the dialects that legitimately survive into the
    // output are legalized. StableHLO is not the only source dialect this pass
    // sees: HloFunctionImporter emits mhlo ops (e.g. mhlo.bitcast, mhlo.erf)
    // for constructs StableHLO has no equivalent for. If the target were only
    // "stablehlo is illegal", those would be legal by default, survive inside
    // the nv_tensor_ir.graph body, still verify and print, and only fail much
    // later when nothing can lower them.
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation*) { return false; });
    target
        .addLegalDialect<mlir::nv_tensor_ir::TensorIRDialect,
                         mlir::arith::ArithDialect, mlir::func::FuncDialect>();
    // The module is the operation being converted; the builtin dialect is
    // otherwise illegal so that a leftover unrealized_conversion_cast is
    // flagged too.
    target.addLegalOp<mlir::ModuleOp>();
    target.addDynamicallyLegalOp<mlir::nv_tensor_ir::ResultsOp>(
        [&](mlir::nv_tensor_ir::ResultsOp op) {
          return type_converter.isLegal(op);
        });

    mlir::RewritePatternSet patterns(context);
    populateLegalizeStablehloToTensorIrPatterns(patterns, type_converter,
                                                context);

    if (mlir::failed(mlir::applyPartialConversion(module, target,
                                                  std::move(patterns)))) {
      // Deliberately no early return: the conversion rolled back, and the walk
      // below names every op that was left illegal.
      signalPassFailure();
    }

    // Some operands of the source IR only survive as attributes on the
    // nv_tensor_ir side: a stablehlo.reduce init value becomes the `identity`
    // attribute of reduce/reduce_ud, for example. Their (already converted)
    // defining ops are then dead, and would otherwise be printed as unused
    // rank-0 constants in the graph body. The dialect conversion driver does
    // not clean those up, so do it here.
    //
    // This has to run to a fixpoint: deadness cascades, and a single walk only
    // catches the first layer. `walk` is post-order, so an op is always visited
    // before the op that consumes it, and a constant feeding a dead convert
    // still looks live on the pass that removes the convert. Post-order also
    // guarantees a nested op is erased before its parent, so erasing in
    // collection order can never leave a dangling pointer.
    bool erased_any = true;
    while (erased_any) {
      erased_any = false;
      llvm::SmallVector<mlir::Operation*> dead_ops;
      module.walk([&](mlir::Operation* op) {
        if (op != module.getOperation() && mlir::isOpTriviallyDead(op)) {
          dead_ops.push_back(op);
        }
      });
      for (mlir::Operation* op : dead_ops) {
        op->erase();
        erased_any = true;
      }
    }

    // applyPartialConversion only reports ops its patterns were asked to
    // convert, so walk the module and flag anything the target considers
    // illegal, naming the offending op.
    bool found_illegal_op = false;
    module.walk([&](mlir::Operation* op) {
      if (op == module.getOperation()) {
        return;
      }
      if (!target.isLegal(op)) {
        op->emitOpError() << "operation was not legalized to nv_tensor_ir "
                             "dialect";
        found_illegal_op = true;
      }
    });

    if (found_illegal_op) {
      signalPassFailure();
      return;
    }

    // Runs last, on a fully legalized graph, so that it only ever sees
    // nv_tensor_ir types and can insert nv_tensor_ir.reshape directly.
    for (auto graph : module.getOps<mlir::nv_tensor_ir::GraphOp>()) {
      PromoteRank0Tensors(graph);
    }
  }
};

}  // namespace

}  // namespace mlir::nv_tensor_ir::xla
