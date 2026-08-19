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

#include <complex>
#include <cstdint>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // IWYU pragma: keep
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"  // IWYU pragma: keep
#include "xla/codegen/xtile/ir/xtile_ops.h"

namespace xla {
namespace xtile {

#define GEN_PASS_DEF_EXPANDXTILECOMPLEXOPSPASS
#include "xla/codegen/xtile/ir/transforms/passes.h.inc"

namespace {

using mlir::ArrayRef;
using mlir::ComplexType;
using mlir::FloatType;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::SmallVector;
using mlir::Type;
using mlir::UnrealizedConversionCastOp;
using mlir::Value;
using mlir::ValueRange;
using xtile::EntryFuncOp;
using xtile::ExtractTileOp;
using xtile::InsertTileOp;

namespace ma = ::mlir::arith;
namespace mm = ::mlir::math;
namespace shlo = ::mlir::stablehlo;

// Appends 2 to the shape of the given shaped type.
SmallVector<int64_t> ExpandShape(llvm::ArrayRef<int64_t> shape) {
  SmallVector<int64_t> new_shape = llvm::to_vector(shape);
  new_shape.push_back(2);
  return new_shape;
}

Type GetExpandedType(Type type) {
  auto shaped_type = mlir::dyn_cast<ShapedType>(type);
  if (!shaped_type) {
    return type;
  }
  auto complex_type = mlir::dyn_cast<ComplexType>(shaped_type.getElementType());
  if (!complex_type) {
    return type;
  }
  if (auto tensor_type = mlir::dyn_cast<RankedTensorType>(type)) {
    return RankedTensorType::get(ExpandShape(shaped_type.getShape()),
                                 complex_type.getElementType());
  }
  if (auto memref_type = mlir::dyn_cast<MemRefType>(type)) {
    return MemRefType::get(ExpandShape(shaped_type.getShape()),
                           complex_type.getElementType(),
                           /*layout=*/mlir::MemRefLayoutAttrInterface(),
                           memref_type.getMemorySpace());
  }
  return type;
}

bool HasComplex(Type type) {
  if (auto shaped_type = mlir::dyn_cast<ShapedType>(type)) {
    return mlir::isa<ComplexType>(shaped_type.getElementType());
  }
  return mlir::isa<ComplexType>(type);
}

Value UnwrapCast(Value value, PatternRewriter& rewriter) {
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    return cast.getOperand(0);
  }
  if (!HasComplex(value.getType())) {
    return value;
  }
  return UnrealizedConversionCastOp::create(
             rewriter, value.getLoc(), GetExpandedType(value.getType()), value)
      .getResult(0);
}

Value ExtractFloatingPointTensor(Value complex_val, Location loc,
                                 PatternRewriter& rewriter,
                                 bool is_real = true) {
  auto complex_tensor_type =
      mlir::cast<RankedTensorType>(complex_val.getType());
  auto shape = complex_tensor_type.getShape();
  int64_t rank = complex_tensor_type.getRank();
  auto elem_type = complex_tensor_type.getElementType();

  SmallVector<int64_t> slice_shape(shape.begin(), shape.end() - 1);
  slice_shape.push_back(1);
  auto slice_type = RankedTensorType::get(slice_shape, elem_type);

  SmallVector<int64_t> orig_shape(shape.begin(), shape.end() - 1);
  auto orig_type = RankedTensorType::get(orig_shape, elem_type);

  SmallVector<int64_t> start_indices(rank, 0);
  start_indices.back() = is_real ? 0 : 1;
  SmallVector<int64_t> limit_indices(shape.begin(), shape.end());
  limit_indices.back() = is_real ? 1 : 2;
  SmallVector<int64_t> strides(rank, 1);

  auto slice = shlo::SliceOp::create(rewriter, loc, slice_type, complex_val,
                                     start_indices, limit_indices, strides);
  return shlo::ReshapeOp::create(rewriter, loc, orig_type, slice.getResult());
}

Value ExtractReal(Value complex_val, Location loc, PatternRewriter& rewriter) {
  return ExtractFloatingPointTensor(complex_val, loc, rewriter,
                                    /*is_real=*/true);
}

Value ExtractImag(Value complex_val, Location loc, PatternRewriter& rewriter) {
  return ExtractFloatingPointTensor(complex_val, loc, rewriter,
                                    /*is_real=*/false);
}

Value ConcatRealAndImag(Value real, Value imag, Location loc,
                        PatternRewriter& rewriter) {
  auto real_type = mlir::cast<RankedTensorType>(real.getType());
  ArrayRef<int64_t> shape = real_type.getShape();
  Type elem_type = real_type.getElementType();

  SmallVector<int64_t> slice_shape(shape.begin(), shape.end());
  slice_shape.push_back(1);
  auto slice_type = RankedTensorType::get(slice_shape, elem_type);

  auto expanded_type = RankedTensorType::get(ExpandShape(shape), elem_type);

  auto real_reshaped = shlo::ReshapeOp::create(rewriter, loc, slice_type, real);
  auto imag_reshaped = shlo::ReshapeOp::create(rewriter, loc, slice_type, imag);

  return shlo::ConcatenateOp::create(
      rewriter, loc, expanded_type,
      ValueRange{real_reshaped.getResult(), imag_reshaped.getResult()},
      /*dimension=*/real_type.getRank());
}

Value GetConstant(PatternRewriter& rewriter, Location loc,
                  RankedTensorType type, double value) {
  auto float_type = mlir::cast<FloatType>(type.getElementType());
  auto attr = mlir::DenseElementsAttr::get(
      type, rewriter.getFloatAttr(float_type, value));
  return ma::ConstantOp::create(rewriter, loc, type, attr);
}

Value GetInfConstant(PatternRewriter& rewriter, Location loc,
                     RankedTensorType type, bool negative = false) {
  auto float_type = mlir::cast<FloatType>(type.getElementType());
  auto ap_float =
      llvm::APFloat::getInf(float_type.getFloatSemantics(), negative);
  auto attr = mlir::DenseElementsAttr::get(
      type, rewriter.getFloatAttr(float_type, ap_float));
  return ma::ConstantOp::create(rewriter, loc, type, attr);
}

Value ComputeAbs(Value real, Value imag, Location loc,
                 PatternRewriter& rewriter) {
  auto tensor_type = mlir::cast<RankedTensorType>(real.getType());
  Value one = GetConstant(rewriter, loc, tensor_type, 1.0);
  Value abs_a = mm::AbsFOp::create(rewriter, loc, real);
  Value abs_b = mm::AbsFOp::create(rewriter, loc, imag);

  Value max = ma::MaximumFOp::create(rewriter, loc, abs_a, abs_b);
  Value min = ma::MinimumFOp::create(rewriter, loc, abs_a, abs_b);

  Value ratio = ma::DivFOp::create(rewriter, loc, min, max);
  Value ratio_sq = ma::MulFOp::create(rewriter, loc, ratio, ratio);
  Value ratio_sq_plus_one = ma::AddFOp::create(rewriter, loc, ratio_sq, one);

  Value sqrt = mm::SqrtOp::create(rewriter, loc, ratio_sq_plus_one);
  Value result = ma::MulFOp::create(rewriter, loc, max, sqrt);

  Value is_nan =
      ma::CmpFOp::create(rewriter, loc, ma::CmpFPredicate::UNO, result, result);
  return ma::SelectOp::create(rewriter, loc, is_nan, min, result);
}

// Follows powOpConversionImplementation in ComplexToStandard.cpp upstream.
// Converts lhs^y = (a+bi)^(c+di) to
//    (a*a+b*b)^(0.5c) * exp(-d*atan2(b,a)) * (cos(q) + i*sin(q)),
//    where q = c*atan2(b,a)+0.5d*ln(a*a+b*b)
Value ComputePower(Value a, Value b, Value c, Value d, Location loc,
                   PatternRewriter& rewriter) {
  auto tensor_type = mlir::cast<RankedTensorType>(a.getType());

  Value abs = ComputeAbs(a, b, loc, rewriter);
  Value abs_to_c = mm::PowFOp::create(rewriter, loc, abs, c);

  Value neg_d = ma::NegFOp::create(rewriter, loc, d);
  Value arg_lhs = mm::Atan2Op::create(rewriter, loc, b, a);
  Value neg_d_arg_lhs = ma::MulFOp::create(rewriter, loc, neg_d, arg_lhs);
  Value exp_neg_d_arg_lhs = mm::ExpOp::create(rewriter, loc, neg_d_arg_lhs);

  Value coeff = ma::MulFOp::create(rewriter, loc, abs_to_c, exp_neg_d_arg_lhs);
  Value ln_abs = mm::LogOp::create(rewriter, loc, abs);
  Value c_arg_lhs = ma::MulFOp::create(rewriter, loc, c, arg_lhs);
  Value d_ln_abs = ma::MulFOp::create(rewriter, loc, d, ln_abs);
  Value q = ma::AddFOp::create(rewriter, loc, c_arg_lhs, d_ln_abs);
  Value cos_q = mm::CosOp::create(rewriter, loc, q);
  Value sin_q = mm::SinOp::create(rewriter, loc, q);

  Value inf = GetInfConstant(rewriter, loc, tensor_type);
  Value zero = GetConstant(rewriter, loc, tensor_type, 0.0);
  Value one = GetConstant(rewriter, loc, tensor_type, 1.0);

  // Case 0:
  // d^c is 0 if d is 0 and c > 0. 0^0 is defined to be 1.0, see
  // Branch Cuts for Complex Elementary Functions or Much Ado About
  // Nothing's Sign Bit, W. Kahan, Section 10.
  Value abs_eq_zero =
      ma::CmpFOp::create(rewriter, loc, ma::CmpFPredicate::OEQ, abs, zero);
  Value d_eq_zero =
      ma::CmpFOp::create(rewriter, loc, ma::CmpFPredicate::OEQ, d, zero);
  Value c_eq_zero =
      ma::CmpFOp::create(rewriter, loc, ma::CmpFPredicate::OEQ, c, zero);
  Value b_eq_zero =
      ma::CmpFOp::create(rewriter, loc, ma::CmpFPredicate::OEQ, b, zero);

  Value zero_le_c =
      ma::CmpFOp::create(rewriter, loc, ma::CmpFPredicate::OLE, zero, c);
  Value coeff_cos_q = ma::MulFOp::create(rewriter, loc, coeff, cos_q);
  Value coeff_sin_q = ma::MulFOp::create(rewriter, loc, coeff, sin_q);

  Value complex_one_or_zero_real =
      ma::SelectOp::create(rewriter, loc, c_eq_zero, one, zero);
  Value complex_one_or_zero_imag = zero;

  Value abs_and_d = ma::AndIOp::create(rewriter, loc, abs_eq_zero, d_eq_zero);
  Value cond0 = ma::AndIOp::create(rewriter, loc, abs_and_d, zero_le_c);

  Value cutoff0_real = ma::SelectOp::create(
      rewriter, loc, cond0, complex_one_or_zero_real, coeff_cos_q);
  Value cutoff0_imag = ma::SelectOp::create(
      rewriter, loc, cond0, complex_one_or_zero_imag, coeff_sin_q);

  // Case 1:
  // x^0 is defined to be 1 for any x, see
  // Branch Cuts for Complex Elementary Functions or Much Ado About
  // Nothing's Sign Bit, W. Kahan, Section 10.
  Value rhs_eq_zero = ma::AndIOp::create(rewriter, loc, c_eq_zero, d_eq_zero);
  Value cutoff1_real =
      ma::SelectOp::create(rewriter, loc, rhs_eq_zero, one, cutoff0_real);
  Value cutoff1_imag =
      ma::SelectOp::create(rewriter, loc, rhs_eq_zero, zero, cutoff0_imag);

  // Case 2:
  // 1^(c + d*i) = 1 + 0*i
  Value a_eq_one =
      ma::CmpFOp::create(rewriter, loc, ma::CmpFPredicate::OEQ, a, one);
  Value lhs_eq_one = ma::AndIOp::create(rewriter, loc, a_eq_one, b_eq_zero);
  Value cutoff2_real =
      ma::SelectOp::create(rewriter, loc, lhs_eq_one, one, cutoff1_real);
  Value cutoff2_imag =
      ma::SelectOp::create(rewriter, loc, lhs_eq_one, zero, cutoff1_imag);

  // Case 3:
  // inf^(c + 0*i) = inf + 0*i, c > 0
  Value a_eq_inf =
      ma::CmpFOp::create(rewriter, loc, ma::CmpFPredicate::OEQ, a, inf);
  Value lhs_eq_inf = ma::AndIOp::create(rewriter, loc, a_eq_inf, b_eq_zero);
  Value c_gt_zero =
      ma::CmpFOp::create(rewriter, loc, ma::CmpFPredicate::OGT, c, zero);
  Value rhs_gt_zero = ma::AndIOp::create(rewriter, loc, d_eq_zero, c_gt_zero);
  Value cond3 = ma::AndIOp::create(rewriter, loc, lhs_eq_inf, rhs_gt_zero);
  Value cutoff3_real =
      ma::SelectOp::create(rewriter, loc, cond3, inf, cutoff2_real);
  Value cutoff3_imag =
      ma::SelectOp::create(rewriter, loc, cond3, zero, cutoff2_imag);

  // Case 4:
  // inf^(c + 0*i) = 0 + 0*i, c < 0
  Value c_lt_zero =
      ma::CmpFOp::create(rewriter, loc, ma::CmpFPredicate::OLT, c, zero);
  Value rhs_lt_zero = ma::AndIOp::create(rewriter, loc, d_eq_zero, c_lt_zero);
  Value cond4 = ma::AndIOp::create(rewriter, loc, lhs_eq_inf, rhs_lt_zero);
  Value cutoff4_real =
      ma::SelectOp::create(rewriter, loc, cond4, zero, cutoff3_real);
  Value cutoff4_imag =
      ma::SelectOp::create(rewriter, loc, cond4, zero, cutoff3_imag);

  return ConcatRealAndImag(cutoff4_real, cutoff4_imag, loc, rewriter);
}

LogicalResult RewriteFunctionSignatures(EntryFuncOp op,
                                        PatternRewriter& rewriter) {
  if (op.getNumResults() != 0) {
    return rewriter.notifyMatchFailure(
        op, "function with non-zero results not supported");
  }
  auto input_types = op.getFunctionType().getInputs();
  if (llvm::none_of(input_types, HasComplex)) {
    return rewriter.notifyMatchFailure(op, "nothing to expand");
  }
  mlir::Location loc = op.getLoc();
  if (op.getBody().empty()) {
    return rewriter.notifyMatchFailure(op, "empty function body");
  }
  mlir::Block* entry_block = &op.getBody().front();

  // Cast all function arguments to the original type.
  SmallVector<Type> new_operand_types(input_types);
  rewriter.setInsertionPointToStart(entry_block);
  for (auto&& [index, operand_type] : llvm::enumerate(new_operand_types)) {
    if (!HasComplex(operand_type)) {
      continue;
    }
    mlir::BlockArgument func_argument = op.getArgument(index);
    auto cast_to_orig_type = UnrealizedConversionCastOp::create(
        rewriter, loc, operand_type, func_argument);
    func_argument.replaceAllUsesExcept(cast_to_orig_type.getResult(0),
                                       cast_to_orig_type);
    operand_type = GetExpandedType(operand_type);
  }
  // Replace the function arguments with the new types.
  for (auto [arg, arg_type] :
       llvm::zip(entry_block->getArguments(), new_operand_types)) {
    arg.setType(arg_type);
  }
  // Update function signature.
  op.setType(rewriter.getFunctionType(new_operand_types, {}));
  return mlir::success();
}

LogicalResult RewriteExtractTileOp(ExtractTileOp op,
                                   PatternRewriter& rewriter) {
  auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
  if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }

  mlir::Location loc = op.getLoc();
  auto new_tensor_type = GetExpandedType(tensor_type);
  Value source = UnwrapCast(op.getSource(), rewriter);

  SmallVector<Value> offsets(op.getOffsets().begin(), op.getOffsets().end());
  offsets.push_back(ma::ConstantIndexOp::create(rewriter, loc, 0));

  SmallVector<int64_t> tile_strides = llvm::to_vector(op.getStrides());
  tile_strides.push_back(1);

  auto new_extract =
      ExtractTileOp::create(rewriter, loc, new_tensor_type, source, offsets,
                            ExpandShape(op.getFullTileShape()), tile_strides);

  auto cast_to_orig_type = UnrealizedConversionCastOp::create(
      rewriter, loc, tensor_type, new_extract.getResult());
  rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
  return mlir::success();
}

LogicalResult RewriteInsertTileOp(InsertTileOp op, PatternRewriter& rewriter) {
  auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getSource().getType());
  if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }

  auto loc = op.getLoc();
  Value source = UnwrapCast(op.getSource(), rewriter);
  Value destination = UnwrapCast(op.getDestination(), rewriter);

  SmallVector<Value> offsets(op.getOffsets().begin(), op.getOffsets().end());
  offsets.push_back(ma::ConstantIndexOp::create(rewriter, loc, 0));

  SmallVector<int64_t> tile_strides = llvm::to_vector(op.getStrides());
  tile_strides.push_back(1);

  InsertTileOp::create(rewriter, loc, source, destination, offsets,
                       ExpandShape(op.getFullTileShape()), tile_strides);
  rewriter.eraseOp(op);
  return mlir::success();
}

LogicalResult RewriteAddOp(shlo::AddOp op, PatternRewriter& rewriter) {
  auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
  if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }
  auto loc = op.getLoc();
  auto new_type = GetExpandedType(tensor_type);
  Value lhs = UnwrapCast(op.getLhs(), rewriter);
  Value rhs = UnwrapCast(op.getRhs(), rewriter);
  auto new_op = shlo::AddOp::create(rewriter, loc, new_type, lhs, rhs);
  auto cast_to_orig_type = UnrealizedConversionCastOp::create(
      rewriter, loc, tensor_type, new_op.getResult());
  rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
  return mlir::success();
}

LogicalResult RewriteSubtractOp(shlo::SubtractOp op,
                                PatternRewriter& rewriter) {
  auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
  if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }

  auto loc = op.getLoc();
  auto new_type = GetExpandedType(tensor_type);
  Value lhs = UnwrapCast(op.getLhs(), rewriter);
  Value rhs = UnwrapCast(op.getRhs(), rewriter);

  auto new_op = shlo::SubtractOp::create(rewriter, loc, new_type, lhs, rhs);
  auto cast_to_orig_type = UnrealizedConversionCastOp::create(
      rewriter, loc, tensor_type, new_op.getResult());
  rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
  return mlir::success();
}

LogicalResult RewriteMulOp(shlo::MulOp op, PatternRewriter& rewriter) {
  auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
  if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }
  auto loc = op.getLoc();
  Value lhs = UnwrapCast(op.getLhs(), rewriter);
  Value rhs = UnwrapCast(op.getRhs(), rewriter);

  Value a = ExtractReal(lhs, loc, rewriter);
  Value b = ExtractImag(lhs, loc, rewriter);
  Value c = ExtractReal(rhs, loc, rewriter);
  Value d = ExtractImag(rhs, loc, rewriter);

  Value ac = shlo::MulOp::create(rewriter, loc, a, c);
  Value bd = shlo::MulOp::create(rewriter, loc, b, d);
  Value real = shlo::SubtractOp::create(rewriter, loc, ac, bd);

  Value ad = shlo::MulOp::create(rewriter, loc, a, d);
  Value bc = shlo::MulOp::create(rewriter, loc, b, c);
  Value imag = shlo::AddOp::create(rewriter, loc, ad, bc);

  Value combined = ConcatRealAndImag(real, imag, loc, rewriter);
  auto cast_to_orig_type =
      UnrealizedConversionCastOp::create(rewriter, loc, tensor_type, combined);
  rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
  return mlir::success();
}

LogicalResult RewriteNegOp(shlo::NegOp op, PatternRewriter& rewriter) {
  auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
  if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }
  auto loc = op.getLoc();
  auto new_type = GetExpandedType(tensor_type);
  Value operand = UnwrapCast(op.getOperand(), rewriter);
  auto new_op = shlo::NegOp::create(rewriter, loc, new_type, operand);
  auto cast_to_orig_type = UnrealizedConversionCastOp::create(
      rewriter, loc, tensor_type, new_op.getResult());
  rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
  return mlir::success();
}

LogicalResult RewriteRealOp(shlo::RealOp op, PatternRewriter& rewriter) {
  auto tensor_type =
      mlir::dyn_cast<RankedTensorType>(op.getOperand().getType());
  if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }
  Value complex_val = UnwrapCast(op.getOperand(), rewriter);
  Value real = ExtractReal(complex_val, op.getLoc(), rewriter);
  rewriter.replaceOp(op, real);
  return mlir::success();
}

LogicalResult RewriteImagOp(shlo::ImagOp op, PatternRewriter& rewriter) {
  auto tensor_type =
      mlir::dyn_cast<RankedTensorType>(op.getOperand().getType());
  if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }
  Value complex_val = UnwrapCast(op.getOperand(), rewriter);
  Value imag = ExtractImag(complex_val, op.getLoc(), rewriter);
  rewriter.replaceOp(op, imag);
  return mlir::success();
}

LogicalResult RewriteComplexOp(shlo::ComplexOp op, PatternRewriter& rewriter) {
  auto result_type = mlir::dyn_cast<RankedTensorType>(op.getType());
  if (!result_type || !mlir::isa<ComplexType>(result_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }
  Value combined =
      ConcatRealAndImag(op.getLhs(), op.getRhs(), op.getLoc(), rewriter);
  auto cast_to_orig_type = UnrealizedConversionCastOp::create(
      rewriter, op.getLoc(), result_type, combined);
  rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
  return mlir::success();
}

template <typename T>
mlir::DenseElementsAttr ExpandComplexDenseAttr(
    mlir::RankedTensorType new_type, mlir::DenseElementsAttr dense_attr) {
  SmallVector<T> values;
  values.reserve(dense_attr.getNumElements() * 2);
  for (const auto& val : dense_attr.getValues<std::complex<T>>()) {
    values.push_back(val.real());
    values.push_back(val.imag());
  }
  return mlir::DenseElementsAttr::get(new_type, llvm::ArrayRef(values));
}

LogicalResult RewriteConstantOp(ma::ConstantOp op, PatternRewriter& rewriter) {
  auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
  if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }
  auto complex_type = mlir::cast<ComplexType>(tensor_type.getElementType());
  Type elem_type = complex_type.getElementType();
  auto new_type =
      RankedTensorType::get(ExpandShape(tensor_type.getShape()), elem_type);

  auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
  if (!dense_attr) {
    return rewriter.notifyMatchFailure(op, "not a dense elements attr");
  }

  mlir::Location loc = op.getLoc();
  mlir::DenseElementsAttr new_attr;
  if (elem_type.isF32()) {
    new_attr = ExpandComplexDenseAttr<float>(new_type, dense_attr);
  } else if (elem_type.isF64()) {
    new_attr = ExpandComplexDenseAttr<double>(new_type, dense_attr);
  } else {
    return rewriter.notifyMatchFailure(op, "unsupported complex element type");
  }
  auto new_op = ma::ConstantOp::create(rewriter, loc, new_type, new_attr);
  auto cast_to_orig_type = UnrealizedConversionCastOp::create(
      rewriter, loc, tensor_type, new_op.getResult());
  rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
  return mlir::success();
}

LogicalResult RewriteBroadcastInDimOp(shlo::BroadcastInDimOp op,
                                      PatternRewriter& rewriter) {
  auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
  if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }

  auto loc = op.getLoc();
  auto new_type = GetExpandedType(tensor_type);
  Value operand = UnwrapCast(op.getOperand(), rewriter);

  SmallVector<int64_t> new_broadcast_dimensions(
      op.getBroadcastDimensions().begin(), op.getBroadcastDimensions().end());
  new_broadcast_dimensions.push_back(tensor_type.getRank());

  auto new_op = shlo::BroadcastInDimOp::create(
      rewriter, loc, new_type, operand,
      rewriter.getDenseI64ArrayAttr(new_broadcast_dimensions));
  auto cast_to_orig_type = UnrealizedConversionCastOp::create(
      rewriter, loc, tensor_type, new_op.getResult());
  rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
  return mlir::success();
}

// Follows DivOpConversion in ComplexToStandard.cpp upstream for
// ComplexRangeFlags::basic.
LogicalResult RewriteDivOp(shlo::DivOp op, PatternRewriter& rewriter) {
  auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
  if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }
  auto loc = op.getLoc();
  Value lhs = UnwrapCast(op.getLhs(), rewriter);
  Value rhs = UnwrapCast(op.getRhs(), rewriter);

  Value a = ExtractReal(lhs, loc, rewriter);
  Value b = ExtractImag(lhs, loc, rewriter);
  Value c = ExtractReal(rhs, loc, rewriter);
  Value d = ExtractImag(rhs, loc, rewriter);

  Value c_sq = shlo::MulOp::create(rewriter, loc, c, c);
  Value d_sq = shlo::MulOp::create(rewriter, loc, d, d);
  Value denom = shlo::AddOp::create(rewriter, loc, c_sq, d_sq);

  Value ac = shlo::MulOp::create(rewriter, loc, a, c);
  Value bd = shlo::MulOp::create(rewriter, loc, b, d);
  Value real_num = shlo::AddOp::create(rewriter, loc, ac, bd);

  Value bc = shlo::MulOp::create(rewriter, loc, b, c);
  Value ad = shlo::MulOp::create(rewriter, loc, a, d);
  Value imag_num = shlo::SubtractOp::create(rewriter, loc, bc, ad);

  Value real = shlo::DivOp::create(rewriter, loc, real_num, denom);
  Value imag = shlo::DivOp::create(rewriter, loc, imag_num, denom);

  Value combined = ConcatRealAndImag(real, imag, loc, rewriter);
  auto cast_to_orig_type =
      UnrealizedConversionCastOp::create(rewriter, loc, tensor_type, combined);
  rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
  return mlir::success();
}

// Follows AbsOpConversion in ComplexToStandard.cpp upstream for AbsFn::abs.
LogicalResult RewriteAbsOp(shlo::AbsOp op, PatternRewriter& rewriter) {
  auto operand_type =
      mlir::dyn_cast<RankedTensorType>(op.getOperand().getType());
  if (!operand_type || !mlir::isa<ComplexType>(operand_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }
  auto loc = op.getLoc();
  Value complex_val = UnwrapCast(op.getOperand(), rewriter);
  Value a = ExtractReal(complex_val, loc, rewriter);
  Value b = ExtractImag(complex_val, loc, rewriter);

  Value abs = ComputeAbs(a, b, loc, rewriter);
  rewriter.replaceOp(op, abs);
  return mlir::success();
}

LogicalResult RewriteCompareOp(shlo::CompareOp op, PatternRewriter& rewriter) {
  auto lhs_type = mlir::dyn_cast<RankedTensorType>(op.getLhs().getType());
  if (!lhs_type || !mlir::isa<ComplexType>(lhs_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }
  auto dir = op.getComparisonDirection();
  if (dir != shlo::ComparisonDirection::EQ &&
      dir != shlo::ComparisonDirection::NE) {
    return rewriter.notifyMatchFailure(
        op, "unsupported complex comparison direction");
  }
  auto loc = op.getLoc();
  Value lhs = UnwrapCast(op.getLhs(), rewriter);
  Value rhs = UnwrapCast(op.getRhs(), rewriter);

  Value a = ExtractReal(lhs, loc, rewriter);
  Value b = ExtractImag(lhs, loc, rewriter);
  Value c = ExtractReal(rhs, loc, rewriter);
  Value d = ExtractImag(rhs, loc, rewriter);

  if (dir == shlo::ComparisonDirection::EQ) {
    Value real_eq = shlo::CompareOp::create(rewriter, loc, a, c,
                                            shlo::ComparisonDirection::EQ);
    Value imag_eq = shlo::CompareOp::create(rewriter, loc, b, d,
                                            shlo::ComparisonDirection::EQ);
    Value res = shlo::AndOp::create(rewriter, loc, real_eq, imag_eq);
    rewriter.replaceOp(op, res);
    return mlir::success();
  }
  if (dir == shlo::ComparisonDirection::NE) {
    Value real_ne = shlo::CompareOp::create(rewriter, loc, a, c,
                                            shlo::ComparisonDirection::NE);
    Value imag_ne = shlo::CompareOp::create(rewriter, loc, b, d,
                                            shlo::ComparisonDirection::NE);
    Value res = shlo::OrOp::create(rewriter, loc, real_ne, imag_ne);
    rewriter.replaceOp(op, res);
    return mlir::success();
  }
  return rewriter.notifyMatchFailure(
      op, "unsupported complex comparison direction");
}

LogicalResult RewritePowOp(shlo::PowOp op, PatternRewriter& rewriter) {
  auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
  if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }
  auto loc = op.getLoc();
  Value lhs = UnwrapCast(op.getLhs(), rewriter);
  Value rhs = UnwrapCast(op.getRhs(), rewriter);

  Value a = ExtractReal(lhs, loc, rewriter);
  Value b = ExtractImag(lhs, loc, rewriter);
  Value c = ExtractReal(rhs, loc, rewriter);
  Value d = ExtractImag(rhs, loc, rewriter);

  Value combined = ComputePower(a, b, c, d, loc, rewriter);
  auto cast_to_orig_type =
      UnrealizedConversionCastOp::create(rewriter, loc, tensor_type, combined);
  rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
  return mlir::success();
}

LogicalResult RewriteArithSelectOp(ma::SelectOp op, PatternRewriter& rewriter) {
  auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
  if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }
  auto loc = op.getLoc();
  Value cond = op.getCondition();
  Value true_val = UnwrapCast(op.getTrueValue(), rewriter);
  Value false_val = UnwrapCast(op.getFalseValue(), rewriter);

  Value true_real = ExtractReal(true_val, loc, rewriter);
  Value true_imag = ExtractImag(true_val, loc, rewriter);
  Value false_real = ExtractReal(false_val, loc, rewriter);
  Value false_imag = ExtractImag(false_val, loc, rewriter);

  Value res_real =
      ma::SelectOp::create(rewriter, loc, cond, true_real, false_real);
  Value res_imag =
      ma::SelectOp::create(rewriter, loc, cond, true_imag, false_imag);

  Value combined = ConcatRealAndImag(res_real, res_imag, loc, rewriter);
  auto cast_to_orig_type =
      UnrealizedConversionCastOp::create(rewriter, loc, tensor_type, combined);
  rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
  return mlir::success();
}

LogicalResult RewriteStablehloSelectOp(shlo::SelectOp op,
                                       PatternRewriter& rewriter) {
  auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
  if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }
  auto loc = op.getLoc();
  Value cond = op.getPred();
  Value true_val = UnwrapCast(op.getOnTrue(), rewriter);
  Value false_val = UnwrapCast(op.getOnFalse(), rewriter);

  Value true_real = ExtractReal(true_val, loc, rewriter);
  Value true_imag = ExtractImag(true_val, loc, rewriter);
  Value false_real = ExtractReal(false_val, loc, rewriter);
  Value false_imag = ExtractImag(false_val, loc, rewriter);

  Value res_real =
      shlo::SelectOp::create(rewriter, loc, cond, true_real, false_real);
  Value res_imag =
      shlo::SelectOp::create(rewriter, loc, cond, true_imag, false_imag);

  Value combined = ConcatRealAndImag(res_real, res_imag, loc, rewriter);
  auto cast_to_orig_type =
      UnrealizedConversionCastOp::create(rewriter, loc, tensor_type, combined);
  rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
  return mlir::success();
}

LogicalResult RewriteReshapeOp(shlo::ReshapeOp op, PatternRewriter& rewriter) {
  auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
  if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }

  auto loc = op.getLoc();
  auto new_type = GetExpandedType(tensor_type);
  Value operand = UnwrapCast(op.getOperand(), rewriter);

  auto new_op = shlo::ReshapeOp::create(rewriter, loc, new_type, operand);
  auto cast_to_orig_type = UnrealizedConversionCastOp::create(
      rewriter, loc, tensor_type, new_op.getResult());
  rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
  return mlir::success();
}

LogicalResult RewriteTransposeOp(shlo::TransposeOp op,
                                 PatternRewriter& rewriter) {
  auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
  if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
    return rewriter.notifyMatchFailure(op, "not a complex tensor");
  }

  auto loc = op.getLoc();
  auto new_type = GetExpandedType(tensor_type);
  Value operand = UnwrapCast(op.getOperand(), rewriter);

  SmallVector<int64_t> new_permutation(op.getPermutation().begin(),
                                       op.getPermutation().end());
  auto operand_orig_type =
      mlir::cast<RankedTensorType>(op.getOperand().getType());
  new_permutation.push_back(operand_orig_type.getRank());

  auto new_op =
      shlo::TransposeOp::create(rewriter, loc, new_type, operand,
                                rewriter.getDenseI64ArrayAttr(new_permutation));
  auto cast_to_orig_type = UnrealizedConversionCastOp::create(
      rewriter, loc, tensor_type, new_op.getResult());
  rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
  return mlir::success();
}

class ExpandXtileComplexOpsPass
    : public impl::ExpandXtileComplexOpsPassBase<ExpandXtileComplexOpsPass> {
 public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);

    patterns.add(RewriteAbsOp);
    patterns.add(RewriteAddOp);
    patterns.add(RewriteArithSelectOp);
    patterns.add(RewriteBroadcastInDimOp);
    patterns.add(RewriteCompareOp);
    patterns.add(RewriteComplexOp);
    patterns.add(RewriteConstantOp);
    patterns.add(RewriteDivOp);
    patterns.add(RewriteExtractTileOp);
    patterns.add(RewriteFunctionSignatures);
    patterns.add(RewriteImagOp);
    patterns.add(RewriteInsertTileOp);
    patterns.add(RewriteMulOp);
    patterns.add(RewriteNegOp);
    patterns.add(RewritePowOp);
    patterns.add(RewriteRealOp);
    patterns.add(RewriteReshapeOp);
    patterns.add(RewriteStablehloSelectOp);
    patterns.add(RewriteSubtractOp);
    patterns.add(RewriteTransposeOp);

    if (mlir::failed(
            mlir::applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
    // Check if there are no unrealized_conversion_casts from/to complex types.
    bool module_has_casts =
        module
            .walk([](UnrealizedConversionCastOp op) {
              if (llvm::any_of(mlir::TypeRange(op.getOperands()), HasComplex) ||
                  llvm::any_of(mlir::TypeRange(op.getResults()), HasComplex)) {
                return mlir::WalkResult::interrupt();
              }
              return mlir::WalkResult::advance();
            })
            .wasInterrupted();
    if (module_has_casts) {
      llvm::outs() << "ExpandXtileComplexOpsPass failed to converge";
      signalPassFailure();
      return;
    }
  }
};

}  // namespace
}  // namespace xtile
}  // namespace xla
