/* Copyright 2025 The OpenXLA Authors.

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
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/gpu/codegen/triton/collective_emitter.h"
#include "xla/backends/gpu/codegen/triton/dot_algorithms.h"
#include "xla/backends/gpu/codegen/triton/emitter_helpers.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/hlo/translate/mhlo_to_hlo/attribute_exporter.h"
#include "xla/service/algorithm_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/tensor_float_32_utils.h"
#include "third_party/triton/include/triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

namespace ttir = ::mlir::triton;

#define GEN_PASS_DEF_STABLEHLOLOWERTOTRITONPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

class LowerTranspose : public mlir::OpRewritePattern<stablehlo::TransposeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      stablehlo::TransposeOp op,
      mlir::PatternRewriter& rewriter) const override {
    SmallVector<int32_t> permutation =
        llvm::to_vector_of<int32_t>(op.getPermutation());
    rewriter.replaceOpWithNewOp<ttir::TransOp>(op, op.getResult().getType(),
                                               op.getOperand(), permutation);
    return mlir::success();
  }
};

class LowerIotaToMakeRange : public mlir::OpRewritePattern<stablehlo::IotaOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      stablehlo::IotaOp op, mlir::PatternRewriter& rewriter) const override {
    auto result_type = op.getResult().getType();

    if (result_type.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), "tt.make_range is only supported for 1D outputs.");
    }

    if (!result_type.getElementType().isInteger(32)) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), "tt.make_range is only supported for integer types.");
    }

    if (result_type.getElementType().isUnsignedInteger(32)) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          "lowering to tt.make_range is only supported for 32 bit signed "
          "integers.");
    }

    auto iota_end = result_type.getDimSize(0);

    rewriter.replaceOpWithNewOp<ttir::MakeRangeOp>(op, result_type,
                                                   /*start=*/0, iota_end);
    return mlir::success();
  }
};

class LowerBroadcastInDim
    : public mlir::OpRewritePattern<stablehlo::BroadcastInDimOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      stablehlo::BroadcastInDimOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    auto input_tensor = op.getOperand();
    auto input_shape = input_tensor.getType().getShape();
    auto output_shape = op.getResult().getType().getShape();
    auto broadcast_dims = op.getBroadcastDimensions();

    if (input_shape.empty()) {
      auto broadcast_dim_input = op.getOperand();

      auto extracted = mlir::tensor::ExtractOp::create(rewriter, op.getLoc(),
                                                       broadcast_dim_input);

      rewriter.replaceOpWithNewOp<ttir::SplatOp>(op, op.getResult().getType(),
                                                 extracted);
      return mlir::success();
    }
    int64_t axis = 0;
    int64_t input_dim_id = 0;
    for (int output_dim_id = 0; output_dim_id < output_shape.size();
         output_dim_id++) {
      if (input_dim_id < broadcast_dims.size() &&
          output_dim_id == broadcast_dims[input_dim_id]) {
        // The dim is not broadcasted. Validate matching dim sizes.
        CHECK_EQ(input_shape[input_dim_id], output_shape[output_dim_id]);
        ++input_dim_id;
        axis = output_dim_id + 1;
        continue;
      }
      input_tensor = ttir::ExpandDimsOp::create(builder, input_tensor, axis);
    }
    rewriter.replaceOpWithNewOp<ttir::BroadcastOp>(op, op.getResult().getType(),
                                                   input_tensor);

    return mlir::success();
  }
};

class LowerReduce : public mlir::OpRewritePattern<stablehlo::ReduceOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      stablehlo::ReduceOp op, mlir::PatternRewriter& rewriter) const override {
    if (mlir::failed(VerifyOpIsCompatibleWithTritonReduce(op, rewriter))) {
      return mlir::failure();
    }

    int32_t axis = op.getDimensions()[0];

    // In case shlo returns a 0 rank tensor triton needs to return a scalar as
    // triton doesn't support 0 rank tensors.
    SmallVector<Type> adjusted_result_types;
    adjusted_result_types.reserve(op.getNumResults());
    for (auto result : op.getResults()) {
      auto shaped_type = cast<mlir::ShapedType>(result.getType());
      if (shaped_type.getRank() == 0) {
        adjusted_result_types.push_back(shaped_type.getElementType());
      } else {
        adjusted_result_types.push_back(shaped_type);
      }
    }

    auto triton_reduce_op = ttir::ReduceOp::create(
        rewriter, op.getLoc(), adjusted_result_types, op.getInputs(), axis);
    Region& triton_reduce_region = triton_reduce_op.getCombineOp();

    mlir::Block& old_block = op.getBody().front();
    llvm::SmallVector<Type> arg_types;
    llvm::SmallVector<mlir::Location> arg_locs;
    for (auto old_arg_type : old_block.getArgumentTypes()) {
      arg_types.push_back(
          llvm::cast<ShapedType>(old_arg_type).getElementType());
      arg_locs.push_back(op.getLoc());
    }
    rewriter.createBlock(&triton_reduce_region, triton_reduce_region.begin(),
                         arg_types, arg_locs);

    mlir::IRMapping mapping;
    Block& triton_reduce_region_block = triton_reduce_region.front();
    rewriter.setInsertionPointToStart(&triton_reduce_region_block);
    for (auto [old_arg, new_arg] :
         llvm::zip(old_block.getArguments(),
                   triton_reduce_region_block.getArguments())) {
      auto to_tensor_op = mlir::tensor::FromElementsOp::create(
          rewriter, op.getLoc(), old_arg.getType(), new_arg);
      mapping.map(old_arg, to_tensor_op);
    }

    for (mlir::Operation& op : old_block.without_terminator()) {
      rewriter.clone(op, mapping);
    }

    SmallVector<Value> return_operands;
    for (Value operand : old_block.getTerminator()->getOperands()) {
      return_operands.push_back(mlir::tensor::ExtractOp::create(
          rewriter, op->getLoc(), mapping.lookupOrDefault(operand)));
    }
    ttir::ReduceReturnOp::create(rewriter, op.getLoc(), return_operands);

    // Replace usages of the original op results. If the original result was a
    // 0-rank tensor, we need to wrap the scalar result of tt.reduce in a
    // tensor.to_tensor op.
    rewriter.setInsertionPointAfter(triton_reduce_op);
    llvm::SmallVector<Value> new_results;
    for (const auto& triton_result : triton_reduce_op.getResults()) {
      if (mlir::isa<mlir::ShapedType>(triton_result.getType())) {
        new_results.push_back(triton_result);
      } else {
        new_results.push_back(mlir::tensor::FromElementsOp::create(
            rewriter, op.getLoc(), op.getType(0), triton_result));
      }
    }

    rewriter.replaceOp(op, new_results);
    return mlir::success();
  }

  // Verifies that the stablehlo reduce op can be lowered to a triton reduce
  // op.
  // This checks that proper emitting of `tensor.from_elements` and
  // `tensor.extract` on reducer inputs and outputs has happened. It also checks
  // that `tensor.extract` was emitted on the result of the reduce operation if
  // the result is a zero rank tensor.
  mlir::LogicalResult VerifyOpIsCompatibleWithTritonReduce(
      stablehlo::ReduceOp op, mlir::PatternRewriter& rewriter) const {
    // Check that the reduction is along a single dimension.
    auto dimensions = op.getDimensions();
    if (dimensions.size() != 1) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), "tt.reduce only supports single dimension reductions.");
    }

    return mlir::success();
  }
};

class LowerReshape : public mlir::OpRewritePattern<stablehlo::ReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      stablehlo::ReshapeOp op, mlir::PatternRewriter& rewriter) const override {
    bool input_is_0d = op.getOperand().getType().getRank() == 0;
    bool output_is_0d = op.getType().getRank() == 0;

    if (input_is_0d && output_is_0d) {
      rewriter.replaceAllUsesWith(op, op.getOperand());
      return mlir::success();
    }

    if (input_is_0d) {
      auto to_scalar = mlir::tensor::ExtractOp::create(rewriter, op->getLoc(),
                                                       op.getOperand());
      rewriter.replaceOpWithNewOp<ttir::SplatOp>(op, op.getType(), to_scalar);
      return mlir::success();
    }

    if (output_is_0d) {
      // We know the input dimensions must be all 1s as reshape input-output
      // must have the same number of elements.
      return LowerRank0ToReduce(op, rewriter);
    }

    // Conservatively prevent Triton from reordering elements within the tile.
    // TODO(b/353637689): see if this restriction can be lifted.
    bool allow_reorder = false;
    rewriter.replaceOpWithNewOp<ttir::ReshapeOp>(
        op, op.getResult().getType(), op.getOperand(), allow_reorder);
    return mlir::success();
  }

  static mlir::LogicalResult LowerRank0ToReduce(
      stablehlo::ReshapeOp op, mlir::PatternRewriter& rewriter) {
    auto input_tensor_type = op.getOperand().getType();

    // First, reshape to a 1D tensor if not already the case. This is needed
    // because triton::ReduceOp can only reduce 1 dimension at a time.
    auto single_dim_tensor = op.getOperand();
    if (input_tensor_type.getRank() > 1) {
      Type output_tensor_type =
          mlir::RankedTensorType::get({1}, input_tensor_type.getElementType());
      single_dim_tensor = ttir::ReshapeOp::create(
          rewriter, op.getLoc(), output_tensor_type, single_dim_tensor,
          /*allow_reorder=*/true);
    }

    // Second, reduce to a scalar.
    ttir::ReduceOp reduction = ttir::ReduceOp::create(
        rewriter, op.getLoc(), single_dim_tensor, /*axis=*/0);

    auto element_type = input_tensor_type.getElementType();
    mlir::Location loc = op.getLoc();
    mlir::Block* reducer =
        rewriter.createBlock(&reduction->getRegion(0), /*insertPt=*/{},
                             /*argTypes=*/
                             {element_type, element_type},
                             /*locs=*/{loc, loc});

    rewriter.setInsertionPointToStart(reducer);
    auto create_binary_op = [&](auto op_type) -> Value {
      return op_type.create(rewriter, reducer->getArgument(0).getLoc(),
                            reducer->getArgument(0), reducer->getArgument(1));
    };
    Value result = mlir::isa<mlir::IntegerType>(element_type)
                       ? create_binary_op(arith::AddIOp())
                       : create_binary_op(arith::AddFOp());
    ttir::ReduceReturnOp::create(rewriter, result.getLoc(), {result});

    rewriter.setInsertionPointAfter(reduction);
    rewriter.replaceOpWithNewOp<mlir::tensor::FromElementsOp>(
        op, op.getType(), reduction.getResult());

    return mlir::success();
  }
};

namespace {

LogicalResult PopulateOperandPrecision(PatternRewriter& rewriter,
                                       stablehlo::DotGeneralOp op,
                                       stablehlo::Precision& lhs_precision,
                                       stablehlo::Precision& rhs_precision) {
  auto precision_config = op.getPrecisionConfig();

  if (!precision_config.has_value()) {
    return rewriter.notifyMatchFailure(op->getLoc(),
                                       "Dot op must have precision config.");
  }

  if (precision_config.value().size() != 2) {
    return rewriter.notifyMatchFailure(
        op->getLoc(),
        "Dot op must have exactly two precisions. One for lhs and one for "
        "rhs.");
  }

  auto lhs_precision_attr =
      mlir::cast<stablehlo::PrecisionAttr>(precision_config.value()[0]);
  auto rhs_precision_attr =
      mlir::cast<stablehlo::PrecisionAttr>(precision_config.value()[1]);

  lhs_precision = lhs_precision_attr.getValue();
  rhs_precision = rhs_precision_attr.getValue();

  return mlir::success();
}

::xla::PrecisionConfig::Precision StableHloPrecisionToXlaPrecision(
    stablehlo::Precision precision) {
  switch (precision) {
    case stablehlo::Precision::DEFAULT:
      return ::xla::PrecisionConfig::DEFAULT;
    case stablehlo::Precision::HIGH:
      return ::xla::PrecisionConfig::HIGH;
    case stablehlo::Precision::HIGHEST:
      return ::xla::PrecisionConfig::HIGHEST;
    default:
      LOG(FATAL) << "Unsupported precision";
  }
}

// Triton implementations of dot algorithms.

struct TritonPrecisionSpec {
  ::xla::PrecisionConfig::Algorithm algorithm;
  // Encodes `tt.dot`'s `inputPrecision` attribute.
  ttir::InputPrecision ttir_input_precision;
};

mlir::Type ElementType(mlir::Value v) { return mlir::getElementTypeOrSelf(v); }

using AlgorithmEmitter = absl::StatusOr<Value> (*)(
    mlir::ImplicitLocOpBuilder&, const ::xla::xtile::DotOperands&,
    const TritonPrecisionSpec&);

absl::StatusOr<Value> EmitDotAlgUnset(
    mlir::ImplicitLocOpBuilder& b,
    const ::xla::xtile::DotOperands& dot_operands,
    const TritonPrecisionSpec& precision_spec) {
  // Execute matrix multiplication of input tiles and pass the accumulator.
  // TODO(manany): Should be looked into once we enable Hopper workloads.
  // maxNumImpreciseAcc flag was introduced for Hopper to accumulate in a
  // lower precision than the output type. The change was introduced here:
  // https://github.com/openai/triton/commit/31b0c521427109a8eda609b58d756c380b21599a
  Value lhs = dot_operands.lhs;
  Value rhs = dot_operands.rhs;
  Value acc = dot_operands.accumulator;

  int max_num_imprecise_acc = 0;
  if (ElementType(lhs).isFloat(8) || ElementType(rhs).isFloat(8)) {
    // For fp8 dots, disable accumulator promotion to mimick cuBLAS. It may make
    // sense to enable frequent accumulator promotion at higher matmul
    // precisions set in the config.
    max_num_imprecise_acc = std::numeric_limits<int>::max();
  }

  return ttir::DotOp::create(
      b, lhs, rhs, acc,
      /*inputPrecision=*/precision_spec.ttir_input_precision,
      /*maxNumImpreciseAcc=*/max_num_imprecise_acc);
}

absl::StatusOr<Value> EmitRegularDot(
    mlir::ImplicitLocOpBuilder& b,
    const ::xla::xtile::DotOperands& dot_operands,
    const TritonPrecisionSpec& precision_spec) {
  Value lhs = dot_operands.lhs;
  Value rhs = dot_operands.rhs;

  int max_num_imprecise_acc = 0;
  if (ElementType(lhs).isFloat(8) || ElementType(rhs).isFloat(8)) {
    // For fp8 dots, disable accumulator promotion to mimick cuBLAS. It may make
    // sense to enable frequent accumulator promotion at higher matmul
    // precisions set in the config.
    max_num_imprecise_acc = std::numeric_limits<int>::max();
  }

  // Cast F32 inputs to BF16 if the algorithm is BF16_BF16_F32.
  // TODO(bchetioui): abstract this.
  if (precision_spec.algorithm ==
      ::xla::PrecisionConfig::ALG_DOT_BF16_BF16_F32) {
    if (ElementType(lhs).isF32()) {
      lhs = ::xla::xtile::Cast(b, lhs, b.getBF16Type());
    }

    if (ElementType(rhs).isF32()) {
      rhs = ::xla::xtile::Cast(b, rhs, b.getBF16Type());
    }
  }

  return ttir::DotOp::create(
      b, dot_operands.lhs, dot_operands.rhs, dot_operands.accumulator,
      /*inputPrecision=*/precision_spec.ttir_input_precision,
      /*maxNumImpreciseAcc=*/max_num_imprecise_acc);
}

// If lhs is 1.0, we will have lhs_high = 1.0 and lhs_low = 0.0.
// If rhs is +infinity, we will have:
// +infinity * 1.0 = +infinity
// +infinity * 0.0 = NaN
// We would get the wrong result if we sum these partial products. Instead, we
// must override any accumulated result if the last partial product is
// non-finite. See b/115844437.
Value ZeroNaNs(mlir::ImplicitLocOpBuilder& b, Value input) {
  Value positive_inf = ::xla::xtile::CreateConst<float>(
      b, b.getF32Type(), std::numeric_limits<float>::infinity(),
      mlir::cast<ShapedType>(input.getType()).getShape());
  Value abs_input = math::AbsFOp::create(b, input);
  Value is_finite = arith::CmpFOp::create(b, arith::CmpFPredicate::OGT,
                                          positive_inf, abs_input);
  return arith::SelectOp::create(b, is_finite, input,
                                 ::xla::xtile::ZerosLike(b, input));
}

absl::Status ExpectType(Value v, Type expected_type) {
  if (ElementType(v) != expected_type) {
    std::string expected_type_str, actual_type_str;
    {
      llvm::raw_string_ostream os_expected(expected_type_str);
      llvm::raw_string_ostream os_actual(actual_type_str);
      expected_type.print(os_expected);
      ElementType(v).print(os_actual);
    }
    return absl::FailedPreconditionError(absl::StrCat(
        "Expected type ", expected_type_str, " but got ", actual_type_str));
  }
  return absl::OkStatus();
}

std::vector<Value> SplitF32(mlir::ImplicitLocOpBuilder& b, Value input,
                            int split_count) {
  std::vector<Value> split_inputs;
  split_inputs.reserve(split_count);
  for (int i = 0; i < split_count; ++i) {
    Value input_as_bf16 = ::xla::xtile::Cast(b, input, b.getBF16Type());
    if (i != split_count - 1) {
      Value input_as_f32 = ::xla::xtile::Cast(b, input_as_bf16, b.getF32Type());
      input = arith::SubFOp::create(b, input, input_as_f32);
    }
    split_inputs.push_back(input_as_bf16);
  }
  return split_inputs;
}

Value IEEEDot(mlir::ImplicitLocOpBuilder& b, Value lhs, Value rhs, Value acc) {
  return ttir::DotOp::create(b, lhs, rhs, acc,
                             /*inputPrecision=*/ttir::InputPrecision::IEEE,
                             /*maxNumImpreciseAcc=*/0);
}

// Leverages BF16 datatype for F32 matmul computation. It follows the guidance
// from https://arxiv.org/pdf/1904.06376.pdf.
absl::StatusOr<Value> EmitBF16x9Matmul(
    mlir::ImplicitLocOpBuilder& b,
    const ::xla::xtile::DotOperands& dot_operands,
    const TritonPrecisionSpec& precision_spec) {
  constexpr int kNumParts = 3;
  constexpr int kHigh = 0;
  constexpr int kMid = 1;
  constexpr int kLow = 2;

  Type f32 = b.getF32Type();
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.lhs, f32));
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.rhs, f32));
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.accumulator, f32));

  std::vector<Value> lhs_parts = SplitF32(b, dot_operands.lhs, kNumParts);
  std::vector<Value> rhs_parts = SplitF32(b, dot_operands.rhs, kNumParts);

  Value result = ::xla::xtile::ZerosLike(b, dot_operands.accumulator);

  result = IEEEDot(b, lhs_parts[kLow], rhs_parts[kLow], result);
  result = IEEEDot(b, lhs_parts[kMid], rhs_parts[kLow], result);
  result = IEEEDot(b, lhs_parts[kLow], rhs_parts[kMid], result);

  result = IEEEDot(b, lhs_parts[kMid], rhs_parts[kMid], result);

  result = IEEEDot(b, lhs_parts[kLow], rhs_parts[kHigh], result);
  result = IEEEDot(b, lhs_parts[kHigh], rhs_parts[kLow], result);

  result = IEEEDot(b, lhs_parts[kMid], rhs_parts[kHigh], result);
  result = IEEEDot(b, lhs_parts[kHigh], rhs_parts[kMid], result);

  result = ZeroNaNs(b, result);
  result = IEEEDot(b, lhs_parts[kHigh], rhs_parts[kHigh], result);
  result = arith::AddFOp::create(b, dot_operands.accumulator, result);
  return result;
}

// Leverages BF16 datatype for F32 matmul computation. It follows the guidance
// from https://arxiv.org/pdf/1904.06376.pdf.
absl::StatusOr<Value> EmitBF16x6Matmul(
    mlir::ImplicitLocOpBuilder& b,
    const ::xla::xtile::DotOperands& dot_operands,
    const TritonPrecisionSpec& precision_spec) {
  constexpr int kNumParts = 3;
  constexpr int kHigh = 0;
  constexpr int kMid = 1;
  constexpr int kLow = 2;

  Type f32 = b.getF32Type();
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.lhs, f32));
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.rhs, f32));
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.accumulator, f32));

  std::vector<Value> lhs_parts = SplitF32(b, dot_operands.lhs, kNumParts);
  std::vector<Value> rhs_parts = SplitF32(b, dot_operands.rhs, kNumParts);

  Value result = ::xla::xtile::ZerosLike(b, dot_operands.accumulator);

  result = IEEEDot(b, lhs_parts[kMid], rhs_parts[kMid], result);

  result = IEEEDot(b, lhs_parts[kLow], rhs_parts[kHigh], result);
  result = IEEEDot(b, lhs_parts[kHigh], rhs_parts[kLow], result);

  result = IEEEDot(b, lhs_parts[kMid], rhs_parts[kHigh], result);
  result = IEEEDot(b, lhs_parts[kHigh], rhs_parts[kMid], result);

  result = ZeroNaNs(b, result);
  result = IEEEDot(b, lhs_parts[kHigh], rhs_parts[kHigh], result);
  result = arith::AddFOp::create(b, dot_operands.accumulator, result);
  return result;
}

// Compute F32 matmul with 3 BF16 dots. It is less accurate than
// EmitBF16x6Matmul.
absl::StatusOr<Value> EmitBF16x3Matmul(
    mlir::ImplicitLocOpBuilder& b,
    const ::xla::xtile::DotOperands& dot_operands,
    const TritonPrecisionSpec& precision_spec) {
  constexpr int kNumParts = 2;
  constexpr int kHigh = 0;
  constexpr int kLow = 1;

  Type f32 = b.getF32Type();
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.lhs, f32));
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.rhs, f32));
  TF_RETURN_IF_ERROR(ExpectType(dot_operands.accumulator, f32));

  std::vector<Value> lhs_bf16 = SplitF32(b, dot_operands.lhs, kNumParts);
  std::vector<Value> rhs_bf16 = SplitF32(b, dot_operands.rhs, kNumParts);

  Value result = ::xla::xtile::ZerosLike(b, dot_operands.accumulator);
  result = IEEEDot(b, lhs_bf16[kLow], rhs_bf16[kHigh], result);
  result = IEEEDot(b, lhs_bf16[kHigh], rhs_bf16[kLow], result);
  result = ZeroNaNs(b, result);
  result = IEEEDot(b, lhs_bf16[kHigh], rhs_bf16[kHigh], result);
  result = arith::AddFOp::create(b, dot_operands.accumulator, result);
  return result;
}

// Returns an emitter for the given dot algorithm. Raises an
// `UnimplementedError` if the algorithm is not supported.
absl::StatusOr<AlgorithmEmitter> GetAlgorithmEmitter(
    const ::xla::PrecisionConfig::Algorithm algorithm) {
  switch (algorithm) {
    case ::xla::PrecisionConfig::ALG_UNSET:
      return EmitDotAlgUnset;
    case ::xla::PrecisionConfig::ALG_DOT_F16_F16_F16:
    case ::xla::PrecisionConfig::ALG_DOT_F32_F32_F32:
    case ::xla::PrecisionConfig::ALG_DOT_F64_F64_F64:
    case ::xla::PrecisionConfig::ALG_DOT_F16_F16_F32:
    case ::xla::PrecisionConfig::ALG_DOT_BF16_BF16_BF16:
    case ::xla::PrecisionConfig::ALG_DOT_BF16_BF16_F32:
      return EmitRegularDot;
    case ::xla::PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
      return EmitBF16x3Matmul;
    case ::xla::PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
      return EmitBF16x6Matmul;
    case ::xla::PrecisionConfig::ALG_DOT_TF32_TF32_F32:
      // TODO(bchetioui): this should be factored out of EmitRegularDot.
      return EmitRegularDot;
    case ::xla::PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
      // TODO(bchetioui): this should be factored out of EmitRegularDot.
      return EmitRegularDot;
    case ::xla::PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
      return EmitBF16x9Matmul;
    case ::xla::PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32:
    case ::xla::PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM:
    default:
      break;
  }

  // Couldn't find an algorithm emitter for this algorithm. Raise an error.
  return absl::UnimplementedError(
      absl::StrCat("This algorithm is not supported yet: ",
                   ::xla::PrecisionConfig::Algorithm_Name(algorithm)));
}

bool IsTf32Allowed(const ::xla::xtile::PrecisionSpec& precision_spec) {
  if (precision_spec.algorithm == ::xla::PrecisionConfig::ALG_UNSET) {
    return tsl::tensor_float_32_execution_enabled() &&
           StableHloPrecisionToXlaPrecision(
               precision_spec.lhs_operand_precision) ==
               ::xla::PrecisionConfig::DEFAULT &&
           StableHloPrecisionToXlaPrecision(
               precision_spec.rhs_operand_precision) ==
               ::xla::PrecisionConfig::DEFAULT;
  }
  return ::xla::algorithm_util::HasTf32InputType(precision_spec.algorithm);
}

ttir::InputPrecision InferDotPrecision(
    const ::xla::xtile::PrecisionSpec& precision_spec) {
  if (precision_spec.algorithm ==
      ::xla::PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3) {
    return ttir::InputPrecision::TF32x3;
  }

  return IsTf32Allowed(precision_spec) ? ttir::InputPrecision::TF32
                                       : ttir::InputPrecision::IEEE;
}

LogicalResult RewriteDotGeneralToTritonDot(mlir::PatternRewriter& rewriter,
                                           stablehlo::DotGeneralOp op,
                                           mlir::Operation* add_op,
                                           Value accumulator) {
  auto dot_algorithm = op.getAlgorithm();

  auto hlo_algorithm_or_status =
      dot_algorithm.has_value()
          ? ::xla::ConvertDotAlgorithm(dot_algorithm.value())
          : ::xla::PrecisionConfig::ALG_UNSET;

  if (!hlo_algorithm_or_status.ok()) {
    return rewriter.notifyMatchFailure(
        op->getLoc(),
        "Dot op must have algorithm set to be converted to "
        "triton dot.");
  }

  auto hlo_algorithm = hlo_algorithm_or_status.value();
  auto algorithm_emitter_or_status = GetAlgorithmEmitter(hlo_algorithm);

  if (!algorithm_emitter_or_status.ok()) {
    return rewriter.notifyMatchFailure(
        op->getLoc(),
        absl::StrCat("Algorithm emitter not found with error: ",
                     algorithm_emitter_or_status.status().message()));
  }

  auto algorithm_emitter = algorithm_emitter_or_status.value();

  mlir::ImplicitLocOpBuilder builder(op->getLoc(), rewriter);

  ::xla::xtile::DotOperands dot_operands{op.getLhs(), op.getRhs(), accumulator};

  stablehlo::Precision lhs_precision;
  stablehlo::Precision rhs_precision;

  if (mlir::failed(PopulateOperandPrecision(rewriter, op, lhs_precision,
                                            rhs_precision))) {
    return mlir::failure();
  }

  ::xla::xtile::PrecisionSpec precision_spec{hlo_algorithm, lhs_precision,
                                             rhs_precision};

  TritonPrecisionSpec triton_precision_spec{hlo_algorithm,
                                            InferDotPrecision(precision_spec)};

  auto triton_dot_op_or_result =
      algorithm_emitter(builder, dot_operands, triton_precision_spec);

  if (!triton_dot_op_or_result.ok()) {
    return rewriter.notifyMatchFailure(
        op->getLoc(), absl::StrCat("Algorithm emitter failed with error: ",
                                   triton_dot_op_or_result.status().message()));
  }

  auto triton_dot_op = triton_dot_op_or_result.value();

  rewriter.replaceAllOpUsesWith(add_op, op.getResult());
  rewriter.replaceOp(op, triton_dot_op);

  return mlir::success();
}

}  // namespace

class LowerDotGeneral : public mlir::OpRewritePattern<stablehlo::DotGeneralOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      stablehlo::DotGeneralOp op,
      mlir::PatternRewriter& rewriter) const override {
    if (std::distance(op->getUsers().begin(), op->getUsers().end()) != 1) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          "Dot op must have exactly one user in order to be lowered to "
          "triton.");
    }

    mlir::Operation* add_op = dyn_cast<arith::AddFOp>(*op->getUsers().begin());
    if (!add_op) {
      add_op = dyn_cast<arith::AddIOp>(*op->getUsers().begin());
    }

    if (!add_op) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          "Dot op must be consumed by an AddOp in order to be convertible to "
          "triton dot.");
    }

    // Accumulator is the operand of add that is not the dot operation.
    auto accumulator = add_op->getOperand(1) == op ? add_op->getOperand(0)
                                                   : add_op->getOperand(1);

    if (mlir::failed(
            RewriteDotGeneralToTritonDot(rewriter, op, add_op, accumulator))) {
      return mlir::failure();
    }
    return mlir::success();
  }
};

// The main memory space on a device (HBM).
static constexpr auto kGlobalAddressSpace =
    static_cast<std::underlying_type_t<mlir::NVVM::NVVMMemorySpace>>(
        mlir::NVVM::NVVMMemorySpace::Global);

class LowerAllReduce : public mlir::OpRewritePattern<stablehlo::AllReduceOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

 private:
  mlir::LogicalResult matchAndRewrite(
      stablehlo::AllReduceOp op,
      mlir::PatternRewriter& rewriter) const override {
    if (op.getOperands().size() != 1) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          "AllReduce op must have exactly one operand in order to be lowered "
          "to triton.");
    }

    // Find the entry function and get the arguments from there.
    auto fn = op->getParentOp();
    while (fn && !dyn_cast<::xla::xtile::EntryFuncOp>(fn)) {
      fn = fn->getParentOp();
    }

    if (!fn) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          "AllReduce op must be in an XTile entry function in order to be "
          "lowered to triton.");
    }

    auto xtile_entry_fn = mlir::cast<::xla::xtile::EntryFuncOp>(fn);

    // Variadics are not supported yet so we can fix inputs to 1.
    // Which means 2 arguments for input/output one for scratch buffers and 3
    // metadata arguments. Plus 1 for the tile index for a total of 7.
    const int32_t num_input_output_args = op.getNumOperands() * 2;
    const int32_t num_scratch_buffers = op.getNumOperands();
    static constexpr int32_t kNumTileIndexArgs = 1;
    if (xtile_entry_fn.getNumArguments() !=
        (num_input_output_args + num_scratch_buffers +
         ::xla::gpu::kNumCollectiveMetadataArgs + kNumTileIndexArgs)) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          absl::StrCat("AllReduce op must have ",
                       num_input_output_args + num_scratch_buffers +
                           ::xla::gpu::kNumCollectiveMetadataArgs +
                           kNumTileIndexArgs,
                       " arguments in order to "
                       "be lowered to triton, but it has ",
                       xtile_entry_fn.getNumArguments()));
    }

    // Opaque arguments start after the input/output arguments.
    const int32_t start_idx = num_input_output_args;
    mlir::Value device_rank = xtile_entry_fn.getArgument(start_idx);
    CHECK(device_rank.getType().isInteger(32));
    mlir::Value signal_value = xtile_entry_fn.getArgument(start_idx + 1);
    CHECK(signal_value.getType().isInteger(32));
    // !tt.ptr<!tt.ptr<i32>>
    mlir::Value signal_buffers = xtile_entry_fn.getArgument(start_idx + 2);
    // !tt.ptr<!tt.ptr<i64>>
    mlir::Value remote_input_buffers =
        xtile_entry_fn.getArgument(start_idx + 3);

    // We assume the input to all reduce is an xtile::ExtractTileOp, or that the
    // parent of the input is an xtile::ExtractTileOp (edge case for booleans).
    auto input_tile = op.getOperand(0);
    auto input_tile_op =
        dyn_cast<::xla::xtile::ExtractTileOp>(input_tile.getDefiningOp());

    if (!input_tile_op && input_tile.getDefiningOp()->getNumOperands() > 0) {
      // Go one place up - this is an edge case for booleans.
      input_tile_op = dyn_cast<::xla::xtile::ExtractTileOp>(
          input_tile.getDefiningOp()->getOperand(0).getDefiningOp());
    }
    if (!input_tile_op) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          "AllReduce op must have an extract tile op as operand in order to be "
          "lowered to triton.");
    }

    auto non_tiled_input_shape = input_tile_op.getSource().getType().getShape();

    const int64_t num_elements = std::accumulate(
        non_tiled_input_shape.begin(), non_tiled_input_shape.end(), /*init=*/1,
        std::multiplies<int64_t>());

    absl::AnyInvocable<::xla::xtile::TensorValue(mlir::ImplicitLocOpBuilder&,
                                                 ::xla::xtile::TensorValue,
                                                 ::xla::xtile::TensorValue)>
        reduce_computation_emitter = nullptr;

    // We have to perform this verification before emitting any triton specific
    // mlir so that we can fall back if the computation is invalid.
    if (mlir::failed(verifyReductionComputationAndPopulateComputationEmitter(
            rewriter, op, reduce_computation_emitter))) {
      return mlir::failure();
    }

    mlir::ImplicitLocOpBuilder builder(op->getLoc(), rewriter);

    // 1. Scatter phase: Copy local tile to the remote buffer of the current
    // rank.
    const auto ptr_to_i64_type =
        ttir::PointerType::get(builder.getI64Type(), kGlobalAddressSpace);
    auto remote_input_buffers_i64 =
        ttir::BitcastOp::create(builder, ptr_to_i64_type, remote_input_buffers);

    const mlir::Type i64_type = builder.getI64Type();
    const mlir::Type elem_type =
        mlir::getElementTypeOrSelf(input_tile.getType());
    const mlir::Type elem_storage_type = ::xla::xtile::StorageType(elem_type);
    const auto ptr_to_elem_type =
        ttir::PointerType::get(elem_storage_type, kGlobalAddressSpace);
    constexpr int32_t kBitsPerByte = 8;
    const int64_t remote_buffer_size =
        num_elements *
        (elem_storage_type.getIntOrFloatBitWidth() / kBitsPerByte);
    Value buffer_index = arith::AndIOp::create(
        builder, i64_type,
        arith::ExtSIOp::create(builder, i64_type, signal_value),
        arith::ConstantOp::create(builder, i64_type,
                                  builder.getI64IntegerAttr(1)));
    Value buffer_offset = arith::MulIOp::create(
        builder, i64_type, buffer_index,
        arith::ConstantOp::create(
            builder, i64_type, builder.getI64IntegerAttr(remote_buffer_size)));
    // Helper function to get the buffer pointer for a given signal value.
    const auto get_buffer_ptr =
        [&](mlir::Value buffer_ptr_base) -> mlir::Value {
      return ttir::AddPtrOp::create(builder, ptr_to_elem_type, buffer_ptr_base,
                                    buffer_offset);
    };

    mlir::ArrayRef<int64_t> remote_shape = non_tiled_input_shape;
    const mlir::MemRefType remote_memref_type =
        mlir::MemRefType::get(remote_shape, elem_storage_type);
    // Scoped to reuse variable names during reduction phase.
    {
      Value remote_buf_ptr_addr = ttir::AddPtrOp::create(
          builder, ptr_to_i64_type, remote_input_buffers_i64, device_rank);
      Value remote_buf_i64 =
          ttir::LoadOp::create(builder, remote_buf_ptr_addr,
                               ttir::CacheModifier::NONE,     //
                               ttir::EvictionPolicy::NORMAL,  //
                               false);                        // isVolatile
      Value remote_buf_ptr_base = ttir::IntToPtrOp::create(
          builder, ptr_to_elem_type, remote_buf_i64,
          llvm::ArrayRef<mlir::NamedAttribute>{
              ::xla::xtile::GetDivisibilityAttr(builder)});
      Value remote_buf_ptr = get_buffer_ptr(remote_buf_ptr_base);
      mlir::Value remote_buf_memref =
          PtrToMemrefOp::create(builder, remote_memref_type, remote_buf_ptr);
      // Workaround(i1_to_i8_workaround) as in fusion_emitter.
      // The parameter extraction casts the storage type to the logical type.
      // But for copying to the remote buffer we need to cast it back to the
      // storage type. Downstream passes should be able to optimize this away.
      mlir::Value storage_tile = input_tile;
      if (elem_storage_type != elem_type) {
        storage_tile = mlir::cast<::xla::xtile::TensorValue>(
            ::xla::xtile::Cast(builder, input_tile, elem_storage_type));
      }
      ::xla::xtile::InsertTileOp::create(
          builder, storage_tile, remote_buf_memref, input_tile_op.getOffsets(),
          input_tile_op.getTile().getType().getShape(),
          input_tile_op.getStrides());
    }

    // 2. Synchronization phase: Wait for all ranks to complete the scatter.
    int64_t world_size = op.getReplicaGroups().size();
    BlockBarrierOp::create(builder, signal_buffers, device_rank, signal_value,
                           builder.getI32IntegerAttr(world_size));

    // 3. Reduce phase: Load tiles from all ranks and reduce them.
    const auto load_tile_for_rank = [&](int64_t rank) {
      Value rank_idx = arith::ConstantOp::create(
          builder, builder.getI64Type(), builder.getI64IntegerAttr(rank));
      Value remote_buf_ptr_addr = ttir::AddPtrOp::create(
          builder, ptr_to_i64_type, remote_input_buffers_i64, rank_idx);
      Value remote_buf_i64 =
          ttir::LoadOp::create(builder, remote_buf_ptr_addr,
                               ttir::CacheModifier::NONE,     //
                               ttir::EvictionPolicy::NORMAL,  //
                               false);                        // isVolatile
      Value remote_buf_ptr_base =
          ttir::IntToPtrOp::create(builder, ptr_to_elem_type, remote_buf_i64);
      Value remote_buf_ptr = get_buffer_ptr(remote_buf_ptr_base);
      Value remote_buf_memref =
          PtrToMemrefOp::create(builder, remote_memref_type, remote_buf_ptr);

      auto tensor_type = mlir::RankedTensorType::get(
          input_tile_op.getTile().getType().getShape(), elem_storage_type);

      ::xla::xtile::TensorValue next_tile = ::xla::xtile::ExtractTileOp::create(
          builder, tensor_type, remote_buf_memref, input_tile_op.getOffsets(),
          input_tile_op.getTile().getType().getShape(),
          input_tile_op.getStrides());
      // # Workaround(i1_to_i8_workaround) as in fusion_emitter.
      // See fusion emitter for more details.
      if (elem_storage_type != elem_type) {
        next_tile = mlir::cast<::xla::xtile::TensorValue>(
            ::xla::xtile::Cast(builder, next_tile, elem_type));
      }
      return next_tile;
    };

    ::xla::xtile::TensorValue accumulator = load_tile_for_rank(0);

    for (int rank = 1; rank < world_size; ++rank) {
      ::xla::xtile::TensorValue next_tile = load_tile_for_rank(rank);

      accumulator = reduce_computation_emitter(builder, accumulator, next_tile);
    }

    rewriter.replaceOp(op, accumulator.getDefiningOp());
    return mlir::success();
  }

  mlir::LogicalResult verifyReductionComputationAndPopulateComputationEmitter(
      mlir::PatternRewriter& rewriter, stablehlo::AllReduceOp op,
      absl::AnyInvocable<::xla::xtile::TensorValue(
          mlir::ImplicitLocOpBuilder&, ::xla::xtile::TensorValue,
          ::xla::xtile::TensorValue)>& computation_emitter) const {
    // At the moment we expect only one operation in the reduction computation
    // to be relevant.
    // We can't simply reuse the ops from the reduction computation block since
    // they operate on types with different shapes (i.e. 0 ranked vs non zero
    // ranked tensors).
    auto& reduction_computation_region = op.getComputation();
    int num_ops_to_emit = 0;
    for (auto& block : reduction_computation_region.getBlocks()) {
      for (auto& block_op : block.without_terminator()) {
        if (auto add_op = dyn_cast<arith::AddFOp>(block_op)) {
          computation_emitter = [](mlir::ImplicitLocOpBuilder& builder,
                                   ::xla::xtile::TensorValue accumulator,
                                   ::xla::xtile::TensorValue next_tile) {
            return mlir::cast<::xla::xtile::TensorValue>(
                arith::AddFOp::create(builder, accumulator.getType(),
                                      accumulator, next_tile)
                    .getResult());
          };
          num_ops_to_emit++;
        } else if (auto ori_op = dyn_cast<arith::OrIOp>(block_op)) {
          computation_emitter = [](mlir::ImplicitLocOpBuilder& builder,
                                   ::xla::xtile::TensorValue accumulator,
                                   ::xla::xtile::TensorValue next_tile) {
            return mlir::cast<::xla::xtile::TensorValue>(
                arith::OrIOp::create(builder, accumulator.getType(),
                                     accumulator, next_tile)
                    .getResult());
          };
          num_ops_to_emit++;

        } else if (dyn_cast<tensor::ExtractOp>(block_op) ||
                   (dyn_cast<tensor::FromElementsOp>(block_op))) {
          // These ops are not relevant to the reduction and are just emitted so
          // that we have a valid stablehlo all reduce op.
          // We don't emit them, but they don't count towards our only one op in
          // the reduction computation requirement.
        } else {
          return rewriter.notifyMatchFailure(
              op.getLoc(), "Unsupperted operation in reduction computation.");
        }
      }
    }

    if (num_ops_to_emit != 1) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          "AllReduce op must have exactly one relevant operation in order to "
          "be lowered to triton.");
    }

    return mlir::success();
  }
};

class StableHLOLowerToTritonPass
    : public impl::StableHLOLowerToTritonPassBase<StableHLOLowerToTritonPass> {
 public:
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<LowerTranspose, LowerIotaToMakeRange, LowerBroadcastInDim,
                 LowerReduce, LowerReshape, LowerDotGeneral, LowerAllReduce>(
        mlir_context);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateStableHLOLowerToTritonPass() {
  return std::make_unique<StableHLOLowerToTritonPass>();
}

}  // namespace mlir::triton::xla
