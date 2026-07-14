/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This pass converts a TFLite unsigned integer graph to the signed integer
// domain, with adaptors at input and output tensors. This is needed because
// TOSA precision is implemented in the signed integer domain. This pass:
// 1. Matches TFL::QConst with unsigned integers, generates TFL::QConst with
// signed integers with values remapped.
// 2. Inserts tosa.RESCALE unsigned -> signed at graph inputs
// 3. Inserts tosa.RESCALE signed -> unsigned at graph outputs

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"                // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"                // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"          // from @llvm-project
#include "mlir/IR/Builders.h"                            // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"                   // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                        // from @llvm-project
#include "mlir/IR/PatternMatch.h"                        // from @llvm-project
#include "mlir/Pass/PassRegistry.h"                      // from @llvm-project
#include "mlir/Support/LLVM.h"                           // from @llvm-project
#include "mlir/Support/LogicalResult.h"                  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_common.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_utils.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"

#define PASS_NAME "tosa-convert-tfl-uint-to-int"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace tosa {
namespace {

#define GEN_PASS_DEF_TOSACONVERTTFLUNSIGNEDINTTOSIGNEDPASS
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

// Validate that zero point follows TOSA requirements:
// - int8/uint8 can have zero point within their valid range
// - uint16 zero point must be either 0 or 32768
// - All other types must have zero point equal to 0
// Ref:
// https://github.com/arm/tosa-specification/blob/9fe5e964e2193f0e345670f7f4098beecd7fd6eb/tosa.xml#L2479
bool isValidTOSAZeroPoint(int32_t zp, unsigned bit_width, bool is_unsigned) {
  if (bit_width == 8) {
    // int8/uint8 can have any zero point in their valid range
    if (is_unsigned) {
      return zp >= 0 && zp <= 255;
    } else {
      return zp >= -128 && zp <= 127;
    }
  } else if (bit_width == 16 && is_unsigned) {
    // uint16 must have zp of 0 or 32768
    return (zp == 0 || zp == 32768);
  } else {
    // All other types (int16, int32, uint32) must have zero point = 0
    return zp == 0;
  }
}

// Adjust zero points according to TOSA requirements
int32_t adjustZeroPointForTOSA(int32_t zp, unsigned bit_width, bool is_unsigned) {
  if (bit_width == 16 && is_unsigned) {
    // uint16 must have zp of 0 or 32768
    if (zp != 0 && zp != 32768) {
      // TOSA doesn't support other zero points for uint16. Round to nearest
      // valid value. This leads to precision loss in the rescaled type.
      return (zp < 16384) ? 0 : 32768;
    }
    return zp;
  } else if (bit_width == 8) {
    // int8/uint8 can have any zero point in their valid range
    return zp;
  } else {
    // All other types must have zero point = 0. This also leads to precision loss.
    return 0;
  }
}

// Only support 8, 16, and 32-bit integers as tosa.rescale op only supports
// these bitWidths as both input and output types
bool isUnsupportedBitWidth(unsigned bit_width) {
  return (bit_width != 8 && bit_width != 16 && bit_width != 32);
}

// Create a signed quantized type from an unsigned one
// Returns the signed quantized type and the adjusted zero point
std::pair<mlir::quant::UniformQuantizedType, int32_t>
createSignedQuantizedTypeFromUnsigned(
    OpBuilder& builder, mlir::quant::UniformQuantizedType unsigned_quant_type) {
  unsigned bit_width = unsigned_quant_type.getStorageTypeIntegralWidth();

  int32_t unsigned_zp = unsigned_quant_type.getZeroPoint();

  // Calculate real value range
  double real_min = static_cast<double>(
                        unsigned_quant_type.getStorageTypeMin() - unsigned_zp) *
                    unsigned_quant_type.getScale();
  double real_max = static_cast<double>(
                        unsigned_quant_type.getStorageTypeMax() - unsigned_zp) *
                    unsigned_quant_type.getScale();

  // Determine signed storage range
  int64_t signed_min = quant::AnyQuantizedType::getDefaultMinimumForInteger(
      /* isSignedInteger = */ true, bit_width);
  int64_t signed_max = quant::AnyQuantizedType::getDefaultMaximumForInteger(
      /* isSignedInteger = */ true, bit_width);

  // Adjust for narrow range
  if (unsigned_quant_type.getStorageTypeMin() == 1) {
    signed_min++;
  }

  // Compute optimal scale and zero point
  double real_range = real_max - real_min;
  double storage_range = static_cast<double>(signed_max - signed_min);
  double new_scale = real_range / storage_range;

  // Compute zero point to align real_min with signed_min
  int32_t signed_zp =
      static_cast<int32_t>(std::round(signed_min - real_min / new_scale));

  int32_t adjusted_zp =
      adjustZeroPointForTOSA(signed_zp, bit_width, /*is_unsigned=*/false);

  // If zero point was changed by TOSA constraints, recompute scale
  if (adjusted_zp != signed_zp) {
    // With the adjusted zp, recompute scale to ensure coverage of real range
    // real_min = new_scale * (signed_min - adjusted_zp)
    // real_max = new_scale * (signed_max - adjusted_zp)
    double scale_for_min =
        real_min / static_cast<double>(signed_min - adjusted_zp);
    double scale_for_max =
        real_max / static_cast<double>(signed_max - adjusted_zp);
    new_scale = std::max(scale_for_min, scale_for_max);
  }

  signed_zp = adjusted_zp;

  auto signed_quant_type = quant::UniformQuantizedType::getChecked(
      builder.getUnknownLoc(), quant::QuantizationFlags::Signed,
      builder.getIntegerType(bit_width), unsigned_quant_type.getExpressedType(),
      new_scale, signed_zp, signed_min, signed_max);

  return {signed_quant_type, signed_zp};
}

// Pattern for converting unsigned QConst ops
struct ConvertUnsignedQConstOp : public RewritePattern {
  explicit ConvertUnsignedQConstOp(MLIRContext* context)
      : RewritePattern(TFL::QConstOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& builder) const override {
    auto tfl_qconst_op = cast<TFL::QConstOp>(op);

    // Skip if it's not ranked tensor type.
    auto output_type =
        dyn_cast<mlir::RankedTensorType>(tfl_qconst_op.getResult().getType());
    if (!output_type)
      return builder.notifyMatchFailure(op, "not ranked tensor");

    // Skip if output is not per-tensor quantized type.
    auto output_element_type = dyn_cast<mlir::quant::UniformQuantizedType>(
        output_type.getElementType());
    if (!output_element_type)
      return builder.notifyMatchFailure(
          op, "not per-tensor quantized tfl.qconst op");

    // Skip if output is already signed.
    if (output_element_type.isSigned()) {
      return builder.notifyMatchFailure(
          op, "expected output type to be unsigned quantized type");
    }

    // Get the bit_width from the quantized type itself
    unsigned bit_width = output_element_type.getStorageTypeIntegralWidth();

    if (isUnsupportedBitWidth(bit_width)) {
      return builder.notifyMatchFailure(
          op, "only 8, 16, and 32-bit integers are supported");
    }

    int32_t unsigned_zp = output_element_type.getZeroPoint();
    if (!isValidTOSAZeroPoint(unsigned_zp, bit_width, /*is_unsigned=*/true)) {
      return builder.notifyMatchFailure(op,
                                        "Zeropoint is not supported by TOSA.");
    }

    // Calculate the zero-point offset for this bit_width
    auto [signed_quant_type, signed_zp] =
        createSignedQuantizedTypeFromUnsigned(builder, output_element_type);

    auto dst_qconst_type = TypeAttr::get(
        RankedTensorType::get(output_type.getShape(), signed_quant_type));

    Type dst_dense_element_type = builder.getIntegerType(bit_width);

    mlir::DenseElementsAttr src_dense_attr =
        mlir::cast<DenseElementsAttr>(tfl_qconst_op.getValue());

    int64_t value_offset = unsigned_zp - signed_zp;
    auto dst_dense_attr = src_dense_attr.mapValues(
        dst_dense_element_type,
        [bit_width, value_offset](const APInt& in) -> APInt {
          int64_t in_i64 = in.getLimitedValue();
          int64_t out_i64 = in_i64 - value_offset;
          return APInt(bit_width, out_i64, true);
        });

    builder.replaceOpWithNewOp<TFL::QConstOp>(op, dst_qconst_type,
                                              dst_dense_attr);

    return success();
  }
};

// returns true iff @a type is a shaped type with element type that is unsigned
// integer. If it is, then update the rescaled type, input_zp, and output_zp to
// use to rescale type to signed type with adjusted zero point.
bool ConvertUnsignedToSignedRescale(OpBuilder& builder, const Type type,
                          Type& rescaled_type, int32_t& unsigned_zp,
                          int32_t& output_zp, unsigned& bit_width) {
  auto shaped_type = dyn_cast<mlir::ShapedType>(type);
  if (!shaped_type) return false;

  auto element_type = shaped_type.getElementType();
  auto unsigned_element_quant_type =
      dyn_cast<mlir::quant::UniformQuantizedType>(element_type);

  // Check if it's unsigned quantized type
  if (unsigned_element_quant_type && !unsigned_element_quant_type.isSigned()) {
    bit_width = unsigned_element_quant_type.getStorageTypeIntegralWidth();
    if (isUnsupportedBitWidth(bit_width)) {
      return false;
    }

    unsigned_zp = unsigned_element_quant_type.getZeroPoint();
    // Use helper function to create signed quantized type
    auto [signed_quant_type, signed_zp] = createSignedQuantizedTypeFromUnsigned(
        builder, unsigned_element_quant_type);

    output_zp = signed_zp;
    rescaled_type = shaped_type.clone(signed_quant_type);
    return true;
  }

  // Check for plain unsigned integer types
  if (element_type.isUnsignedInteger()) {
    bit_width = element_type.getIntOrFloatBitWidth();
    if (isUnsupportedBitWidth(bit_width)) {
      return false;
    }

    int64_t zp_offset = 1LL << (bit_width - 1);
    unsigned_zp = 0;
    output_zp = adjustZeroPointForTOSA(unsigned_zp - zp_offset, bit_width,
                                       /*is_unsigned=*/false);
    // Determine signed storage range
    int64_t signed_min = quant::AnyQuantizedType::getDefaultMinimumForInteger(
        /* isSignedInteger = */ true, bit_width);
    int64_t signed_max = quant::AnyQuantizedType::getDefaultMaximumForInteger(
        /* isSignedInteger = */ true, bit_width);

    rescaled_type = shaped_type.clone(quant::UniformQuantizedType::getChecked(
        builder.getUnknownLoc(), quant::QuantizationFlags::Signed,
        builder.getIntegerType(bit_width), builder.getF32Type(),
        /* scale = */ 1.0,
        /* zeroPoint = */ output_zp,
        /* storageTypeMin = */ signed_min,
        /* storageTypeMax = */ signed_max));

    return true;
  }

  return false;
}

LogicalResult convert_graph_unsigned_tensor(mlir::MLIRContext& context,
                                            mlir::func::FuncOp& function) {
  size_t num_blocks_in_main = 0;
  mlir::Region* region = function.getCallableRegion();
  OpBuilder builder(&context);
  auto loc = function.getLoc();

  for (mlir::Block& bb : region->getBlocks()) {
    // Always have one block for each region right now
    num_blocks_in_main++;
    if (num_blocks_in_main > 1) {
      return function.emitError("Invalid MLIR: multiple blocks in a region");
    }

    if (!bb.isEntryBlock()) {
      return function.emitError("Invalid MLIR: block must be entry block");
    }

    // Create multiplier and shift tensors (these are always i32 and i8
    // respectively)
    auto multiplier = tosa::getConstTensorInt<int32_t>(builder, loc, {1 << 30});
    auto shift = tosa::getConstTensorInt<int8_t>(builder, loc, {30});

    // Insert rescale unsigned->signed for input arguments
    for (Value arg : bb.getArguments()) {
      auto shaped_type = dyn_cast<ShapedType>(arg.getType());
      if (!shaped_type) continue;

      // Check if zeropoint is supported by TOSA if quantized unsigned type
      auto element_type = shaped_type.getElementType();
      auto unsigned_element_quant_type =
          dyn_cast<mlir::quant::UniformQuantizedType>(element_type);

      if (unsigned_element_quant_type &&
          !unsigned_element_quant_type.isSigned()) {
        unsigned bit_width =
            unsigned_element_quant_type.getStorageTypeIntegralWidth();
        int32_t unsigned_zp = unsigned_element_quant_type.getZeroPoint();

        if (!isValidTOSAZeroPoint(unsigned_zp, bit_width, /*is_unsigned=*/true)) {
          return function.emitError()
                 << "Input argument has unsigned quantized type with zero "
                    "point "
                 << unsigned_zp
                 << " which is not supported by TOSA for bitwidth " << bit_width
                 << ".";
        }
      }

      Type rescaled_type;
      int32_t rescale_input_zp_val, rescale_output_zp_val;
      unsigned bit_width;

      if (!ConvertUnsignedToSignedRescale(builder, arg.getType(), rescaled_type,
                                rescale_input_zp_val, rescale_output_zp_val,
                                bit_width))
        continue;

      // Keep original input_val use with tmp_val.
      auto tmp_const_type =
          RankedTensorType::get({1}, builder.getIntegerType(bit_width));
      auto tmp_const_attr =
          DenseElementsAttr::get(tmp_const_type, {APInt(bit_width, 0)});
      Value tmp_val =
          builder.create<TFL::ConstOp>(loc, tmp_const_type, tmp_const_attr);
      arg.replaceAllUsesWith(tmp_val);

      // Create zero point constants with appropriate bit_width
      // input_zp must match input bit_width, output_zp must match
      // output bit_width
      auto rescale_input_zp = createZeroPointTensor(
          builder, loc, tmp_const_type, rescale_input_zp_val);
      auto rescale_output_zp = createZeroPointTensor(
          builder, loc, tmp_const_type, rescale_output_zp_val);

      if (!rescale_input_zp || !rescale_output_zp) {
        return function.emitError("Failed to create input zero point tensors.");
      }

      const auto rounding_mode_attr = tosa::RoundingModeAttr::get(
          builder.getContext(), tosa::RoundingMode::SINGLE_ROUND);
      auto rescale_op = builder.create<tosa::RescaleOp>(
          loc, rescaled_type, arg, multiplier, shift, rescale_input_zp.value(),
          rescale_output_zp.value(),
          /* scale32 = */ builder.getBoolAttr(true),
          /* rounding_mode = */ rounding_mode_attr,
          /* per_channel = */ builder.getBoolAttr(false),
          /* input_unsigned = */ builder.getBoolAttr(true),     // unsigned ->
          /* output_unsigned = */ builder.getBoolAttr(false));  // signed

      Operation* op_rescale_op = static_cast<Operation*>(rescale_op);
      bb.push_front(op_rescale_op);
      tmp_val.replaceAllUsesWith(rescale_op.getResult());
      tmp_val.getDefiningOp()->erase();
      bb.push_front(rescale_output_zp.value().getDefiningOp());
      bb.push_front(rescale_input_zp.value().getDefiningOp());
    }

    bb.push_front(shift.getDefiningOp());
    bb.push_front(multiplier.getDefiningOp());

    // Record types of original graph output before we convert intermediate
    // tensors.
    auto terminator = bb.getTerminator();
    SmallVector<Type, 4> output_types;
    for (Value val : terminator->getOperands()) {
      output_types.push_back(val.getType());
    }

    // Convert intermediate tensors.
    for (auto& op : bb) {
      if (llvm::dyn_cast<tosa::ConstOp>(&op)) {
        // Skip tosa const ops created during rescaling.
        continue;
      }

      for (Value output_val : op.getResults()) {
        auto shaped_type = dyn_cast<ShapedType>(output_val.getType());
        if (!shaped_type) continue;
        Type new_type;
        int32_t unused_input_zp, unused_output_zp;
        unsigned bit_width;

        if (ConvertUnsignedToSignedRescale(builder, output_val.getType(), new_type,
                                 unused_input_zp, unused_output_zp, bit_width)) {
          output_val.setType(new_type);
        }
      }
    }

    if (terminator->getNumOperands() != output_types.size()) {
      return function.emitError(
          "Terminator's operand mismatch with number of outputs in graph");
    }

    // Insert signed->unsigned rescale before all terminator's operands
    for (int32_t i = 0; i < terminator->getNumOperands(); i++) {
      auto defining_op = terminator->getOperand(i).getDefiningOp();
      // Skip if operand of terminator is block arg (nullptr in this case)
      if (!defining_op) continue;
      Value input_val = defining_op->getResult(0);

      // Check if graph output is unsigned type.
      auto unsigned_output_type = dyn_cast<mlir::ShapedType>(output_types[i]);
      if (!unsigned_output_type) continue;

      // Check if graph output is unsigned type.
      Type rescaled_type;
      int32_t unsigned_zp_val, unused_output_zp_val;
      unsigned bit_width;

      if (!ConvertUnsignedToSignedRescale(builder, output_types[i], rescaled_type,
                                unsigned_zp_val, unused_output_zp_val,
                                bit_width))
        continue;

      // convert terminator operand type back to original output_type.
      auto terminator_operand_type =
          dyn_cast<mlir::ShapedType>(terminator->getOperand(i).getType());
      if (!terminator_operand_type) continue;
      int operand_zp_val = 0;
      auto quantized_type = dyn_cast<mlir::quant::UniformQuantizedType>(
          terminator_operand_type.getElementType());
      if (quantized_type) {
        operand_zp_val = adjustZeroPointForTOSA(quantized_type.getZeroPoint(),
                                                bit_width, /*is_unsigned=*/true);
      }

      // Keep original input_val use with tmp_val.
      auto tmp_const_type =
          RankedTensorType::get({1}, builder.getIntegerType(bit_width));
      auto tmp_const_attr =
          DenseElementsAttr::get(tmp_const_type, {APInt(bit_width, 0)});
      Value tmp_val =
          builder.create<TFL::ConstOp>(loc, tmp_const_type, tmp_const_attr);
      input_val.replaceUsesWithIf(tmp_val, [&terminator](OpOperand& use) {
        return use.getOwner() == terminator;
      });

      // Create zero point constants with appropriate bit_width
      // input_zp must match input bit_width (signed), output_zp must match
      // output bit_width (unsigned)
      auto rescale_input_zp =
          createZeroPointTensor(builder, loc, tmp_const_type, operand_zp_val);
      auto rescale_output_zp =
          createZeroPointTensor(builder, loc, tmp_const_type, unsigned_zp_val);

      if (!rescale_input_zp || !rescale_output_zp) {
        return function.emitError("Failed to create input zero point tensors.");
      }

      const auto rounding_mode_attr = tosa::RoundingModeAttr::get(
          builder.getContext(), tosa::RoundingMode::SINGLE_ROUND);
      auto rescale_op = builder.create<tosa::RescaleOp>(
          loc, unsigned_output_type, input_val, multiplier, shift,
          rescale_input_zp.value(), rescale_output_zp.value(),
          /* scale32 = */ builder.getBoolAttr(true),
          /* rounding_mode = */ rounding_mode_attr,
          /* per_channel = */ builder.getBoolAttr(false),
          /* input_unsigned = */ builder.getBoolAttr(false),   // signed ->
          /* output_unsigned = */ builder.getBoolAttr(true));  // unsigned

      Operation* op_rescale_op = static_cast<Operation*>(rescale_op);
      bb.push_back(op_rescale_op);
      op_rescale_op->moveBefore(terminator);
      tmp_val.replaceAllUsesWith(rescale_op.getResult());
      tmp_val.getDefiningOp()->erase();
      bb.push_front(rescale_output_zp.value().getDefiningOp());
      bb.push_front(rescale_input_zp.value().getDefiningOp());
    }
  }

  return success();
}

class ConvertTFLUnsignedToSigned
    : public impl::TosaConvertTFLUnsignedIntToSignedPassBase<
          ConvertTFLUnsignedToSigned> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    auto& ctx = getContext();
    mlir::func::FuncOp func = getOperation();

    func.walk([&](Operation* op) {
      if (isa<TosaOp>(op)) {
        // Run this before calling convert_graph_uint8_tensor as rescaling
        // introduces tosa ops
        op->emitError(
            "tosa operations are not expected in this pass. Run "
            "tosa-convert-tfl-uint-to-int before tosa-legalize-tfl");
      }
    });

    // Convert uint const tensor. const needs to be handled specifically.
    patterns.add<ConvertUnsignedQConstOp>(&ctx);
    (void)applyPatternsGreedily(func, std::move(patterns));

    // Replace uint tensor in the graph and insert rescale as needed.
    (void)convert_graph_unsigned_tensor(ctx, func);
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertTFLUnsignedIntToSignedPass() {
  return std::make_unique<ConvertTFLUnsignedToSigned>();
}

}  // namespace tosa
}  // namespace mlir
