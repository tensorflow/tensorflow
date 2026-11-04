/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

// This transformation pass propagates QSV information through the model.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/transforms/quantization/quant_utils.h"

namespace mlir {
namespace TFL {
namespace {

#define GEN_PASS_DEF_BIASQUANTIZERPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Returns the quantization dimension for the bias.
//
// Some kernels allow 1D bias, broadcasting it inside the kernel. In this case,
// the `quantizedDimension` is 0 for per-channel quantization. However, for
// some kernels that require the bias to be already broadcasted to match the
// accumulation shape, the very last dimension should be used. This function
// returns the last dimension of the bias.
//
// Returns -1 for scalar bias. Returns std::nullopt if bias is none.
std::optional<int> GetAdjustedQuantizationDimension(
    RequiresQuantizedBiasInterface op, int bias_index) {
  int adjusted_quant_dim = -1;
  if (op->getNumOperands() > bias_index) {
    const mlir::Value bias = op->getOperand(bias_index);
    const Type bias_type = bias.getType();
    if (mlir::isa<NoneType>(bias_type)) {
      return std::nullopt;
    }

    if (const auto shaped_bias_type =
            mlir::dyn_cast<mlir::ShapedType>(bias_type)) {
      if (shaped_bias_type.hasRank()) {
        const int bias_rank = shaped_bias_type.getRank();
        adjusted_quant_dim = bias_rank > 1 ? bias_rank - 1 : 0;
      }
    }
  }
  return adjusted_quant_dim;
}

// Reads the elements of a float constant (f32, f16 or bf16), which may be
// stored as a dense resource, as doubles.
std::optional<SmallVector<double>> ReadFloatElements(ElementsAttr attr) {
  auto float_type = dyn_cast<FloatType>(attr.getElementType());
  if (!float_type) return std::nullopt;
  SmallVector<double> values;
  values.reserve(attr.getNumElements());
  auto append = [&](APFloat value) {
    bool loses_info;
    value.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven,
                  &loses_info);
    values.push_back(value.convertToDouble());
  };
  if (auto dense = dyn_cast<DenseFPElementsAttr>(attr)) {
    for (APFloat value : dense.getValues<APFloat>()) append(value);
    return values;
  }
  auto resource = dyn_cast<DenseResourceElementsAttr>(attr);
  if (!resource || !resource.getRawHandle().getBlob()) return std::nullopt;
  ArrayRef<char> data = resource.getRawHandle().getBlob()->getData();
  const unsigned bit_width = float_type.getWidth();
  if (bit_width != 16 && bit_width != 32) return std::nullopt;
  const size_t bytes_per_element = bit_width / 8;
  if (data.size() != attr.getNumElements() * bytes_per_element) {
    return std::nullopt;
  }
  for (size_t offset = 0; offset < data.size(); offset += bytes_per_element) {
    // Read with the element's own width so the value lands in the low bits on
    // hosts of either byte order.
    uint64_t bits;
    if (bit_width == 16) {
      uint16_t value;
      std::memcpy(&value, data.data() + offset, sizeof(value));
      bits = value;
    } else {
      uint32_t value;
      std::memcpy(&value, data.data() + offset, sizeof(value));
      bits = value;
    }
    append(APFloat(float_type.getFloatSemantics(), APInt(bit_width, bits)));
  }
  return values;
}

// Returns a per output channel quantized type standing in for a constant
// `tfl.blockwise_dequantize` filter of a `tfl.fully_connected`, from which the
// bias scale is derived. The filter has no single per-channel scale when it
// has several blocks along its input dimension, so each channel uses the
// largest of its block scales. Any positive bias scale is valid because the
// FULLY_CONNECTED kernel dequantizes a bias with its own scale; the largest one
// keeps the int32 bias from overflowing.
std::optional<quant::QuantizedType> GetQTypeForBlockwiseFilter(
    Operation* op, int operand_index) {
  // Only biases of ops whose filter `FuseQDQPass` folds into a blockwise
  // quantized filter are quantized. Otherwise the op stays a float op and an
  // int32 bias would need a DEQUANTIZE that the runtime does not support.
  // If this returns std::nullopt for an op that `CanFoldBlockwiseWeightOperand`
  // accepts (e.g. a non-positive block scale or unreadable scales), the bias
  // stays float and `FuseQDQPass` keeps the float `tfl.blockwise_dequantize`
  // filter, i.e. the op falls back to weight-only.
  if (!isa<FullyConnectedOp>(op) || operand_index != 1 ||
      !CanFoldBlockwiseWeightOperand(op, operand_index,
                                     op->getOperand(operand_index))) {
    return std::nullopt;
  }
  auto bw_deq =
      op->getOperand(operand_index).getDefiningOp<BlockwiseDequantizeOp>();
  auto filter_type = cast<RankedTensorType>(bw_deq.getType());
  if (!filter_type.hasStaticShape()) return std::nullopt;
  ElementsAttr scales_attr = GetConstantElements(bw_deq.getScales());
  auto scales_type = dyn_cast<RankedTensorType>(scales_attr.getType());
  if (!scales_type || scales_type.getRank() != 2) return std::nullopt;

  const int64_t num_units = filter_type.getDimSize(0);
  const int64_t num_unit_blocks = scales_type.getDimSize(0);
  const int64_t num_depth_blocks = scales_type.getDimSize(1);
  if (num_units <= 0 || num_unit_blocks <= 0 || num_depth_blocks <= 0 ||
      bw_deq.getBlockShape().size() != 2) {
    return std::nullopt;
  }
  const int64_t unit_block_size =
      cast<IntegerAttr>(bw_deq.getBlockShape()[0]).getInt();
  if (unit_block_size <= 0) return std::nullopt;

  std::optional<SmallVector<double>> maybe_block_scales =
      ReadFloatElements(scales_attr);
  if (!maybe_block_scales.has_value() ||
      static_cast<int64_t>(maybe_block_scales->size()) !=
          num_unit_blocks * num_depth_blocks) {
    return std::nullopt;
  }
  const SmallVector<double>& block_scales = *maybe_block_scales;
  SmallVector<double> channel_scales(num_units, 0.0);
  for (int64_t n = 0; n < num_units; ++n) {
    const int64_t unit_block = num_unit_blocks == 1 ? 0 : n / unit_block_size;
    if (unit_block >= num_unit_blocks) return std::nullopt;
    for (int64_t kb = 0; kb < num_depth_blocks; ++kb) {
      channel_scales[n] = std::max(
          channel_scales[n], block_scales[unit_block * num_depth_blocks + kb]);
    }
    if (!(channel_scales[n] > 0.0) || !std::isfinite(channel_scales[n])) {
      return std::nullopt;
    }
  }

  Builder builder(op->getContext());
  SmallVector<int64_t> zero_points(num_units, 0);
  return quant::UniformQuantizedPerAxisType::getChecked(
      op->getLoc(), quant::QuantizationFlags::Signed, builder.getI8Type(),
      builder.getF32Type(), channel_scales, zero_points,
      /*quantizedDimension=*/0, /*storageTypeMin=*/-128,
      /*storageTypeMax=*/127);
}

std::optional<quant::QuantizedType> GetBiasQuantizedType(
    RequiresQuantizedBiasInterface op) {
  int bias_index = op.GetBiasOperandIndex();
  std::vector<int> non_bias_operand_indices = op.GetNonBiasOperandIndices();

  std::optional<int> adjusted_quant_dim_opt =
      GetAdjustedQuantizationDimension(op, bias_index);
  if (!adjusted_quant_dim_opt) {
    return std::nullopt;
  }

  std::vector<QuantizedType> op_types{};
  op_types.reserve(non_bias_operand_indices.size());
  for (const int non_bias_operand_index : non_bias_operand_indices) {
    auto operand_qtype =
        GetQTypeFromDefiningDequantize(op->getOperand(non_bias_operand_index));
    if (!operand_qtype.has_value()) {
      operand_qtype =
          GetQTypeForBlockwiseFilter(op.getOperation(), non_bias_operand_index);
    }
    if (operand_qtype.has_value()) {
      op_types.push_back(operand_qtype.value());
    }
  }
  if (op_types.size() < non_bias_operand_indices.size()) {
    // Not all the non-bias operands are quantized so not quantizing the bias.
    // This could for example happen in weight-only.
    return std::nullopt;
  }
  return mlir::TFL::GetUniformQuantizedTypeForBias(op_types,
                                                   *adjusted_quant_dim_opt);
}

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

class QuantizeBias
    : public OpInterfaceRewritePattern<RequiresQuantizedBiasInterface> {
  using OpInterfaceRewritePattern<
      RequiresQuantizedBiasInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(mlir::RequiresQuantizedBiasInterface op,
                                PatternRewriter& rewriter) const override {
    auto new_bias_qtype = GetBiasQuantizedType(op);
    if (!new_bias_qtype) {
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "Failed to get bias quantized type");
    }
    auto bias_value = op->getOperand(op.GetBiasOperandIndex());
    auto existing_bias_qtype = GetQTypeFromDefiningDequantize(bias_value);
    if (existing_bias_qtype && *existing_bias_qtype == *new_bias_qtype) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), "Bias already quantized with the same type");
    }

    // Provide the target_op to ensure QDQ is only inserted for this use.
    if (failed(InsertQDQ(bias_value, *new_bias_qtype, rewriter,
                         op.getOperation()))) {
      op->emitError("Failed to insert QDQ before bias");
      return failure();
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct BiasQuantizerPass
    : public impl::BiasQuantizerPassBase<BiasQuantizerPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BiasQuantizerPass)

  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

void BiasQuantizerPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  mlir::ModuleOp module = getOperation();

  RewritePatternSet patterns(ctx);
  patterns.add<QuantizeBias>(ctx);

  GreedyRewriteConfig greedy_config;
  greedy_config.enableFolding(true);

  if (failed(
          applyPatternsGreedily(module, std::move(patterns), greedy_config))) {
    module.emitError("Failed to apply BiasQuantizerPass patterns.");
    signalPassFailure();
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// Pass Creation Function
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<mlir::ModuleOp>> CreateBiasQuantizerPass() {
  return std::make_unique<BiasQuantizerPass>();
}

}  // namespace TFL
}  // namespace mlir
