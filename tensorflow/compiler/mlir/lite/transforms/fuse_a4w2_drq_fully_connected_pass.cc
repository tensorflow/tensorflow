/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

namespace mlir {
namespace TFL {
namespace {

#define GEN_PASS_DEF_FUSEA4W2DRQFULLYCONNECTEDPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// The contract named by `kA4W2SpecName`. Every one of these is baked into the
// runtime kernel rather than carried in the flatbuffer, so the pattern below
// has to verify each one explicitly: a graph that differs in any of them would
// still be serialized as `cint2_fp32_int4_e8m0_drq` and then silently evaluated
// with the wrong numerics.
constexpr char kA4W2SpecName[] = "cint2_fp32_int4_e8m0_drq";
// Activations are quantized in blocks of 32 along the contracting axis.
constexpr int64_t kActBlockSize = 32;
// Weights use a grid centered between the integers; see
// `pure_observer.CENTERED_ZERO_POINT` on the JAX side and the `+ 0.5f` in
// `EvalA4W2DRQ` on the runtime side.
constexpr double kCenteredZeroPoint = -0.5;

// Returns true if `value` is a constant whose elements are all exactly
// `expected`.
bool IsSplatFloatConstant(mlir::Value value, double expected) {
  if (!value || mlir::isa<NoneType>(value.getType())) return false;

  ElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) {
    auto const_op = value.getDefiningOp<TFL::ConstOp>();
    if (!const_op) return false;
    attr = const_op.getValue();
  }

  auto fp_attr = mlir::dyn_cast<DenseFPElementsAttr>(attr);
  if (!fp_attr) return false;
  return llvm::all_of(fp_attr.getValues<APFloat>(), [expected](APFloat v) {
    return v.convertToDouble() == expected;
  });
}

// Pattern to match an a4w2 dynamic range fully connected pattern:
//
//   %act_q:3 = tfl.blockwise_quantize(%x, block_shape=[..., 32], ...)
//   %act_dq = tfl.blockwise_dequantize(%act_q#0, %act_q#1, %act_q#2, ...)
//   %w_dq = tfl.blockwise_dequantize(%q_w, %scales, %zp, block_shape=[1, K])
//   %res = tfl.fully_connected(%act_dq, %w_dq, %bias)
//
// and fuse it into a single DRQ fully connected op:
//
//   %q_w_per_axis = tfl.pseudo_qconst(...) : tensor<NxK x
//   !quant.uniform<i2:f32:0, {scales}>> %res = tfl.fully_connected(%x,
//   %q_w_per_axis, %bias)
//          { tfl.quant_spec = { spec = "cint2_fp32_int4_e8m0_drq", act_dilation
//          = ... }
//          }
struct FuseA4W2DRQFullyConnectedPattern
    : public OpRewritePattern<TFL::FullyConnectedOp> {
  using OpRewritePattern<TFL::FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fc,
                                PatternRewriter& rewriter) const override {
    // The rewrite below builds a single-result op, so a shuffled-weights
    // fully_connected (which has a second result for the shuffled input)
    // cannot be replaced by it.
    if (fc.getNumResults() != 1) return failure();
    if (fc.getWeightsFormat() != "DEFAULT") return failure();

    // 1. Match activations side.
    mlir::Value act_val = fc.getInput();
    auto act_deq = act_val.getDefiningOp<BlockwiseDequantizeOp>();
    if (!act_deq) return failure();

    mlir::Value act_q_val = act_deq.getInput();
    auto act_q = act_q_val.getDefiningOp<BlockwiseQuantizeOp>();
    if (!act_q) return failure();

    // Consuming `act_q`'s quantized result is not on its own enough to make
    // `act_deq` its inverse: the dequantize takes its own scales and zero
    // points, while every parameter of the fused op is read off `act_q`. A
    // dequantize wired to different quantization parameters describes a
    // different computation and must not be folded away here.
    //
    // The block shape does not need its own check: the verifier ties the
    // scales grid to it, so once the scales are required to be `act_q`'s own
    // result a differing block shape cannot verify.
    if (act_deq.getScales() != act_q.getScale()) return failure();
    if (act_deq.getZeroPoints() != act_q.getZeroPoint()) return failure();

    if (!act_q.getSymmetric()) return failure();

    auto act_type =
        mlir::dyn_cast<RankedTensorType>(act_q.getOutput().getType());
    if (!act_type || !act_type.getElementType().isInteger(4)) return failure();

    // The kernel derives the scale as `exp2(ceil(log2(raw_scale)))`, i.e. it
    // assumes an e8m0 scale. An f32 scale would quantize to different values.
    if (!mlir::isa<Float8E8M0FNUType>(act_q.getScaleType())) return failure();

    // Verify block shape on activations: innermost contracting dimension block
    // size must be 32, outer dims must be 1.
    ArrayAttr act_block_shape = act_q.getBlockShapeAttr();
    if (!act_block_shape || act_block_shape.empty()) return failure();
    const int64_t act_rank = act_block_shape.size();
    if (mlir::cast<IntegerAttr>(act_block_shape[act_rank - 1]).getInt() !=
        kActBlockSize) {
      return failure();
    }
    for (int64_t i = 0; i < act_rank - 1; ++i) {
      if (mlir::cast<IntegerAttr>(act_block_shape[i]).getInt() != 1) {
        return failure();
      }
    }

    mlir::Value real_input = act_q.getInput();
    float act_dilation = act_q.getRangeDilation().convertToFloat();

    // 2. Match weights side.
    mlir::Value filter_val = fc.getFilter();
    auto weight_deq = filter_val.getDefiningOp<BlockwiseDequantizeOp>();
    if (!weight_deq) return failure();

    // The kernel reconstructs the weights as `(q + 0.5) * scale`, i.e. it
    // assumes a grid centered between the integers. Applying that to weights
    // quantized on a plain symmetric grid would shift every value by half a
    // step, so the centered zero point has to be verified rather than assumed.
    if (weight_deq.getSymmetric()) return failure();
    if (!IsSplatFloatConstant(weight_deq.getZeroPoints(), kCenteredZeroPoint)) {
      return failure();
    }

    mlir::Value q_weight_val = weight_deq.getInput();
    ElementsAttr q_weight_attr;
    if (!matchPattern(q_weight_val, m_Constant(&q_weight_attr))) {
      if (auto const_op = q_weight_val.getDefiningOp<TFL::ConstOp>()) {
        q_weight_attr = const_op.getValue();
      } else {
        return failure();
      }
    }
    auto q_weight_type =
        mlir::dyn_cast<RankedTensorType>(q_weight_attr.getType());
    if (!q_weight_type || q_weight_type.getRank() != 2) return failure();
    if (!q_weight_type.getElementType().isInteger(2)) return failure();

    int64_t num_units = q_weight_type.getDimSize(0);
    int64_t input_size = q_weight_type.getDimSize(1);

    // Verify weight block shape: [1, input_size] (per-channel across output
    // units).
    ArrayAttr weight_block_shape = weight_deq.getBlockShapeAttr();
    if (!weight_block_shape || weight_block_shape.size() != 2) return failure();
    if (mlir::cast<IntegerAttr>(weight_block_shape[0]).getInt() != 1 ||
        mlir::cast<IntegerAttr>(weight_block_shape[1]).getInt() != input_size) {
      return failure();
    }

    // Extract per-channel scales.
    ElementsAttr scales_attr;
    if (!matchPattern(weight_deq.getScales(), m_Constant(&scales_attr))) {
      if (auto const_op =
              weight_deq.getScales().getDefiningOp<TFL::ConstOp>()) {
        scales_attr = const_op.getValue();
      } else {
        return failure();
      }
    }

    SmallVector<double> per_channel_scales;
    per_channel_scales.reserve(num_units);
    if (auto dense_scales = mlir::dyn_cast<DenseFPElementsAttr>(scales_attr)) {
      for (const auto& fp : dense_scales.getValues<APFloat>()) {
        per_channel_scales.push_back(fp.convertToDouble());
      }
    } else {
      return failure();
    }
    if (per_channel_scales.size() != static_cast<size_t>(num_units)) {
      return failure();
    }

    // 3. Create UniformQuantizedPerAxisType for the weights.
    MLIRContext* ctx = fc.getContext();
    Type expressed_type = Float32Type::get(ctx);
    Type storage_type = IntegerType::get(ctx, 2);
    int64_t qmin = -2;
    int64_t qmax = 1;
    SmallVector<int64_t> zero_points(num_units, 0);
    int32_t quantized_dimension = 0;

    auto quant_type = quant::UniformQuantizedPerAxisType::get(
        /*flags=*/quant::QuantizationFlags::Signed, storage_type,
        expressed_type, per_channel_scales, zero_points, quantized_dimension,
        qmin, qmax);

    RankedTensorType new_filter_type =
        RankedTensorType::get(q_weight_type.getShape(), quant_type);

    mlir::Value new_filter = rewriter.create<TFL::QConstOp>(
        fc.getLoc(), TypeAttr::get(new_filter_type), q_weight_attr);

    // 4. Create tfl.quant_spec attribute dictionary.
    SmallVector<NamedAttribute, 2> spec_entries;
    spec_entries.push_back(
        rewriter.getNamedAttr("spec", rewriter.getStringAttr(kA4W2SpecName)));
    spec_entries.push_back(rewriter.getNamedAttr(
        "act_dilation", rewriter.getF32FloatAttr(act_dilation)));
    DictionaryAttr quant_spec_dict = rewriter.getDictionaryAttr(spec_entries);

    // 5. Replace the FullyConnectedOp.
    auto new_fc = rewriter.create<TFL::FullyConnectedOp>(
        fc.getLoc(), fc.getType(0), real_input, new_filter, fc.getBias(),
        fc.getFusedActivationFunctionAttr(), fc.getWeightsFormatAttr(),
        fc.getKeepNumDimsAttr(), fc.getAsymmetricQuantizeInputsAttr());
    new_fc->setAttr(TFL::kQuantSpecAttrName, quant_spec_dict);

    rewriter.replaceOp(fc, new_fc.getOutput());
    return success();
  }
};

struct FuseA4W2DRQFullyConnectedPass
    : public impl::FuseA4W2DRQFullyConnectedPassBase<
          FuseA4W2DRQFullyConnectedPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseA4W2DRQFullyConnectedPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext* context = func.getContext();

    RewritePatternSet patterns(context);
    patterns.add<FuseA4W2DRQFullyConnectedPattern>(context);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateFuseA4W2DRQFullyConnectedPass() {
  return std::make_unique<FuseA4W2DRQFullyConnectedPass>();
}

}  // namespace TFL
}  // namespace mlir
