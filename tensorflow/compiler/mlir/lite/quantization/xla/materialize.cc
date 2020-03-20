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

// This transformation pass quantize the constant and rewrite the quantization
// ops by xla_hlo primitive ops.
#include <cstdint>
#include <iterator>
#include <numeric>
#include <string>

#include "absl/memory/memory.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Quant/QuantOps.h"  // TF:llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/xla/client/lib/quantize.h"

//===----------------------------------------------------------------------===//
// The pass to materialize the quantization results by xla primitive ops.
//
namespace mlir {
namespace xla_hlo {

namespace {

// This pattern matches the "constant->qcast->dcast" pattern and replaces it by
// "quantized constant->xla_hlo.dequantize". If it only matches the
// "non-constant->qcast->dcast" pattern, it will remove both the "qcast->dcast".
// We chain the pattern as a whole to bypass the type checks of the normal
// xla_hlo ops.
// TODO(fengliuai): make this pass work for bf16 input.
class RewriteDequantize : public OpRewritePattern<quant::DequantizeCastOp> {
 public:
  explicit RewriteDequantize(int64_t size, MLIRContext *context)
      : OpRewritePattern<quant::DequantizeCastOp>(context), size_(size) {}

  LogicalResult matchAndRewrite(quant::DequantizeCastOp op,
                                PatternRewriter &rewriter) const override {
    // quant.dcast
    // xla_hlo dequantize only takes min/max, so let's recover them from
    // the quantization parameters.
    Value dcast = op.arg();
    auto type = quant::QuantizedType::getQuantizedElementType(dcast.getType());
    if (!type || !type.isa<quant::UniformQuantizedType>()) {
      return failure();
    }
    auto qtype = type.cast<quant::UniformQuantizedType>();
    double scale = qtype.getScale();
    int64_t zero_point = qtype.getZeroPoint();
    float min = scale * (qtype.getStorageTypeMin() - zero_point);
    float max = scale * (qtype.getStorageTypeMax() - zero_point);

    // quant.qcast
    auto qcast =
        llvm::dyn_cast_or_null<quant::QuantizeCastOp>(dcast.getDefiningOp());
    if (!qcast) return failure();

    // constant
    DenseFPElementsAttr attr;
    // If it isn't a floating-point constant or the size is too small, let's
    // remove the quantization. Also the last dimension size should be a
    // multiplier of 4, so the shape isn't broken during packing and unpacking.
    if (!matchPattern(qcast.arg(), m_Constant(&attr)) ||
        attr.getNumElements() <= size_ ||
        attr.getType().getDimSize(attr.getType().getRank() - 1) % 4 != 0) {
      op.getResult().replaceAllUsesWith(qcast.arg());
      return success();
    }
    // TODO(fengliuai): implement transpose if it has high dimension.

    // Create the quantized result
    auto quantized_result =
        quant::Quantize(attr, qtype).dyn_cast_or_null<DenseIntElementsAttr>();
    if (!quantized_result) {
      return failure();
    }

    // Pack the uint8 bits to uint32. The shape is changed from from
    // [n0, n1, ..., nk] to [n0, n1, ..., nk / 4].
    std::vector<uint8_t> raw_data;
    for (auto d : quantized_result.getValues<uint8_t>()) {
      raw_data.push_back(d);
    }
    // The packing might increase the data size by paddings.
    auto packed_data = xla::PackToUint32<uint8_t>(raw_data);
    auto packed_shape = attr.getType().getShape().vec();
    int lower_dims = std::accumulate(
        packed_shape.begin(),
        std::next(packed_shape.begin(), packed_shape.size() - 1), 1,
        std::multiplies<int>());
    packed_shape[packed_shape.size() - 1] = packed_data.size() / lower_dims;
    auto packed_type =
        RankedTensorType::get(packed_shape, rewriter.getIntegerType(32));

    auto packed_quantized_result =
        DenseElementsAttr::get<uint32_t>(packed_type, packed_data);
    auto quantized_constant =
        rewriter.create<ConstantOp>(qcast.getLoc(), packed_quantized_result);

    // Create the xla dequantize op with bf16 output
    auto dequantized_type = RankedTensorType::get(attr.getType().getShape(),
                                                  rewriter.getBF16Type());
    auto dequantize = rewriter.create<DequantizeOp>(
        qcast.getLoc(), dequantized_type, quantized_constant,
        rewriter.getF32FloatAttr(min), rewriter.getF32FloatAttr(max),
        rewriter.getStringAttr("MIN_COMBINED"), rewriter.getBoolAttr(false),
        rewriter.getBoolAttr(false));

    // Convert bf16 output back to f32
    rewriter.replaceOpWithNewOp<ConvertOp>(op, op.getResult().getType(),
                                           dequantize);
    return success();
  }

 private:
  int64_t size_;
};

// Materialize the quantization results by hlo primitive ops.
struct MaterializeToXlaPass : public FunctionPass<MaterializeToXlaPass> {
  explicit MaterializeToXlaPass() = default;
  MaterializeToXlaPass(const MaterializeToXlaPass &) {}

  void runOnFunction() override;
};

void MaterializeToXlaPass::runOnFunction() {
  FuncOp func = getFunction();
  MLIRContext *ctx = &getContext();

  OwningRewritePatternList patterns;
  // TODO(fengliuai): make the size 6 configurable.
  patterns.insert<RewriteDequantize>(6, ctx);

  applyPatternsGreedily(func, patterns);
}

}  // namespace

// Creates an instance of the xla_hlo dialect quantization propagation pass.
std::unique_ptr<OpPassBase<FuncOp>> CreateMaterializeToXlaPass() {
  return std::make_unique<MaterializeToXlaPass>();
}

static PassRegistration<MaterializeToXlaPass> pass(
    "xla-hlo-materialize-quant",
    "Materialize the quantization results by xla primitve ops");

}  // namespace xla_hlo
}  // namespace mlir
