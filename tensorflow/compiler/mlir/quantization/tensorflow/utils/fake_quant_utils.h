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

// This header file defines common utils used by TF-Quant transformation
// passes to work with tf.FakeQuant* ops. Copied and modified from
// //third_party/tensorflow/compiler/mlir/lite/utils/fake_quant_utils.h
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_UTILS_FAKE_QUANT_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_UTILS_FAKE_QUANT_UTILS_H_

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/common/tf_quantization_lib/tf_quantization_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"

namespace mlir {
namespace quant {

template <class TFFakeQuantOp>
struct FetchMinMaxAttrs {
  using AttrType = FloatAttr;
  bool operator()(TFFakeQuantOp tf_op, AttrType &min_value,
                  AttrType &max_value) const {
    min_value = tf_op.getMinAttr();
    max_value = tf_op.getMaxAttr();
    return true;  // Successfully matched and fetched.
  }
};

template <class TFFakeQuantOp>
struct FetchConstantMinMaxInputs {
  using AttrType = DenseFPElementsAttr;
  bool operator()(TFFakeQuantOp tf_op, AttrType &min_value,
                  AttrType &max_value) const {
    Value min = tf_op.getMin(), max = tf_op.getMax();
    if (auto min_id = min.getDefiningOp<TF::IdentityOp>()) {
      min = min_id.getInput();
    }
    if (auto max_id = max.getDefiningOp<TF::IdentityOp>()) {
      max = max_id.getInput();
    }

    if (!matchPattern(min, m_Constant(&min_value))) {
      return false;
    }
    if (!matchPattern(max, m_Constant(&max_value))) {
      return false;
    }
    return true;  // Successfully matched and fetched.
  }
};

// Inserts a "quant.qcast" and "quant.dcast" op pair (QDQs) in place of the
// tf.FakeQyantWithMinMax{Vars|VarsPerChannel|Args}Op
// before the op being constant folded. Since the constant
// folding logic will use a "arith.constant" op to replace the
// "tf.FakeQuantWithMinMaxVarsOp", the "quant.qcast" op is used to preserve
// the quantization parameters as a TypeAttr and "quant.dcast" op used to
// convert the output type to the next op. Here are the transformations:
//
// input   min cst       max cst              input
//  \       |             |                     |
//   \  (tf.Identity) (tf.Identity)   =>   quant.qcast
//    \     |             |                     |
//       tf.FakeQuantWithMinMaxVars        quant.dcast
//                   |                          |
//
// Warns if the (most likely unwanted, currently not quite correctly handled)
// case of back-to-back tf.FakeQuant occurs
//
//             tf.FakeQuant*
//                   |
//             tf.FakeQuant*
//
template <typename TFFakeQuantOp, bool PerAxis, class FetchMinMax>
class ConvertFakeQuantOpToQuantOps {
 public:
  explicit ConvertFakeQuantOpToQuantOps(bool use_fake_quant_num_bits)
      : use_fake_quant_num_bits_(use_fake_quant_num_bits) {}

  FetchMinMax fetch_min_max_;

  using FetchAttrType = typename FetchMinMax::AttrType;
  LogicalResult matchAndRewrite(TFFakeQuantOp tf_op,
                                OpBuilder &rewriter) const {
    if (tf_op.getNumBits() != 8) {
      return failure();
    }

    // Extract the min/max constant values from the operands. We also consider
    // a special case that there are tf.Identity ops between the min/max
    // constants and the tf.FakeQuantWithMinMaxVarsOp.
    FetchAttrType min_value, max_value;
    if (!fetch_min_max_(tf_op, min_value, max_value)) {
      return failure();
    }

    Value input = tf_op.getInputs();
    int quant_dim = -1;
    auto input_type = mlir::cast<ShapedType>(input.getType());
    if (PerAxis) {
      if (!input_type.hasRank()) {
        tf_op.emitError("The input should have known rank for per-channel op.");
        return failure();
      }
      // This is a special case that the quant_dim is the last dimensions.
      quant_dim = input_type.getRank() - 1;
    }
    // Use the min/max from the operands and the num_bits and narrow_range
    // attribute to create the quantization parameter for the new quantize op.
    rewriter.setInsertionPointAfter(tf_op.getOperation());
    IntegerAttr num_bits = rewriter.getI64IntegerAttr(tf_op.getNumBits());
    BoolAttr narrow_range = rewriter.getBoolAttr(tf_op.getNarrowRange());
    Type res_type = tf_op.getType();
    TypeAttr qtype = tf_quant::GetQuantizedTypeAttr(
        rewriter, input_type, min_value, max_value, quant_dim, num_bits,
        narrow_range, /*is_signed=*/true, /*legacy_float_scale=*/false,
        use_fake_quant_num_bits_);
    if (!qtype) {
      return failure();
    }

    // Finally, use the quantization parameter to create the quantize and
    // dequantize ops, and insert them between the tf.FakeQuantWithMinMaxVarsOp
    // and its users.
    auto quantize = rewriter.create<mlir::quant::ir::QuantizeCastOp>(
        tf_op.getLoc(), qtype.getValue(), input);
    auto dequantize = rewriter.create<mlir::quant::ir::DequantizeCastOp>(
        tf_op.getLoc(), res_type, quantize.getResult());
    tf_op.getOutputs().replaceAllUsesWith(dequantize);

    return success();
  }

  bool use_fake_quant_num_bits_;
};

// Removes the wrapper of the tf.FakeQuant* ops and creates the quant.qcast
// and quant.dcast pairs before tf.FakeQuant* ops are being folded.
LogicalResult ConvertFakeQuantOps(func::FuncOp func, MLIRContext *ctx,
                                  bool use_fake_quant_num_bits);

}  // namespace quant
}  // namespace mlir
#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_UTILS_FAKE_QUANT_UTILS_H_
