/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This header file defines common utils used by TFLite transformation
// passes to work with tf.FakeQuant* ops.
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_FAKE_QUANT_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_FAKE_QUANT_UTILS_H_

#include <string>
#include <vector>

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"

namespace mlir {
namespace TFL {

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
    if (!matchPattern(min, m_Constant(&min_value))) {
      return false;
    }
    if (!matchPattern(max, m_Constant(&max_value))) {
      return false;
    }
    return true;  // Successfully matched and fetched.
  }
};

// Inserts a "tfl.quantize" and "tfl.dequantize" op pair (QDQs) after the
// tf.FakeQyantWithMinMax{Vars|VarsPerChannel|Args}Op
// before the op being constant folded. Since the constant
// folding logic will use a "arith.constant" op to replace the
// "tf.FakeQuantWithMinMaxVarsOp", the "tfl.quantize" op is used to preserve
// the quantization parameters as a TypeAttr and "tfl.dequantize" op used to
// convert the output type to the next op. Here are the transformations:
//
// input   min cst       max cst          input   min cst       max cst
//  \       |             |                \       |             |
//   \  (tf.Identity) (tf.Identity)   =>    \  (tf.Identity) (tf.Identity)
//    \     |             |                  \     |             |
//       tf.FakeQuantWithMinMaxVars       tf.FakeQuantWithMinMaxVars
//                   |                                 |
//                                                tfl.quantize
//                                                     |
//                                                tfl.dequantize
//                                                     |
// If the input is a constant, the result pattern will eventually converted to
//
//            quant-emulated input
//                   |
//               tfl.quantize
//                   |
//              tfl.dequantize
//                   |
//
//
// Warns if the (most likely unwanted, currently not quite correctly handled)
// case of back-to-back tf.FakeQuant occurs
//
//             tf.FakeQuant*
//                   |
//             tf.FakeQuant*
//
template <typename TFFakeQuantOp, bool PerAxis, class FetchMinMax>
class InsertTFLQuantOpsAfterTFFakeQuantOp {
 public:
  explicit InsertTFLQuantOpsAfterTFFakeQuantOp(bool use_fake_quant_num_bits)
      : use_fake_quant_num_bits_(use_fake_quant_num_bits) {}

  FetchMinMax fetch_min_max_;

  using FetchAttrType = typename FetchMinMax::AttrType;
  LogicalResult matchAndRewrite(TFFakeQuantOp tf_op,
                                OpBuilder &rewriter) const {
    // We don't want to insert quantize/dequantize if the quantize op exists.
    auto res = tf_op.getOutputs();
    if (!res.hasOneUse() || isa<QuantizeOp>(*res.user_begin())) {
      return failure();
    }

    // Extract the min/max constant values from the operands. We also consider
    // a special case that there are tf.Identity ops between the min/max
    // constants and the tf.FakeQuantWithMinMaxVarsOp.

    FetchAttrType min_value, max_value;
    if (!fetch_min_max_(tf_op, min_value, max_value)) {
      return failure();
    }

    int quant_dim = -1;
    if (PerAxis) {
      // This is a special case that the quant_dim is the last dimensions.
      quant_dim = mlir::cast<ShapedType>(res.getType()).getRank() - 1;
    }
    // Use the min/max from the operands and the num_bits and narrow_range
    // attribute to create the quantization parameter for the new quantize op.
    rewriter.setInsertionPointAfter(tf_op.getOperation());
    IntegerAttr num_bits = rewriter.getI64IntegerAttr(tf_op.getNumBits());
    BoolAttr narrow_range = rewriter.getBoolAttr(tf_op.getNarrowRange());
    Type res_type = tf_op.getType();
    TypeAttr qtype = quant::GetQuantizedTypeAttr(
        rewriter, res_type, min_value, max_value, quant_dim, num_bits,
        narrow_range, /*is_signed=*/false, /*legacy_float_scale=*/false,
        use_fake_quant_num_bits_);
    if (!qtype) {
      return failure();
    }

    // Finally, use the quantization parameter to create the quantize and
    // dequantize ops, and insert them between the tf.FakeQuantWithMinMaxVarsOp
    // and its users.
    Value value = tf_op.getOutputs();
    auto quantize = rewriter.create<TFL::QuantizeOp>(
        tf_op.getLoc(), qtype.getValue(), value, qtype);
    auto dequantize = rewriter.create<TFL::DequantizeOp>(
        tf_op.getLoc(), res_type, quantize.getOutput());
    value.replaceAllUsesWith(dequantize);
    quantize.getOperation()->replaceUsesOfWith(dequantize, value);

    return success();
  }

  bool use_fake_quant_num_bits_;
};

// Removes the wrapper of the tf.FakeQuant* ops and creates the tfl.quantize
// and tfl.dequantize pairs before tf.FakeQuant* being foled.
LogicalResult ConvertFakeQuantOps(func::FuncOp func, MLIRContext *ctx,
                                  bool use_fake_quant_num_bits = false);

// Returns the names of all the considered tf.FakeQuant* ops.
std::vector<std::string> AllTfFakeQuantOps();

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_FAKE_QUANT_UTILS_H_
