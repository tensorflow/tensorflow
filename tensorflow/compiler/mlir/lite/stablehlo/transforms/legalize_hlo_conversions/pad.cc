/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/pad.h"

#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {

ConversionState BuildConversionState(mhlo::PadOp mhlo_pad,
                                     ConversionPatternRewriter& rewriter) {
  ConversionState state{
      /*.shlo_op=*/mhlo_pad.getOperation(),
      /*.rewriter=*/rewriter,
      /*.last_tf_op=*/nullptr,
  };
  return state;
}

// Converts the given StableHLO Pad operation to a chain of TFLite operations.
//
// StableHLO Pad allows dilating, padding and cropping its input, in that order.
// This can be implemented in TFLite as a sequence of these operations. Note
// that all operations do not always need to be called: if there is no dilation
// (resp. pad, crop) we do not need to add it to the chain.
//
// TFLite does not provide a crop operation, the StridedSlice one is used
// instead.
LogicalResult ConvertPadOp::matchAndRewrite(
    mhlo::PadOp mhlo_pad, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // We don't need to match the pad op as we always know how to convert it.
  ConversionState state = BuildConversionState(mhlo_pad, rewriter);

  // Dilate when interior padding is specified different from 0.
  AddDilateOpIfRequired(state, mhlo_pad.getInteriorPadding(),
                        mhlo_pad.getPaddingValue(),
                        /*is_padding=*/true);
  // Pad when padding has positive values.
  AddPadOpIfRequired(state, mhlo_pad.getEdgePaddingLow(),
                     mhlo_pad.getEdgePaddingHigh(), mhlo_pad.getPaddingValue());
  // Crop when padding has negative values.
  //
  // Note that there is no crop operation in TFLite so we use the StridedSlice
  // operation instead.
  const DenseElementsAttr strides_data = CreateDenseElementsAttr(
      state.rewriter,
      llvm::SmallVector<int64_t, 6>(state.GetOperandShape().size(), 1));
  AddStridedSliceOpIfRequired(state, mhlo_pad.getEdgePaddingLow(),
                              mhlo_pad.getEdgePaddingHigh(), strides_data);

  if (state.last_tf_op) {
    rewriter.replaceOp(mhlo_pad, state.last_tf_op);
  } else {
    rewriter.replaceOp(mhlo_pad, mhlo_pad.getOperand());
  }
  return success();
}

}  // namespace odml
}  // namespace mlir
