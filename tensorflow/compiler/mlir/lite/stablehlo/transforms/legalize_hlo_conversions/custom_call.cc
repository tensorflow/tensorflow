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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/custom_call.h"

#include <optional>

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {

LogicalResult ConvertCustomCallOp::matchAndRewrite(
    mhlo::CustomCallOp mhlo_custom_call, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto tfl_custom = rewriter.create<TFL::CustomOp>(
      mhlo_custom_call.getLoc(), mhlo_custom_call.getResultTypes(),
      mhlo_custom_call.getInputs());
  tfl_custom.setCustomCodeAttr(
      rewriter.getStringAttr(mhlo_custom_call.getCallTargetName()));

  if (auto bc = mhlo_custom_call.getBackendConfig()) {
    if (auto stringattr = mlir::dyn_cast_or_null<mlir::StringAttr>(*bc)) {
      tfl_custom.setCustomOptionAttr(
          TFL::ConstBytesAttr::get(rewriter.getContext(), stringattr));
    }
  } else {
    tfl_custom.setCustomOptionAttr(
        TFL::ConstBytesAttr::get(rewriter.getContext(), ""));
  }

  rewriter.replaceOp(mhlo_custom_call, tfl_custom);
  return success();
}

std::optional<bool> IsCustomCallLegal(mhlo::CustomCallOp op) {
  if (op.getCallTargetName().starts_with("custom_call.")) {
    auto bc = op.getBackendConfig();
    if (!bc || mlir::isa<mlir::StringAttr>(*bc)) {
      return false;
    }
  }

  return true;
}

}  // namespace odml
}  // namespace mlir
