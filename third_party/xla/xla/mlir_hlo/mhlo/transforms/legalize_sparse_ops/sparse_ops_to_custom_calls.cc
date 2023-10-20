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

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace {

StringAttr getOperationTargetName(Operation* op) {
  // Strips off `dialect` from `dialect.opName`.
  StringRef opName = op->getName().getIdentifier().strref().split(".").second;
  return StringAttr::get(op->getContext(), "sparse_tensor_" + opName);
}

}  // namespace
namespace mhlo {

template <typename OpTy>
class SparseOpToCustomCallConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    NamedAttribute callTargetName =
        rewriter.getNamedAttr("call_target_name", getOperationTargetName(op));
    rewriter.replaceOpWithNewOp<mhlo::CustomCallOp>(op, op->getResultTypes(),
                                                    adaptor.getOperands(),
                                                    ArrayRef{callTargetName});
    return success();
  }
};

void populateLegalizeSparseOpsToCustomCallPatterns(
    MLIRContext* context, TypeConverter& typeConverter,
    RewritePatternSet* patterns) {
  patterns->add<SparseOpToCustomCallConverter<sparse_tensor::AssembleOp>,
                SparseOpToCustomCallConverter<sparse_tensor::DisassembleOp>,
                SparseOpToCustomCallConverter<sparse_tensor::ConvertOp>>(
      typeConverter, context);
}

}  // namespace mhlo
}  // namespace mlir
