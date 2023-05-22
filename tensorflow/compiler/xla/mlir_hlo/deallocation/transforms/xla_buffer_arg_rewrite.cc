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

#include <memory>

#include "deallocation/transforms/passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace deallocation {
namespace {

#define GEN_PASS_DEF_XLABUFFERARGREWRITEPASS
#include "deallocation/transforms/passes.h.inc"

constexpr char kInputMapping[] = "xla_framework.input_mapping";
constexpr char kResultMapping[] = "xla_framework.result_mapping";
constexpr char kResultInnerMapping[] = "xla_framework.result_inner_mapping";

struct XlaBufferArgRewritePass
    : public impl::XlaBufferArgRewritePassBase<XlaBufferArgRewritePass> {
  void runOnOperation() override {
    func::FuncOp op = getOperation();
    if (!op->hasAttr(kResultMapping)) return;

    // Collect result arguments and input arguments.
    auto results = llvm::to_vector(
        llvm::make_filter_range(op.getArguments(), [&](auto arg) {
          return op.getArgAttr(arg.getArgNumber(), kInputMapping) == nullptr;
        }));
    auto args =
        llvm::to_vector(llvm::map_range(op.getArguments(), [&](auto arg) {
          auto buffer = op.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),
                                                         kInputMapping);
          return buffer ? buffer.getInt() : -1;
        }));

    SmallVector<int64_t> resultMapping;
    if (auto innerMapping = op->getAttrOfType<ArrayAttr>(kResultInnerMapping)) {
      resultMapping = llvm::to_vector(llvm::map_range(
          innerMapping.getAsValueRange<IntegerAttr>(),
          [](const APInt& value) { return value.getSExtValue(); }));
    } else if (auto mapping = op->getAttrOfType<IntegerAttr>(kResultMapping)) {
      resultMapping = {mapping.getInt()};
    }

    if (resultMapping.size() != results.size()) {
      op.emitOpError(
          "number of result arguments does not match size of mapping");
      signalPassFailure();
      return;
    }

    for (auto [bufferIndex, result] : llvm::zip(resultMapping, results)) {
      // If the result doesn't alias any argument, add the
      // `deallocation.restrict` attribute to signal to the buffer reuse pass
      // that this buffer is guaranteed not to alias any other argument.
      if (!llvm::is_contained(args, bufferIndex)) {
        op.setArgAttr(result.getArgNumber(), "deallocation.restrict",
                      OpBuilder(op).getBoolAttr(true));
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createXlaBufferArgRewritePass() {
  return std::make_unique<XlaBufferArgRewritePass>();
}

}  // namespace deallocation
}  // namespace mlir
