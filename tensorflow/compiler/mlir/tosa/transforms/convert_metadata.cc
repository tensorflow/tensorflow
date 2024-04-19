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
#include <string>

#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir::tosa {

#define GEN_PASS_DEF_CONVERTFUNCTIONMETADATA
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

namespace {

// Extract the input and output names
static void splitFunctionIONames(StringAttr namesAttr,
                                 llvm::SmallVectorImpl<std::string> &names) {
  SmallVector<StringRef, 4> namesRef;
  llvm::SplitString(namesAttr.getValue(), namesRef, ",");
  for (auto nameRef : namesRef) {
    names.push_back(nameRef.str());
  }
}

class ConvertFunctionMetadataPass
    : public impl::ConvertFunctionMetadataBase<ConvertFunctionMetadataPass> {
 public:
  void runOnOperation() override {
    auto funcOp = getOperation();

    // Setup entry functions for compilation and preserve the
    // associated metadata. Note that TFLite uses `tf.entry_function`.
    auto entryFunctionAttr =
        funcOp->getAttrOfType<DictionaryAttr>("tf.entry_function");
    if (entryFunctionAttr) {
      setupEntryPointAttrs(funcOp, entryFunctionAttr);
    }
  }

 private:
  // TF/TFL pack their I/O names in a dictionary, convert into arg attributes.
  void setupEntryPointAttrs(func::FuncOp funcOp,
                            DictionaryAttr entryFunctionAttr) {
    funcOp.setPublic();

    if (funcOp.getNumArguments() > 0) {
      auto inputsAttr =
          dyn_cast_or_null<StringAttr>(entryFunctionAttr.get("inputs"));
      if (!inputsAttr) {
        funcOp.emitError() << "functions with tf.entry_function must have "
                              "input names to be handled by backend";
        return signalPassFailure();
      }
      SmallVector<std::string, 4> inputNames;
      splitFunctionIONames(inputsAttr, inputNames);
      if (inputNames.size() != funcOp.getNumArguments()) {
        funcOp.emitError()
            << "tf.entry_function attribute malformed: inputs don't "
               "match the function signature";
        return signalPassFailure();
      }
      for (auto [i, name] : llvm::enumerate(inputNames)) {
        funcOp.setArgAttr(i, "ml_program.identifier",
                          StringAttr::get(&getContext(), name));
      }
    }
    if (funcOp.getNumResults() > 0) {
      auto outputsAttr =
          dyn_cast_or_null<StringAttr>(entryFunctionAttr.get("outputs"));
      if (!outputsAttr) {
        funcOp.emitError() << "functions with tf.entry_function must have "
                              "output names to be handled by backend";
        return signalPassFailure();
      }
      SmallVector<std::string, 4> outputNames;
      splitFunctionIONames(outputsAttr, outputNames);
      if (outputNames.size() != funcOp.getNumResults()) {
        funcOp.emitError()
            << "tf.entry_function attribute malformed: outputs don't "
               "match the function signature";
        return signalPassFailure();
      }
      for (auto [i, name] : llvm::enumerate(outputNames)) {
        funcOp.setResultAttr(i, "ml_program.identifier",
                             StringAttr::get(&getContext(), name));
      }
    }
  }
};
}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertFunctionMetadataPass() {
  return std::make_unique<ConvertFunctionMetadataPass>();
}

}  // namespace mlir::tosa
