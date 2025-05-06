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
#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_passes.h"

// Required when using LLVM_DEBUG macro.
#define DEBUG_TYPE "tf-mark-functions-noinline"

namespace mlir {
namespace tf_quant {
namespace {

// Name of the boolean attribute indicating whether the function can be
// inlined or not.
constexpr StringRef kTfNoinlineAttr = "tf._noinline";

// This pass marks functions with the attribute `tf._noinline = true` so that
// they aren't inlined by the `InlinerPass`. The names of the functions to be
// marked noinline should be specified by the `noinline-functions` option.
class TFMarkFunctionsNoinlinePass
    : public PassWrapper<TFMarkFunctionsNoinlinePass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TFMarkFunctionsNoinlinePass)

  explicit TFMarkFunctionsNoinlinePass()
      : TFMarkFunctionsNoinlinePass(
            /*noinline_functions=*/ArrayRef<std::string>{}) {}

  // `noinline_functions` is a list of function names to be marked noinline.
  explicit TFMarkFunctionsNoinlinePass(
      const ArrayRef<std::string> noinline_functions)
      : noinline_functions_(CreateNoinlineFunctionsOption(noinline_functions)) {
  }

  TFMarkFunctionsNoinlinePass(const TFMarkFunctionsNoinlinePass& other)
      : TFMarkFunctionsNoinlinePass() {
    noinline_functions_ = other.noinline_functions_;
  }

  StringRef getArgument() const final { return "tf-mark-functions-noinline"; }

  StringRef getDescription() const final {
    return "Marks a function whose name is in `noinline-functions` option with "
           "the attribute `tf._noinline = true`. This attributes the function "
           "from being inlined by the `InlinerPass`.";
  }

  void runOnOperation() override;

 private:
  ListOption<std::string> CreateNoinlineFunctionsOption(
      const ArrayRef<std::string> noinline_functions) {
    return {*this, "noinline-functions",
            llvm::cl::desc(
                "Name of the functions that should be marked "
                "tf._noinline = true to prevent inlining. The name of the "
                "function should exactly match to be marked noinline."),
            llvm::cl::list_init<std::string>(noinline_functions),
            llvm::cl::ZeroOrMore};
  }

  // Gets a set of function names from `noinline_functions_`.
  llvm::StringSet<> GetNoinlineFunctionsSet() {
    llvm::StringSet<> noinline_functions;
    noinline_functions.insert(noinline_functions_.begin(),
                              noinline_functions_.end());
    return noinline_functions;
  }

  // Names of the functions to be marked noinline.
  ListOption<std::string> noinline_functions_;
};

void TFMarkFunctionsNoinlinePass::runOnOperation() {
  const llvm::StringSet<> noinline_functions = GetNoinlineFunctionsSet();

  func::FuncOp func_op = getOperation();
  Builder builder(&getContext());

  // Adds the `tf._noinline = true` attribute to the function if the name
  // matches.
  if (noinline_functions.contains(func_op.getSymName())) {
    func_op->setAttr(kTfNoinlineAttr, builder.getBoolAttr(true));
    LLVM_DEBUG(llvm::dbgs()
               << "Marked tf._noinline = true: " << func_op.getSymName());
  }
}

static PassRegistration<TFMarkFunctionsNoinlinePass> pass{};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateTFMarkFunctionsNoinlinePass(
    const ArrayRef<std::string> noinline_functions) {
  return std::make_unique<TFMarkFunctionsNoinlinePass>(noinline_functions);
}

}  // namespace tf_quant
}  // namespace mlir
