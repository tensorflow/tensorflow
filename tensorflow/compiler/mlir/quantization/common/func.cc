/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/common/func.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"

namespace mlir::quant {
namespace {

using ::tensorflow::kDefaultServingSignatureDefKey;
using ::tensorflow::kImportModelDefaultGraphFuncName;

// Returns true iff the function's symbol is public.
bool IsPublicFuncOp(func::FuncOp func_op) {
  return SymbolTable::getSymbolVisibility(&*func_op) ==
         SymbolTable::Visibility::Public;
}

}  // namespace

func::FuncOp FindMainFuncOp(ModuleOp module_op) {
  if (const auto main_func_op = module_op.lookupSymbol<func::FuncOp>(
          kImportModelDefaultGraphFuncName);
      main_func_op != nullptr && IsPublicFuncOp(main_func_op)) {
    return main_func_op;
  }

  if (const auto serving_default_func_op =
          module_op.lookupSymbol<func::FuncOp>(kDefaultServingSignatureDefKey);
      serving_default_func_op != nullptr &&
      IsPublicFuncOp(serving_default_func_op)) {
    return serving_default_func_op;
  }

  return nullptr;
}

}  // namespace mlir::quant
