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

#include "tensorflow/compiler/mlir/tensorflow/utils/stablehlo_custom_call.h"

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace TF {
namespace {

// jax2tf sets `stablehlo.custom_call`'s target name as `tf.call_tf_function`
// to represent calling a TF host callback function.
constexpr llvm::StringRef kTfTargetName = "tf.call_tf_function";

// `tf.backend_config` is a DictionaryAttr, JAX2TF sets the value of its
// string attribute `caller_name` to the TF host callback function's name.
constexpr llvm::StringRef kTfBackendConfigAttrName = "tf.backend_config";
constexpr llvm::StringRef kCalledFuncAttrName = "called_func";

}  // namespace

bool IsTfFuncCustomCall(stablehlo::CustomCallOp op) {
  return op.getCallTargetName() == kTfTargetName;
}

DictionaryAttr GetTfBackendConfig(stablehlo::CustomCallOp op) {
  return op->getAttrOfType<DictionaryAttr>(kTfBackendConfigAttrName);
}

FailureOr<SymbolRefAttr> GetTfFuncCustomCallFuncName(
    stablehlo::CustomCallOp op) {
  if (!IsTfFuncCustomCall(op)) {
    return success(nullptr);
  }

  auto config = GetTfBackendConfig(op);
  if (config == nullptr) {
    op.emitOpError() << "does not have dictionary attribute '"
                     << kTfBackendConfigAttrName << "'";
    return failure();
  }

  auto f = config.get(kCalledFuncAttrName);
  if (f == nullptr) {
    op.emitOpError() << "does not have attribute '" << kCalledFuncAttrName
                     << "' in its dictionary attribute '"
                     << kTfBackendConfigAttrName << "'";
    return failure();
  }

  if (auto attr = mlir::dyn_cast<FlatSymbolRefAttr>(f)) {
    return attr;
  }

  op.emitOpError() << "'s attribute '" << kCalledFuncAttrName
                   << "' is neither StringAttr nor FlatSymbolRefAttr";
  return failure();
}

}  // namespace TF
}  // namespace mlir
