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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_STABLEHLO_CUSTOM_CALL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_STABLEHLO_CUSTOM_CALL_H_

#include "llvm/ADT/StringRef.h"
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

namespace mlir {
namespace TF {

// Returns whether the op calls a TF callback function.
bool IsTfHostCallback(stablehlo::CustomCallOp op);

// Returns the `caller_name` string attribute in the `tf.backend_config`
// dictionary attribute.
//
// If the op does not call TF host callback, returns nullptr.
// Otherwise, if the op does not have `caller_name`, returns failure.
FailureOr<StringAttr> GetTfHostCallbackName(stablehlo::CustomCallOp op);

// Sets the `caller_name` string attribute to `name` in the `tf.backend_config`
// dictionary attribute.
// Creates `tf.backend_config` attribute if it does not already exist.
void SetTfHostCallbackName(stablehlo::CustomCallOp op, FlatSymbolRefAttr f);

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_STABLEHLO_CUSTOM_CALL_H_
