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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_TEST_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_TEST_UTILS_H_

#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace mlir {
namespace mhlo {
namespace test {

// Given a raw string, return a ModuleOp that can be used with the given
// MLIRContext.
tsl::StatusOr<OwningOpRef<ModuleOp>> GetMlirModuleFromString(
    absl::string_view module_string, MLIRContext* mlir_context);

}  // namespace test
}  // namespace mhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_TEST_UTILS_H_
