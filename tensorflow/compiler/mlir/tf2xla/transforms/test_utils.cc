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

#include "tensorflow/compiler/mlir/tf2xla/transforms/test_utils.h"

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/register_common_dialects.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "xla/tsl/platform/statusor.h"

namespace mlir {
namespace hlo {
namespace test {

using ::mlir::DialectRegistry;
using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OwningOpRef;
using ::tsl::StatusOr;

absl::StatusOr<OwningOpRef<ModuleOp>> GetMlirModuleFromString(
    absl::string_view module_string, MLIRContext* context) {
  DialectRegistry mlir_registry;
  RegisterCommonToolingDialects(mlir_registry);
  context->appendDialectRegistry(mlir_registry);

  OwningOpRef<ModuleOp> mlir_module;
  auto status =
      tensorflow::DeserializeMlirModule(module_string, context, &mlir_module);
  if (!status.ok()) {
    return status;
  }
  return mlir_module;
}

}  // namespace test
}  // namespace hlo
}  // namespace mlir
