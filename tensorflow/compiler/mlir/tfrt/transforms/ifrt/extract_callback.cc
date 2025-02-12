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

#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/extract_callback.h"

#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h.inc"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h.inc"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/visitor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace ifrt_serving {

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ExtractCallbackModule(
    mlir::ModuleOp module, absl::string_view callback_key) {
  // Find the entry function name first.
  mlir::func::FuncOp callback_entry_func;
  module.walk([&](mlir::func::FuncOp func) {
    if (func.getSymName().str() == callback_key) {
      callback_entry_func = func;
      return mlir::WalkResult::skip();
    }
    return mlir::WalkResult::advance();
  });

  if (!callback_entry_func) {
    return absl::NotFoundError(
        absl::StrCat("Callback key ", callback_key, " not found"));
  }

  mlir::StatusScopedDiagnosticHandler diag_handler(module->getContext());
  auto entry_function_name = callback_entry_func.getSymName();
  auto submodule = mlir::TF::CreatePrunedModule(module, entry_function_name);
  if (mlir::failed(submodule)) {
    return diag_handler.ConsumeStatus();
  }

  // Remove the attribute inherited from saved model loading. They impose
  // additional constraint on public functions that are not necessary.
  submodule->get()->removeAttr("tf_saved_model.semantics");
  submodule->get().walk([&](mlir::func::FuncOp func) {
    if (func.getSymName() == entry_function_name) {
      func.setPublic();
    }
  });
  return std::move(*submodule);
}

}  // namespace ifrt_serving
}  // namespace tensorflow
