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

#include "tensorflow/compiler/mlir/tensorflow/utils/mlprogram_util.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/mlprogram.h"

namespace tensorflow {

void RegisterMlProgramPasses() {
  mlir::registerPassPipeline(
      "tf-lower-to-mlprogram-and-hlo", "Lower TF to ml_program + mhlo",
      [](mlir::OpPassManager& pm, llvm::StringRef options,
         llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)>
             errorHandler) {
        tensorflow::PopulateLowerToMlProgramAndHloPipeline(pm);
        return mlir::success();
      },
      [](llvm::function_ref<void(const mlir::detail::PassOptions&)>) {});
}

}  // namespace tensorflow
