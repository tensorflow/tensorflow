/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_LARGE_CONSTANT_FOLD_PASS_OPTIONS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_LARGE_CONSTANT_FOLD_PASS_OPTIONS_H_

#include "llvm/Support/CommandLine.h"
#include "mlir/Pass/PassOptions.h"  // from @llvm-project

namespace mlir::TFL {

struct LargeConstantFoldPassOptions : public mlir::detail::PassOptions {
  Option<bool> fold_fp16_resource_casts{
      *this, "fold-fp16-resource-casts",
      llvm::cl::desc(
          "Whether to fold 16-bit float (fp16/bf16) resource casts."),
      llvm::cl::init(false)};
  Option<bool> fold_elementwise_ops{
      *this, "fold-elementwise-ops",
      llvm::cl::desc("Whether to fold elementwise operations on resources."),
      llvm::cl::init(false)};
};

}  // namespace mlir::TFL

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_LARGE_CONSTANT_FOLD_PASS_OPTIONS_H_
