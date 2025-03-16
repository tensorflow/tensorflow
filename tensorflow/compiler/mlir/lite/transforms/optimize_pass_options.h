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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZE_PASS_OPTIONS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZE_PASS_OPTIONS_H_

#include "llvm/Support/CommandLine.h"
#include "mlir/Pass/PassOptions.h"  // from @llvm-project

namespace mlir {
namespace TFL {

////////////////////////////////////////////////////////////////////////////////
// Pass Options
////////////////////////////////////////////////////////////////////////////////

struct OptimizePassOptions : public mlir::detail::PassOptions {
  mlir::detail::PassOptions::Option<bool> enable_canonicalization{
      *this, "enable-canonicalization",
      llvm::cl::desc("Enable canonicalization in the optimize pass"),
      llvm::cl::init(true)};
  mlir::detail::PassOptions::Option<bool> disable_fuse_mul_and_fc{
      *this, "disable-fuse-mul-and-fc",
      llvm::cl::desc("Disable fuse mul and fc in the optimize pass"),
      llvm::cl::init(false)};
  mlir::detail::PassOptions::Option<bool> enable_strict_qdq_mode{
      *this, "enable-strict-qdq-mode",
      llvm::cl::desc("Enable strict QDQ mode in the optimize pass"),
      llvm::cl::init(false)};
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZE_PASS_OPTIONS_H_
