/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_MATH_MATH_COMPILER_LIB_H_
#define XLA_CODEGEN_MATH_MATH_COMPILER_LIB_H_

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Module.h"

namespace xla::codegen::math {

// Runs passes that integrate the injected xla.* math functions.
// Should inline, CSE, DCE, and CP and remove unused functions.
void RunInlineAndOptPasses(llvm::Module& module);

// Removes the specified functions from the llvm.compiler.used array.
// If all functions are removed, the array is removed entirely.
// If only some functions are removed, the array is replaced with a new one
// containing the remaining functions.
// This is necessary to allow unused vectorized xla.* math functions to be
// removed by GlobalDCEPass.
void RemoveFromCompilerUsed(
    llvm::Module& module,
    absl::flat_hash_set<absl::string_view> replaced_functions);

}  // namespace xla::codegen::math

#endif  // XLA_CODEGEN_MATH_MATH_COMPILER_LIB_H_
