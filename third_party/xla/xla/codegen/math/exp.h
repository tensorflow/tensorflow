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

#ifndef XLA_CODEGEN_MATH_EXP_H_
#define XLA_CODEGEN_MATH_EXP_H_

#include <cstddef>
#include <string>

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

namespace xla::codegen::math {

llvm::Function* CreateExpF64(llvm::Module* module, llvm::Type* input_type);
std::string ExpF64FunctionName(size_t num_elements);

}  // namespace xla::codegen::math

#endif  // XLA_CODEGEN_MATH_EXP_H_
