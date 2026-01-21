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

#include <memory>
#include <string>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_ll.h"

int main(int argc, char** argv) {
  const std::string& bitcode_view = llvm_ir::kEigenUnaryLlIr;

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = xla::codegen::ParseEmbeddedBitcode(
      context, bitcode_view, "embedded_bitcode");

  module->print(llvm::outs(), nullptr);

  return 0;
}
