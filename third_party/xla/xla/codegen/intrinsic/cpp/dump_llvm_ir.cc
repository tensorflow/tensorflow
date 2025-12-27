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

#include <iostream>
#include <memory>
#include <string>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_ll.h"

int main(int argc, char** argv) {
  // 1. Get the bitcode data from the generated header
  const std::string& bitcode_view = llvm_ir::kEigenUnaryLlIr;

  // 2. Wrap it in a MemoryBuffer
  std::unique_ptr<llvm::MemoryBuffer> buffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(bitcode_view.data(), bitcode_view.size()),
      "embedded_bitcode",
      /*RequiresNullTerminator=*/false);

  // 3. Parse bitcode into a Module
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);

  if (!module) {
    std::cerr << "Error parsing embedded bitcode: "
              << diagnostic.getMessage().str() << std::endl;
    return 1;
  }

  // 4. Print the module as textual IR to stdout
  module->print(llvm::outs(), nullptr);

  return 0;
}
