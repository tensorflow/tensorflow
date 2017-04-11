/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/utils.h"

#include "tensorflow/core/platform/logging.h"

#include "external/llvm/include/llvm/IR/LLVMContext.h"
#include "external/llvm/include/llvm/IR/Module.h"
#include "external/llvm/include/llvm/IRReader/IRReader.h"
#include "external/llvm/include/llvm/Support/SourceMgr.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace {

static void DieWithSMDiagnosticError(llvm::SMDiagnostic* diagnostic) {
  LOG(FATAL) << diagnostic->getLineNo() << ":" << diagnostic->getColumnNo()
             << ": " << diagnostic->getMessage().str();
}

}  // namespace

namespace xla {
namespace gpu {

std::unique_ptr<llvm::Module> LoadIRModule(const string& filename,
                                           llvm::LLVMContext* llvm_context) {
  llvm::SMDiagnostic diagnostic_err;
  std::unique_ptr<llvm::Module> module(
      llvm::parseIRFile(llvm::StringRef(filename.data(), filename.size()),
                        diagnostic_err, *llvm_context));

  if (module == nullptr) {
    DieWithSMDiagnosticError(&diagnostic_err);
  }

  return module;
}

string ReplaceFilenameExtension(tensorflow::StringPiece filename,
                                tensorflow::StringPiece new_extension) {
  auto pos = filename.rfind('.');
  tensorflow::StringPiece stem =
      pos == tensorflow::StringPiece::npos
          ? filename
          : tensorflow::StringPiece(filename.data(), pos);
  return tensorflow::strings::StrCat(stem, ".", new_extension);
}

}  // namespace gpu
}  // namespace xla
