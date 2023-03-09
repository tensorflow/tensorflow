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

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/tsl/platform/logging.h"

namespace {

static void DieWithSMDiagnosticError(llvm::SMDiagnostic* diagnostic) {
  LOG(FATAL) << diagnostic->getFilename().str() << ":"
             << diagnostic->getLineNo() << ":" << diagnostic->getColumnNo()
             << ": " << diagnostic->getMessage().str();
}

}  // namespace

namespace xla {
namespace gpu {

std::unique_ptr<llvm::Module> LoadIRModule(const std::string& filename,
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

std::string ReplaceFilenameExtension(absl::string_view filename,
                                     absl::string_view new_extension) {
  auto pos = filename.rfind('.');
  absl::string_view stem = pos == absl::string_view::npos
                               ? filename
                               : absl::string_view(filename.data(), pos);
  return absl::StrCat(stem, ".", new_extension);
}

}  // namespace gpu
}  // namespace xla
