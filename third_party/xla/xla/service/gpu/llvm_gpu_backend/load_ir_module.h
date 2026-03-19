/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_LLVM_GPU_BACKEND_LOAD_IR_MODULE_H_
#define XLA_SERVICE_GPU_LLVM_GPU_BACKEND_LOAD_IR_MODULE_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"

namespace llvm {
class LLVMContext;
class Module;
}  // namespace llvm

namespace xla::gpu {

// Convenience function for loading a LLVM module from an IR file. The module
// is created in the given LLVM context.
//
// If loading fails for some reason, dies printing a diagnostic error.
std::unique_ptr<llvm::Module> LoadIRModule(const std::string& filename,
                                           llvm::LLVMContext* llvm_context);

// Convenience function for replacing the extension of the given filename.
// If the filename has no extension, the new extension is appended to its name.
//
// For example:
//   ReplaceFilenameExtension("/foo/baz.txt", "cc") --> "/foo/baz.cc"
std::string ReplaceFilenameExtension(absl::string_view filename,
                                     absl::string_view new_extension);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_LLVM_GPU_BACKEND_LOAD_IR_MODULE_H_
