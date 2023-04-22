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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_LLVM_GPU_BACKEND_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_LLVM_GPU_BACKEND_UTILS_H_

#include <memory>
#include <string>
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/types.h"

namespace llvm {
class LLVMContext;
class Module;
}

namespace xla {
namespace gpu {

// Convenience function for loading a LLVM module from an IR file. The module
// is created in the given LLVM context.
//
// If loading fails for some reason, dies printing a diagnostic error.
std::unique_ptr<llvm::Module> LoadIRModule(const string& filename,
                                           llvm::LLVMContext* llvm_context);

// Convenience function for replacing the extension of the given filename.
// If the filename has no extension, the new extension is appended to its name.
//
// For example:
//   ReplaceFilenameExtension("/foo/baz.txt", "cc") --> "/foo/baz.cc"
string ReplaceFilenameExtension(absl::string_view filename,
                                absl::string_view new_extension);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_LLVM_GPU_BACKEND_UTILS_H_
