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

// LLVM-based compiler backend.
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_LLVM_GPU_BACKEND_GPU_BACKEND_LIB_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_LLVM_GPU_BACKEND_GPU_BACKEND_LIB_H_

#include <string>
#include <utility>

#include "llvm/IR/Module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace xla {
namespace gpu {

// Compiles the argument module and returns it. libdevice_dir_path is the parent
// directory of the libdevice bitcode libraries. The contents of the module may
// be changed.
//
// The Compile.* interfaces each create their own llvm::LLVMContext objects for
// thread safety, but note that LLVM's multithreaded support is very
// preliminary; multithreaded use is not recommended at this time.
StatusOr<string> CompileToPtx(llvm::Module* module,
                              std::pair<int, int> compute_capability,
                              const HloModuleConfig& hlo_module_config,
                              const string& libdevice_dir_path);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_LLVM_GPU_BACKEND_GPU_BACKEND_LIB_H_
