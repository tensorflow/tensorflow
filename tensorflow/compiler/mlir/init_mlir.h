/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_INIT_MLIR_H_
#define TENSORFLOW_COMPILER_MLIR_INIT_MLIR_H_

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

namespace tensorflow {

// Initializer to perform both InitLLVM and TF's InitMain initialization.
// InitMain also performs flag parsing and '--' is used to separate flags passed
// to it: Flags before the first '--' are parsed by InitMain and argc and argv
// progressed to the flags post. If there is no separator, then no flags are
// parsed by InitMain and argc/argv left unadjusted.
class InitMlir {
 public:
  InitMlir(int *argc, char ***argv);

 private:
  llvm::InitLLVM init_llvm_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_INIT_MLIR_H_
