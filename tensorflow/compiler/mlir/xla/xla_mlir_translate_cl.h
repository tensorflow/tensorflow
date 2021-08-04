/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_XLA_MLIR_TRANSLATE_CL_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_XLA_MLIR_TRANSLATE_CL_H_

#include "llvm/Support/CommandLine.h"

// This file contains command-line options aimed to provide the parameters
// required by the MLIR module to XLA HLO conversion. It is only intended to be
// included by binaries.

extern llvm::cl::opt<bool> emit_use_tuple_arg;
extern llvm::cl::opt<bool> emit_return_tuple;
extern llvm::cl::opt<bool> optimize_xla_hlo;
extern llvm::cl::opt<bool> prefer_tf2xla;

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_XLA_MLIR_TRANSLATE_CL_H_
