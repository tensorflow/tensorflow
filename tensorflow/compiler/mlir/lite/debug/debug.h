/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_DEBUG_DEBUG_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_DEBUG_DEBUG_H_

#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/debug/debug_options.pb.h"

namespace tensorflow {

// Initializes the pass manager with default options that make debugging easier.
// The `out` method parameter is exposed for testing purposes and not intended
// to be specified by client code.
void InitPassManager(mlir::PassManager& pm,
                     const converter::DebugOptions& options,
                     llvm::raw_ostream& out = llvm::outs());

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_DEBUG_DEBUG_H_
