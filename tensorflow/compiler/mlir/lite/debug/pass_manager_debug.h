/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_DEBUG_PASS_MANAGER_DEBUG_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_DEBUG_PASS_MANAGER_DEBUG_H_

#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/debug/debug_options.pb.h"

namespace mlir::TFL {

// Sets up `pm` for debugging as requested by `options`: IR dumps to
// `ir_dump_dir` (with a crash reproducer), IR printing to `out`, RSS logging
// and pass timing. Also adds the error collector instrumentation.
//
// Same behavior as `tensorflow::InitPassManager` in debug.h, except that the
// dump directory must be on the local file system and the regexes use
// `llvm::Regex` (POSIX extended) syntax instead of RE2. Used by the StableHLO
// to TFLite converter so that it does not depend on TSL.
//
// The `out` parameter is exposed for testing purposes and not intended to be
// specified by client code.
void InitPassManager(mlir::PassManager& pm,
                     const tensorflow::converter::DebugOptions& options,
                     llvm::raw_ostream& out = llvm::outs());

}  // namespace mlir::TFL

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_DEBUG_PASS_MANAGER_DEBUG_H_
