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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_EXPORT_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_EXPORT_H_

#include "absl/status/status.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project

namespace mlir::TFL {

// Writes `module` as a TFLite flatbuffer model to `stream`.
//
// This is the exporter of the StableHLO to TFLite pipeline (models from JAX
// and PyTorch). The module holds TFL ops and the VHLO ops that weren't
// legalized to TFL. Every function becomes a subgraph; functions with a
// `tf.entry_function` attribute come first and become signatures.
//
// Constant data and large custom options are written after the flatbuffer and
// referenced by offset, which supports models larger than 2GB. Identical
// constants are stored once. Resource blobs of constants are released once
// written, so on success `module` must not be used afterwards. Invalid
// constants are detected before anything is released, so on such errors
// `module` is intact for error reporting.
//
// On failure, errors are also emitted as MLIR diagnostics on the ops.
absl::Status ExportFlatbuffer(ModuleOp module, llvm::raw_pwrite_stream& stream);

}  // namespace mlir::TFL

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_EXPORT_H_
