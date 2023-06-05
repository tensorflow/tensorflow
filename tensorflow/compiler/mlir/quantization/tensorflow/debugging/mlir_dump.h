/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_DEBUGGING_MLIR_DUMP_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_DEBUGGING_MLIR_DUMP_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project

namespace tensorflow {
namespace quantization {

// Enables IR printing for `pm`. When the passes are run, the IRs will be dumped
// to `out_stream`.
void EnableIrPrinting(llvm::raw_ostream &out_stream, mlir::PassManager &pm);

// If verbosity level >= 1, this will dump intermediate IRs of passes to a file.
// The file path is given by prefixing `name`.mlir with the value of the
// TF_QUANT_MLIR_DUMP_PREFIX env variable. Returns `nullptr` iff the verbosity
// level < 1 or TF_QUANT_MLIR_DUMP_PREFIX is not set or set to an empty string.
// The returned ostream instance should live until the pass run is complete.
absl::StatusOr<std::unique_ptr<llvm::raw_ostream>> MaybeEnableIrPrinting(
    mlir::PassManager &pm, const absl::string_view name);

}  // namespace quantization
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_DEBUGGING_MLIR_DUMP_H_
