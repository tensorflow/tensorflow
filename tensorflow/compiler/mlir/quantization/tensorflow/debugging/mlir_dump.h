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

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project

namespace tensorflow {
namespace quantization {

// Enables IR printing for `pm`. When the passes are run, each pass will dump to
// its own file with prefix `file_name_prefix`.
void EnableIrPrinting(mlir::PassManager &pm,
                      absl::string_view file_name_prefix);

// If verbosity level >= 1, this will dump intermediate IRs of passes to a file.
// The dumped mlir files with be under a directory determined by
// the TF_QUANT_MLIR_DUMP_PREFIX env variable. The PassManager will dump to a
// new file for each pass. The file name will have the format
// {file_name_prefix}_{pass_number}_{pass_name}_{before|after}.mlir.
// * `file_name_prefix` is from input.
// * `pass_number` increments from 1 for each pass.
// * `pass_name` is the name of the pass.
// * `before|after` indicates whether the dump occurs before or after the pass.
absl::Status MaybeEnableIrPrinting(mlir::PassManager &pm,
                                   absl::string_view file_name_prefix);

}  // namespace quantization
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_DEBUGGING_MLIR_DUMP_H_
