// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_EXECUTION_METADATA_EXPORTER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_EXECUTION_METADATA_EXPORTER_H_

#include <string>

#include "llvm/ADT/Optional.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project

namespace tflite {

// Returns serialized string for the generated flatbuffer.
llvm::Optional<std::string> ExportRuntimeMetadata(mlir::ModuleOp module);

}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_EXECUTION_METADATA_EXPORTER_H_
