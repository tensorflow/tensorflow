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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_CONSTANTS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_CONSTANTS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace quant {

// Name of the save function. The "tf_quant__" prefix is for avoiding conflict
// with existing function's name.
inline constexpr StringRef kTfQuantSaveFuncName = "tf_quant__save";

// Name of the TensorFlow Operation to be fetched to save the variables to
// checkpoint. This save op follows the SavedModel's load semantics, so it
// should return the file prefix of the checkpoint as a string tensor.
inline constexpr StringRef kTfQuantSaveOpName = "tf_quant__save_op";

// Name the file prefix string tensor. The tensor is used to identify the prefix
// to the checkpoint where the variables are saved / loaded. This may be present
// in a function argument's "tf_saved_model.index_path" attribute to identify
// the file prefix function argument.
inline constexpr StringRef kTfFilePrefix = "__tf_file_prefix";

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_CONSTANTS_H_
