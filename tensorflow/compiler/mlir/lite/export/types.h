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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_TYPES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_TYPES_H_

#include "absl/status/statusor.h"
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"

namespace mlir::TFL::fb_export {

// Returns the TFLite tensor type of an MLIR element type. Quantized types map
// to their storage type.
absl::StatusOr<tflite::TensorType> GetTensorType(Type element_type);

}  // namespace mlir::TFL::fb_export

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_TYPES_H_
