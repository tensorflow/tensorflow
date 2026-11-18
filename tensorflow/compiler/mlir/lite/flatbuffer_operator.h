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

// Conversions of TFLite operators between MLIR and FlatBuffer. The writers
// (MLIR -> FlatBuffer) are in flatbuffer_operator_writers.h, which this header
// includes; export-only users can depend on that header alone
// (:flatbuffer_tflite_operator_writers). This header adds the readers
// (FlatBuffer operator options -> MLIR attributes) used by the importer.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_FLATBUFFER_OPERATOR_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_FLATBUFFER_OPERATOR_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/flatbuffer_operator_writers.h"  // IWYU pragma: export
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"

namespace mlir {

// Populates the array of mlir::NamedAttributes corresponding to the given
// tflite::FlatbufferOptionsUnion.
// We use an out parameter per LLVM convention
void BuiltinOptionsToAttributes(
    tflite::BuiltinOptionsUnion op_union, mlir::Builder builder,
    // NOLINTNEXTLINE
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes);

// Populates the `custom_code` and `custom_options` to attributes.
// `custom_code` is used to identify CustomOp.
// `custom_options` are opaque attribute used to store information for this
// custom op.
absl::Status CustomOptionsToAttributes(
    const std::string& custom_code, llvm::ArrayRef<uint8_t> custom_options,
    mlir::Builder builder,
    // NOLINTNEXTLINE
    Location loc, llvm::SmallVectorImpl<mlir::NamedAttribute>* attributes);

// TODO(zichuanwei@): Populate Builtin_options_2 manual for now, should automate
// these in the future
void BuiltinOptions2ToAttributes(
    tflite::BuiltinOptions2Union op_union, mlir::Builder builder,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes);

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_FLATBUFFER_OPERATOR_H_
