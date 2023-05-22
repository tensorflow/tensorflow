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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_FLATBUFFER_OPERATOR_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_FLATBUFFER_OPERATOR_H_

#include <stdint.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumeBundleQueries.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mlir {

// Returns the MLIR op name for the flatbuffer operator corresponding to
// `op_code`.
std::string GetMlirOpNameFromOpCode(const ::tflite::OperatorCodeT &op_code);

// Returns the builtin op code for the given MLIR operation on success; emits
// error and returns std::nullopt on failure.
std::optional<tflite::BuiltinOperator> GetBuiltinOpCode(Operation *mlir_op);

// Packs the given MLIR operation into a TFLite FlatBuffer operator object.
// Returns the FlatBuffer offset for the operator on success; emits error and
// returns std::nullopt on failure.
std::optional<flatbuffers::Offset<tflite::Operator>> CreateFlatBufferOperator(
    Operation *mlir_op, uint32_t opcode_index,
    const std::vector<int32_t> &operands, const std::vector<int32_t> &results,
    const std::vector<int32_t> &intermediates,
    flatbuffers::FlatBufferBuilder *fbb);

// Populates the array of mlir::NamedAttributes corresponding to the given
// tflite::FlatbufferOptionsUnion.
// We use an out parameter per LLVM convention
void BuiltinOptionsToAttributes(
    tflite::BuiltinOptionsUnion op_union, mlir::Builder builder,
    // NOLINTNEXTLINE
    llvm::SmallVectorImpl<mlir::NamedAttribute> &attributes);

// While the last several tensors could be optional tensors for an tfl op, the
// number of input operands could vary. This function gets the min/max number of
// operands from tflite op name.
llvm::MinMax OperandNumbersMinMax(llvm::StringRef op_name);

// Populates the `custom_code` and `custom_options` to attributes.
// `custom_code` is used to identify CustomOp.
// `custom_options` are opaque attribute used to store infomations for this
// custom op.
tensorflow::Status CustomOptionsToAttributes(
    const std::string &custom_code, const std::vector<uint8_t> &custom_options,
    mlir::Builder builder,
    // NOLINTNEXTLINE
    Location loc, llvm::SmallVectorImpl<mlir::NamedAttribute> *attributes);

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_FLATBUFFER_OPERATOR_H_
