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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_VHLO_OPERATORS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_VHLO_OPERATORS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"

namespace mlir::TFL::fb_export {

// What serializing an operator needs from the model being built.
class ModelContext {
 public:
  virtual ~ModelContext() = default;

  virtual flatbuffers::FlatBufferBuilder& builder() = 0;

  // Returns the index of the operator code for `op` (and `custom_code`, for
  // custom ops), adding it if needed.
  virtual uint32_t GetOpcodeIndex(tflite::BuiltinOperator op,
                                  llvm::StringRef custom_code = "") = 0;

  // Returns the index of the subgraph of function `name`.
  virtual std::optional<int32_t> GetFunctionSubgraph(llvm::StringRef name) = 0;

  // Builds a subgraph from `region`, an operator body, and returns its index.
  virtual std::optional<int32_t> BuildRegionSubgraph(
      Region& region, tflite::BuiltinOperator op) = 0;
};

// Serializes a VHLO op as a TFLite STABLEHLO_* operator. Supports the ops the
// TFLite runtime has kernels for, element-wise ops without options and custom
// calls.
// Emits an error and returns std::nullopt for other ops.
std::optional<flatbuffers::Offset<tflite::Operator>> BuildVhloOperator(
    Operation* op, const std::vector<int32_t>& inputs,
    const std::vector<int32_t>& outputs, ModelContext& context);

}  // namespace mlir::TFL::fb_export

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_VHLO_OPERATORS_H_
