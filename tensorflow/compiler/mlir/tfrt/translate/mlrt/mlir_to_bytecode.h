/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSLATE_MLRT_MLIR_TO_BYTECODE_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSLATE_MLRT_MLIR_TO_BYTECODE_H_

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"

namespace mlrt {

class ModuleEmitterContext;

// Defines a custom attribute encoding registry. Users can register custom
// attribute encoding for their dialects in this registry. If no custom encoder
// is registered for a dialect, the default encoding with a limited support, the
// EncodeSimpleAttribute() below, will be used.
class AttributeEncoderRegistry {
 public:
  using EncoderFn = std::function<absl::StatusOr<std::string>(
      const ModuleEmitterContext&, mlir::Attribute)>;

  void Register(absl::string_view dialect, EncoderFn encoder) {
    encoders_[dialect] = std::move(encoder);
  }

  // Returns the encoder for the specified dialect. It can be nullptr if it is
  // not registered for this dialect. The returned reference will be invalidated
  // if Register() is called.
  const EncoderFn* Get(absl::string_view dialect) const {
    auto iter = encoders_.find(dialect);
    if (iter != encoders_.end()) return &iter->second;
    return nullptr;
  }

 private:
  absl::flat_hash_map<std::string, EncoderFn> encoders_;
};

class ModuleEmitterContext {
 public:
  explicit ModuleEmitterContext(
      const AttributeEncoderRegistry* attribute_encoder_registry)
      : attribute_encoder_registry_(*attribute_encoder_registry) {}

  void AddKernelName(std::string name) {
    AddData(std::move(name), kernels_, kernel_id_map_);
  }

  int GetKernelId(llvm::StringRef name) const {
    return kernel_id_map_.at(name);
  }

  absl::Status AddAttribute(mlir::Operation* op, mlir::Attribute attr);

  int GetAttributeId(mlir::Attribute attr) const {
    return attribute_id_map_.lookup(attr);
  }

  int AddFunction(mlir::func::FuncOp func);

  int GetFunctionId(absl::string_view name) const {
    return function_name_id_map_.at(name);
  }

  absl::Span<const std::string> kernels() const { return kernels_; }
  absl::Span<const std::string> attributes() const { return attributes_; }
  absl::Span<const mlir::func::FuncOp> functions() const { return functions_; }

 private:
  int AddData(std::string data, std::vector<std::string>& data_vector,
              absl::flat_hash_map<std::string, int>& data_map) {
    auto iter = data_map.find(data);
    if (iter != data_map.end()) return iter->second;

    int id = data_vector.size();
    data_map[data] = id;
    data_vector.push_back(std::move(data));
    return id;
  }

  absl::StatusOr<std::string> DefaultEncodeAttribute(mlir::Attribute attr);

  const AttributeEncoderRegistry& attribute_encoder_registry_;

  std::vector<std::string> kernels_;
  absl::flat_hash_map<std::string, int> kernel_id_map_;

  std::vector<std::string> attributes_;
  llvm::DenseMap<mlir::Attribute, int> attribute_id_map_;
  absl::flat_hash_map<std::string, int> attribute_data_id_map_;

  std::vector<mlir::func::FuncOp> functions_;
  absl::flat_hash_map<std::string, int> function_name_id_map_;
};

// Encodes a few simple attributes. Users can use this function in their custom
// attribute encoder.
std::optional<std::string> EncodeSimpleAttribute(
    const ModuleEmitterContext& module_context, mlir::Attribute attr);

absl::StatusOr<bc::Buffer> EmitExecutable(
    const AttributeEncoderRegistry& attribute_encoder_registry,
    mlir::ModuleOp module);

}  // namespace mlrt

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSLATE_MLRT_MLIR_TO_BYTECODE_H_
