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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_SERIALIZER_FLATBUFFER_TRANSLATOR_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_SERIALIZER_FLATBUFFER_TRANSLATOR_H_

#include <stddef.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/Optional.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/stablehlo/schema/schema_generated.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"

template <typename T>
using BufferOffset = flatbuffers::Offset<T>;

template <typename T>
using VectorBufferOffset = flatbuffers::Offset<flatbuffers::Vector<T>>;

using CustomOptionsOffset = VectorBufferOffset<uint8_t>;

// Use initial buffer size in flatbuffer builder to be same as the initial size
// used by the TOCO export. (It does not explain rationale for this choice.)
// This number is currently inherited from Tflite
constexpr size_t kInitialBufferSize = 10240;

namespace mlir {
namespace odml {

// Translates an MLIR module in mhlo dialect to TFLite FlatBuffer.
class Translator {
 public:
  // Translates the given MLIR module into TFLite FlatBuffer format and returns
  // the serialized output. Returns std::nullopt on unsupported, invalid inputs
  // or internal error.
  static llvm::Optional<std::string> Translate(
      ModuleOp module, const toco::TocoFlags& toco_flags,
      const std::unordered_set<std::string>& tags,
      tensorflow::OpOrArgNameMapper* op_or_arg_name_mapper,
      const std::map<std::string, std::string>& metadata);

 private:
  enum class OpType : char { kStablehloOp };
  explicit Translator(ModuleOp module, const toco::TocoFlags& toco_flags,
                      const std::unordered_set<std::string>& saved_model_tags,
                      tensorflow::OpOrArgNameMapper* op_or_arg_name_mapper,
                      const std::map<std::string, std::string>& metadata)
      : module_(module),
        name_mapper_(*op_or_arg_name_mapper),
        builder_(kInitialBufferSize),
        saved_model_tags_(saved_model_tags) {
    // The first buffer must be empty according to the schema definition.
    empty_buffer_ = ::stablehlo::flatbuf::CreateBuffer(builder_);
    buffers_.push_back(empty_buffer_);
    stablehlo_dialect_ =
        module.getContext()
            ->getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
    // Right now the TF executor dialect is still needed to build NodeDef.
    module.getContext()
        ->getOrLoadDialect<mlir::tf_executor::TensorFlowExecutorDialect>();
  }

  llvm::Optional<std::string> TranslateInternal();

  // Returns TFLite buffer populated with constant value if the operation is
  // TFLite constant operation. Otherwise, returns an empty buffer. Emits error
  // and returns std::nullopt on failure.
  llvm::Optional<BufferOffset<::stablehlo::flatbuf::Buffer>> BuildBuffer(
      Value value);

  // Builds TFLite tensor from the given value. `buffer_idx` is index of the
  // corresponding buffer. Emits error and returns std::nullopt on failure.
  llvm::Optional<BufferOffset<::stablehlo::flatbuf::Tensor>> BuildTensor(
      Value value, const std::string& name, unsigned buffer_idx);

  // Returns opcode index for op identified by the op_name, if already
  // available. Otherwise, creates a new OperatorCode using the given `builtin`
  // operator and associates it with `op_name`.
  uint32_t GetOpcodeIndex(const std::string& op_name,
                          ::stablehlo::flatbuf::OperatorCode op_code);

  // Builds operator for the given operation with specified operand and result
  // tensor indices. Emits an error and returns std::nullopt on failure.
  llvm::Optional<BufferOffset<::stablehlo::flatbuf::Operator>> BuildOperator(
      Operation* inst, std::vector<int32_t> operands,
      const std::vector<int32_t>& results);

  // Build a subgraph with a given name out of the region either corresponding
  // to a function's body or while op. Modifies *region by calling
  // ExtractControlEdges.
  llvm::Optional<BufferOffset<::stablehlo::flatbuf::SubGraph>> BuildSubGraph(
      const std::string& name, Region* region, int index);

  // Uses the tf.entry_function attribute (if set) to initialize the op to name
  // mapping.
  void InitializeNamesFromAttribute(mlir::func::FuncOp fn,
                                    bool* has_input_attr);

  // Returns a unique name for `val`.
  std::string UniqueName(mlir::Value val);

  ModuleOp module_;

  tensorflow::OpOrArgNameMapper& name_mapper_;

  flatbuffers::FlatBufferBuilder builder_;
  BufferOffset<::stablehlo::flatbuf::Buffer> empty_buffer_;

  std::vector<BufferOffset<::stablehlo::flatbuf::Buffer>> buffers_;
  // Maps subgraph index and tensor name in the graph to the tensor index.
  absl::flat_hash_map<int, absl::flat_hash_map<std::string, int>>
      tensor_index_map_;

  // Maps op name to index of the corresponding OperatorCode in opcodes_ vector.
  absl::flat_hash_map<std::string, uint32_t> opcode_index_map_;
  std::vector<int32_t> opcodes_;

  // Maps function name to index of the corresponding subgraph in the FlatBuffer
  // model.
  absl::flat_hash_map<std::string, int> subgraph_index_map_;
  absl::flat_hash_set<OpType> enabled_op_types_;

  // Points to stablehlo dialects & mhlo dialects, respectively. nullptr if the
  // dialect is not registered.
  Dialect* stablehlo_dialect_;

  // Set of saved model tags, if any.
  const std::unordered_set<std::string> saved_model_tags_;
  // Map of key value pairs of metadata to export.
  const std::map<std::string, std::string> metadata_;
  // A mapping table to mlir::Operation objects for TFL subgraph and operator
  // index in a flatbuffer.
  std::vector<std::vector<Operation*>> subgraph_op_inst_map_;

  const std::string tf_entry_function_ = "tf.entry_function";
};

}  // namespace odml
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_SERIALIZER_FLATBUFFER_TRANSLATOR_H_
