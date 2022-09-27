/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
// Library to write a flatbuffer of a currently loaded TFLite model/subgraph.

#ifndef TENSORFLOW_LITE_TOOLS_SERIALIZATION_WRITER_LIB_H_
#define TENSORFLOW_LITE_TOOLS_SERIALIZATION_WRITER_LIB_H_
#include <iostream>
#include <string>
#include <unordered_map>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/reflection/schema_generated.h"
#include "tensorflow/lite/tools/serialization/enum_mapping.h"
#include "tensorflow/lite/version.h"

namespace tflite {

struct OpCode {
  int builtin;
  std::string custom;
};

// Forward declaration.
class SubgraphWriter;

// Handles writing a full TFLite model (with 1 or more subgraphs) to a
// serialized TF lite file format.
// TODO(b/174708523): Support custom I/O or unused tensors later.
class ModelWriter {
 public:
  // CustomWriter allows the delegate to customize the write to the flatbuffer.
  typedef flatbuffers::Offset<Operator> (*CustomWriter)(
      flatbuffers::FlatBufferBuilder* fbb, Subgraph* subgraph, int node_index,
      flatbuffers::Offset<flatbuffers::Vector<uint8_t>>* output_options,
      CustomOptionsFormat* custom_options_format);

  // Construct a writer for the specified `interpreter`. Then, use
  // .Write() or .GetBuffer(...) to extract the data.
  explicit ModelWriter(Interpreter* interpreter);

  // Same as above, except takes subgraphs as input.
  explicit ModelWriter(const std::vector<Subgraph*>& subgraphs);

  // For initializing the ModelWriter internal data.
  void Init(const std::vector<Subgraph*>& subgraphs);

  // Get a buffer and size of a serialized flatbuffer.
  TfLiteStatus GetBuffer(std::unique_ptr<uint8_t[]>* out, size_t* size);
  // Write the serialized flatbuffer to the prescribed `filename`.
  TfLiteStatus Write(const std::string& filename);

  // Specifies unused tensors on the target subgraph.
  void SetUnusedTensors(int subgraph_index,
                        const std::set<int>& unused_tensors);

  // Specifies custom inputs, outputs, and execution_plan to target subgraph.
  TfLiteStatus SetCustomInputOutput(int subgraph_index,
                                    const std::vector<int>& inputs,
                                    const std::vector<int>& outputs,
                                    const std::vector<int>& execution_plan);

  // Registers a custom writer for a custom op. The customization allows the
  // caller to change the custom data.
  TfLiteStatus RegisterCustomWriter(const std::string& custom_name,
                                    CustomWriter custom_writer);

 private:
  template <class T>
  using Offset = flatbuffers::Offset<T>;
  Offset<flatbuffers::Vector<Offset<OperatorCode>>> CreateOpCodeTable(
      flatbuffers::FlatBufferBuilder* fbb);
  Offset<flatbuffers::Vector<Offset<Buffer>>> ExportBuffers(
      flatbuffers::FlatBufferBuilder* fbb);

  // List of subgraph writers owned by this model writer.
  // There is one subgraph writer for each subgraph in the model.
  std::vector<SubgraphWriter> subgraph_writers_;

  // This data corresponds to the overall model (rather than individual
  // subgraphs), so we define common fields. Keep track of byte buffers
  std::vector<std::pair<const uint8_t*, size_t>> buffers_;
  // List of used opcodes
  std::vector<OpCode> opcodes_;
  std::unordered_map<int, int> builtin_op_to_opcode_;
};

// Handles writing TensorFlow Lite running subgraph to a serialized TF lite
// file format.
// TODO(b/174708523): Reconcile into ModelWriter?
class SubgraphWriter {
 public:
  friend class ModelWriter;

  typedef flatbuffers::Offset<Operator> (*CustomWriter)(
      flatbuffers::FlatBufferBuilder* fbb, Subgraph* subgraph, int node_index,
      flatbuffers::Offset<flatbuffers::Vector<uint8_t>>* output_options,
      CustomOptionsFormat* custom_options_format);

  // Construct a subgraph writer for the specified `subgraph`. Then, use
  // .Write() or .GetBuffer(...) to extract the data.
  explicit SubgraphWriter(Subgraph* subgraph)
      : subgraph_(subgraph),
        inputs_(subgraph->inputs()),
        outputs_(subgraph->outputs()),
        execution_plan_(subgraph->execution_plan()) {
    buffers_ = &buffers_data_;
    opcodes_ = &opcodes_data_;
    builtin_op_to_opcode_ = &builtin_op_to_opcode_data_;
    buffers_->push_back(std::make_pair(nullptr, 0));
  }

  // Get a buffer and size of a serialized flatbuffer.
  TfLiteStatus GetBuffer(std::unique_ptr<uint8_t[]>* out, size_t* size);
  // Write the serialized flatbuffer to the prescribed `filename`.
  TfLiteStatus Write(const std::string& filename);
  // Registers a custom writer for a custom op. The customization allows the
  // caller to change the custom data.
  TfLiteStatus RegisterCustomWriter(const std::string& custom_name,
                                    CustomWriter custom_writer);
  // Tensors that are unused and shouldn't be written.
  void SetUnusedTensors(const std::set<int>& unused_tensors) {
    unused_tensors_ = unused_tensors;
  }
  // Sets custom inputs, outputs, and execution_plan so that a portion of the
  // subgraph is written to the buffer instead of the whole subgraph.
  TfLiteStatus SetCustomInputOutput(const std::vector<int>& inputs,
                                    const std::vector<int>& outputs,
                                    const std::vector<int>& execution_plan);

 private:
  // Used by ModelWriter.
  explicit SubgraphWriter(
      Subgraph* subgraph,
      std::vector<std::pair<const uint8_t*, size_t>>* external_buffers,
      std::vector<OpCode>* external_opcodes,
      std::unordered_map<int, int>* external_builtin_op_to_opcode)
      : subgraph_(subgraph),
        inputs_(subgraph->inputs()),
        outputs_(subgraph->outputs()),
        execution_plan_(subgraph->execution_plan()) {
    buffers_ = external_buffers;
    opcodes_ = external_opcodes;
    builtin_op_to_opcode_ = external_builtin_op_to_opcode;
    buffers_->push_back(std::make_pair(nullptr, 0));
  }

  // Used by ModelWriter to populate data specific to this subgraph.
  // Global stuff (like opcodes & buffers) is populated into buffers_, opcodes_,
  // etc. & populated in the Flatbuffer by ModelWriter.
  flatbuffers::Offset<SubGraph> PopulateAndGetOffset(
      flatbuffers::FlatBufferBuilder* builder,
      const std::string& subgraph_name);

  template <class T>
  using Offset = flatbuffers::Offset<T>;
  template <class T_OUTPUT, class T_INPUT>
  Offset<flatbuffers::Vector<T_OUTPUT>> ExportVector(
      flatbuffers::FlatBufferBuilder* fbb, const T_INPUT& v);
  Offset<flatbuffers::Vector<Offset<Tensor>>> ExportTensors(
      flatbuffers::FlatBufferBuilder* fbb);
  Offset<flatbuffers::Vector<Offset<Operator>>> ExportOperators(
      flatbuffers::FlatBufferBuilder* fbb);
  Offset<flatbuffers::Vector<Offset<OperatorCode>>> CreateOpCodeTable(
      flatbuffers::FlatBufferBuilder* fbb);
  Offset<flatbuffers::Vector<Offset<Buffer>>> ExportBuffers(
      flatbuffers::FlatBufferBuilder* fbb);

  template <class T>
  std::vector<int> RemapTensorIndicesToWritten(const T& input);

  // Checks if given `input`, `output`, and `execution_plan` represents a valid
  // model within the Subgraph.
  TfLiteStatus CheckInputOutput(const std::vector<int>& inputs,
                                const std::vector<int>& outputs,
                                const std::vector<int>& execution_plan);

  int GetOpCodeForBuiltin(int builtin_op_index) {
    // auto it = builtin_op_to_opcode_.find(builtin_op_index);
    std::pair<decltype(builtin_op_to_opcode_data_)::iterator, bool> result =
        builtin_op_to_opcode_->insert(
            std::make_pair(builtin_op_index, opcodes_->size()));
    if (result.second) {
      opcodes_->push_back({builtin_op_index, ""});
    }
    return result.first->second;
  }

  int GetOpCodeForCustom(const std::string& custom_name) {
    std::pair<decltype(custom_op_to_opcode_)::iterator, bool> result =
        custom_op_to_opcode_.insert(
            std::make_pair(custom_name, opcodes_->size()));
    if (result.second) {
      opcodes_->push_back({BuiltinOperator_CUSTOM, custom_name});
    }
    return result.first->second;
  }

  // The subgraph we are writing
  Subgraph* subgraph_;
  // Input tensor indices to be written.
  std::vector<int> inputs_;
  // Output tensor indices to be written.
  std::vector<int> outputs_;
  // Order of nodes to be written.
  std::vector<int> execution_plan_;
  // List of op codes and mappings from builtin or custom op to opcode
  std::set<int> unused_tensors_;
  // For every tensor index in the subgraph, the index in the written.
  // This is different due to temporary and unused tensors not being written.
  std::vector<int> tensor_to_written_tensor_;
  std::unordered_map<std::string, int> custom_op_to_opcode_;
  std::unordered_map<std::string, CustomWriter> custom_op_to_writer_;

  // We use pointers for these, since they may be provided by ModelWriter.
  // Keep track of byte buffers
  std::vector<std::pair<const uint8_t*, size_t>>* buffers_;
  // List of used opcodes
  std::vector<OpCode>* opcodes_;
  std::unordered_map<int, int>* builtin_op_to_opcode_;

  // These are used if SubgraphWriter is being used directly.
  std::vector<std::pair<const uint8_t*, size_t>> buffers_data_;
  // List of used opcodes
  std::vector<OpCode> opcodes_data_;
  std::unordered_map<int, int> builtin_op_to_opcode_data_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_SERIALIZATION_WRITER_LIB_H_
