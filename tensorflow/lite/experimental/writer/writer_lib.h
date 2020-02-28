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
// Writes a flatbuffer of a currently loaded TensorFlow Lite subgraph.
//
// Usage:
//  From command line:
//   bazel run third_party/tensorflow/lite/experimental/writer:writer
//     -- foo.tflite foo.out.tflite
//
// From C++
//   std::unique_ptr<Interpreter> interpreter;
//   // Build Interpreter however
//   // ... <omitted>
//   SubgraphWriter(&interpreter->primary_subgraph()).Write("output.tflite");
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_WRITER_WRITER_LIB_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_WRITER_WRITER_LIB_H_
#include <iostream>
#include <unordered_map>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/writer/enum_mapping.h"
#include "tensorflow/lite/schema/reflection/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {

// Handles writing TensorFlow Lite running subgraph to a serialized TF lite
// file format.
class SubgraphWriter {
 public:
  typedef flatbuffers::Offset<Operator> (*CustomWriter)(
      flatbuffers::FlatBufferBuilder* fbb, Subgraph* subgraph, int node_index,
      flatbuffers::Offset<flatbuffers::Vector<uint8_t>>* output_options,
      CustomOptionsFormat* custom_options_format);

  // Construct an subgraph writer for the specified `subgraph`. Then,
  // a uses .Write() or .GetBuffer(...)  to extract the data.
  explicit SubgraphWriter(Subgraph* subgraph) : subgraph_(subgraph) {
    buffers_.push_back(std::make_pair(nullptr, 0));
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

 private:
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

  int GetOpCodeForBuiltin(int builtin_op_index) {
    // auto it = builtin_op_to_opcode_.find(builtin_op_index);
    std::pair<decltype(builtin_op_to_opcode_)::iterator, bool> result =
        builtin_op_to_opcode_.insert(
            std::make_pair(builtin_op_index, opcodes_.size()));
    if (result.second) {
      opcodes_.push_back({builtin_op_index, ""});
    }
    return result.first->second;
  }

  int GetOpCodeForCustom(const std::string& custom_name) {
    std::pair<decltype(custom_op_to_opcode_)::iterator, bool> result =
        custom_op_to_opcode_.insert(
            std::make_pair(custom_name, opcodes_.size()));
    if (result.second) {
      opcodes_.push_back({BuiltinOperator_CUSTOM, custom_name});
    }
    return result.first->second;
  }

  // The subgraph we are writing
  Subgraph* subgraph_;
  // Keep track of byte buffers
  std::vector<std::pair<const uint8_t*, size_t>> buffers_;
  // List of op codes and mappings from builtin or custom op to opcode
  struct OpCode {
    int builtin;
    std::string custom;
  };
  std::set<int> unused_tensors_;
  // For every tensor index in the subgraph, the index in the written.
  // This is different due to temporary and unused tensors not being written.
  std::vector<int> tensor_to_written_tensor_;
  // List of used opcodes
  std::vector<OpCode> opcodes_;
  std::unordered_map<int, int> builtin_op_to_opcode_;
  std::unordered_map<std::string, int> custom_op_to_opcode_;
  std::unordered_map<std::string, CustomWriter> custom_op_to_writer_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_WRITER_WRITER_LIB_H_
