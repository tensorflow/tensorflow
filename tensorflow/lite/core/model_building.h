/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_CORE_MODEL_BUILDING_H_
#define TENSORFLOW_LITE_CORE_MODEL_BUILDING_H_

// This is an EXPERIMENTAL API to programmatically build TFLite graphs. It may
// change or be removed at any time. Use it at your own risk.

#include <array>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"

namespace tflite {
namespace model_builder {

class InterpreterInfo;
class Graph;

namespace internal {

template <class T, class Tag>
struct StrongType {
  constexpr StrongType() = default;
  constexpr explicit StrongType(const T& v) : val(v) {}

  T val;
};

template <typename H, class T, class Tag>
H AbslHashValue(H h, const StrongType<T, Tag>& v) {
  return H::combine(std::move(h), v.val);
}

template <class T, class Tag>
bool operator==(const StrongType<T, Tag>& lhs, const StrongType<T, Tag>& rhs) {
  return lhs.val == rhs.val;
}

}  // namespace internal

using OwningErasedPtr = std::unique_ptr<void, void (*)(void*)>;
using GraphIdx = internal::StrongType<int, class GraphTag>;
using TensorIdx = internal::StrongType<int, class TensorTag>;
using BufferIdx = internal::StrongType<int, class BufferTag>;

struct NoQuantization {};

struct [[nodiscard]] AffineQuantization {
  std::vector<float> zero_points;
  std::vector<float> scales;
  int axis;
};

using Quantization = std::variant<NoQuantization, AffineQuantization>;

// Represents a buffer in the TFLite graph.
//
// Copyable but you shouldn't create such an object by yourself. Use the
// `NewConstantTensor` family of functions with Builder for that.
//
// Each buffer is attached to a builder instance. Use `AddConstantTensor` to
// make it available to a graph.
class [[nodiscard]] Buffer {
 public:
  static constexpr const BufferIdx kNoBuffer{-1};

  Buffer(const Buffer&) = default;
  Buffer& operator=(const Buffer&) = default;

 private:
  Buffer(InterpreterInfo* builder, BufferIdx buffer_idx)
      : builder_(builder), buffer_idx_(buffer_idx) {}

  friend class Helper;
  friend class Tensor;

  InterpreterInfo* builder_;
  BufferIdx buffer_idx_;
};

// Assigns data to be managed by the given Buffer.
template <TfLiteType kType, class T>
void Assign(Buffer b, std::vector<int> shape, const std::vector<T>& data,
            Quantization quantization) {
  using Storage = TfLiteTypeToType<kType>::Type;
  std::unique_ptr<Storage[]> buffer_data(new Storage[data.size()]);
  std::copy(begin(data), end(data), buffer_data.get());
  Assign(
      b, kType, std::move(shape),
      reinterpret_cast<char*>(buffer_data.release()),
      [](char* data) { delete[] reinterpret_cast<Storage*>(data); },
      sizeof(Storage) * data.size(), std::move(quantization));
}

// Assigns data to be managed by the given Buffer.
void Assign(Buffer b, TfLiteType type, std::vector<int> shape, char* data,
            void (*deleter)(char*), size_t bytes, Quantization quantization);

// Represents a tensor in the TFLite graph.
//
// Copyable but you shouldn't create such an object by yourself. Use the
// `NewInput` family of functions with a Graph for that.
//
// Each tensor is attached to a particular graph. Don't mix tensors created by
// different graphs in operations.
//
// ```cpp
// Tensor a = graph.NewInput(kTfLiteInt32);
// Tensor b = NewInput(graph, kTfLiteFloat32);
// auto [c, d] = NewInputs<2>(graph, kTfLiteFloat32);
// ```
class [[nodiscard]] Tensor {
 public:
  Tensor(const Tensor&) = default;
  Tensor& operator=(const Tensor&) = default;

  explicit Tensor(Buffer buffer) : builder_(buffer.builder_) {}

 private:
  Tensor() = default;
  Tensor(InterpreterInfo* builder, TensorIdx tensor_idx, GraphIdx graph_idx)
      : builder_(builder), tensor_idx_(tensor_idx), graph_idx_(graph_idx) {}

  friend class Helper;
  template <size_t count>
  friend std::array<Tensor, count> NewInputs(Graph graph, TfLiteType type);

  InterpreterInfo* builder_;
  TensorIdx tensor_idx_{-1};
  GraphIdx graph_idx_{-1};
};

// Represents a subgraph in the TFLite interpreter.
//
// Copyable but you shouldn't create such an object by yourself. Use the
// `NewGraph` function with a GraphBuilder for that.
//
// ```cpp
// Graph a = builder.NewGraph();
// ```
class [[nodiscard]] Graph {
 public:
  // Returns a new input for the given graph.
  //
  // See also: `NewInputs<Count>()`.
  friend Tensor NewInput(Graph& graph, TfLiteType type);

  friend Tensor NewConstantTensor(Graph& graph, Buffer buffer);

 private:
  Graph(InterpreterInfo* builder, GraphIdx graph_idx)
      : builder_(builder), graph_idx_(graph_idx) {}

  friend class Helper;

  InterpreterInfo* builder_;
  GraphIdx graph_idx_;
};

// Returns an array of `Count` inputs for the given `graph`.
//
// Useful to declare multiple similar tensors in the same graph.
//
// ```cpp
// auto [t1, t2] = NewInputs<2>(graph, kTfLiteFloat32);
// ```
template <size_t count>
[[nodiscard]] std::array<Tensor, count> NewInputs(Graph graph,
                                                  TfLiteType type) {
  std::array<Tensor, count> tensors{};
  for (size_t i = 0; i < count; ++i) {
    tensors[i] = NewInput(graph, type);
  }
  return tensors;
}

// Allows building a TFLite interpreter programmatically.
//
// ```cpp
// ModelBuilder builder;
// Graph graph = builder.NewGraph();
//
// auto [in1, in2] = NewInputs<2>(graph, kTfLiteInt32);
// Tensor sum = Add(in1, in2);
// Tensor abs = Abs(in1)
// Tensor out = Mul(sum, abs1);
// MarkOuput(out);
//
// builder.Build(interpreter);
// ```
class ModelBuilder {
 public:
  ModelBuilder();

  // Applies the graphs that were defined with this builder (and related
  // `Graph`/`Tensor` objects) to the given TFLite `Interpreter`.
  //
  // Note: calling this on an interpreter that has already been set up is
  // unsupported.
  void Build(Interpreter& interpreter);

  // Returns a new graph managed by the given builder.
  friend Graph NewGraph(ModelBuilder& builder);

  // Returns a new buffer that can be used as a constant tensor with graphs.
  friend Buffer NewConstantBuffer(ModelBuilder& builder);

 private:
  friend class Helper;

  OwningErasedPtr impl_;
};

template <TfLiteType kType, class T>
Buffer NewConstantBuffer(ModelBuilder& builder, std::vector<int> shape,
                         const std::vector<T>& data,
                         Quantization quantization) {
  Buffer buffer = NewConstantBuffer(builder);
  Assign<kType>(buffer, std::move(shape), std::move(data),
                std::move(quantization));
  return buffer;
}

void SetShape(Tensor tensor, std::vector<int> shape);

// Marks the given tensor as an output of the graph it is attached to.
void MarkOutput(Tensor tensor);

// Marks the given tensors as outputs of the graph they are attached to.
inline void MarkOutputs(const std::vector<Tensor>& tensors) {
  for (const Tensor& t : tensors) {
    MarkOutput(t);
  }
}

// Marks the given tensors as outputs of the graph they are attached to.
inline void MarkOutputs(std::initializer_list<Tensor> tensors) {
  for (const Tensor& t : tensors) {
    MarkOutput(t);
  }
}

// Marks the given tensors as outputs of the graph they are attached to.
template <class... Ts>
void MarkOutputs(Ts... tensors) {
  (MarkOutput(tensors), ...);
}

// Creates an ABS operation with `tensor` as the input and returns the tensor
// representing the result.
//
// The resulting operation is added to the `Graph` the `tensor` is related to.
Tensor Abs(Tensor tensor);

// Creates an ADD operation with `lhs` and `rhs` as the inputs and returns the
// tensor representing the result.
//
// - `lhs` and `rhs` must be from the same `Graph`.
//
// The resulting operation is added to the `Graph` the tensors are related to.
Tensor Add(Tensor lhs, Tensor rhs);

// Creates an MUL operation with `lhs` and `rhs` as the inputs and returns the
// tensor representing the result.
//
// - `lhs` and `rhs` must be from the same `Graph`.
//
// The resulting operation is added to the `Graph` the tensors are related to.
Tensor Mul(Tensor lhs, Tensor rhs);

// Creates a TRANSPOSE operation with `tensor` and `permutation` as the inputs
// and returns the tensor representing the result.
//
// - `tensor` and `permutation` must be from the same `Graph`.
//
// The resulting operation is added to the `Graph` the `tensor` is related to.
Tensor Transpose(Tensor tensor, Tensor permutation);

// Creates a FULLY_CONNECTED operation and returns its output tensor handle.
//
// - A tensor referencing `weights` is added to `input`'s `Graph`.
// - The types and and shapes must be compatible with what TFLite supports.
//
// The resulting operation is added to the `Graph` the `input` is related to.
Tensor FullyConnected(Tensor input, Buffer weights);

// Creates a STABLEHLO_COMPOSITE operation named `name` and falling back to
// `subgraph`.
//
// `inputs` is associated to the subgraph inputs.
//
// - All the `inputs` must be from the same `Graph`.
// - `subgraph` must be associated to the same `GraphBuilder` as the `inputs`'
//   `Graph`.
//
// The resulting operation is added to the `Graph` the `inputs` are related to.
//
// Returns a list of tensors representing the subgraph outputs.
std::vector<Tensor> StableHLOComposite(const char* name, const Graph& subgraph,
                                       const std::vector<Tensor>& inputs);

}  // namespace model_builder
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_MODEL_BUILDING_H_
