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

// This is an EXPERIMENTAL API to programatically build TFLite graphs. It may
// change or be removed at any time. Use it at your own risk.

#include <array>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {

namespace model_builder {

using OwningErasedPtr = std::unique_ptr<void, void (*)(void*)>;

class InterpreterInfo;
struct Helper;

namespace internal {

template <class T, class Tag>
struct StrongType {
  StrongType() = default;
  explicit StrongType(const T& v) : val(v) {}

  T val;
};

using GraphIdx = StrongType<int, class GraphTag>;
using TensorIdx = StrongType<int, class TensorTag>;

}  // namespace internal

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

 private:
  Tensor(InterpreterInfo* builder, internal::TensorIdx tensor_idx,
         internal::GraphIdx graph_idx)
      : builder_(builder), tensor_idx_(tensor_idx), graph_idx_(graph_idx) {}

  friend class Helper;

  InterpreterInfo* builder_;
  internal::TensorIdx tensor_idx_;
  internal::GraphIdx graph_idx_;
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

 private:
  Graph(InterpreterInfo* builder, internal::GraphIdx graph_idx)
      : builder_(builder), graph_idx_(graph_idx) {}

  friend class Helper;

  InterpreterInfo* builder_;
  internal::GraphIdx graph_idx_;
};

namespace internal {

template <size_t... Is>
std::array<Tensor, sizeof...(Is)> NewInputsImpl(std::index_sequence<Is...>,
                                                Graph graph, TfLiteType type);
}  // namespace internal

// Returns an array of `Count` inputs for the given `graph`.
//
// Useful to declare multiple similar tensors in the same graph.
//
// ```cpp
// auto [t1, t2] = NewInputs<2>(graph, kTfLiteFloat32);
// ```
template <size_t Count>
[[nodiscard]] std::array<Tensor, Count> NewInputs(Graph graph,
                                                  TfLiteType type) {
  return internal::NewInputsImpl(std::make_index_sequence<Count>{}, graph,
                                 type);
}

// Allows building a TFLite interpreter programatically.
//
// ```cpp
// GraphBuilder builder;
// Graph grap = builder.NewGraph();
//
// auto [in1, in2] = NewInputs<2>(kTfLiteInt32);
// Tensor sum = Add(in1, in2);
// Tensor abs1 = Abs(in1)
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

 private:
  friend class Helper;

  OwningErasedPtr impl_;
};

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

namespace internal {

template <size_t... Is>
std::array<Tensor, sizeof...(Is)> NewInputsImpl(std::index_sequence<Is...>,
                                                Graph graph, TfLiteType type) {
  return std::array<Tensor, sizeof...(Is)>{
      ((void)Is, NewInput(graph, type))...};
}

}  // namespace internal

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

// Creates a STABLEHLO_COMPOSITE operation named `name` and falling back to
// `subgraph`.
//
// `inputs` is associated to the subgraph inputs.
//
// - all the `inputs` must be from the same `Graph`.
// - `subgraph` but be associated to the same `GraphBuilder` as the `inputs`'
// `Graph`.
//
// The resulting operation is added to the `Graph` the `inputs` are related to.
//
// Returns a list of tensors representing the subgraph outputs.
std::vector<Tensor> StableHLOComposite(const char* name, const Graph& subgraph,
                                       const std::vector<Tensor>& inputs);

}  // namespace model_builder
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_MODEL_BUILDING_H_
