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
#include "tensorflow/lite/core/model_building.h"

#include <algorithm>
#include <array>

// TODO: Change assert to TFLITE logging.
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace model_builder {

namespace {

// Returns a functor that provides overloads based on the
// functors passed to it.
//
// Useful when used in conjunction with `std::visit`.
template <class... Ts>
class Overload : public Ts... {
 public:
  explicit Overload(Ts&&... ts) : Ts(static_cast<Ts&&>(ts))... {}
  using Ts::operator()...;
};

template <class... Ts>
Overload(Ts&&...) -> Overload<Ts...>;

}  // namespace

struct BufferInfo {
  void AssignOwning(char* data_src, void (*deleter)(char*), size_t data_bytes) {
    data = std::unique_ptr<char, void (*)(char*)>(data_src, deleter);
    bytes = data_bytes;
  }

  void AssignNonOwning(char* data_src, size_t data_bytes) {
    data = std::unique_ptr<char, void (*)(char*)>(data_src, DontDelete);
    bytes = data_bytes;
  }

  template <class T>
  T* DataAs() noexcept {
    return reinterpret_cast<T*>(data.get());
  }

  template <class T>
  const T* DataAs() const noexcept {
    return reinterpret_cast<T*>(data.get());
  }

  // No-op cleanup for non-owned data.
  static constexpr void DontDelete(char*) {}

  BufferIdx idx;
  TfLiteType type;
  std::unique_ptr<char, void (*)(char*)> data{nullptr, DontDelete};
  size_t bytes = 0;
  std::vector<int> shape;
  Quantization quantization;
};

struct TensorInfo {
  TensorInfo() = default;
  // This object holds information that is part of a graph. A inadvertent copy
  // made by not returning a reference to the object stored in an GraphInfo will
  // lead to issues (namely no modification done to the graph).
  //
  // The copy constructor is made explicit to allow copying the TensorInfo in
  // the rare cases that would actually need a copy.
  //
  // NOLINTNEXTLINE(*-explicit-constructor)
  explicit TensorInfo(const TensorInfo&) = default;
  TensorInfo& operator=(const TensorInfo&) = delete;

  TensorInfo(TensorInfo&&) = default;
  TensorInfo& operator=(TensorInfo&&) = default;

  TensorInfo(int idx, TfLiteType type) : idx(idx), type(type) {}

  // Index in GraphInfo tensors.
  TensorIdx idx;
  TfLiteType type;
  // Index in InterpreterInfo buffers.
  BufferIdx buffer_idx = Buffer::kNoBuffer;
  // Shape when the tensor is not backed by a buffer.
  std::vector<int> shape;
};

struct OpInfo {
  BuiltinOperator op;
  OwningErasedPtr params;
  // Indices in GraphInfo tensors.
  std::vector<TensorIdx> inputs;
  // Indices in GraphInfo tensors.
  std::vector<TensorIdx> outputs;
  TfLiteRegistration registration;
};

struct GraphInfo {
  GraphInfo() = default;
  // This object holds information that is part of a graph. A inadvertent copy
  // made by not returning a reference to the object stored in an
  // InterpreterInfo will lead to issues (namely no modification done to the
  // graph).
  //
  // The copy constructor is made explicit to allow copying the GraphInfo in the
  // rare cases that would actually need a copy.
  //
  // NOLINTNEXTLINE(*-explicit-constructor)
  explicit GraphInfo(const GraphInfo&) = default;
  GraphInfo& operator=(const GraphInfo&) = delete;

  GraphInfo(GraphInfo&&) = default;
  GraphInfo& operator=(GraphInfo&&) = default;

  // Index in InterpreterInfo subgraphs.
  GraphIdx idx;
  std::vector<OpInfo> ops;
  std::vector<TensorInfo> tensors;
  // Indices in tensors.
  std::vector<TensorIdx> inputs;
  // Indices in tensors.
  std::vector<TensorIdx> outputs;

  enum TensorRole {
    kNone,
    kInput = 1,
    kOutput = 1 << 1,
  };

  TensorInfo& NewTensor(TfLiteType type, const int role = TensorRole::kNone) {
    const int idx = tensors.size();
    tensors.emplace_back(idx, type);
    if (role & TensorRole::kInput) {
      inputs.push_back(tensors.back().idx);
    }
    if (role & TensorRole::kOutput) {
      outputs.push_back(tensors.back().idx);
    }
    return tensors.back();
  }

  const TensorInfo& NewInput(TfLiteType type) {
    return NewTensor(type, TensorRole::kInput);
  }

  const TensorInfo& NewOutput(TfLiteType type) {
    return NewTensor(type, TensorRole::kOutput);
  }

  const TensorInfo& GetOutput(int i) const {
    const TensorIdx idx = outputs.at(i);
    return tensors.at(idx.val);
  }
};

struct InterpreterInfo {
  std::vector<GraphInfo> subgraphs;
  std::vector<BufferInfo> buffers;

  GraphInfo& GetGraph(GraphIdx graph_idx) {
    return subgraphs.at(graph_idx.val);
  }

  // Adds a new graph to the interpreter.
  //
  // Calling this function may invalidate existing references (to GraphInfo
  // objects, Graph object are fine as they only hold the graph index).
  GraphInfo& NewGraph() {
    const GraphIdx idx(subgraphs.size());
    subgraphs.emplace_back();
    subgraphs.back().idx = idx;
    return subgraphs.back();
  }

  BufferInfo& NewBuffer() {
    buffers.emplace_back();
    BufferInfo& buffer = buffers.back();
    buffer.idx = BufferIdx(buffers.size() - 1);
    return buffer;
  }
};

TfLiteQuantization ToTfLiteQuantization(Quantization quantization) {
  TfLiteQuantization q{/*.type=*/kTfLiteNoQuantization};
  std::visit(
      Overload([&q](NoQuantization) { q.type = kTfLiteNoQuantization; },
               [&q](const AffineQuantization& src) {
                 q.type = kTfLiteAffineQuantization;
                 q.params = calloc(sizeof(TfLiteAffineQuantization), 1);
                 TfLiteAffineQuantization& qa =
                     *reinterpret_cast<TfLiteAffineQuantization*>(q.params);
                 qa.quantized_dimension = src.axis;
                 qa.scale = BuildTfLiteArray<float>(src.scales).release();
                 qa.zero_point =
                     BuildTfLiteArray<int>(src.zero_points).release();
               }),
      quantization);
  return q;
}

void Apply(InterpreterInfo& interpreter_info, GraphInfo& graph,
           Subgraph& subgraph) {
  // Maps graph indices to subgraph indices.
  absl::flat_hash_map<TensorIdx, int> tensor_id_to_idx;
  int first_new_tensor_index;
  subgraph.AddTensors(graph.tensors.size(), &first_new_tensor_index);
  for (int i = 0; i < graph.tensors.size(); ++i) {
    tensor_id_to_idx.insert({graph.tensors[i].idx, i + first_new_tensor_index});
    TfLiteTensor& tensor = *subgraph.tensor(i + first_new_tensor_index);
    tensor.type = graph.tensors[i].type;
  }

  auto GetTensorIds = [&](const std::vector<TensorIdx>& ts) {
    std::vector<int> indices(ts.size());
    for (int i = 0; i < ts.size(); ++i) {
      indices[i] = tensor_id_to_idx[graph.tensors[ts[i].val].idx];
    }
    return indices;
  };

  const std::vector<int> input_indices = GetTensorIds(graph.inputs);
  subgraph.SetInputs(input_indices);
  const std::vector<int> output_indices = GetTensorIds(graph.outputs);
  subgraph.SetOutputs(output_indices);
  for (TensorIdx i(0); i.val < graph.tensors.size(); ++i.val) {
    const TensorInfo& tensor = graph.tensors[i.val];
    if (tensor.buffer_idx.val == -1) {
      if (subgraph.SetTensorParametersReadWrite(
              tensor_id_to_idx[i], tensor.type, /*name=*/"",
              /*ndims=*/tensor.shape.size(),
              /*dims=*/tensor.shape.data(), /*quantization=*/{},
              /*is_variable=*/false) != kTfLiteOk) {
        std::terminate();
      }
    } else {
      const BufferInfo& buffer =
          interpreter_info.buffers[tensor.buffer_idx.val];
      if (subgraph.SetTensorParametersReadOnly(
              tensor_id_to_idx[i], tensor.type, "", buffer.shape,
              ToTfLiteQuantization(buffer.quantization), buffer.DataAs<char>(),
              buffer.bytes,
              /*allocation=*/nullptr, /*sparsity=*/nullptr,
              /*buffer_identifier=*/buffer.idx.val) != kTfLiteOk) {
        std::terminate();
      }
    }
  }

  for (OpInfo& op : graph.ops) {
    const TfLiteRegistration& reg = op.registration;
    int node_index;
    subgraph.AddNodeWithParameters(GetTensorIds(op.inputs),
                                   GetTensorIds(op.outputs), {}, nullptr, 0,
                                   op.params.release(), &reg, &node_index);
  }
}

struct Helper {
  static InterpreterInfo& GetInterpreterInfo(ModelBuilder& builder) {
    return *reinterpret_cast<InterpreterInfo*>(builder.impl_.get());
  }

  static InterpreterInfo& GetInterpreterInfo(const Tensor tensor) {
    return *tensor.builder_;
  }

  static InterpreterInfo& GetInterpreterInfo(const Buffer buffer) {
    return *buffer.builder_;
  }

  static InterpreterInfo& GetInterpreterInfo(const Graph graph) {
    return *graph.builder_;
  }

  static GraphInfo& GetGraphInfo(const Tensor a) {
    InterpreterInfo& ii = Helper::GetInterpreterInfo(a);
    return ii.subgraphs.at(a.graph_idx_.val);
  }

  static GraphInfo& GetGraphInfo(const Graph a) {
    InterpreterInfo& ii = Helper::GetInterpreterInfo(a);
    return ii.subgraphs.at(a.graph_idx_.val);
  }

  static BufferInfo& GetBufferInfo(const Buffer a) {
    InterpreterInfo& ii = Helper::GetInterpreterInfo(a);
    return ii.buffers.at(a.buffer_idx_.val);
  }

  static BufferInfo* MaybeGetBufferInfo(const Tensor a) {
    InterpreterInfo& ii = Helper::GetInterpreterInfo(a);
    const TensorInfo& tensor_info = Helper::GetTensorInfo(a);
    if (tensor_info.buffer_idx.val >= 0) {
      return &ii.buffers[tensor_info.buffer_idx.val];
    }
    return nullptr;
  }

  static TensorInfo& GetTensorInfo(const Tensor& a) {
    InterpreterInfo& ii = Helper::GetInterpreterInfo(a);
    return ii.subgraphs.at(a.graph_idx_.val).tensors.at(a.tensor_idx_.val);
  }

  static TensorIdx GetTensorIndex(const Tensor& a) { return a.tensor_idx_; }

  static TfLiteType GetTensorType(const Tensor& a) {
    return Helper::GetTensorInfo(a).type;
  }

  static const std::vector<int>& GetTensorShape(const Tensor& a) {
    TensorInfo& tensor_info = Helper::GetTensorInfo(a);
    if (tensor_info.buffer_idx.val >= 0) {
      InterpreterInfo& ii = Helper::GetInterpreterInfo(a);
      return ii.buffers.at(tensor_info.buffer_idx.val).shape;
    }
    return tensor_info.shape;
  }

  static Graph BuildGraph(InterpreterInfo& interpreter_info,
                          const GraphInfo& graph) {
    return Graph(&interpreter_info, graph.idx);
  }

  static Tensor BuildTensor(InterpreterInfo& interpreter_info,
                            const GraphInfo& graph, const TensorInfo& tensor) {
    return Tensor(&interpreter_info, tensor.idx, graph.idx);
  }

  static Buffer BuildBuffer(InterpreterInfo& interpreter_info,
                            const BufferInfo& buffer) {
    return Buffer(&interpreter_info, buffer.idx);
  }
};

namespace internal {

BufferInfo& GetBufferInfo(Buffer buffer) {
  return Helper::GetBufferInfo(buffer);
}

}  // namespace internal

void Assign(Buffer b, TfLiteType type, std::vector<int> shape, char* data,
            void (*deleter)(char*), size_t bytes, Quantization quantization) {
  BufferInfo& buffer_info = Helper::GetBufferInfo(b);
  buffer_info.AssignOwning(data, deleter, bytes);
  buffer_info.type = type;
  buffer_info.shape = std::move(shape);
  buffer_info.quantization = std::move(quantization);
}

ModelBuilder::ModelBuilder()
    : impl_(new InterpreterInfo(), [](void* data) {
        delete reinterpret_cast<InterpreterInfo*>(data);
      }) {}

Graph NewGraph(ModelBuilder& builder) {
  InterpreterInfo& interpreter_info = Helper::GetInterpreterInfo(builder);
  const GraphInfo& graph = interpreter_info.NewGraph();
  return Helper::BuildGraph(interpreter_info, graph);
}

Buffer NewConstantBuffer(ModelBuilder& builder) {
  InterpreterInfo& interpreter_info = Helper::GetInterpreterInfo(builder);
  BufferInfo& buffer_info = interpreter_info.NewBuffer();
  return Helper::BuildBuffer(interpreter_info, buffer_info);
}

void ModelBuilder::Build(Interpreter& interpreter) {
  InterpreterInfo& interpreter_info = Helper::GetInterpreterInfo(*this);
  int first_new_subgraph_index = 0;
  interpreter.AddSubgraphs(interpreter_info.subgraphs.size(),
                           &first_new_subgraph_index);
  for (int i = 0; i < interpreter_info.subgraphs.size(); ++i) {
    auto&& a = interpreter_info.subgraphs[i];
    auto&& b = *interpreter.subgraph(i);
    Apply(interpreter_info, a, b);
  }
}

Tensor NewInput(Graph& g, TfLiteType type) {
  GraphInfo& graph = g.builder_->GetGraph(g.graph_idx_);
  const TensorInfo& tensor = graph.NewInput(type);
  return Helper::BuildTensor(*g.builder_, graph, tensor);
}

Tensor NewConstantTensor(InterpreterInfo& builder, GraphInfo& graph,
                         Buffer buffer) {
  BufferInfo& buffer_info = Helper::GetBufferInfo(buffer);
  TensorInfo& tensor = graph.NewTensor(buffer_info.type);
  tensor.buffer_idx = buffer_info.idx;
  return Helper::BuildTensor(builder, graph, tensor);
}

Tensor NewConstantTensor(Graph& g, Buffer buffer) {
  GraphInfo& graph = Helper::GetGraphInfo(g);
  return NewConstantTensor(*g.builder_, graph, buffer);
}

void MarkOutput(Tensor tensor) {
  GraphInfo& graph = Helper::GetGraphInfo(tensor);
  graph.outputs.push_back(Helper::GetTensorIndex(tensor));
}

void SetShape(Tensor tensor, std::vector<int> shape) {
  TensorInfo& tensor_info = Helper::GetTensorInfo(tensor);
  tensor_info.shape = std::move(shape);
}

namespace {

template <class T>
static OwningErasedPtr AllocateParam(T val) {
  OwningErasedPtr ptr(calloc(1, sizeof(T)),
                      [](void* data) { free(reinterpret_cast<T*>(data)); });
  *reinterpret_cast<T*>(ptr.get()) = val;
  return ptr;
}

OwningErasedPtr NoParam() {
  return OwningErasedPtr(nullptr, [](void*) {});
}

struct SameGraphResult {
  bool same_graph = false;
  const GraphInfo* const graph = nullptr;
  friend bool operator==(const SameGraphResult& lhs,
                         const SameGraphResult& rhs) {
    return lhs.same_graph == rhs.same_graph && lhs.graph == rhs.graph;
  }
};

// Checks that all elements in a container return the same `GraphInfo` for
// `Helper::GetGraphInfo()`.
template <class Container,
          class SFINAE = decltype(begin(std::declval<Container>()))>
SameGraphResult FromSameGraphImpl(const Container& container) {
  auto it = begin(container);
  const auto end_it = end(container);
  if (it == end_it) {
    return SameGraphResult{false, nullptr};
  }
  const GraphInfo* const graph = &Helper::GetGraphInfo(*it);
  for (; it != end_it; ++it) {
    if (graph != &Helper::GetGraphInfo(*it)) {
      return SameGraphResult{false, nullptr};
    }
  }
  return SameGraphResult{true, graph};
}

// Returns the `GraphInfo` associated to the `graph`.
SameGraphResult FromSameGraphImpl(const Graph& graph) {
  return {true, &Helper::GetGraphInfo(graph)};
}

// Returns the `GraphInfo` associated to the `tensor`.
SameGraphResult FromSameGraphImpl(const Tensor& tensor) {
  return {true, &Helper::GetGraphInfo(tensor)};
}

// Checks that all parameters are associated to the same `GraphInfo` object.
//
// An overload of `Helper::GetGraphInfo()` must exist for every argument type.
template <class T, class... Ts>
bool FromSameGraph(const T& first, const Ts&... args) {
  std::array res{FromSameGraphImpl(first), FromSameGraphImpl(args)...};
  return std::count(begin(res), end(res), res[0]) == res.size();
}

}  // namespace

Tensor UnaryOp(BuiltinOperator op, TfLiteRegistration registration,
               OwningErasedPtr params, const Tensor& input_tensor,
               TfLiteType output_type) {
  InterpreterInfo& interpreter_info = Helper::GetInterpreterInfo(input_tensor);
  GraphInfo& graph = Helper::GetGraphInfo(input_tensor);
  TensorInfo& output = graph.NewTensor(output_type);
  output.shape = Helper::GetTensorShape(input_tensor);
  registration.builtin_code = op;
  graph.ops.push_back(OpInfo{op,
                             std::move(params),
                             {Helper::GetTensorIndex(input_tensor)},
                             {output.idx},
                             registration});
  return Helper::BuildTensor(interpreter_info, graph, output);
}

Tensor BinaryOp(BuiltinOperator op, TfLiteRegistration registration,
                OwningErasedPtr params, const Tensor& lhs_tensor,
                const Tensor& rhs_tensor, TfLiteType output_type) {
  assert(FromSameGraph(lhs_tensor, rhs_tensor) &&
         "LHS and RHS are not from the same Graph.");
  InterpreterInfo& interpreter_info = Helper::GetInterpreterInfo(lhs_tensor);
  GraphInfo& graph = Helper::GetGraphInfo(lhs_tensor);
  const TensorInfo& output = graph.NewTensor(output_type);
  registration.builtin_code = op;
  graph.ops.push_back(OpInfo{
      op,
      std::move(params),
      {Helper::GetTensorIndex(lhs_tensor), Helper::GetTensorIndex(rhs_tensor)},
      {output.idx},
      registration});
  return Helper::BuildTensor(interpreter_info, graph, output);
}

Tensor Abs(Tensor tensor) {
  return UnaryOp(BuiltinOperator_ABS, *ops::builtin::Register_ABS(), NoParam(),
                 tensor, Helper::GetTensorType(tensor));
}

Tensor Add(Tensor lhs, Tensor rhs) {
  return BinaryOp(BuiltinOperator_ADD, *ops::builtin::Register_ADD(),
                  AllocateParam(TfLiteAddParams()), lhs, rhs,
                  Helper::GetTensorType(lhs));
}

Tensor Mul(Tensor lhs, Tensor rhs) {
  return BinaryOp(BuiltinOperator_MUL, *ops::builtin::Register_MUL(),
                  AllocateParam(TfLiteMulParams()), lhs, rhs,
                  Helper::GetTensorType(lhs));
}

Tensor Transpose(Tensor tensor, Tensor permutation) {
  Tensor output =
      BinaryOp(BuiltinOperator_TRANSPOSE, *ops::builtin::Register_TRANSPOSE(),
               NoParam(), tensor, permutation, Helper::GetTensorType(tensor));

  if (const BufferInfo* buffer_info = Helper::MaybeGetBufferInfo(permutation);
      buffer_info) {
    SetShape(output, buffer_info->shape);
  }

  return output;
}

Tensor FullyConnected(Tensor input, Buffer weights_buffer) {
  InterpreterInfo& interpreter_info = Helper::GetInterpreterInfo(input);
  GraphInfo& graph = Helper::GetGraphInfo(input);
  Tensor weights = NewConstantTensor(interpreter_info, graph, weights_buffer);
  const TensorInfo& input_info = Helper::GetTensorInfo(input);
  TensorInfo& output = graph.NewTensor(input_info.type);
  const std::vector<int> input_shape = Helper::GetTensorShape(input);
  const std::vector<int> weights_shape = Helper::GetTensorShape(weights);
  output.shape = {input_shape.at(0), weights_shape.at(0)};
  TfLiteRegistration registration = *ops::builtin::Register_FULLY_CONNECTED();
  registration.builtin_code = BuiltinOperator_FULLY_CONNECTED;
  OwningErasedPtr params = AllocateParam(TfLiteFullyConnectedParams());
  graph.ops.push_back(
      OpInfo{/*op=*/BuiltinOperator_FULLY_CONNECTED,
             /*params=*/std::move(params),
             /*inputs=*/
             {Helper::GetTensorIndex(input), Helper::GetTensorIndex(weights)},
             /*outputs=*/{output.idx},
             /*registration=*/registration});
  return Helper::BuildTensor(interpreter_info, graph, output);
}

std::vector<Tensor> StableHLOComposite(const char* name, const Graph& subgraph,
                                       const std::vector<Tensor>& inputs) {
  assert(FromSameGraph(inputs) && "inputs are not from the same Graph.");
  assert(!FromSameGraph(inputs, subgraph) && "inputs belong to the subgraph.");
  InterpreterInfo& interpreter_info = Helper::GetInterpreterInfo(subgraph);
  assert(!inputs.empty());
  std::vector<TensorIdx> input_indices;
  std::vector<TensorIdx> output_indices;
  std::vector<Tensor> outputs;
  GraphInfo& graph_info = Helper::GetGraphInfo(inputs[0]);
  GraphInfo& subgraph_info = Helper::GetGraphInfo(subgraph);

  output_indices.reserve(subgraph_info.outputs.size());
  for (const TensorIdx output_idx : subgraph_info.outputs) {
    const TensorInfo& out =
        graph_info.NewTensor(subgraph_info.tensors.at(output_idx.val).type);
    output_indices.push_back(out.idx);
    outputs.push_back(Helper::BuildTensor(interpreter_info, graph_info, out));
  }

  input_indices.reserve(inputs.size());
  for (const Tensor& t : inputs) {
    input_indices.push_back(Helper::GetTensorIndex(t));
  }

  const TfLiteStablehloCompositeParams params{
      .name = name,
      .subgraph_index = static_cast<int32_t>(subgraph_info.idx.val),
      .version = 1};

  OpInfo op{.op = BuiltinOperator_STABLEHLO_COMPOSITE,
            .params = AllocateParam(params),
            .inputs = input_indices,
            .outputs = output_indices,
            .registration = *ops::builtin::Register_STABLEHLO_COMPOSITE()};
  op.registration.builtin_code = BuiltinOperator_STABLEHLO_COMPOSITE;

  graph_info.ops.push_back(std::move(op));

  return outputs;
}

}  // namespace model_builder
}  // namespace tflite
