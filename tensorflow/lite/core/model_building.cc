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

#include <array>
// TODO: Change assert to TFLITE logging.
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace model_builder {

namespace internal {

template <typename H, class T, class Tag>
H AbslHashValue(H h, const StrongType<T, Tag>& v) {
  return H::combine(std::move(h), v.val);
}

template <class T, class Tag>
bool operator==(const StrongType<T, Tag>& lhs, const StrongType<T, Tag>& rhs) {
  return lhs.val == rhs.val;
}

}  // namespace internal

struct TensorInfo {
  TensorInfo() = default;
  // This object holds information that is part of a graph. A inadvertant copy
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
  internal::TensorIdx idx;
  TfLiteType type;
};

struct OpInfo {
  BuiltinOperator op;
  OwningErasedPtr params;
  // Indices in GraphInfo tensors.
  std::vector<internal::TensorIdx> inputs;
  // Indices in GraphInfo tensors.
  std::vector<internal::TensorIdx> outputs;
  TfLiteRegistration registration;
};

struct GraphInfo {
  GraphInfo() = default;
  // This object holds information that is part of a graph. A inadvertant copy
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
  internal::GraphIdx idx;
  std::vector<OpInfo> ops;
  std::vector<TensorInfo> tensors;
  // Indices in tensors.
  std::vector<internal::TensorIdx> inputs;
  // Indices in tensors.
  std::vector<internal::TensorIdx> outputs;

  enum TensorRole {
    kNone,
    kInput = 1,
    kOutput = 1 << 1,
  };

  const TensorInfo& NewTensor(TfLiteType type,
                              const int role = TensorRole::kNone) {
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
    const internal::TensorIdx idx = outputs.at(i);
    return tensors.at(idx.val);
  }
};

struct InterpreterInfo {
  std::vector<GraphInfo> subgraphs;

  GraphInfo& GetGraph(internal::GraphIdx graph_idx) {
    return subgraphs.at(graph_idx.val);
  }

  // Adds a new graph to the interpreter.
  //
  // Calling this function may invalidate existing references (to GraphInfo
  // objects, Graph object are fine as they only hold the graph index).
  GraphInfo& NewGraph() {
    const internal::GraphIdx idx(subgraphs.size());
    subgraphs.emplace_back();
    subgraphs.back().idx = idx;
    return subgraphs.back();
  }
};

void Apply(GraphInfo& graph, Subgraph& subgraph) {
  // Maps graph indices to subgraph indices.
  absl::flat_hash_map<internal::TensorIdx, int> tensor_id_to_idx;
  int first_new_tensor_index;
  subgraph.AddTensors(graph.tensors.size(), &first_new_tensor_index);
  for (int i = 0; i < graph.tensors.size(); ++i) {
    tensor_id_to_idx.insert({graph.tensors[i].idx, i + first_new_tensor_index});
  }

  auto GetTensorIds = [&](const std::vector<internal::TensorIdx>& ts) {
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
  for (internal::TensorIdx i(0); i.val < graph.tensors.size(); ++i.val) {
    if (subgraph.SetTensorParametersReadWrite(
            tensor_id_to_idx[i], graph.tensors[i.val].type, "", 0, nullptr, {},
            false) != kTfLiteOk) {
      std::terminate();
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

  static const TensorInfo& GetTensorInfo(const Tensor& a) {
    InterpreterInfo& ii = Helper::GetInterpreterInfo(a);
    return ii.subgraphs.at(a.graph_idx_.val).tensors.at(a.tensor_idx_.val);
  }

  static internal::TensorIdx GetTensorIndex(const Tensor& a) {
    return a.tensor_idx_;
  }

  static TfLiteType GetTensorType(const Tensor& a) {
    return Helper::GetTensorInfo(a).type;
  }

  static Graph BuildGraph(InterpreterInfo& interpreter_info,
                          const GraphInfo& graph) {
    return Graph(&interpreter_info, graph.idx);
  }

  static Tensor BuildTensor(InterpreterInfo& interpreter_info,
                            const GraphInfo& graph, const TensorInfo& tensor) {
    return Tensor(&interpreter_info, tensor.idx, graph.idx);
  }
};

ModelBuilder::ModelBuilder()
    : impl_(new InterpreterInfo(), [](void* data) {
        delete reinterpret_cast<InterpreterInfo*>(data);
      }) {}

Graph NewGraph(ModelBuilder& builder) {
  InterpreterInfo& interpreter_info = Helper::GetInterpreterInfo(builder);
  const GraphInfo& graph = interpreter_info.NewGraph();
  return Helper::BuildGraph(interpreter_info, graph);
}

void ModelBuilder::Build(Interpreter& interpreter) {
  InterpreterInfo& interpreter_info = Helper::GetInterpreterInfo(*this);
  int first_new_subgraph_index = 0;
  interpreter.AddSubgraphs(interpreter_info.subgraphs.size(),
                           &first_new_subgraph_index);
  for (int i = 0; i < interpreter_info.subgraphs.size(); ++i) {
    Apply(interpreter_info.subgraphs[i], *interpreter.subgraph(i));
  }
}

Tensor NewInput(Graph& g, TfLiteType type) {
  GraphInfo& graph = g.builder_->GetGraph(g.graph_idx_);
  const TensorInfo& tensor = graph.NewInput(type);
  return Helper::BuildTensor(*g.builder_, graph, tensor);
}

void MarkOutput(Tensor tensor) {
  GraphInfo& graph = Helper::GetGraphInfo(tensor);
  graph.outputs.push_back(Helper::GetTensorIndex(tensor));
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
  const TensorInfo& output = graph.NewTensor(output_type);
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

Tensor Add(Tensor lhs, Tensor brhs) {
  return BinaryOp(BuiltinOperator_ADD, *ops::builtin::Register_ADD(),
                  AllocateParam(TfLiteAddParams()), lhs, brhs,
                  Helper::GetTensorType(lhs));
}

Tensor Mul(Tensor alhs, Tensor rhs) {
  return BinaryOp(BuiltinOperator_MUL, *ops::builtin::Register_MUL(),
                  AllocateParam(TfLiteMulParams()), alhs, rhs,
                  Helper::GetTensorType(alhs));
}

Tensor Transpose(Tensor tensor, Tensor permutation) {
  return BinaryOp(BuiltinOperator_TRANSPOSE,
                  *ops::builtin::Register_TRANSPOSE(), NoParam(), tensor,
                  permutation, Helper::GetTensorType(tensor));
}

std::vector<Tensor> StableHLOComposite(const char* name, const Graph& subgraph,
                                       const std::vector<Tensor>& inputs) {
  assert(FromSameGraph(inputs) && "inputs are not from the same Graph.");
  assert(!FromSameGraph(inputs, subgraph) && "inputs belong to the subgraph.");
  InterpreterInfo& interpreter_info = Helper::GetInterpreterInfo(subgraph);
  assert(!inputs.empty());
  std::vector<internal::TensorIdx> input_indices;
  std::vector<internal::TensorIdx> output_indices;
  std::vector<Tensor> outputs;
  GraphInfo& graph_info = Helper::GetGraphInfo(inputs[0]);
  GraphInfo& subgraph_info = Helper::GetGraphInfo(subgraph);

  output_indices.reserve(subgraph_info.outputs.size());
  for (const internal::TensorIdx output_idx : subgraph_info.outputs) {
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
