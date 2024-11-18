// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expruns or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Utility types for mapping LiteRt IR to arbitrary backend specific
// types. Implementations of these types define mapping for ops and tensors
// that may be used in a stndalone fashion. They also may be composed
// to create lowerings of entire graphs with topology.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_EXAMPLES_EXAMPLE_CONVERT_TYPES_IMPL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_EXAMPLES_EXAMPLE_CONVERT_TYPES_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/litert_convert.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/litert_convert_types.h"

namespace litert {
namespace example {

// Example implementations for litert_convert_types used for testing and
// edification.

// Dummy tensor type definition.
using Type = std::string;

// Example backend op type.
struct ExampleOp {
  int code;
  std::vector<Type> input_types;
  std::vector<Type> output_types;
};

// Example backend tensor type.
struct ExampleTensor {
  Type type;
};

// Example tensor conversion logic (litert tensor -> backend tensor).
inline Expected<ExampleTensor> ExampleConvertTensor(
    const Tensor& litert_tensor) {
  return ExampleTensor{
      std::string(litert_tensor.Name().begin(), litert_tensor.Name().end())};
}

// Example op legalization implementation.
template <LiteRtOpCode OpCode = kLiteRtOpCodeTflCustom>
class ExampleOpLegalization : public Legalization<ExampleOp, ExampleTensor> {
 public:
  using Base::Result;
  using Legalization::Base;
  using Ptr = std::unique_ptr<ExampleOpLegalization>;
  using PtrVec = std::vector<Ptr>;

  // Build the legalization.
  static Ptr Create() { return std::make_unique<ExampleOpLegalization>(); }

  // Get the op to match on.
  LiteRtOpCode OpToMatch() const override { return OpCode; }

  ExampleOpLegalization() : Legalization(ExampleConvertTensor) {}

 private:
  // Convert the given litert op to backend op.
  Expected<Result> Convert(const Op& litert_op, TensorVec& inputs,
                           TensorVec& outputs) override {
    if (inputs.empty()) {
      return Expected<Result>(NoMatch{});
    }
    ExampleOp result;
    result.code = static_cast<int>(litert_op.Code());
    for (auto& input : inputs) {
      result.input_types.push_back(
          std::string(input.type.begin(), input.type.end()));
    }
    for (auto& output : outputs) {
      result.output_types.push_back(
          std::string(output.type.begin(), output.type.end()));
    }
    return Expected<Result>(result);
  }
};

// We can simply alias the legalizer to use with our op defs.
using ExampleLegalizer = Legalizer<ExampleOp, ExampleTensor>;

// Example implementation of compiler capability check.
inline bool ExampleCapability(ExampleOp backend_op) {
  return backend_op.code == kLiteRtOpCodeTflMul;
}

// Example "context" used to store state during graph construction.
struct ExampleGraphContext {
  // Context information for a single converted subgraph partition.
  struct Info {
    // All converted ops.
    std::vector<ExampleOp> backend_ops;

    // The names of all tensors in the subgraph.
    std::vector<Type> tensor_names;

    // Tag whether or not graph has been finalized.
    bool finalized = false;

    // The name the graph was initialized with.
    std::string name;
  } info;

  // List of info for each graph converted.
  std::vector<Info> infos;

  // Get the info for the current subgraph.
  Info& Cur() { return infos.back(); }
};

// Example tensor finalizer factory that captures converted tensor names.
inline TensorFinalizer<ExampleTensor> MakeExampleTensorFinalizer(
    ExampleGraphContext& graph_context) {
  return [&](ExampleTensor& backend_tensor) -> LiteRtStatus {
    graph_context.Cur().tensor_names.push_back(backend_tensor.type);
    return kLiteRtStatusOk;
  };
}

// Example op finalizer factory that captures converted op code.
inline OpFinalizer<ExampleOp> MakeExampleOpFinalizer(
    ExampleGraphContext& graph_context) {
  return [&](ExampleOp& backend_op) -> LiteRtStatus {
    graph_context.Cur().backend_ops.push_back(backend_op);
    return kLiteRtStatusOk;
  };
}

// Example graph initializer factory that adds the name to the graph context.
inline GraphInitializer MakeExampleGraphInitializer(
    ExampleGraphContext& graph_context) {
  return [&](absl::string_view name) -> LiteRtStatus {
    graph_context.infos.emplace_back();
    graph_context.Cur().name.assign(name.begin(), name.end());
    return kLiteRtStatusOk;
  };
}

// Example graph finalizer factory that tags context as being finalized.
inline GraphFinalizer MakeExampleGraphFinalizer(
    ExampleGraphContext& graph_context) {
  return [&]() -> LiteRtStatus {
    graph_context.Cur().finalized = true;
    return kLiteRtStatusOk;
  };
}

// Specialization of GraphConverter for example IR.
using ExampleGraphConverter = GraphConverter<ExampleOp, ExampleTensor>;

}  // namespace example
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_EXAMPLES_EXAMPLE_CONVERT_TYPES_IMPL_H_
