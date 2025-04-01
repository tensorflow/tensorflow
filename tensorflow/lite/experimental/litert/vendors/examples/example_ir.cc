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
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/vendors/examples/example_ir.h"

#include <ostream>
#include <string>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"

namespace litert::example {

namespace {

template <class It>
void PrintWithCommas(It start, It end, std::ostream& out) {
  for (auto it = start; it < end; ++it) {
    out << std::to_string(*it);
    if (it != end - 1) {
      out << ", ";
    }
  }
}

}  // namespace

LiteRtStatus ExampleGraphBuilder::RegisterOp(ExampleOp& op) {
  switch (op.op_code) {
    case ExampleOpType::ADD:
      example_graph_ << "ADD";
      break;
    case ExampleOpType::MUL:
      example_graph_ << "MUL";
      break;
    case ExampleOpType::RELU:
      example_graph_ << "RELU";
      break;
  }
  example_graph_ << "(";
  PrintWithCommas(op.inputs.cbegin(), op.inputs.cend(), example_graph_);
  example_graph_ << ")->(";
  PrintWithCommas(op.outputs.cbegin(), op.outputs.cend(), example_graph_);
  example_graph_ << ")";
  return kLiteRtStatusOk;
}

LiteRtStatus ExampleGraphBuilder::RegisterTensor(ExampleTensor& tensor) {
  example_graph_ << std::to_string(tensor.id);
  switch (tensor.type) {
    case ExampleTensorType::FLOAT:
      example_graph_ << "FLOAT";
      break;
    case ExampleTensorType::INT:
      example_graph_ << "INT";
      break;
  }
  example_graph_ << "[";
  PrintWithCommas(tensor.dims.cbegin(), tensor.dims.cend(), example_graph_);
  example_graph_ << "]";
  return kLiteRtStatusOk;
}

LiteRtStatus ExampleGraphBuilder::FinalizeGraph() {
  example_graph_ << "FINALIZED";
  return kLiteRtStatusOk;
}

void ExampleGraphBuilder::InitGraph(std::string graph_name) {
  example_graph_ << "name=" << graph_name << "\n";
}

std::string ExampleGraphBuilder::Serialize() const {
  return example_graph_.str();
}

}  // namespace litert::example
