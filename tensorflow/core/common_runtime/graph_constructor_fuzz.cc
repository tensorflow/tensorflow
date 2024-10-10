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

#include <fuzzer/FuzzedDataProvider.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow::fuzzing {
namespace {

void FuzzImportGraphDef(const GraphDef& graph_def) {
  ImportGraphDefOptions options;
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  ImportGraphDef(options, graph_def, graph.get(), nullptr, nullptr)
      .IgnoreError();
}
FUZZ_TEST(GraphDefFuzz, FuzzImportGraphDef);

void FuzzGraphEndToEndSimpleFixedInput(const GraphDef& graph_def) {
  // Load an arbitrary graph and run a session on it using simple input.
  ImportGraphDefOptions options;
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  absl::Status status =
      ImportGraphDef(options, graph_def, graph.get(), nullptr, nullptr);
  if (!status.ok()) {
    return;
  }
  GraphDef gdef;
  graph->ToGraphDef(&gdef);
  SessionOptions sess_options;
  std::unique_ptr<Session> sess =
      std::unique_ptr<Session>(NewSession(sess_options));
  status = sess->Create(gdef);
  if (!status.ok()) {
    return;
  }

  // Use the same input for each fuzz iteration. The benefit of this is the
  // fuzzer will focus on exploring graphs that match this input, which
  // gives it a more narrow space to search, as opposed to any given input.
  Tensor p1(DT_FLOAT, TensorShape({1}));
  p1.scalar<float>()() = 1.0;
  Tensor p2(DT_FLOAT, TensorShape({1}));
  p2.scalar<float>()() = 2.0;
  std::vector<std::pair<string, Tensor>> inputs = {{"Placeholder", p1},
                                                   {"Placeholder_1", p2}};
  std::vector<string> output_names = {"O_FUZZ"};
  std::vector<string> target_names;
  std::vector<Tensor> outputs;
  status = sess->Run(inputs, output_names, target_names, &outputs);
}
FUZZ_TEST(GraphDefFuzz, FuzzGraphEndToEndSimpleFixedInput);

void FuzzGraphEndToEndAllStatic(const GraphDef& graph_def) {
  // Load an arbitrary graph and run a session on it. No input or output is
  // provided and the reason is we aim for the graph itself to embed all
  // values needed for the computations. In this sense we enable the fuzzer
  // to explore any arbitrary graph computation.
  ImportGraphDefOptions options;
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  absl::Status status =
      ImportGraphDef(options, graph_def, graph.get(), nullptr, nullptr);
  if (!status.ok()) {
    return;
  }
  GraphDef gdef;
  graph->ToGraphDef(&gdef);
  SessionOptions sess_options;
  auto sess = std::unique_ptr<Session>(NewSession(sess_options));
  status = sess->Create(gdef);
  if (!status.ok()) {
    return;
  }

  std::vector<std::pair<string, Tensor>> inputs = {};
  std::vector<string> output_names = {};
  std::vector<string> target_names = {};
  std::vector<Tensor> outputs = {};
  status = sess->Run(inputs, output_names, target_names, &outputs);
}
FUZZ_TEST(GraphDefFuzz, FuzzGraphEndToEndAllStatic);

Node* FindNode(const string& name, Graph* graph) {
  for (Node* n : graph->nodes()) {
    if (n->name() == name) return n;
  }
  return nullptr;
}

bool HasNode(const string& name, Graph* graph) {
  return FindNode(name, graph) != nullptr;
}

class EmptyErrorCollector : public protobuf::io::ErrorCollector {
 public:
  EmptyErrorCollector() {}
  ~EmptyErrorCollector() override {}
  void AddError(int line, int column, const std::string& message) override {
    // log error
  }
  void AddWarning(int line, int column, const std::string& message) override {
    // log warning
  }
};

std::vector<std::string> ops = {
    "Abs",
    "Acos",
    "Acosh",
    "Add",
    "AddV2",
    "Asin",
    "Asinh",
    "Atan",
    "Atan2",
    "Atanh",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "Ceil",
    "ClipByValue",
    "ComplexAbs",
    "Conj",
    "Cos",
    "Cosh",
    "Cross",
    "Digamma",
    "Div",
    "DivNoNan",
    "Equal",
    "Erf",
    "Erfc",
    "Erfinv",
    "Exp",
    "Expint",
    "Expm1",
    "Floor",
    "FloorDiv",
    "FloorMod",
    "FresnelCos",
    "FresnelSin",
    "Greater",
    "GreaterEqual",
    "Igamma",
    "Inv",
    "Invert",
    "IsFinite",
    "IsInf",
    "LeftShift",
    "Less",
    "LessEqual",
    "Lgamma",
    "Log1p",
    "Maximum",
    "Minimum",
    "Mod",
    "Mul",
    "MulNoNan",
    "Ndtri",
    "Neg",
    "NextAfter",
    "NotEqual",
    "Polygamma",
    "Pow",
    "RandomGammaGrad",
    "RealDiv",
    "Reciprocal",
    "RightShift",
    "Rint",
    "Round",
    "Rsqrt",
    "RsqrtGrad",
    "Select",
    "SelectV2",
    "Sigmoid",
    "SigmoidGrad",
    "Sign",
    "Sin",
    "Sinh",
    "Sqrt",
    "SqrtGrad",
    "Square",
    "Sub",
    "Tan",
    "TanhGrad",
    "TruncateDiv",
    "TruncateMod",
    "Xdivy",
    "Xlog1py",
    "Xlogy",
    "Zeta",
};

std::vector<std::string> types = {"DT_INT32", "DT_FLOAT"};

std::string generate_node(const std::string name, const std::string op,
                          const std::string type,
                          std::vector<std::string> inputs,
                          bool includeAttrValue, std::string keytype) {
  std::string node_start = "";
  std::string node_inputs = "";
  std::string attr_key = "";
  std::string attr_value = "";
  std::string node_end = "";

  node_start += std::string(
      "node {\n"
      "  name: \"" +
      std::string(name) +
      "\"\n"
      "  op: \"" +
      std::string(op) + "\"\n");

  // Go through all inputs
  node_inputs += "";
  for (auto& input : inputs) {
    node_inputs += "  input: \"" + input + "\"\n";
  }

  attr_key += std::string(
      "  attr {\n"
      "    key: \"" +
      std::string(keytype) +
      "\"\n"
      "    value: {\n"
      "      type: " +
      std::string(type) +
      "\n"
      "    }\n"
      "  }\n");

  if (includeAttrValue) {
    attr_value += std::string(
        "  attr {\n"
        "    key: \"shape\"\n"
        "    value {\n"
        "      shape {\n"
        "        unknown_rank: true\n"
        "      }\n"
        "    }\n"
        "  }\n");
  }

  node_end += "}\n";

  return node_start + node_inputs + attr_key + attr_value + node_end;
}

void FuzzGraphEndToEndFDP(std::vector<uint8_t> data) {
  // Fuzzer that assembles a graph that has a high chance of being a working
  // graph. Specifically, the nodes are connected to each other in terms of
  // naming, and the string representing ops are actual ops from Tensorflow.
  // Types (DT_FLOAT) is not necessarily compatible in the graph, although many
  // of the generated graph will be well defined in that sense.
  // TODOs:
  // - Extend to all possible ops.
  // - Extend to all types (currently only DT_FLOAT and DT_INT32 are used).
  //   - I think complex types are particularly useful as they will open up
  //     for more ops.
  // - Extend the possible set of inputs, in particular in terms of structure.
  //
  // It would be smart to approach the above TODOs in a manner that keeps
  // the fuzzer so it has a high chance of making valid graphs. For example,
  // when extending types it may become smart to implement features for
  // mapping ops to the set of input types they accept, and then ensure
  // only such types are created in the graph.
  FuzzedDataProvider fdp(data.data(), data.size());

  std::vector<std::string> names_used;
  std::string graphFdp = "";

  // Create a set of nodes for the graph
  // Max number of nodes in the graph
  int MAX_NODES = 8;

  // The actual number of nodes in the graph
  int nodes_in_graph = fdp.ConsumeIntegralInRange<int>(3, MAX_NODES);

  // Add initial placeholders
  graphFdp +=
      generate_node("N0", "Placeholder",
                    types[fdp.ConsumeIntegralInRange<int>(0, types.size() - 1)],
                    {}, true, "dtype");
  graphFdp +=
      generate_node("N1", "Placeholder",
                    types[fdp.ConsumeIntegralInRange<int>(0, types.size() - 1)],
                    {}, true, "dtype");

  names_used.push_back("N0");
  names_used.push_back("N1");

  // Create all the nodes in the graph that will do computations.
  // We start at the third node because we've already added 2.
  std::string last_node = "";
  for (int i = 2; i < nodes_in_graph; i++) {
    std::string name = "N" + std::to_string(i);
    last_node = name;
    std::vector<std::string> inputs;

    bool should_include_inputs = fdp.ConsumeBool();
    if (should_include_inputs) {
      int inputs_to_include = fdp.ConsumeIntegralInRange<int>(1, 3);
      for (int j = 0; j < inputs_to_include; j++) {
        inputs.push_back(names_used[fdp.ConsumeIntegralInRange<int>(
            0, names_used.size() - 1)]);
      }
    }

    std::string op = ops[fdp.ConsumeIntegralInRange<int>(0, ops.size() - 1)];
    std::string type =
        types[fdp.ConsumeIntegralInRange<int>(0, types.size() - 1)];

    graphFdp += generate_node(name, op, type, inputs, false, "T");
    // Add the name of the node to the used nodes.
    names_used.push_back(name);
  }

  graphFdp += "versions { producer: 21 min_consumer: 12 }";

  // For debugging
  // std::cout << ">>>>>>>>>>>>>>>>>>>>>>>\n";
  // std::cout << graphFdp;
  // std::cout << "<<<<<<<<<<<<<<<<<<<<<<<\n";

  // Convert the ASCII graph to an actual graph
  GraphDef gdef_;
  ImportGraphDefOptions opts;
  EmptyErrorCollector emptyErrorCollector;
  protobuf::TextFormat::Parser parser;
  parser.RecordErrorsTo(&emptyErrorCollector);
  bool parsed = parser.ParseFromString(graphFdp, &gdef_);
  if (!parsed) {
    return;
  }

  std::unique_ptr<Graph> graph = std::make_unique<Graph>(OpRegistry::Global());
  absl::Status s = ImportGraphDef(opts, gdef_, graph.get(), nullptr, nullptr);
  if (!s.ok()) {
    return;
  }

  // Ensure at this point we actually do have our placeholder nodes and our last
  // node.
  if (!HasNode("N0", graph.get())) return;
  if (!HasNode("N1", graph.get())) return;
  if (!HasNode(last_node, graph.get())) return;

  // Create a session with our graph.
  GraphDef gdef;
  graph->ToGraphDef(&gdef);
  std::unique_ptr<Session> sess(NewSession(SessionOptions()));
  s = sess->Create(gdef);
  if (!s.ok()) {
    return;
  }

  // Create input tensors and run the session.
  // TODO: Add more options here, although probably not make it too general or
  // large.
  std::vector<Tensor> input_tensors;
  for (int i = 0; i < 2; i++) {
    int type_choice = fdp.ConsumeIntegral<int>();

    Tensor input_tensor;
    if (type_choice % 4 == 0) {
      input_tensor = Tensor(DT_FLOAT, TensorShape({2}));
      input_tensor.vec<float>()(0) = fdp.ConsumeFloatingPoint<float>();
      input_tensor.vec<float>()(1) = fdp.ConsumeFloatingPoint<float>();
    } else if (type_choice % 4 == 1) {
      input_tensor = Tensor(DT_INT32, TensorShape({1}));
      input_tensor.scalar<int>()() = fdp.ConsumeIntegral<int>();
    } else if (type_choice % 4 == 2) {
      input_tensor = Tensor(DT_FLOAT, TensorShape({1}));
      input_tensor.scalar<float>()() = fdp.ConsumeFloatingPoint<float>();
    } else if (type_choice % 4 == 3) {
      input_tensor = Tensor(DT_FLOAT, TensorShape({2, 2}));
      input_tensor.matrix<float>()(0, 0) = fdp.ConsumeFloatingPoint<float>();
      input_tensor.matrix<float>()(0, 1) = fdp.ConsumeFloatingPoint<float>();
      input_tensor.matrix<float>()(1, 0) = fdp.ConsumeFloatingPoint<float>();
      input_tensor.matrix<float>()(1, 1) = fdp.ConsumeFloatingPoint<float>();
    }
    input_tensors.push_back(input_tensor);
  }

  std::vector<std::pair<string, Tensor>> inputs = {{"N0", input_tensors[0]},
                                                   {"N1", input_tensors[1]}};
  std::vector<string> output_names = {last_node};
  std::vector<string> target_names;
  std::vector<Tensor> outputs;
  s = sess->Run(inputs, output_names, target_names, &outputs);
  if (!s.ok()) {
    return;
  }
}
FUZZ_TEST(GraphDefFuzz, FuzzGraphEndToEndFDP);

}  // namespace
}  // namespace tensorflow::fuzzing
