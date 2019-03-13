/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPH_BENCHMARK_TESTLIB_H_
#define TENSORFLOW_CORE_GRAPH_BENCHMARK_TESTLIB_H_

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"

namespace tensorflow {
namespace test {

REGISTER_OP("Input").Output("y: float");
REGISTER_OP("Output")
    .Input("x: N * float")
    .Attr("N: int >= 1")
    .Output("y: float");
REGISTER_OP("In2Out1").Input("a: float").Input("b: float").Output("y: float");
REGISTER_OP("In4Out1")
    .Input("a: float")
    .Input("b: float")
    .Input("c: float")
    .Input("d: float")
    .Output("y: float");
REGISTER_OP("In8Out1")
    .Input("a: float")
    .Input("b: float")
    .Input("c: float")
    .Input("d: float")
    .Input("e: float")
    .Input("f: float")
    .Input("g: float")
    .Input("h: float")
    .Output("y: float");
REGISTER_OP("In16Out1")
    .Input("a: float")
    .Input("b: float")
    .Input("c: float")
    .Input("d: float")
    .Input("e: float")
    .Input("f: float")
    .Input("g: float")
    .Input("h: float")
    .Input("i: float")
    .Input("j: float")
    .Input("k: float")
    .Input("l: float")
    .Input("m: float")
    .Input("n: float")
    .Input("o: float")
    .Input("p: float")
    .Output("y: float");

GraphDef CreateGraphDef(int num_nodes, int num_edges_per_node) {
  const int kNumInNodes = 10 * num_edges_per_node;
  string s;
  for (int in = 0; in < kNumInNodes; in++) {
    s += absl::PrintF("node { name: 'in%04d' op: 'Input' }", in);
  }
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  for (int op = 0; op < num_nodes; op++) {
    s += absl::PrintF("node { name: 'op%05d' op: 'In%dOut1' input: [ ", op,
                      num_edges_per_node);
    for (int edge = 0; edge < num_edges_per_node - 1; ++edge) {
      s += absl::PrintF("'in%04d', ", rnd.Uniform(kNumInNodes));
    }
    s += absl::PrintF("'in%04d' ] } ", rnd.Uniform(kNumInNodes));
  }
  // Add a single sink node. Otherwise a lot of time is spent in
  // FixupSourceAndSinkEdges().
  s += absl::PrintF("node { name: 'out' op: 'Output' input: [ ");
  for (int op = 0; op < num_nodes - 1; op++) {
    s += absl::PrintF("'op%05d', ", op);
  }
  s += absl::PrintF("'op%05d' ], attr: { key: 'N' value { i: %d } } } ",
                    num_nodes - 1, num_nodes);
  GraphDef graph_def;
  CHECK(protobuf::TextFormat::ParseFromString(s, &graph_def));
  return graph_def;
}

GraphDef CreateRandomGraph(int size) {
  random::PhiloxRandom philox(0x12345);
  random::SimplePhilox rnd(&philox);

  string prefix = "long_node_name_prefix_to_measure_string_copy_overhead";

  GraphDef graph;
  for (int i = 0; i < size; ++i) {
    const string name = absl::StrCat(prefix, i);
    const uint32 num_inputs = rnd.Uniform(std::min(i, 5));

    NodeDef node;
    node.set_name(name);
    for (int n = 0; n < num_inputs; ++n) {
      const uint32 input_node = rnd.Uniform(i);
      node.add_input(absl::StrCat(prefix, input_node));
    }

    *graph.add_node() = std::move(node);
  }

  return graph;
}

}  // namespace test
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_BENCHMARK_TESTLIB_H_
