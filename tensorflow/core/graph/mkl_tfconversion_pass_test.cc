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

#ifdef INTEL_MKL

#include "tensorflow/core/graph/mkl_tfconversion_pass.h"
#include "tensorflow/core/graph/mkl_graph_util.h"

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class MklToTfConversionPass : public ::testing::Test {
 public:
  MklToTfConversionPass() : graph_(OpRegistry::Global()) {}

  static void InitGraph(const string& s, Graph* graph) {
    GraphDef graph_def;

    auto parser = protobuf::TextFormat::Parser();
    CHECK(parser.MergeFromString(s, &graph_def)) << s;
    GraphConstructorOptions opts;
    TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, graph));
  }

  void InitGraph(const string& s) {
    InitGraph(s, &graph_);
    original_ = CanonicalGraphString(&graph_);
  }

  static bool IncludeNode(const Node* n) { return n->IsOp(); }

  static string EdgeId(const Node* n, int index) {
    if (index == 0) {
      return n->name();
    } else if (index == Graph::kControlSlot) {
      return strings::StrCat(n->name(), ":control");
    } else {
      return strings::StrCat(n->name(), ":", index);
    }
  }

  string CanonicalGraphString(Graph* g) {
    std::vector<string> nodes;
    std::vector<string> edges;
    for (const Node* n : g->nodes()) {
      if (IncludeNode(n)) {
        nodes.push_back(strings::StrCat(n->name(), "(", n->type_string(), ")"));
      }
    }
    for (const Edge* e : g->edges()) {
      if (IncludeNode(e->src()) && IncludeNode(e->dst())) {
        edges.push_back(strings::StrCat(EdgeId(e->src(), e->src_output()), "->",
                                        EdgeId(e->dst(), e->dst_input())));
      }
    }
    // Canonicalize
    std::sort(nodes.begin(), nodes.end());
    std::sort(edges.begin(), edges.end());
    return strings::StrCat(str_util::Join(nodes, ";"), "|",
                           str_util::Join(edges, ";"));
  }

  string DoRunMklToTfConversionPass() {
    string before = CanonicalGraphString(&graph_);
    LOG(ERROR) << "Before MklToTf conversion pass: " << before;

    std::unique_ptr<Graph>* ug = new std::unique_ptr<Graph>(&graph_);
    InsertMklToTfConversionNodes(ug);

    string result = CanonicalGraphString(&graph_);
    LOG(ERROR) << "After MklToTf conversion pass:  " << result;
    return result;
  }

  const string& OriginalGraph() const { return original_; }

  Graph graph_;
  string original_;
};

REGISTER_OP("Input").Output("o: float").SetIsStateful();
REGISTER_OP("HalfInput").Output("o: half").SetIsStateful();
REGISTER_OP("_MklInput").Output("o: uint8").SetIsStateful();

TEST_F(MklToTfConversionPass, Basic) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoRunMklToTfConversionPass(),
            "A(Input);B(Input);C(Mul);D(Mul)|"
            "A->C;A->D;B->C:1;B->D:1");
}

// MklConv2D followed by Non-Mkl layer
// C=MklConv2D(A,M,B,N); E=Sub(C,D) (for interleaved ordering)
// C=MklConv2D(A,B,M,N); E=Sub(C,D) (for contiguous ordering)
TEST_F(MklToTfConversionPass, Positive) {
  if (kTensorOrdering == MklTfTensorOrdering::TENSORS_INTERLEAVED) {
    InitGraph(
        "node { name: 'A' op: 'Input'}"
        "node { name: 'M' op: '_MklInput'}"
        "node { name: 'B' op: 'Input'}"
        "node { name: 'N' op: '_MklInput'}"
        "node { name: 'C' op: '_MklConv2D'"
        " attr { key: 'T'                value { type: DT_FLOAT } }"
        " attr { key: 'data_format'      value { s: 'NCHW' } }"
        " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } "
        "}"
        " attr { key: 'padding'          value { s: 'SAME' } }"
        " input: ['A', 'M', 'B', 'N']}"
        "node { name: 'D' op: 'Input'}"
        "node { name: 'E' op: 'Sub'"
        " attr {key: 'T'                 value { type: DT_FLOAT } }"
        " input: ['C', 'D']}");
    EXPECT_EQ(DoRunMklToTfConversionPass(),
              "A(Input);B(Input);C(_MklConv2D);D(Input);E(Sub);M(_MklInput);"
              "Mkl2Tf/_0(_MklToTf);N(_MklInput)|A->C;B->C:2;C->Mkl2Tf/_0;"
              "C:1->Mkl2Tf/_0:1;D->E:1;M->C:1;Mkl2Tf/_0->E;N->C:3");
  } else {
    CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);
    InitGraph(
        "node { name: 'A' op: 'Input'}"
        "node { name: 'B' op: 'Input'}"
        "node { name: 'M' op: '_MklInput'}"
        "node { name: 'N' op: '_MklInput'}"
        "node { name: 'C' op: '_MklConv2D'"
        " attr { key: 'T'                value { type: DT_FLOAT } }"
        " attr { key: 'data_format'      value { s: 'NCHW' } }"
        " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } "
        "}"
        " attr { key: 'padding'          value { s: 'SAME' } }"
        " input: ['A', 'B', 'M', 'N']}"
        "node { name: 'D' op: 'Input'}"
        "node { name: 'E' op: 'Sub'"
        " attr {key: 'T'                 value { type: DT_FLOAT } }"
        " input: ['C', 'D']}");
    EXPECT_EQ(DoRunMklToTfConversionPass(),
              "A(Input);B(Input);C(_MklConv2D);D(Input);E(Sub);M(_MklInput);"
              "Mkl2Tf/_0(_MklToTf);N(_MklInput)|A->C;B->C:1;C->Mkl2Tf/_0;"
              "C:2->Mkl2Tf/_0:1;D->E:1;M->C:2;Mkl2Tf/_0->E;N->C:3");
  }
}

// MklConv2D followed by MklToTf op followed by Non-Mkl layer.
// C=MklConv2D(A,M,B,N); D=MklToTf(C:0, C:1) F=Sub(D,E) (for interleaved)
// C=MklConv2D(A,B,M,N); D=MklToTf(C:0, C:2) F=Sub(D,E) (for contiguous)
// MklToTf node should not be inserted again.
TEST_F(MklToTfConversionPass, Negative_DoubleInsert) {
  if (kTensorOrdering == MklTfTensorOrdering::TENSORS_INTERLEAVED) {
    InitGraph(
        "node { name: 'A' op: 'Input'}"
        "node { name: 'M' op: '_MklInput'}"
        "node { name: 'B' op: 'Input'}"
        "node { name: 'N' op: '_MklInput'}"
        "node { name: 'C' op: '_MklConv2D'"
        " attr { key: 'T'                value { type: DT_FLOAT } }"
        " attr { key: 'data_format'      value { s: 'NCHW' } }"
        " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } "
        "}"
        " attr { key: 'padding'          value { s: 'SAME' } }"
        " input: ['A', 'M', 'B', 'N']}"
        "node { name: 'D' op: '_MklToTf'"
        " attr { key: 'T'                value { type: DT_FLOAT } }"
        " attr { key: 'data_format'      value { s: 'NCHW' } }"
        " input: ['C:0', 'C:1']}"
        "node { name: 'E' op: 'Input'}"
        "node { name: 'F' op: 'Sub'"
        " attr {key: 'T'                 value { type: DT_FLOAT } }"
        " input: ['D', 'E']}");
    EXPECT_EQ(DoRunMklToTfConversionPass(),
              "A(Input);B(Input);C(_MklConv2D);D(_MklToTf);E(Input);"
              "F(Sub);M(_MklInput);N(_MklInput)|"
              "A->C;B->C:2;C->D;C:1->D:1;D->F;E->F:1;M->C:1;N->C:3");
  } else {
    CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);
    InitGraph(
        "node { name: 'A' op: 'Input'}"
        "node { name: 'B' op: 'Input'}"
        "node { name: 'M' op: '_MklInput'}"
        "node { name: 'N' op: '_MklInput'}"
        "node { name: 'C' op: '_MklConv2D'"
        " attr { key: 'T'                value { type: DT_FLOAT } }"
        " attr { key: 'data_format'      value { s: 'NCHW' } }"
        " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } "
        "}"
        " attr { key: 'padding'          value { s: 'SAME' } }"
        " input: ['A', 'B', 'M', 'N']}"
        "node { name: 'D' op: '_MklToTf'"
        " attr { key: 'T'                value { type: DT_FLOAT } }"
        " attr { key: 'data_format'      value { s: 'NCHW' } }"
        " input: ['C:0', 'C:2']}"
        "node { name: 'E' op: 'Input'}"
        "node { name: 'F' op: 'Sub'"
        " attr {key: 'T'                 value { type: DT_FLOAT } }"
        " input: ['D', 'E']}");
    EXPECT_EQ(DoRunMklToTfConversionPass(),
              "A(Input);B(Input);C(_MklConv2D);D(_MklToTf);E(Input);"
              "F(Sub);M(_MklInput);N(_MklInput)|"
              "A->C;B->C:1;C->D;C:2->D:1;D->F;E->F:1;M->C:2;N->C:3");
  }
}

// C=Conv2D(A,B); E=BiasAdd(C,D); Z=Sub(E,Y);
// There is no Mkl layer so no conversion op should be inserted.
TEST_F(MklToTfConversionPass, Negative_NoMklLayer) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'BiasAdd'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['C', 'D'] }"
      "node { name: 'Y' op: 'Input'}"
      "node { name: 'Z' op: 'Sub'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['E', 'Y']}");
  EXPECT_EQ(DoRunMklToTfConversionPass(),
            "A(Input);B(Input);C(Conv2D);D(Input);E(BiasAdd);Y(Input);Z(Sub)|"
            "A->C;B->C:1;C->E;D->E:1;E->Z;Y->Z:1");
}

static void BM_RunMklToTfConversionPass(int iters, int op_nodes) {
  testing::StopTiming();
  string s;
  for (int in = 0; in < 10; in++) {
    s += strings::Printf("node { name: 'in%04d' op: 'Input'}", in);
  }
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  for (int op = 0; op < op_nodes; op++) {
    s += strings::Printf(
        "node { name: 'op%04d' op: 'Mul' attr { key: 'T' value { "
        "type: DT_FLOAT } } input: ['in%04d', 'in%04d' ] }",
        op, rnd.Uniform(10), rnd.Uniform(10));
  }

  bool first = true;
  while (iters > 0) {
    Graph* graph = new Graph(OpRegistry::Global());
    MklToTfConversionPass::InitGraph(s, graph);
    int N = graph->num_node_ids();
    if (first) {
      testing::SetLabel(strings::StrCat("Per graph node.  Nodes: ", N));
      first = false;
    }
    {
      testing::StartTiming();
      std::unique_ptr<Graph> ug(graph);
      InsertMklToTfConversionNodes(&ug);
      testing::StopTiming();
    }
    iters -= N;  // Our benchmark units are individual graph nodes,
                 // not whole graphs
    // delete graph;
  }
}
BENCHMARK(BM_RunMklToTfConversionPass)->Arg(1000)->Arg(10000);

}  // namespace
}  // namespace tensorflow

#endif /* INTEL_MKL */
