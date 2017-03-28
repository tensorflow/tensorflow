/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/mkl_optimizer_merge.h"

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

class OptimizerMergeTest : public ::testing::Test {
 public:
  OptimizerMergeTest() : graph_(OpRegistry::Global()) {}

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

  string DoNodeMerge() {
    string before = CanonicalGraphString(&graph_);
    LOG(ERROR) << "Before node merge optimize: " << before;

    std::unique_ptr<Graph>* ug = new std::unique_ptr<Graph>(&graph_);
    OptimizeNodeMerge(ug);

    string result = CanonicalGraphString(&graph_);
    LOG(ERROR) << "After node merge optimize:  " << result;
    return result;
  }

  const string& OriginalGraph() const { return original_; }

  Graph graph_;
  string original_;
};

REGISTER_OP("Input").Output("o: float").SetIsStateful();
REGISTER_OP("MklInput").Output("o: uint8").SetIsStateful();

TEST_F(OptimizerMergeTest, Basic) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoNodeMerge(),
            "A(Input);B(Input);C(Mul);D(Mul)|"
            "A->C;A->D;B->C:1;B->D:1");
}

// Test set 1: Conv2D + AddBias

// C=MklConv2D(A,M,B,N); E=BiasAdd(C,D); Z=Sub(E,Y)
TEST_F(OptimizerMergeTest, Conv2DWithBias_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'M' op: 'MklInput'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'N' op: 'MklInput'}"
      "node { name: 'C' op: 'MklConv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'M', 'B', 'N']}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'BiasAdd'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['C', 'D'] }"
      "node { name: 'Y' op: 'Input'}"
      "node { name: 'Z' op: 'Sub'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['E', 'Y']}");
  EXPECT_EQ(DoNodeMerge(),
            "A(Input);B(Input);D(Input);DMT/_0(Const);E(MklConv2DWithBias);"
            "M(MklInput);N(MklInput);Y(Input);Z(Sub)|A->E;B->E:2;D->E:4;"
            "DMT/_0->E:5;E->Z;M->E:1;N->E:3;Y->Z:1");
}

// C=Conv2D(A,B); E=BiasAdd(C,D); Z=Sub(E,Y);
// We do not merge in this case as op is Conv2D and not MklConv2D.
TEST_F(OptimizerMergeTest, Conv2DWithBias_Negative_NoMklConv2D) {
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
  EXPECT_EQ(DoNodeMerge(),
            "A(Input);B(Input);C(Conv2D);D(Input);E(BiasAdd);Y(Input);Z(Sub)|"
            "A->C;B->C:1;C->E;D->E:1;E->Z;Y->Z:1");
}

// Graph contains only MklConv2D, no AddBias.
TEST_F(OptimizerMergeTest, Conv2DWithBias_Negative_NoAddBias) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'M' op: 'MklInput'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'N' op: 'MklInput'}"
      "node { name: 'C' op: 'MklConv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'M', 'B', 'N']}");
  EXPECT_EQ(DoNodeMerge(),
            "A(Input);B(Input);C(MklConv2D);M(MklInput);N(MklInput)|"
            "A->C;B->C:2;M->C:1;N->C:3");
}

// MklConv2D output does not go to BiasAdd.
TEST_F(OptimizerMergeTest, Conv2DWithBias_Negative_Dataflow1) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'M' op: 'MklInput'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'N' op: 'MklInput'}"
      "node { name: 'C' op: 'MklConv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'M', 'B', 'N']}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'BiasAdd'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['D', 'E'] }");  // Output of MklConv2D does not go to BiasAdd.
  EXPECT_EQ(DoNodeMerge(),
            "A(Input);B(Input);C(MklConv2D);D(Input);E(Input);F(BiasAdd);"
            "M(MklInput);N(MklInput)|A->C;B->C:2;D->F;E->F:1;M->C:1;N->C:3");
}

// MklConv2D has two outgoing edges: BiasAdd and some other dummy node (Add).
// Merge should not be done in such case.
TEST_F(OptimizerMergeTest, Conv2DWithBias_Negative_Dataflow2) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'M' op: 'MklInput'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'N' op: 'MklInput'}"
      "node { name: 'C' op: 'MklConv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'M', 'B', 'N']}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'BiasAdd'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['D', 'E'] }"  // Conv2D has two outputs.
                              // No merge should happen.
      "node { name: 'G' op: 'Add'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['C', 'E'] }");
  EXPECT_EQ(DoNodeMerge(),
            "A(Input);B(Input);C(MklConv2D);D(Input);E(Input);F(BiasAdd);"
            "G(Add);M(MklInput);N(MklInput)|A->C;B->C:2;C->G;D->F;"
            "E->F:1;E->G:1;M->C:1;N->C:3");
}

// data_format attribute value mismatch. Merge should not be done
// in such case.
TEST_F(OptimizerMergeTest, Conv2DWithBias_Negative_AttrMismatch) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'M' op: 'MklInput'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'N' op: 'MklInput'}"
      "node { name: 'C' op: 'MklConv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'M', 'B', 'N']}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'BiasAdd'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NHCW' } }"
      " input: ['C', 'D'] }");
  EXPECT_EQ(DoNodeMerge(),
            "A(Input);B(Input);C(MklConv2D);D(Input);E(BiasAdd);M(MklInput);"
            "N(MklInput)|A->C;B->C:2;C->E;D->E:1;M->C:1;N->C:3");
}

#if 0
// This test set is disabled temporarily as we do not enable node rewrite.
// This test set will be enabled when we support Mkl-specific kernels for
// backward bias.
//
// Test set 2: MklConv2D..BiasAddGrad -> Conv2DWithBiasBackpropBias
// rewrite tests

// C=MklConv2D(A,M,B,N); D=Sub(C,A); E=BiasAddGrad(D)
TEST_F(OptimizerMergeTest, Conv2DBackprop_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'M' op: 'MklInput'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'N' op: 'MklInput'}"
      "node { name: 'C' op: 'MklConv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'M', 'B', 'N']}"
      "node { name: 'D' op: 'Sub'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['C', 'A']}"
      "node { name: 'E' op: 'BiasAddGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['D'] }");
  EXPECT_EQ(DoNodeMerge(),
            "A(Input);B(Input);C(MklConv2D);D(Sub);E(Conv2DWithBiasBackpropBias);"
            "M(MklInput);N(MklInput)|A->C;A->D:1;B->C:2;C->D;D->E;M->C:1;N->C:3");
}

// No MklConv2D in context, but Conv2D in context. No rewrite should happen.
// C=Conv2D(A,B); D=Sub(C,A); E=BiasAddGrad(D)
TEST_F(OptimizerMergeTest, Conv2DBackprop_Negative_NoMklConv2D) {
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
      "node { name: 'D' op: 'Sub'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['C', 'A']}"
      "node { name: 'E' op: 'BiasAddGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['D'] }");
  EXPECT_EQ(DoNodeMerge(),
            "A(Input);B(Input);C(Conv2D);D(Sub);E(BiasAddGrad)|"
             "A->C;A->D:1;B->C:1;C->D;D->E");
}

// No Conv2D in the context for BiasAddGrad. No rewrite should happen.
// C=Add(A,B); D=Sub(C,A); E=BiasAddGrad(D)
TEST_F(OptimizerMergeTest, Conv2DBackprop_Negative_NoConv2D) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Add'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Sub'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['C', 'A']}"
      "node { name: 'E' op: 'BiasAddGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['D'] }");
  EXPECT_EQ(DoNodeMerge(),
            "A(Input);B(Input);C(Add);D(Sub);E(BiasAddGrad)|"
             "A->C;A->D:1;B->C:1;C->D;D->E");
}

// No Conv2D in the context for BiasAddGrad, but MatMul in context.
// Rewrite should happen, but name of BiasAddGrad does not change.
// C=MatMul(A,B); D=Sub(C,A); E=BiasAddGrad(D)
TEST_F(OptimizerMergeTest, Conv2DBackprop_Negative_NoConv2D_MatMul) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'MatMul'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'transpose_a'      value { b: false } }"
      " attr { key: 'transpose_b'      value { b: false } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Sub'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['C', 'A']}"
      "node { name: 'E' op: 'BiasAddGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['D'] }");
  EXPECT_EQ(DoNodeMerge(),
            "A(Input);B(Input);C(MatMul);D(Sub);E(BiasAddGrad)|"
             "A->C;A->D:1;B->C:1;C->D;D->E");
}

// Test set 3: MatMul..BiasAddGrad -> BiasAddGrad rewrite tests
// C=MatMul(A,B); D=Sub(C,A); E=BiasAddGrad(D)
TEST_F(OptimizerMergeTest, MatMulBiasAddGrad_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'MatMul'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'transpose_a'      value { b: false } }"
      " attr { key: 'transpose_b'      value { b: false } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Sub'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['C', 'A']}"
      "node { name: 'E' op: 'BiasAddGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['D'] }");
  EXPECT_EQ(DoNodeMerge(),
            "A(Input);B(Input);C(MatMul);D(Sub);E(BiasAddGrad)|"
             "A->C;A->D:1;B->C:1;C->D;D->E");
}

// No MatMul in the context for BiasAddGrad. No rewrite should happen.
// C=Add(A,B); D=Sub(C,A); E=BiasAddGrad(D)
TEST_F(OptimizerMergeTest, MatMulBiasAddGrad_Negative_NoMatMul) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Add'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Sub'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['C', 'A']}"
      "node { name: 'E' op: 'BiasAddGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['D'] }");
  EXPECT_EQ(DoNodeMerge(),
            "A(Input);B(Input);C(Add);D(Sub);E(BiasAddGrad)|"
             "A->C;A->D:1;B->C:1;C->D;D->E");
}
#endif

static void BM_NodeMerge(int iters, int op_nodes) {
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
    OptimizerMergeTest::InitGraph(s, graph);
    int N = graph->num_node_ids();
    if (first) {
      testing::SetLabel(strings::StrCat("Per graph node.  Nodes: ", N));
      first = false;
    }
    {
      testing::StartTiming();
      std::unique_ptr<Graph> ug(graph);
      OptimizeNodeMerge(&ug);
      testing::StopTiming();
    }
    iters -= N;  // Our benchmark units are individual graph nodes,
                 // not whole graphs
    // delete graph;
  }
}
BENCHMARK(BM_NodeMerge)->Arg(1000)->Arg(10000);

}  // namespace
}  // namespace tensorflow

#endif /* INTEL_MKL */
