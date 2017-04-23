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

#include "tensorflow/core/graph/mkl_layout_pass.h"
#include "tensorflow/core/util/mkl_util.h"

#include <algorithm>
#include <string>
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

static void InitGraph(const string& s, Graph* graph) {
  GraphDef graph_def;

  auto parser = protobuf::TextFormat::Parser();
  //  parser.AllowRelaxedWhitespace(true);
  CHECK(parser.MergeFromString(s, &graph_def)) << s;
  GraphConstructorOptions opts;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, graph));
}

class MklLayoutPassTest : public ::testing::Test {
 public:
  MklLayoutPassTest() : graph_(OpRegistry::Global()) {}

  void InitGraph(const string& s) {
    ::tensorflow::InitGraph(s, &graph_);
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

  string DoMklLayoutOptimizationPass() {
    string before = CanonicalGraphString(&graph_);
    LOG(ERROR) << "Before MKL layout rewrite pass: " << before;

    std::unique_ptr<Graph>* ug = new std::unique_ptr<Graph>(&graph_);
    RunMklLayoutRewritePass(ug);

    string result = CanonicalGraphString(&graph_);
    LOG(ERROR) << "After MKL layout rewrite pass:  " << result;
    return result;
  }

  const string& OriginalGraph() const { return original_; }

  Graph graph_;
  string original_;
};

REGISTER_OP("Input").Output("o: float").SetIsStateful();
REGISTER_OP("InputList").Output("o: N * float").Attr("N: int").SetIsStateful();
REGISTER_OP("HalfInput").Output("o: half").SetIsStateful();
REGISTER_OP("Int32Input").Output("o: int32").SetIsStateful();
REGISTER_OP("_MklInput").Output("o: uint8").SetIsStateful();
REGISTER_OP("_MklInput2").Output("o: uint8").Output("o1: uint8").SetIsStateful();

/////////////////////////////////////////////////////////////////////
//  Unit tests related to node merge optiimization
/////////////////////////////////////////////////////////////////////

TEST_F(MklLayoutPassTest, Basic) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Mul);D(Mul)|"
            "A->C;A->D;B->C:1;B->D:1");
}

// Test set 1: Conv2D + AddBias

// C=_MklConv2D(A,M,B,N); E=BiasAdd(C,D); Z=Sub(E,Y) (for interleaved ordering)
// C=_MklConv2D(A,B,M,N); E=BiasAdd(C,D); Z=Sub(E,Y) (for contiguous ordering)
TEST_F(MklLayoutPassTest, NodeMerge_Conv2DWithBias_Positive) {
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
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'B', 'M', 'N']}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'BiasAdd'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['C', 'D'] }"
      "node { name: 'Y' op: 'Input'}"
      "node { name: 'Z' op: 'Sub'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['E', 'Y']}");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);D(Input);DMT/_0(Const);E(_MklConv2DWithBias);"
            "M(_MklInput);N(_MklInput);Y(Input);Z(Sub)|A->E;B->E:1;D->E:2;"
            "DMT/_0->E:5;E->Z;M->E:3;N->E:4;Y->Z:1");
}

// C=_MklConv2D(A,M:1,B,N:1); E=BiasAdd(C,D); Z=Sub(E,Y) (for interleaved)
// C=_MklConv2D(A,B,M:1,N:1); E=BiasAdd(C,D); Z=Sub(E,Y) (for contiguous)
// Test for correct output slots selected
TEST_F(MklLayoutPassTest, NodeMerge_Conv2DWithBias_Positive1) {
  CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'M' op: '_MklInput2'}"
      "node { name: 'N' op: '_MklInput2'}"
      "node { name: 'C' op: '_MklConv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'B', 'M:1', 'N:1']}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'BiasAdd'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['C', 'D'] }"
      "node { name: 'Y' op: 'Input'}"
      "node { name: 'Z' op: 'Sub'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['E', 'Y']}");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);D(Input);DMT/_0(Const);E(_MklConv2DWithBias);"
            "M(_MklInput2);N(_MklInput2);Y(Input);Z(Sub)|A->E;B->E:1;D->E:2;"
            "DMT/_0->E:5;E->Z;M:1->E:3;N:1->E:4;Y->Z:1");
}

// C=Conv2D(A,B); E=BiasAdd(C,D); Z=Sub(E,Y);
// This is a case of node rewrite followed by node merge.
// We will first rewrite Conv2D to _MklConv2D, and then merge _MklConv2D
// with BiasAdd to produce _MklConv2DWithBias.
TEST_F(MklLayoutPassTest, NodeMerge_Conv2DWithBias_Positive2) {
  CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);
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
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);E(_MklConv2DWithBias);Y(Input);Z(Sub)|"
            "A->E;B->E:1;D->E:2;DMT/_0->E:3;DMT/_1->E:4;DMT/_2->E:5;"
            "E->Z;Y->Z:1");
}

// Graph contains only _MklConv2D, no AddBias.
TEST_F(MklLayoutPassTest, NodeMerge_Conv2DWithBias_Negative_NoAddBias) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'M' op: '_MklInput'}"
      "node { name: 'N' op: '_MklInput'}"
      "node { name: 'C' op: '_MklConv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'B', 'M', 'N']}");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklConv2D);M(_MklInput);N(_MklInput)|"
            "A->C;B->C:1;M->C:2;N->C:3");
}

// _MklConv2D output does not go to BiasAdd.
TEST_F(MklLayoutPassTest, NodeMerge_Conv2DWithBias_Negative_Dataflow1) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'M' op: '_MklInput'}"
      "node { name: 'N' op: '_MklInput'}"
      "node { name: 'C' op: '_MklConv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'B', 'M', 'N']}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'BiasAdd'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['D', 'E'] }");  // Output of _MklConv2D does not go to BiasAdd.
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklConv2D);D(Input);E(Input);F(BiasAdd);"
            "M(_MklInput);N(_MklInput)|A->C;B->C:1;D->F;E->F:1;M->C:2;N->C:3");
}

// _MklConv2D has two outgoing edges: BiasAdd and some other dummy node (Add).
// Merge should not be done in such case.
TEST_F(MklLayoutPassTest, NodeMerge_Conv2DWithBias_Negative_Dataflow2) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'M' op: '_MklInput'}"
      "node { name: 'N' op: '_MklInput'}"
      "node { name: 'C' op: '_MklConv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'B', 'M', 'N']}"
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
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklConv2D);D(Input);E(Input);F(BiasAdd);"
            "G(Add);M(_MklInput);N(_MklInput)|A->C;B->C:1;C->G;D->F;"
            "E->F:1;E->G:1;M->C:2;N->C:3");
}

// data_format attribute value mismatch. Merge should not be done
// in such case.
TEST_F(MklLayoutPassTest, NodeMerge_Conv2DWithBias_Negative_AttrMismatch) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'M' op: '_MklInput'}"
      "node { name: 'N' op: '_MklInput'}"
      "node { name: 'C' op: '_MklConv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'B', 'M', 'N']}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'BiasAdd'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NHCW' } }"
      " input: ['C', 'D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklConv2D);D(Input);E(BiasAdd);M(_MklInput);"
            "N(_MklInput)|A->C;B->C:1;C->E;D->E:1;M->C:2;N->C:3");
}

// Disabling Conv2DBackpropBias test for now as we have disabled rewrite
// of BiasAddGrad into BackpropBias
#if 0
// Test set 2: _MklConv2D..BiasAddGrad -> _MklConv2DWithBiasBackpropBias
// rewrite tests

// D=_MklConv2D(A,M,B,N,C,O); E=Sub(D,A); F=BiasAddGrad(E)
TEST_F(MklLayoutPassTest, NodeMerge_Conv2DBackprop_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'M' op: '_MklInput'}"
      "node { name: 'N' op: '_MklInput'}"
      "node { name: 'O' op: '_MklInput'}"
      "node { name: 'D' op: '_MklConv2DWithBias'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'B', 'C', 'M', 'N', 'O']}"
      "node { name: 'E' op: 'Sub'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['D', 'A']}"
      "node { name: 'F' op: 'BiasAddGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['E'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(_MklConv2DWithBias);DMT/_0(Const);"
            "E(Sub);F(_MklConv2DWithBiasBackpropBias);M(_MklInput);N(_MklInput);"
            "O(_MklInput)|A->D;A->E:1;B->D:1;C->D:2;D->E;DMT/_0->F:1;E->F;"
            "M->D:3;N->D:4;O->D:5");
}
#endif

// No _MklConv2D in context, but Conv2D in context.
// Only Conv2D would be rewritten to _MklConv2D, but no rewrite
// for BiasAddGrad should happen.
// C=_MklConv2D(A,M,B,N); D=Sub(C,A); E=BiasAddGrad(D) (for interleaved)
// C=_MklConv2D(A,B,M,N); D=Sub(C,A); E=BiasAddGrad(D) (for contiguous)
TEST_F(MklLayoutPassTest, NodeMerge_Conv2DBackprop_Neg_No_MklConv2DWithBias) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'M' op: '_MklInput'}"
      "node { name: 'N' op: '_MklInput'}"
      "node { name: 'C' op: '_MklConv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'B', 'M', 'N']}"
      "node { name: 'D' op: 'Sub'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['C', 'A']}"
      "node { name: 'E' op: 'BiasAddGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklConv2D);D(Sub);E(BiasAddGrad);"
            "M(_MklInput);N(_MklInput)|A->C;A->D:1;B->C:1;C->D;D->E;"
            "M->C:2;N->C:3");
}

// No Conv2D in the context for BiasAddGrad. No rewrite should happen.
// C=Add(A,B); D=Sub(C,A); E=BiasAddGrad(D)
TEST_F(MklLayoutPassTest, NodeMerge_Conv2DBackprop_Negative_NoConv2D) {
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
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Add);D(Sub);E(BiasAddGrad)|"
            "A->C;A->D:1;B->C:1;C->D;D->E");
}

// No Conv2D in the context for BiasAddGrad, but MatMul in context.
// Rewrite should happen, but name of BiasAddGrad does not change.
// C=MatMul(A,B); D=Sub(C,A); E=BiasAddGrad(D)
TEST_F(MklLayoutPassTest, NodeMerge_Conv2DBackprop_Negative_NoConv2D_MatMul) {
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
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(MatMul);D(Sub);E(BiasAddGrad)|"
            "A->C;A->D:1;B->C:1;C->D;D->E");
}

// Test set 3: MatMul..BiasAddGrad -> BiasAddGrad rewrite tests
// C=MatMul(A,B); D=Sub(C,A); E=BiasAddGrad(D)
TEST_F(MklLayoutPassTest, NodeMerge_MatMulBiasAddGrad_Positive) {
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
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(MatMul);D(Sub);E(BiasAddGrad)|"
            "A->C;A->D:1;B->C:1;C->D;D->E");
}

// No MatMul in the context for BiasAddGrad. No rewrite should happen.
// C=Add(A,B); D=Sub(C,A); E=BiasAddGrad(D)
TEST_F(MklLayoutPassTest, NodeMerge_MatMulBiasAddGrad_Negative_NoMatMul) {
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
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Add);D(Sub);E(BiasAddGrad)|"
            "A->C;A->D:1;B->C:1;C->D;D->E");
}

/////////////////////////////////////////////////////////////////////
//  Unit tests related to rewriting node to Mkl node
/////////////////////////////////////////////////////////////////////

// Single Conv2D Op; No Mkl layer on the input and on the output.
// We will generate dummy Mkl tensor as 2nd input of Conv2D.
TEST_F(MklLayoutPassTest, NodeRewrite_Conv2D_Basic) {
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
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'C'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklConv2D);D(Mul);DMT/_0(Const);DMT/_1(Const)|"
            "A->C;B->C:1;B->D;C->D:1;DMT/_0->C:2;DMT/_1->C:3");
}

// 2 Conv2D Ops in sequence. Both should get transformed and 1st Conv2D will
// have 2 outputs, both of which will be inputs to next Conv2D.
TEST_F(MklLayoutPassTest, NodeRewrite_Conv2D_Positive1) {
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
      "node { name: 'D' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'C']}"
      "node { name: 'E' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklConv2D);D(_MklConv2D);DMT/_0(Const);"
            "DMT/_1(Const);DMT/_2(Const);E(Mul)|A->C;A->D;B->C:1;C->D:1;C->E;"
            "C:1->D:3;D->E:1;DMT/_0->C:2;DMT/_1->C:3;DMT/_2->D:2");
}

// Conv2D with INT32 which is not supported by Mkl
TEST_F(MklLayoutPassTest, NodeRewrite_Conv2D_Negative_UnsupportedType) {
  InitGraph(
      "node { name: 'A' op: 'HalfInput'}"
      "node { name: 'B' op: 'HalfInput'}"
      "node { name: 'C' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_HALF } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_HALF } }"
      " input: ['B', 'C'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(HalfInput);B(HalfInput);C(Conv2D);D(Mul)|"
            "A->C;B->C:1;B->D;C->D:1");
}

// Concat Op test: Concat with no Mkl layer feeding it
TEST_F(MklLayoutPassTest, NodeRewrite_Concat_Basic) {
  InitGraph(
      "node { name: 'A' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_INT32 } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'B' op: 'InputList'"
      " attr { key: 'N'                value { i: 2 } }}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Concat'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'N'                value { i: 2 } }"
      " input: ['A', 'B']}"
      "node { name: 'E' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Const);B(InputList);C(Input);D(_MklConcat);DMT/_0(Const);"
            "DMT/_1(Const);DMT/_2(Const);E(Mul)|A->D;B->D:1;B->D:2;C->E;"
            "D->E:1;DMT/_0->D:3;DMT/_1->D:4;DMT/_2->D:5");
}

// Concat with 2 Mkl layers feeding it
TEST_F(MklLayoutPassTest, NodeRewrite_Concat_Input_Mkl) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'B']}"
      "node { name: 'F' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['C', 'D']}"
      "node { name: 'G' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_INT32 } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'H' op: 'Concat'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'N'                value { i: 2 } }"
      " input: ['G', 'E', 'F']}"
      "node { name: 'I' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'H'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);DMT/_3(Const);DMT/_4(Const);E(_MklConv2D);"
            "F(_MklConv2D);G(Const);H(_MklConcat);I(Mul)|A->E;A->I;B->E:1;C->F;"
            "D->F:1;DMT/_0->F:2;DMT/_1->F:3;DMT/_2->E:2;DMT/_3->E:3;"
            "DMT/_4->H:3;E->H:1;E:1->H:4;F->H:2;F:1->H:5;G->H;H->I:1");
}

// Concat with 1 Mkl and 1 non-Mkl layer feeding it
TEST_F(MklLayoutPassTest, NodeRewrite_Concat_Input_MixedMkl) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'B']}"
      "node { name: 'F' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'D']}"
      "node { name: 'G' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_INT32 } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'H' op: 'Concat'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'N'                value { i: 2 } }"
      " input: ['G', 'E', 'F']}"
      "node { name: 'I' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'H'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);DMT/_3(Const);E(_MklConv2D);F(Mul);G(Const);"
            "H(_MklConcat);I(Mul)|A->E;A->I;B->E:1;C->F;D->F:1;DMT/_0->E:2;"
            "DMT/_1->E:3;DMT/_2->H:3;DMT/_3->H:5;E->H:1;E:1->H:4;F->H:2;"
            "G->H;H->I:1");
}

#if 0
// ConcatV2 Op test: ConcatV2 with no Mkl layer feeding it
TEST_F(MklLayoutPassTest, NodeRewrite_ConcatV2_Basic) {
  InitGraph(
      "node { name: 'A' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_INT32 } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'B' op: 'InputList'"
      " attr { key: 'N'                value { i: 2 } }}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'ConcatV2'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'Tidx'             value { type: DT_INT32 } }"
      " attr { key: 'N'                value { i: 2 } }"
      " input: ['B:0', 'B:1', 'A']}"
      "node { name: 'E' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Const);B(InputList);C(Input);D(_MklConcat);DMT/_0(Const);"
            "DMT/_1(Const);DMT/_2(Const);E(Mul)|A->D:2;B->D;B:1->D:1;C->E;"
            "D->E:1;DMT/_0->D:3;DMT/_1->D:4;DMT/_2->D:5");
}
#endif

// ConcatV2 with 2 Mkl layers feeding it
TEST_F(MklLayoutPassTest, NodeRewrite_ConcatV2_Input_Mkl) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'B']}"
      "node { name: 'F' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['C', 'D']}"
      "node { name: 'G' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_INT32 } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'H' op: 'ConcatV2'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'Tidx'             value { type: DT_INT32 } }"
      " attr { key: 'N'                value { i: 2 } }"
      " input: ['E', 'F', 'G']}"
      "node { name: 'I' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'H'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);DMT/_3(Const);DMT/_4(Const);E(_MklConv2D);"
            "F(_MklConv2D);G(Const);H(_MklConcatV2);I(Mul)|A->E;A->I;B->E:1;C->F;"
            "D->F:1;DMT/_0->F:2;DMT/_1->F:3;DMT/_2->E:2;DMT/_3->E:3;"
            "DMT/_4->H:5;E->H;E:1->H:3;F->H:1;F:1->H:4;G->H:2;H->I:1");
}

// ConcatV2 with 1 Mkl and 1 non-Mkl layer feeding it
TEST_F(MklLayoutPassTest, NodeRewrite_ConcatV2_Input_MixedMkl) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'B']}"
      "node { name: 'F' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'D']}"
      "node { name: 'G' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_INT32 } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'H' op: 'ConcatV2'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'Tidx'             value { type: DT_INT32 } }"
      " attr { key: 'N'                value { i: 2 } }"
      " input: ['E', 'F', 'G']}"
      "node { name: 'I' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'H'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);DMT/_3(Const);E(_MklConv2D);F(Mul);G(Const);"
            "H(_MklConcatV2);I(Mul)|A->E;A->I;B->E:1;C->F;D->F:1;DMT/_0->E:2;"
            "DMT/_1->E:3;DMT/_2->H:4;DMT/_3->H:5;E->H;E:1->H:3;F->H:1;"
            "G->H:2;H->I:1");
}

/////////////////////////////////////////////////////////////////////
//  Unit tests related to rewriting node for workspace edges
/////////////////////////////////////////////////////////////////////

/* Test LRN->MaxPool->MaxPoolGrad->LRNGrad replacement by workspace nodes. */
TEST_F(MklLayoutPassTest, MaxPoolLRN_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'LRN'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'alpha'        value { f: 0.001 } }"
      " attr { key: 'beta'         value { f: 0.75 } }"
      " attr { key: 'bias'         value { f: 1.0 } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'depth_radius' value { i: 2 } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'MaxPool'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:1, i:3, i:3} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:1, i:2, i:2} } }"
      " input: ['B'] }"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'MaxPoolGrad'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:1, i:3, i:3} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:1, i:2, i:2} } }"
      " input: ['B', 'C', 'D'] }"
      "node { name: 'F' op: 'Input'}"
      "node { name: 'G' op: 'LRNGrad'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'alpha'        value { f: 0.001 } }"
      " attr { key: 'beta'         value { f: 0.75 } }"
      " attr { key: 'bias'         value { f: 1.0 } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'depth_radius' value { i: 2 } }"
      " input: ['E', 'F', 'B'] }"
      "node { name: 'H' op: 'Input'}"
      "node { name: 'I' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['H', 'G'] }");
  EXPECT_EQ(
      DoMklLayoutOptimizationPass(),
      "A(Input);B(_MklLRN);C(_MklMaxPool);D(Input);DMT/_0(Const);DMT/_1(Const);"
      "DMT/_2(Const);E(_MklMaxPoolGrad);F(Input);G(_MklLRNGrad);H(Input);I(Mul)|"
      "A->B;B->C;B->E;B->G:2;B:1->G:3;B:2->C:1;B:2->E:4;B:2->G:6;B:3->G:7;"
      "C->E:1;C:1->E:3;C:2->E:5;C:3->E:7;D->E:2;DMT/_0->B:1;DMT/_1->E:6;"
      "DMT/_2->G:5;E->G;E:1->G:4;F->G:1;G->I:1;H->I");
}

/* Test LRN->LRNGrad replacement by workspace nodes. */
TEST_F(MklLayoutPassTest, LRN_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'LRN'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'alpha'        value { f: 0.001 } }"
      " attr { key: 'beta'         value { f: 0.75 } }"
      " attr { key: 'bias'         value { f: 1.0 } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'depth_radius' value { i: 2 } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'LRNGrad'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'alpha'        value { f: 0.001 } }"
      " attr { key: 'beta'         value { f: 0.75 } }"
      " attr { key: 'bias'         value { f: 1.0 } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'depth_radius' value { i: 2 } }"
      " input: ['C', 'D', 'B'] }"
      "node { name: 'F' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'E'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(_MklLRN);C(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);E(_MklLRNGrad);F(Mul)|"
            "A->B;B->E:2;B:1->E:3;B:2->E:6;B:3->E:7;C->E;C->F;D->E:1;"
            "DMT/_0->B:1;DMT/_1->E:4;DMT/_2->E:5;E->F:1");
}

/* Test LRN->LRNGrad replacement when only one of them is present. */
TEST_F(MklLayoutPassTest, LRN_Negative1) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'LRN'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'alpha'        value { f: 0.001 } }"
      " attr { key: 'beta'         value { f: 0.75 } }"
      " attr { key: 'bias'         value { f: 1.0 } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'depth_radius' value { i: 2 } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(_MklLRN);C(Mul);DMT/_0(Const)|"
            "A->B;A->C;B->C:1;DMT/_0->B:1");
}

/* Test LRN->LRNGrad replacement when only one of them is present. */
TEST_F(MklLayoutPassTest, LRN_Negative2) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'LRNGrad'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'alpha'        value { f: 0.001 } }"
      " attr { key: 'beta'         value { f: 0.75 } }"
      " attr { key: 'bias'         value { f: 1.0 } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'depth_radius' value { i: 2 } }"
      " input: ['A', 'B', 'C'] }"
      "node { name: 'E' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(_MklLRNGrad);DMT/_0(Const);"
            "DMT/_1(Const);DMT/_2(Const);DMT/_3(Const);DMT/_4(Const);E(Mul)|"
            "A->D;A->E;B->D:1;C->D:2;D->E:1;DMT/_0->D:3;DMT/_1->D:7;"
            "DMT/_2->D:4;DMT/_3->D:5;DMT/_4->D:6");
}

/* Test LRN->LRNGrad negative case, where single LRN feeds
   2 LRNGrad nodes at different slots. */
TEST_F(MklLayoutPassTest, LRN_Negative3) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'LRN'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'alpha'        value { f: 0.001 } }"
      " attr { key: 'beta'         value { f: 0.75 } }"
      " attr { key: 'bias'         value { f: 1.0 } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'depth_radius' value { i: 2 } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'LRNGrad'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'alpha'        value { f: 0.001 } }"
      " attr { key: 'beta'         value { f: 0.75 } }"
      " attr { key: 'bias'         value { f: 1.0 } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'depth_radius' value { i: 2 } }"
      " input: ['C', 'D', 'B'] }"
      "node { name: 'F' op: 'LRNGrad'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'alpha'        value { f: 0.001 } }"
      " attr { key: 'beta'         value { f: 0.75 } }"
      " attr { key: 'bias'         value { f: 1.0 } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'depth_radius' value { i: 2 } }"
      " input: ['C', 'B', 'D'] }"
      "node { name: 'G' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['E', 'F'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(_MklLRN);C(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);DMT/_3(Const);DMT/_4(Const);DMT/_5(Const);"
            "DMT/_6(Const);E(_MklLRNGrad);F(_MklLRNGrad);G(Mul)|A->B;B->E:2;"
            "B->F:1;B:1->E:3;B:2->E:6;B:2->F:5;B:3->E:7;C->E;C->F;D->E:1;"
            "D->F:2;DMT/_0->B:1;DMT/_1->F:3;DMT/_2->F:7;DMT/_3->F:4;"
            "DMT/_4->F:6;DMT/_5->E:4;DMT/_6->E:5;E->G;F->G:1");
}

/* Test MaxPool->MaxPoolGrad replacement by workspace+rewrite nodes. */
TEST_F(MklLayoutPassTest, NodeWorkspace_MaxPool_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'MaxPool'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:1, i:3, i:3} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:1, i:2, i:2} } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'MaxPoolGrad'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:1, i:3, i:3} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:1, i:2, i:2} } }"
      " input: ['C', 'B', 'D'] }"
      "node { name: 'F' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'E'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(_MklMaxPool);C(Input);D(Input);DMT/_0(Const);"
            "DMT/_1(Const);DMT/_2(Const);E(_MklMaxPoolGrad);F(Mul)|"
            "A->B;B->E:1;B:1->E:3;B:2->E:5;B:3->E:7;C->E;C->F;D->E:2;"
            "DMT/_0->B:1;DMT/_1->E:4;DMT/_2->E:6;E->F:1");
}

// Test MaxPool>MaxPoolGrad replacement when only one of them is present.
// In this case, we will rewrite MaxPool node but workspace edges will not
// be present.
TEST_F(MklLayoutPassTest, NodeWorkspace_MaxPool_Negative1) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'MaxPool'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:1, i:3, i:3} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:1, i:2, i:2} } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(_MklMaxPool);C(Mul);DMT/_0(Const)|"
            "A->B;A->C;B->C:1;DMT/_0->B:1");
}

// Test MaxPoolGrad replacement when only one of them is present.
// In this case, we will rewrite MaxPoolGrad and for workspace tensor and
// its Mkl part, we will generate dummy tensor.
TEST_F(MklLayoutPassTest, NodeWorkspace_MaxPool_Negative2) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'MaxPoolGrad'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:1, i:3, i:3} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:1, i:2, i:2} } }"
      " input: ['A', 'B', 'C'] }"
      "node { name: 'E' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(_MklMaxPoolGrad);DMT/_0(Const);"
            "DMT/_1(Const);DMT/_2(Const);DMT/_3(Const);DMT/_4(Const);E(Mul)|"
            "A->D;A->E;B->D:1;C->D:2;D->E:1;DMT/_0->D:3;DMT/_1->D:7;"
            "DMT/_2->D:4;DMT/_3->D:5;DMT/_4->D:6");
}

/////////////////////////////////////////////////////////////////////

static void BM_MklLayoutRewritePass(int iters, int op_nodes) {
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
    InitGraph(s, graph);
    int N = graph->num_node_ids();
    if (first) {
      testing::SetLabel(strings::StrCat("Per graph node.  Nodes: ", N));
      first = false;
    }
    {
      testing::StartTiming();
      std::unique_ptr<Graph> ug(graph);
      RunMklLayoutRewritePass(&ug);
      testing::StopTiming();
    }
    iters -= N;  // Our benchmark units are individual graph nodes,
                 // not whole graphs
    // delete graph;
  }
}
BENCHMARK(BM_MklLayoutRewritePass)->Arg(1000)->Arg(10000);

}  // namespace
}  // namespace tensorflow

#endif /* INTEL_MKL */
