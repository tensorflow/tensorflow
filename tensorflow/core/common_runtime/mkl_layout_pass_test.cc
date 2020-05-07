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

#if defined(INTEL_MKL) && defined(ENABLE_MKL)

#include "tensorflow/core/common_runtime/mkl_layout_pass.h"

#include <algorithm>
#include <vector>

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
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

// NOTE: Unit tests in this file rely on a topological sorted graph for
// printing. But since sibling nodes of a node in the topologically sorted graph
// can be printed in different orders, tests may fail if the order in which
// sibling nodes are visited is changed.

namespace {

const char kCPUDevice[] = "/job:a/replica:0/task:0/device:CPU:0";
const char kGPUDevice[] = "/job:a/replica:0/task:0/device:GPU:0";

static void InitGraph(const string& s, Graph* graph,
                      const string& device = kCPUDevice) {
  GraphDef graph_def;

  auto parser = protobuf::TextFormat::Parser();
  //  parser.AllowRelaxedWhitespace(true);
  CHECK(parser.MergeFromString(s, &graph_def)) << s;
  GraphConstructorOptions opts;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, graph));

  for (Node* node : graph->nodes()) {
    node->set_assigned_device_name(device);
  }
}

class MklLayoutPassTest : public ::testing::Test {
 public:
  MklLayoutPassTest() : graph_(OpRegistry::Global()) {}
  // Ashraf added
  Node* FindNode(const string& name) {
    for (Node* node : graph_.nodes()) {
      if (node->name() == name) return node;
    }
    LOG(FATAL) << name;
  }

  void InitGraph(const string& s, const string& device = kCPUDevice) {
    ::tensorflow::InitGraph(s, &graph_, device);
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
    return strings::StrCat(absl::StrJoin(nodes, ";"), "|",
                           absl::StrJoin(edges, ";"));
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

  // Returns the attribute value only from the first node
  template <typename T>
  T DoMklLayoutOptimizationPassGetAttrVal(const string& attr,
                                          const string& node_name) {
    DoMklLayoutOptimizationPass();
    T attr_val;
    for (const Node* n : graph_.nodes()) {
      if (IncludeNode(n) && n->type_string() == node_name) {
        TF_CHECK_OK(GetNodeAttr(n->def(), attr, &attr_val));
        return attr_val;
      }
    }
    return attr_val;
  }

  const string& OriginalGraph() const { return original_; }

  Graph graph_;
  string original_;
};

// TODO(nhasabni): remove these two ops later once all of the file is modified
// to use new type-specific ops.
REGISTER_OP("Input").Output("o: float").SetIsStateful();
REGISTER_OP("InputList").Output("o: N * float").Attr("N: int").SetIsStateful();
REGISTER_OP("Output2").Input("i: float").Input("i1: float").SetIsStateful();

REGISTER_OP("Float32Input").Output("o: float").SetIsStateful();
REGISTER_OP("Float32InputList")
    .Output("o: N * float")
    .Attr("N: int")
    .SetIsStateful();
REGISTER_OP("HalfInput").Output("o: half").SetIsStateful();
REGISTER_OP("Int32Input").Output("o: int32").SetIsStateful();
REGISTER_OP("DoubleInput").Output("o: double").SetIsStateful();
REGISTER_OP("QuantizedInput").Output("o: quint8").SetIsStateful();
REGISTER_OP("_MklInput").Output("o: uint8").SetIsStateful();
REGISTER_OP("_MklInput2")
    .Output("o: uint8")
    .Output("o1: uint8")
    .SetIsStateful();
REGISTER_OP("QuantizedUnsignedInt8Input").Output("o: quint8").SetIsStateful();
REGISTER_OP("QuantizedSignedInt8Input").Output("o: qint8").SetIsStateful();
REGISTER_OP("QuantizedSignedInt32Input").Output("o: qint32").SetIsStateful();
REGISTER_OP("Float32Output2")
    .Input("i: float")
    .Input("i1: float")
    .SetIsStateful();
REGISTER_OP("Output").Input("i: float").SetIsStateful();
REGISTER_OP("QInt8Input").Output("o: qint8").SetIsStateful();
REGISTER_OP("QUInt8Input").Output("o: quint8").SetIsStateful();
REGISTER_OP("QInt32Input").Output("o: qint32").SetIsStateful();

#ifdef ENABLE_INTEL_MKL_BFLOAT16
REGISTER_OP("BFloat16Input").Output("o: bfloat16").SetIsStateful();
REGISTER_OP("BFloat16InputList")
    .Output("o: N * bfloat16")
    .Attr("N: int")
    .SetIsStateful();
REGISTER_OP("BFloat16Output2")
    .Input("i: bfloat16")
    .Input("i1: bfloat16")
    .SetIsStateful();
#endif  // ENABLE_INTEL_MKL_BFLOAT16

/////////////////////////////////////////////////////////////////////
// Macros for handling registeration for various types
/////////////////////////////////////////////////////////////////////

#define REGISTER_TEST_FLOAT32(TEST) REGISTER_TEST(TEST, DT_FLOAT, Float32Input);

#ifdef ENABLE_INTEL_MKL_BFLOAT16
#define REGISTER_TEST_BFLOAT16(TEST) \
  REGISTER_TEST(TEST, DT_BFLOAT16, BFloat16Input);

#define REGISTER_TEST_ALL_TYPES(TEST) \
  REGISTER_TEST_FLOAT32(TEST);        \
  REGISTER_TEST_BFLOAT16(TEST);
#else
#define REGISTER_TEST_ALL_TYPES(TEST) REGISTER_TEST_FLOAT32(TEST);
#endif  // ENABLE_INTEL_MKL_BFLOAT16

/////////////////////////////////////////////////////////////////////
//  Unit tests related to node merge optimization
/////////////////////////////////////////////////////////////////////

// clang-format off
TEST_F(MklLayoutPassTest, Basic) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Zeta);D(Zeta)|"
            "A->C;A->D;B->C:1;B->D:1");
}

// Test set 1: Conv2D + AddBias

// C=Conv2D(A,B); E=BiasAdd(C,D); Z=Zeta(E,Y)
#define REGISTER_TEST(NAME, T, INPUT)                                      \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                  \
    CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);    \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                         \
              "node { name: 'B' op: '" #INPUT "'}"                         \
              "node { name: 'C' op: 'Conv2D'"                              \
              " attr { key: 'T'                value { type:" #T " } }"    \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"      \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"       \
              " attr { key: 'strides'          value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " attr { key: 'padding'          value { s: 'SAME' } }"      \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " input: ['A', 'B']}"                                        \
              "node { name: 'D' op: '" #INPUT "'}"                         \
              "node { name: 'E' op: 'BiasAdd'"                             \
              " attr { key: 'T'                value { type:" #T " } }"    \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"      \
              " input: ['C', 'D'] }"                                       \
              "node { name: 'Y' op: '" #INPUT "'}"                         \
              "node { name: 'Z' op: 'Zeta'"                                \
              " attr {key: 'T'                 value { type:" #T " } }"    \
              " input: ['E', 'Y']}");                                      \
                                                                           \
    EXPECT_EQ(                                                             \
        DoMklLayoutOptimizationPass(),                                     \
        "A(" #INPUT ");B(" #INPUT ");D(" #INPUT ");DMT/_0(Const);"         \
        "DMT/_1(Const);DMT/_2(Const);E(_MklConv2DWithBias);Y(" #INPUT ");" \
        "Z(Zeta)|A->E;A:control->DMT/_0:control;A:control->DMT/_1:control;"\
        "A:control->DMT/_2:control;B->E:1;D->E:2;DMT/_0->E:3;DMT/_1->E:4;" \
        "DMT/_2->E:5;E->Z;Y->Z:1");                                        \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DWithBias_Positive);
#undef REGISTER_TEST

// Graph contains only Conv2D, no AddBias.
#define REGISTER_TEST(NAME, T, INPUT)                                      \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                  \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                         \
              "node { name: 'B' op: '" #INPUT "'}"                         \
              "node { name: 'C' op: 'Conv2D'"                              \
              " attr { key: 'T'                value { type:" #T " } }"    \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"      \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"       \
              " attr { key: 'strides'          value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " attr { key: 'padding'          value { s: 'SAME' } }"      \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " input: ['A', 'B']}");                                      \
    EXPECT_EQ(                                                             \
        DoMklLayoutOptimizationPass(),                                     \
        "A(" #INPUT ");B(" #INPUT ");C(_MklConv2D);DMT/_0(Const);"         \
        "DMT/_1(Const)|A->C;A:control->DMT/_0:control;A:control->"         \
        "DMT/_1:control;B->C:1;DMT/_0->C:2;DMT/_1->C:3");                  \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DWithBias_Negative_NoAddBias);
#undef REGISTER_TEST

// Conv2D output does not go to BiasAdd.
#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                             \
              "node { name: 'B' op: '" #INPUT "'}"                             \
              "node { name: 'C' op: 'Conv2D'"                                  \
              " attr { key: 'T'                value { type:" #T "} }"         \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"          \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"           \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "     \
              "i:1, i:1} } }"                                                  \
              " attr { key: 'padding'          value { s: 'SAME' } }"          \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "     \
              "i:1, i:1} } }"                                                  \
              " input: ['A', 'B']}"                                            \
              "node { name: 'D' op: '" #INPUT "'}"                             \
              "node { name: 'E' op: '" #INPUT "'}"                             \
              "node { name: 'F' op: 'BiasAdd'"                                 \
              " attr { key: 'T'                value { type:" #T "} }"         \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"          \
              " input: ['D', 'E'] }");                                         \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                   \
              "A(" #INPUT ");B(" #INPUT ");C(_MklConv2D);D(" #INPUT ");"       \
              "DMT/_0(Const);DMT/_1(Const);E(" #INPUT ");F(BiasAdd)|A->C;"     \
              "A:control->DMT/_0:control;A:control->DMT/_1:control;B->C:1;"    \
              "D->F;DMT/_0->C:2;DMT/_1->C:3;E->F:1");                          \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DWithBias_Negative_Dataflow1);
#undef REGISTER_TEST

// Conv2D has two outgoing edges: BiasAdd and some other dummy node (Zeta).
// Merge should not be done in such case.
#define REGISTER_TEST(NAME, T, INPUT)                                      \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                  \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                         \
              "node { name: 'B' op: '" #INPUT "'}"                         \
              "node { name: 'C' op: 'Conv2D'"                              \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"      \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"       \
              " attr { key: 'strides'          value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " attr { key: 'padding'          value { s: 'SAME' } }"      \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " input: ['A', 'B']}"                                        \
              "node { name: 'D' op: '" #INPUT "'}"                         \
              "node { name: 'E' op: '" #INPUT "'}"                         \
              "node { name: 'F' op: 'BiasAdd'"                             \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"      \
              " input: ['D', 'E'] }"                                       \
              "node { name: 'G' op: 'Zeta'"                                \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " input: ['C', 'E'] }");                                     \
    EXPECT_EQ(                                                             \
        DoMklLayoutOptimizationPass(),                                     \
        "A(" #INPUT ");B(" #INPUT ");C(_MklConv2D);D(" #INPUT ");"         \
        "DMT/_0(Const);DMT/_1(Const);E(" #INPUT ");F(BiasAdd);G(Zeta)|"    \
        "A->C;A:control->DMT/_0:control;A:control->DMT/_1:control;B->C:1;" \
        "C->G;D->F;DMT/_0->C:2;DMT/_1->C:3;E->F:1;E->G:1");                \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DWithBias_Negative_Dataflow2);
#undef REGISTER_TEST

// data_format attribute value mismatch. Merge should not be done
// in such case.
#define REGISTER_TEST(NAME, T, INPUT)                                      \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                  \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                         \
              "node { name: 'B' op: '" #INPUT "'}"                         \
              "node { name: 'C' op: 'Conv2D'"                              \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"      \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"       \
              " attr { key: 'strides'          value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " attr { key: 'padding'          value { s: 'SAME' } }"      \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " input: ['A', 'B']}"                                        \
              "node { name: 'D' op: '" #INPUT "'}"                         \
              "node { name: 'E' op: 'BiasAdd'"                             \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'data_format'      value { s: 'NHCW' } }"      \
              " input: ['C', 'D'] }");                                     \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                               \
              "A(" #INPUT ");B(" #INPUT ");C(_MklConv2D);D(" #INPUT ");"   \
              "DMT/_0(Const);DMT/_1(Const);E(BiasAdd)|A->C;A:control->"    \
              "DMT/_0:control;A:control->DMT/_1:control;B->C:1;C->E;"      \
              "D->E:1;DMT/_0->C:2;DMT/_1->C:3");                           \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DWithBias_Negative_AttrMismatch);
#undef REGISTER_TEST

// Test set 2: BiasAddGrad + Conv2DBackpropFilter fusion tests

#define REGISTER_TEST(NAME, T, INPUT)                                        \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                    \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                           \
              "node { name: 'B' op: 'Int32Input'}"                           \
              "node { name: 'C' op: '" #INPUT "'}"                           \
              "node { name: 'D' op: 'Conv2DBackpropFilter'"                  \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " attr { key: 'padding'          value { s: 'SAME' } }"        \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " input: ['A', 'B', 'C'] }"                                    \
              "node { name: 'E' op: 'BiasAddGrad'"                           \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " input: ['C'] }");                                            \
    EXPECT_EQ(                                                               \
        DoMklLayoutOptimizationPass(),                                       \
        "A(" #INPUT ");B(Int32Input);C(" #INPUT ");"                         \
        "D(_MklConv2DBackpropFilterWithBias);DMT/_0(Const);DMT/_1(Const);"   \
        "DMT/_2(Const)|A->D;A:control->DMT/_0:control;A:control->"           \
        "DMT/_1:control;A:control->DMT/_2:control;B->D:1;C->D:2;"            \
        "DMT/_0->D:3;DMT/_1->D:4;DMT/_2->D:5");                              \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DBackpropFilterFusion_Positive);
#undef REGISTER_TEST

// BiasAddGrad fusion in the presence of BackpropFilter. But nodes do not match
// criteria for rewrite. So rewrite should not happen. 3rd input of
// Conv2DBackpropFilter is different than input to BiasAddGrad.
#define REGISTER_TEST(NAME, T, INPUT)                                        \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                    \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                           \
              "node { name: 'B' op: 'Int32Input'}"                           \
              "node { name: 'C' op: '" #INPUT "'}"                           \
              "node { name: 'D' op: 'Conv2DBackpropFilter'"                  \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " attr { key: 'padding'          value { s: 'SAME' } }"        \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " input: ['A', 'B', 'C'] }"                                    \
              "node { name: 'E' op: 'BiasAddGrad'"                           \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " input: ['A'] }");                                            \
    EXPECT_EQ(                                                               \
        DoMklLayoutOptimizationPass(),                                       \
        "A(" #INPUT ");B(Int32Input);C(" #INPUT ");"                         \
        "D(_MklConv2DBackpropFilter);DMT/_0(Const);DMT/_1(Const);"           \
        "DMT/_2(Const);E(BiasAddGrad)|A->D;A->E;A:control->DMT/_0:control;"  \
        "A:control->DMT/_1:control;A:control->DMT/_2:control;B->D:1;C->D:2;" \
        "DMT/_0->D:3;DMT/_1->D:4;DMT/_2->D:5");                              \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DBackpropFilterFusion_Negative1);
#undef REGISTER_TEST

// BiasAddGrad fusion, but nodes do not match criteria for fusion.
// Different input formats.
#define REGISTER_TEST(NAME, T, INPUT)                                        \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                    \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                           \
              "node { name: 'B' op: 'Int32Input'}"                           \
              "node { name: 'C' op: '" #INPUT "'}"                           \
              "node { name: 'D' op: 'Conv2DBackpropFilter'"                  \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " attr { key: 'padding'          value { s: 'SAME' } }"        \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " input: ['A', 'B', 'C'] }"                                    \
              "node { name: 'E' op: 'BiasAddGrad'"                           \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NHWC' } }"        \
              " input: ['A'] }");                                            \
    EXPECT_EQ(                                                               \
        DoMklLayoutOptimizationPass(),                                       \
        "A(" #INPUT ");B(Int32Input);C(" #INPUT ");"                         \
        "D(_MklConv2DBackpropFilter);DMT/_0(Const);DMT/_1(Const);"           \
        "DMT/_2(Const);E(BiasAddGrad)|A->D;A->E;A:control->DMT/_0:control;"  \
        "A:control->DMT/_1:control;A:control->DMT/_2:control;B->D:1;C->D:2;" \
        "DMT/_0->D:3;DMT/_1->D:4;DMT/_2->D:5");                              \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DBackpropFilterFusion_Negative2);
#undef REGISTER_TEST

// BiasAddGrad fusion in the presence of BackpropFilter only. Fusion is done
// before node rewrite. Check this ordering.
#define REGISTER_TEST(NAME, T, INPUT)                                       \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                   \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                          \
              "node { name: 'B' op: '" #INPUT "'}"                          \
              "node { name: 'C' op: '" #INPUT "'}"                          \
              "node { name: 'M' op: '_MklInput'}"                           \
              "node { name: 'N' op: '_MklInput'}"                           \
              "node { name: 'O' op: '_MklInput'}"                           \
              "node { name: 'D' op: '_MklConv2DWithBias'"                   \
              " attr { key: 'T'                value { type: " #T " } }"    \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"       \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"        \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "  \
              "i:1, i:1} } }"                                               \
              " attr { key: 'padding'          value { s: 'SAME' } }"       \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "  \
              "i:1, i:1} } }"                                               \
              " input: ['A', 'B', 'C', 'M', 'N', 'O']}"                     \
              "node { name: 'E' op: 'Zeta'"                                 \
              " attr {key: 'T'                 value { type: " #T " } }"    \
              " input: ['D', 'A']}"                                         \
              "node { name: 'F' op: 'Int32Input'}"                          \
              "node { name: 'G' op: '_MklConv2DBackpropFilter'"             \
              " attr { key: 'T'                value { type: " #T " } }"    \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"       \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"        \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "  \
              "i:1, i:1} } }"                                               \
              " attr { key: 'padding'          value { s: 'SAME' } }"       \
              " input: ['E', 'F', 'A', 'M', 'N', 'O'] }"                    \
              "node { name: 'H' op: 'BiasAddGrad'"                          \
              " attr { key: 'T'                value { type: " #T " } }"    \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"       \
              " input: ['E'] }");                                           \
    EXPECT_EQ(                                                              \
        DoMklLayoutOptimizationPass(),                                      \
        "A(" #INPUT ");B(" #INPUT ");C(" #INPUT ");D(_MklConv2DWithBias);"  \
        "E(Zeta);F(Int32Input);G(_MklConv2DBackpropFilter);H(BiasAddGrad);" \
        "M(_MklInput);N(_MklInput);O(_MklInput)|A->D;A->E:1;A->G:2;B->D:1;" \
        "C->D:2;D->E;E->G;E->H;F->G:1;M->D:3;M->G:3;N->D:4;N->G:4;O->D:5;"  \
        "O->G:5");                                                          \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DBackpropFilterFusion_Negative3);
#undef REGISTER_TEST

// C=Conv2D(A,B); E=BiasAdd(C,D); Y=Zeta(E,X);
// G=Conv2DBackpropInput(F,B,E)
// This is a case of node rewrite followed by node merge followed by connecting
// filter output of Conv2DWithBias to filter input of Conv2DBackpropInput.
#define REGISTER_TEST(NAME, T, INPUT)                                        \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                    \
    CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);      \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                           \
              "node { name: 'B' op: '" #INPUT "'}"                           \
              "node { name: 'C' op: 'Conv2D'"                                \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " attr { key: 'padding'          value { s: 'SAME' } }"        \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " input: ['A', 'B']}"                                          \
              "node { name: 'D' op: '" #INPUT "'}"                           \
              "node { name: 'E' op: 'BiasAdd'"                               \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " input: ['C', 'D'] }"                                         \
              "node { name: 'X' op: '" #INPUT "'}"                           \
              "node { name: 'Y' op: 'Zeta'"                                  \
              " attr {key: 'T'                 value { type: " #T " } }"     \
              " input: ['E', 'X']}"                                          \
              "node { name: 'F' op: 'Int32Input'}"                           \
              "node { name: 'G' op: 'Conv2DBackpropInput'"                   \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " attr { key: 'padding'          value { s: 'SAME' } }"        \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " input: ['F', 'B', 'E']}"                                     \
              "node { name: 'Z' op: 'Zeta'"                                  \
              " attr {key: 'T'                 value { type: " #T " } }"     \
              " input: ['G', 'X']}");                                        \
    EXPECT_EQ(                                                               \
        DoMklLayoutOptimizationPass(),                                       \
        "A(" #INPUT ");B(" #INPUT ");D(" #INPUT ");DMT/_0(Const);"           \
        "DMT/_1(Const);DMT/_2(Const);DMT/_3(Const);E(_MklConv2DWithBias);"   \
        "F(Int32Input);G(_MklConv2DBackpropInput);X(" #INPUT ");Y(Zeta);"    \
        "Z(Zeta)|A->E;A:control->DMT/_0:control;A:control->DMT/_1:control;"  \
        "A:control->DMT/_2:control;B->E:1;D->E:2;DMT/_0->E:3;"               \
        "DMT/_1->E:4;DMT/_2->E:5;DMT/_3->G:3;E->G:2;E->Y;E:1->G:1;E:2->G:5;" \
        "E:3->G:4;F->G;F:control->DMT/_3:control;G->Z;X->Y:1;X->Z:1");       \
  }
// TODO(nhasabni): Enable bfloat16 test when we enable the operator.
REGISTER_TEST_FLOAT32(NodeMerge_Conv2DWithBias_ConvBpropInput_FilterFwd);
#undef REGISTER_TEST

// Test set 3: Pad + Conv2D fusion
// padding is VALID type
// A = input(image), B = input(paddings), C= Pad = input of conv2D,
// D=input(filter), E = Conv2D, Z = Zeta
// C=Pad(A,B); E=Conv2D(C,D); Z=Zeta(E,Y)
// After layout pass
// _MklPadWithConv2D(A, D, B, DMT/_0, DMT/_1, DMT/_2)
#define REGISTER_TEST(NAME, T, INPUT)                                      \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                  \
    DCHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);   \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                         \
              "node { name: 'B' op: 'Int32Input'}"                         \
              "node { name: 'C' op: 'Pad'"                                 \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'Tpaddings'        value { type: DT_INT32 } }" \
              " input: ['A', 'B']}"                                        \
              "node { name: 'D' op: '" #INPUT "'}"                         \
              "node { name: 'E' op: 'Conv2D'"                              \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'data_format'      value { s: 'NHWC' } }"      \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"       \
              " attr { key: 'strides'          value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " attr { key: 'padding'          value { s: 'VALID' } }"     \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " input: ['C', 'D'] }"                                       \
              "node { name: 'Y' op: '" #INPUT "'}"                         \
              "node { name: 'Z' op: 'Zeta'"                                \
              " attr {key: 'T'                 value { type: " #T " } }"   \
              " input: ['E', 'Y']}");                                      \
    EXPECT_EQ(                                                             \
        DoMklLayoutOptimizationPass(),                                     \
        "A(" #INPUT ");B(Int32Input);D(" #INPUT ");DMT/_0(Const);"         \
        "DMT/_1(Const);DMT/_2(Const);E(_MklPadWithConv2D);Y(" #INPUT ");"  \
        "Z(Zeta)|A->E;A:control->DMT/_0:control;A:control->DMT/_1:control;"\
        "A:control->DMT/_2:control;B->E:2;D->E:1;DMT/_0->E:3;DMT/_1->E:4;" \
        "DMT/_2->E:5;E->Z;Y->Z:1");                                        \
  }
// TODO(nhasabni): Enable bfloat16 test when we enable the operator.
REGISTER_TEST_FLOAT32(NodeMerge_PadWithConv2D_Positive);
#undef REGISTER_TEST

// Test if input control edges do not duplicate after merge.
// If both the merging ops have input control edge from a common op
// then, the merged op will have only one control edge from that
// common op.
// padding is VALID type
// A = input(image), A1 = input, B = input(paddings),
// C= Pad = input of conv2D,
// D=input(filter), E = Conv2D, Z = Zeta
// C=Pad(A,B); E=Conv2D(C,D); Z=Zeta(E,Y)
// A1:control->C:control
// A1:control->E:control
// After layout pass:
// _MklPadWithConv2D(A, D, B, DMT/_0, DMT/_1, DMT/_2)
// A1:control->E:control (only one control edge)
#define REGISTER_TEST(NAME, T, INPUT)                                      \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                  \
    DCHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);   \
    InitGraph("node { name: 'A1' op: '" #INPUT "'}"                        \
              "node { name: 'A' op: '" #INPUT "'}"                         \
              "node { name: 'B' op: 'Int32Input'}"                         \
              "node { name: 'C' op: 'Pad'"                                 \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'Tpaddings'        value { type: DT_INT32 } }" \
              " input: ['A', 'B']}"                                        \
              "node { name: 'D' op: '" #INPUT "'}"                         \
              "node { name: 'E' op: 'Conv2D'"                              \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'data_format'      value { s: 'NHWC' } }"      \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"       \
              " attr { key: 'strides'          value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " attr { key: 'padding'          value { s: 'VALID' } }"     \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " input: ['C', 'D'] }"                                       \
              "node { name: 'Y' op: '" #INPUT "'}"                         \
              "node { name: 'Z' op: 'Zeta'"                                \
              " attr {key: 'T'                 value { type: " #T " } }"   \
              " input: ['E', 'Y']}");                                      \
    Node* a1 = FindNode("A1");                                             \
    Node* c = FindNode("C");                                               \
    Node* e = FindNode("E");                                               \
    const Edge* edge = graph_.AddControlEdge(a1, c);                       \
    const Edge* edge_1 = graph_.AddControlEdge(a1, e);                     \
    ASSERT_NE(edge, nullptr);                                              \
    ASSERT_NE(edge_1, nullptr);                                            \
    EXPECT_EQ(                                                             \
        DoMklLayoutOptimizationPass(),                                     \
        "A(" #INPUT ");A1(" #INPUT ");B(Int32Input);D(" #INPUT ");"        \
        "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);E(_MklPadWithConv2D);" \
        "Y(" #INPUT ");Z(Zeta)|A->E;A1:control->E:control;A:control->"    \
        "DMT/_0:control;A:control->DMT/_1:control;A:control->"           \
        "DMT/_2:control;B->E:2;D->E:1;DMT/_0->E:3;DMT/_1->E:4;"            \
        "DMT/_2->E:5;E->Z;Y->Z:1");                                        \
  }
// TODO(nhasabni): Enable bfloat16 test when we enable the operator.
REGISTER_TEST_FLOAT32(Input_ControlEdge_PadWithConv2D_Positive);
#undef REGISTER_TEST

// Test if output control edges does not duplicate after merge.
// If both the merging ops have output control edge to a common op,
// then after merge, the merged op will have only one control edge
// to that common op.
// padding is VALID type
// A = input(image), B = input(paddings), C= Pad = input of conv2D,
// D=input(filter), E = Conv2D, Z = Zeta
// C=Pad(A,B); E=Conv2D(C,D); Z=Zeta(E,Y)
// C:control->A1:control
// E:control->A1:control
// After layout pass:
// _MklPadWithConv2D(A, D, B, DMT/_0, DMT/_1, DMT/_2)
// E:control->A1:control (only one control edge)
#define REGISTER_TEST(NAME, T, INPUT)                                      \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                  \
    DCHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);   \
    InitGraph("node { name: 'A1' op: '" #INPUT "'}"                        \
              "node { name: 'A' op: '" #INPUT "'}"                         \
              "node { name: 'B' op: 'Int32Input'}"                         \
              "node { name: 'C' op: 'Pad'"                                 \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'Tpaddings'        value { type: DT_INT32 } }" \
              " input: ['A', 'B']}"                                        \
              "node { name: 'D' op: '" #INPUT "'}"                         \
              "node { name: 'E' op: 'Conv2D'"                              \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'data_format'      value { s: 'NHWC' } }"      \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"       \
              " attr { key: 'strides'          value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " attr { key: 'padding'          value { s: 'VALID' } }"     \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " input: ['C', 'D'] }"                                       \
              "node { name: 'Y' op: '" #INPUT "'}"                         \
              "node { name: 'Z' op: 'Zeta'"                                \
              " attr {key: 'T'                 value { type: " #T " } }"   \
              " input: ['E', 'Y']}");                                      \
    Node* a1 = FindNode("A1");                                             \
    Node* c = FindNode("C");                                               \
    Node* e = FindNode("E");                                               \
    const Edge* edge = graph_.AddControlEdge(c, a1);                       \
    const Edge* edge_1 = graph_.AddControlEdge(e, a1);                     \
    ASSERT_NE(edge, nullptr);                                              \
    ASSERT_NE(edge_1, nullptr);                                            \
    EXPECT_EQ(                                                             \
        DoMklLayoutOptimizationPass(),                                     \
        "A(" #INPUT ");A1(" #INPUT ");B(Int32Input);D(" #INPUT ");"        \
        "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);E(_MklPadWithConv2D);"  \
        "Y(" #INPUT ");Z(Zeta)|A->E;A:control->DMT/_0:control;A:control->" \
        "DMT/_1:control;A:control->DMT/_2:control;B->E:2;D->E:1;"          \
        "DMT/_0->E:3;DMT/_1->E:4;DMT/_2->E:5;E->Z;E:control->A1:control;"  \
        "Y->Z:1");                                                         \
  }
// TODO(nhasabni): Enable bfloat16 test when we enable the operator.
REGISTER_TEST_FLOAT32(Output_ControlEdge_PadWithConv2D_Positive);
#undef REGISTER_TEST

// Pad + Conv2D fusion with padding is VALID,
// Input node pointing to both Pad and Conv2D
// A = input(image), B = input(paddings), C= Pad
// E = Conv2D, Z = Zeta
// C=Pad(A,B); E=Conv2D(C,A); Z=Zeta(E,Y)
// After layout pass
// _MklPadWithConv2D(A, A, B, DMT/_0, DMT/_1, DMT/_2)
#define REGISTER_TEST(NAME, T, INPUT)                                      \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                  \
    DCHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);   \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                         \
              "node { name: 'B' op: 'Int32Input'}"                         \
              "node { name: 'C' op: 'Pad'"                                 \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'Tpaddings'        value { type: DT_INT32 } }" \
              " input: ['A', 'B']}"                                        \
              "node { name: 'E' op: 'Conv2D'"                              \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'data_format'      value { s: 'NHWC' } }"      \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"       \
              " attr { key: 'strides'          value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " attr { key: 'padding'          value { s: 'VALID' } }"     \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " input: ['C', 'A'] }"                                       \
              "node { name: 'Y' op: '" #INPUT "'}"                         \
              "node { name: 'Z' op: 'Zeta'"                                \
              " attr {key: 'T'                 value { type: " #T " } }"   \
              " input: ['E', 'Y']}");                                      \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                               \
              "A(" #INPUT ");B(Int32Input);DMT/_0(Const);DMT/_1(Const);"   \
              "DMT/_2(Const);E(_MklPadWithConv2D);Y(" #INPUT ");Z(Zeta)|"  \
              "A->E;A->E:1;A:control->DMT/_0:control;A:control->"          \
              "DMT/_1:control;A:control->DMT/_2:control;B->E:2;DMT/_0->E:3;"\
              "DMT/_1->E:4;DMT/_2->E:5;E->Z;Y->Z:1");                      \
  }
// TODO(nhasabni): Enable bfloat16 test when we enable the operator.
REGISTER_TEST_FLOAT32(NodeMerge_PadWithConv2D_Common_Input);
#undef REGISTER_TEST

// Pad + Conv2D with padding is VALID,
// Input node pointing to both Pad and Conv2D
// Output of both Pad and Conv2D feeds one node (Z as Output2)
// A = input(as image), B = input(as paddings), C= Pad
// E = Conv2D, Z = Output2
// C=Pad(A,B); E=Conv2D(C,A); Z=Output(C,E)
// After layout pass - No merging, since Pad and Conv2D both
// feed to the same node (Z)
#define REGISTER_TEST(NAME, T, INPUT, OUTPUT)                              \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                  \
    DCHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);   \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                         \
              "node { name: 'B' op: 'Int32Input'}"                         \
              "node { name: 'C' op: 'Pad'"                                 \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'Tpaddings'        value { type: DT_INT32 } }" \
              " input: ['A', 'B']}"                                        \
              "node { name: 'E' op: 'Conv2D'"                              \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'data_format'      value { s: 'NHWC' } }"      \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"       \
              " attr { key: 'strides'          value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " attr { key: 'padding'          value { s: 'VALID' } }"     \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " input: ['C', 'A'] }"                                       \
              "node { name: 'Z' op: '" #OUTPUT "'"                         \
              " input: ['C', 'E']}");                                      \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                               \
              "A(" #INPUT ");B(Int32Input);C(Pad);DMT/_0(Const);"          \
              "DMT/_1(Const);E(_MklConv2D);Z(" #OUTPUT ")|A->C;A->E:1;"    \
              "B->C:1;C->E;C->Z;C:control->DMT/_0:control;C:control->"     \
              "DMT/_1:control;DMT/_0->E:2;DMT/_1->E:3;E->Z:1");            \
  }
REGISTER_TEST(NodeMerge_PadWithConv2D_Common_InOutput, DT_FLOAT, Float32Input,
              Float32Output2);
#ifdef ENABLE_INTEL_MKL_BFLOAT16
// TODO(nhasabni): Enable bfloat16 test when we enable the operator.
REGISTER_TEST(NodeMerge_PadWithConv2D_Common_InOutput, DT_BFLOAT16,
              BFloat16Input, BFloat16Output2);
#endif
#undef REGISTER_TEST

// Pad + Conv2D; padding is SAME
// A = input(image), B = input(paddings), C= Pad = input of conv2D,
// D=input(filter), E = Conv2D, Z = Zeta
// C=Pad(A,B); E=Conv2D(C,D); Z=Zeta(E,Y)
// After layout pass - No merging
#define REGISTER_TEST(NAME, T, INPUT)                                      \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                  \
    DCHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);   \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                         \
              "node { name: 'B' op: 'Int32Input'}"                         \
              "node { name: 'C' op: 'Pad'"                                 \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'Tpaddings'        value { type: DT_INT32 } }" \
              " input: ['A', 'B']}"                                        \
              "node { name: 'D' op: '" #INPUT "'}"                         \
              "node { name: 'E' op: 'Conv2D'"                              \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'data_format'      value { s: 'NHWC' } }"      \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"       \
              " attr { key: 'strides'          value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " attr { key: 'padding'          value { s: 'SAME' } }"      \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " input: ['C', 'D'] }"                                       \
              "node { name: 'Y' op: '" #INPUT "'}"                         \
              "node { name: 'Z' op: 'Zeta'"                                \
              " attr {key: 'T'                 value { type: " #T " } }"   \
              " input: ['E', 'Y']}");                                      \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                               \
              "A(" #INPUT ");B(Int32Input);C(Pad);D(" #INPUT ");"          \
              "DMT/_0(Const);DMT/_1(Const);E(_MklConv2D);Y(" #INPUT ");"   \
              "Z(Zeta)|A->C;B->C:1;C->E;C:control->DMT/_0:control;"        \
              "C:control->DMT/_1:control;D->E:1;DMT/_0->E:2;DMT/_1->E:3;"  \
              "E->Z;Y->Z:1");                                              \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_PadWithConv2D_Negative);
#undef REGISTER_TEST

#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
    InitGraph(                                                                 \
        "node { name: 'Input0' op: '" #INPUT "'}"                              \
        "node { name: 'Input1' op: '" #INPUT "'}"                              \
        "node { name: 'Const0' op: 'Const'"                                    \
        "  attr {key: 'dtype'                  value { type: DT_INT32 } }"     \
        "  attr {key: 'value'                  value { "                       \
        "       tensor {"                                                      \
        "       dtype: DT_INT32"                                               \
        "       tensor_shape { dim { size: 4 } }"                              \
        "       tensor_content: "                                              \
        "'\\000\\000\\000\\000\\002\\000\\000\\000\\003\\000\\000\\000\\001\\" \
        "000\\000\\000'"                                                       \
        "        }"                                                            \
        "     }"                                                               \
        "  }"                                                                  \
        "}"                                                                    \
        "node { name: 'Const1' op: 'Const'"                                    \
        "  attr {key: 'dtype'                   value { type: DT_INT32 } }"    \
        "  attr {key: 'value'"                                                 \
        "   value {"                                                           \
        "     tensor {"                                                        \
        "       dtype: DT_INT32"                                               \
        "       tensor_shape {dim {size: 4 }}"                                 \
        "       tensor_content: "                                              \
        "'\\000\\000\\000\\000\\003\\000\\000\\000\\001"                       \
        "\\000\\000\\000\\002\\000\\000\\000'"                                 \
        "     }"                                                               \
        "   }"                                                                 \
        " }"                                                                   \
        "}"                                                                    \
        "node { name: 'Transpose0' op: 'Transpose'"                            \
        " input: ['Input0', 'Const0']"                                         \
        " attr { key: 'T'                       value { type: " #T  "} }"      \
        " attr { key: 'Tperm'                   value { type: DT_INT32 } } }"  \
        "node { name: 'Conv2D'     op: 'Conv2D'"                               \
        " input: ['Transpose0', 'Input1']"                                     \
        " attr { key: 'T'                       value { type: " #T "} }"       \
        " attr { key: 'data_format'             value { s: 'NHWC' }}"          \
        " attr { key: 'dilations'            value {list: {i:1,i:1,i:1,i:1}}}" \
        " attr { key: 'padding'                 value {s: 'SAME'}}"            \
        " attr { key: 'strides'              value {list: {i:1,i:1,i:1,i:1}}}" \
        " attr { key: 'use_cudnn_on_gpu'      value {b: true}}}"               \
        "node { name: 'Transpose1' op: 'Transpose'"                            \
        " input: ['Conv2D', 'Const1' ]"                                        \
        " attr { key: 'T'                       value { type: " #T "}}"        \
        " attr { key: 'Tperm'                   value { type: DT_INT32  }}}"   \
        "node { name: 'Relu' op: 'Relu'"                                       \
        " attr { key: 'T'                value { type: " #T "} }"              \
        " input: ['Transpose1'] }");                                           \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                   \
              "Const0(Const);Const1(Const);Conv2D(_MklConv2D);DMT/_0(Const);"  \
              "DMT/_1(Const);Input0(" #INPUT ");Input1(" #INPUT ");"           \
              "Relu(_MklRelu)|Conv2D->Relu;Conv2D:2->Relu:1;DMT/_0->Conv2D:2;" \
              "DMT/_1->Conv2D:3;Input0->Conv2D;Input0:control->DMT/_0:control;"\
              "Input0:control->DMT/_1:control;Input1->Conv2D:1");              \
}
REGISTER_TEST_ALL_TYPES(NodeMerge_TransposeConv2DTranspose_Positive);
#undef REGISTER_TEST

#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
    InitGraph(                                                                 \
        "node { name: 'Input0' op: '" #INPUT "'}"                              \
        "node { name: 'Input1' op: '" #INPUT "'}"                              \
        "node { name: 'Const0' op: 'Const'"                                    \
        " attr { key: 'dtype'                   value { type: DT_INT32 } }"    \
        " attr { key: 'value'"                                                 \
        "   value {"                                                           \
        "     tensor {"                                                        \
        "       dtype: DT_INT32"                                               \
        "       tensor_shape {dim {size: 4}}"                                  \
        "       tensor_content: "                                              \
        "'\\000\\000\\000\\000\\002\\000\\000\\000\\003\\000\\000\\000\\001\\" \
        "000\\000\\000'"                                                       \
        "     }"                                                               \
        "   }"                                                                 \
        " }"                                                                   \
        "}"                                                                    \
        "node { name: 'Const1' op: 'Const'"                                    \
        "  attr { key: 'dtype'                  value { type: DT_INT32 }}"     \
        "  attr {"                                                             \
        "   key: 'value'"                                                      \
        "   value {"                                                           \
        "     tensor {"                                                        \
        "       dtype: DT_INT32"                                               \
        "       tensor_shape { dim { size: 4 }}"                               \
        "       tensor_content: "                                              \
        "'\\000\\000\\000\\000\\002\\000\\000\\000\\003\\000\\000\\000\\001\\" \
        "000\\000\\000'"                                                       \
        "     }"                                                               \
        "   }"                                                                 \
        " }"                                                                   \
        "}"                                                                    \
        "node {name: 'Transpose0'   op: 'Transpose'"                           \
        " input: ['Input0', 'Const0']"                                         \
        " attr { key: 'T'                       value { type: " #T " }}"       \
        " attr { key: 'Tperm'                   value { type: DT_INT32 }}}"    \
        "node { name: 'Conv2D'  op: 'Conv2D'"                                  \
        " input: ['Transpose0', 'Input1']"                                     \
        " attr { key: 'T'                       value { type: " #T "}}"        \
        " attr { key: 'data_format'             value { s: 'NHWC'  }}"         \
        " attr { key: 'dilations'           value { list: {i:1,i:1,i:1,i:1}}}" \
        " attr { key: 'padding'                 value { s: 'SAME' }}"          \
        " attr { key: 'strides'             value { list: {i:1,i:1,i:1,i:1}}}" \
        " attr { key: 'use_cudnn_on_gpu'        value { b: true }}}"           \
        "node {name: 'Transpose1' op: 'Transpose'"                             \
        " input: ['Conv2D', 'Const1']"                                         \
        " attr { key: 'T'                       value { type: " #T "}}"        \
        " attr { key: 'Tperm'                   value { type: DT_INT32 }}}"    \
        "node { name: 'Relu' op: 'Relu'"                                       \
        " attr { key: 'T'                       value { type: " #T "}}"        \
        " input: ['Transpose1'] }");                                           \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                   \
              "Const0(Const);Const1(Const);Conv2D(_MklConv2D);DMT/_0(Const);"  \
              "DMT/_1(Const);DMT/_2(Const);Input0(" #INPUT ");Input1(" #INPUT  \
              ");Relu(_MklRelu);Transpose0(_MklTranspose);"                    \
              "Transpose1(_MklTranspose)|Const0->Transpose0:1;"                \
              "Const1->Transpose1:1;Conv2D->Transpose1;DMT/_0->Conv2D:2;"      \
              "DMT/_1->Conv2D:3;DMT/_2->Relu:1;Input0->Transpose0;"            \
              "Input1->Conv2D:1;Transpose0->Conv2D;Transpose0:control->DMT/"   \
              "_0:control;Transpose0:control->DMT/_1:control;Transpose1->Relu;"\
              "Transpose1:control->DMT/_2:control");                           \
}
REGISTER_TEST_ALL_TYPES(NodeMerge_TransposeConv2DTranspose_Negative);
#undef REGISTER_TEST


#define REGISTER_TEST(NAME, T, INPUT)                                         \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                     \
    InitGraph(                                                                \
        "node { name: 'Input0' op: '" #INPUT "'}"                             \
        "node { name: 'Input1' op: '" #INPUT "'}"                             \
        "node { name: 'Const0' op: 'Const'"                                   \
        " attr { key: 'dtype' value { type: DT_INT32 } }"                     \
        " attr { key: 'value'"                                                \
        "  value {"                                                           \
        "     tensor {"                                                       \
        "       dtype: DT_INT32"                                              \
        "       tensor_shape { dim {size: 5}}"                                \
        "       tensor_content:"                                              \
        "'\\000\\000\\000\\000\\002\\000\\000\\000\\003\\000\\000\\000\\004"  \
        "\\000\\000\\000\\001\\000\\000\\000'"                                \
        "     }"                                                              \
        "   }"                                                                \
        " }"                                                                  \
        "}"                                                                   \
        "node { name: 'Const1' op: 'Const'"                                   \
        " attr { key: 'dtype' value { type: DT_INT32 } }"                     \
        "  attr { key: 'value'"                                               \
        "    value {"                                                         \
        "      tensor {"                                                      \
        "        dtype: DT_INT32"                                             \
        "        tensor_shape { dim { size: 5 }}"                             \
        "        tensor_content: "                                            \
        "'\\000\\000\\000\\000\\004\\000\\000\\000\\001\\000\\000\\000\\002"  \
        "\\000\\000\\000\\003\\000\\000\\000'"                                \
        "      }"                                                             \
        "    }"                                                               \
        "  }"                                                                 \
        "}"                                                                   \
        "node { name: 'Transpose0' op: 'Transpose'"                           \
        " input: ['Input0', 'Const0']"                                        \
        " attr { key: 'T'                       value { type: " #T " }}"      \
        " attr { key: 'Tperm'                   value { type: DT_INT32 }}}"   \
        "node { name: 'Conv3D' op: 'Conv3D'"                                  \
        "input: ['Transpose0', 'Input1']"                                     \
        " attr { key: 'T'                       value { type: " #T " }}"      \
        " attr { key: 'data_format'             value { s: 'NDHWC'  }}"       \
        " attr { key: 'dilations'      value { list: {i:1,i:1,i:1,i:1,i:1}}}" \
        " attr { key: 'padding'                 value { s: 'SAME' }}"         \
        " attr { key: 'strides'        value { list: {i:1,i:1,i:1,i:1,i:1}}}" \
        " attr { key: 'use_cudnn_on_gpu'        value { b: true }}}"          \
        "node { name: 'Transpose1' op: 'Transpose'"                           \
        " input: ['Conv3D', 'Const1']"                                        \
        " attr { key: 'T'                       value { type: " #T " }}"      \
        " attr { key: 'Tperm'                   value { type: DT_INT32 }}}"   \
        "node { name: 'Relu' op: 'Relu'"                                      \
        " attr { key: 'T'                value { type: " #T " } }"            \
        " input: ['Transpose1'] }");                                          \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                  \
              "Const0(Const);Const1(Const);Conv3D(_MklConv3D);DMT/_0(Const);" \
              "DMT/_1(Const);Input0(" #INPUT ");Input1(" #INPUT ");"          \
              "Relu(_MklRelu)|Conv3D->Relu;Conv3D:2->Relu:1;"                 \
              "DMT/_0->Conv3D:2;DMT/_1->Conv3D:3;Input0->Conv3D;"             \
              "Input0:control->DMT/_0:control;"                               \
              "Input0:control->DMT/_1:control;Input1->Conv3D:1");             \
}
REGISTER_TEST_ALL_TYPES(NodeMerge_TransposeConv3DTranspose_Positive);
#undef REGISTER_TEST

#define REGISTER_TEST(NAME, T, INPUT)                                         \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                     \
    InitGraph(                                                                \
        "node { name: 'Input0' op: '" #INPUT "'}"                             \
        "node { name: 'Input1' op: '" #INPUT "'}"                             \
        "node { name: 'Const0' op: 'Const'"                                   \
        " attr { key: 'dtype' value { type: DT_INT32 } }"                     \
        " attr { key: 'value'"                                                \
        "  value {"                                                           \
        "     tensor {"                                                       \
        "       dtype: DT_INT32"                                              \
        "       tensor_shape { dim {size: 5}}"                                \
        "       tensor_content:"                                              \
        "'\\000\\000\\000\\000\\002\\000\\000\\000\\003\\000\\000\\000\\004"  \
        "\\000\\000\\000\\001\\000\\000\\000'"                                \
        "     }"                                                              \
        "   }"                                                                \
        " }"                                                                  \
        "}"                                                                   \
        "node { name: 'Const1' op: 'Const'"                                   \
        " attr { key: 'dtype' value { type: DT_INT32 } }"                     \
        "  attr { key: 'value'"                                               \
        "    value {"                                                         \
        "      tensor {"                                                      \
        "        dtype: DT_INT32"                                             \
        "        tensor_shape { dim { size: 5 }}"                             \
        "        tensor_content: "                                            \
        "'\\000\\000\\000\\000\\002\\000\\000\\000\\003\\000\\000\\000\\004"  \
        "\\000\\000\\000\\001\\000\\000\\000'"                                \
        "      }"                                                             \
        "    }"                                                               \
        "  }"                                                                 \
        "}"                                                                   \
        "node { name: 'Transpose0' op: 'Transpose'"                           \
        " input: ['Input0', 'Const0']"                                        \
        " attr { key: 'T'                       value { type: " #T " }}"      \
        " attr { key: 'Tperm'                   value { type: DT_INT32 }}}"   \
        "node { name: 'Conv3D' op: 'Conv3D'"                                  \
        "input: ['Transpose0', 'Input1']"                                     \
        " attr { key: 'T'                       value { type: " #T " }}"      \
        " attr { key: 'data_format'             value { s: 'NDHWC'  }}"       \
        " attr { key: 'dilations'      value { list: {i:1,i:1,i:1,i:1,i:1}}}" \
        " attr { key: 'padding'                 value { s: 'SAME' }}"         \
        " attr { key: 'strides'        value { list: {i:1,i:1,i:1,i:1,i:1}}}" \
        " attr { key: 'use_cudnn_on_gpu'        value { b: true }}}"          \
        "node { name: 'Transpose1' op: 'Transpose'"                           \
        " input: ['Conv3D', 'Const1']"                                        \
        " attr { key: 'T'                       value { type: " #T " }}"      \
        " attr { key: 'Tperm'                   value { type: DT_INT32 }}}"   \
        "node { name: 'Relu' op: 'Relu'"                                      \
        " attr { key: 'T'                value { type: " #T " } }"            \
        " input: ['Transpose1'] }");                                          \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                  \
              "Const0(Const);Const1(Const);Conv3D(_MklConv3D);DMT/_0(Const);" \
              "DMT/_1(Const);DMT/_2(Const);Input0(" #INPUT ");"               \
              "Input1(" #INPUT ");Relu(_MklRelu);Transpose0(_MklTranspose);"  \
              "Transpose1(_MklTranspose)|Const0->Transpose0:1;Const1->"       \
              "Transpose1:1;Conv3D->Transpose1;DMT/_0->Conv3D:2;"             \
              "DMT/_1->Conv3D:3;DMT/_2->Relu:1;Input0->Transpose0;Input1->"   \
              "Conv3D:1;Transpose0->Conv3D;Transpose0:control->"              \
              "DMT/_0:control;Transpose0:control->DMT/_1:control;"            \
              "Transpose1->Relu;Transpose1:control->DMT/_2:control");         \
}
REGISTER_TEST_ALL_TYPES(NodeMerge_TransposeConv3DTranspose_Negative);
#undef REGISTER_TEST

#define REGISTER_TEST(NAME, T, INPUT)                                         \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                     \
    InitGraph(                                                                \
        "node { name: 'Input0' op: '" #INPUT "'}"                             \
        "node { name: 'Const0' op: 'Const'"                                   \
        " attr { key: 'dtype' value { type: DT_INT32 } }"                     \
        " attr { key: 'value'"                                                \
        "  value {"                                                           \
        "     tensor {"                                                       \
        "       dtype: DT_INT32"                                              \
        "       tensor_shape { dim {size: 5}}"                                \
        "       tensor_content:"                                              \
        "'\\000\\000\\000\\000\\002\\000\\000\\000\\003\\000\\000\\000\\004"  \
        "\\000\\000\\000\\001\\000\\000\\000'"                                \
        "     }"                                                              \
        "   }"                                                                \
        " }"                                                                  \
        "}"                                                                   \
        "node { name: 'Const1' op: 'Const'"                                   \
        " attr { key: 'dtype' value { type: DT_INT32 } }"                     \
        "  attr { key: 'value'"                                               \
        "    value {"                                                         \
        "      tensor {"                                                      \
        "        dtype: DT_INT32"                                             \
        "        tensor_shape { dim { size: 5 }}"                             \
        "        tensor_content: "                                            \
        "'\\000\\000\\000\\000\\004\\000\\000\\000\\001\\000\\000\\000\\002"  \
        "\\000\\000\\000\\003\\000\\000\\000'"                                \
        "      }"                                                             \
        "    }"                                                               \
        "  }"                                                                 \
        "}"                                                                   \
        "node { name: 'Transpose0' op: 'Transpose'"                           \
        " input: ['Input0', 'Const0']"                                        \
        " attr { key: 'T'                       value { type: " #T " }}"      \
        " attr { key: 'Tperm'                   value { type: DT_INT32 }}}"   \
        "node { name: 'MaxPool3D'   op: 'MaxPool3D'"                          \
        "input: ['Transpose0']"                                               \
        " attr { key: 'T'                       value { type: " #T " }}"      \
        " attr { key: 'data_format'             value { s: 'NDHWC'  }}"       \
        " attr { key: 'padding'                 value { s: 'SAME' }}"         \
        " attr { key: 'strides'        value { list: {i:1,i:2,i:2,i:2,i:1}}}" \
        " attr { key: 'ksize'          value { list: {i:1,i:1,i:1,i:1,i:1}}}}"\
        "node { name: 'Transpose1' op: 'Transpose'"                           \
        " input: ['MaxPool3D', 'Const1']"                                     \
        " attr { key: 'T'                       value { type: " #T " }}"      \
        " attr { key: 'Tperm'                   value { type: DT_INT32 }}}"   \
        "node { name: 'Relu' op: 'Relu'"                                      \
        " attr { key: 'T'                       value { type: " #T " }}"      \
        " input: ['Transpose1'] }");                                          \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                  \
            "Const0(Const);Const1(Const);DMT/_0(Const);Input0(" #INPUT ");"   \
            "MaxPool3D(_MklMaxPool3D);Relu(_MklRelu)|DMT/_0->MaxPool3D:1;"    \
            "Input0->MaxPool3D;Input0:control->DMT/_0:control;"               \
            "MaxPool3D->Relu;MaxPool3D:2->Relu:1");                           \
}
REGISTER_TEST_ALL_TYPES(NodeMerge_TransposeMaxPool3DTranspose_Positive);
#undef REGISTER_TEST

#define REGISTER_TEST(NAME, T, INPUT)                                         \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                     \
    InitGraph(                                                                \
        "node { name: 'Input0' op: '" #INPUT "'}"                             \
        "node { name: 'Const0' op: 'Const'"                                   \
        " attr { key: 'dtype' value { type: DT_INT32 } }"                     \
        " attr { key: 'value'"                                                \
        "  value {"                                                           \
        "     tensor {"                                                       \
        "       dtype: DT_INT32"                                              \
        "       tensor_shape { dim {size: 5}}"                                \
        "       tensor_content:"                                              \
        "'\\000\\000\\000\\000\\002\\000\\000\\000\\003\\000\\000\\000\\004"  \
        "\\000\\000\\000\\001\\000\\000\\000'"                                \
        "     }"                                                              \
        "   }"                                                                \
        " }"                                                                  \
        "}"                                                                   \
        "node { name: 'Const1' op: 'Const'"                                   \
        " attr { key: 'dtype' value { type: DT_INT32 } }"                     \
        "  attr { key: 'value'"                                               \
        "    value {"                                                         \
        "      tensor {"                                                      \
        "        dtype: DT_INT32"                                             \
        "        tensor_shape { dim { size: 5 }}"                             \
        "        tensor_content: "                                            \
        "'\\000\\000\\000\\000\\002\\000\\000\\000\\003\\000\\000\\000\\004"  \
        "\\000\\000\\000\\001\\000\\000\\000'"                                \
        "      }"                                                             \
        "    }"                                                               \
        "  }"                                                                 \
        "}"                                                                   \
        "node { name: 'Transpose0' op: 'Transpose'"                           \
        " input: ['Input0', 'Const0']"                                        \
        " attr { key: 'T'                       value { type: " #T " }}"      \
        " attr { key: 'Tperm'                   value { type: DT_INT32 }}}"   \
        "node { name: 'MaxPool3D'   op: 'MaxPool3D'"                          \
        "input: ['Transpose0']"                                               \
        " attr { key: 'T'                       value { type: " #T " }}"      \
        " attr { key: 'data_format'             value { s: 'NDHWC'  }}"       \
        " attr { key: 'padding'                 value { s: 'SAME' }}"         \
        " attr { key: 'strides'        value { list: {i:1,i:2,i:2,i:2,i:1}}}" \
        " attr { key: 'ksize'          value { list: {i:1,i:1,i:1,i:1,i:1}}}}"\
        "node { name: 'Transpose1' op: 'Transpose'"                           \
        " input: ['MaxPool3D', 'Const1']"                                     \
        " attr { key: 'T'                       value { type: " #T " }}"      \
        " attr { key: 'Tperm'                   value { type: DT_INT32 }}}"   \
        "node { name: 'Relu' op: 'Relu'"                                      \
        " attr { key: 'T'                       value { type: " #T " }}"      \
        " input: ['Transpose1'] }");                                          \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                  \
      "Const0(Const);Const1(Const);DMT/_0(Const);DMT/_1(Const);Input0(" #INPUT\
      ");MaxPool3D(_MklMaxPool3D);Relu(_MklRelu);Transpose0(_MklTranspose);"  \
      "Transpose1(_MklTranspose)|Const0->Transpose0:1;"                       \
      "Const1->Transpose1:1;DMT/_0->MaxPool3D:1;DMT/_1->Relu:1;Input0->"      \
      "Transpose0;MaxPool3D->Transpose1;Transpose0->MaxPool3D;Transpose0:"    \
      "control->DMT/_0:control;Transpose1->Relu;Transpose1:control->"         \
      "DMT/_1:control");                                                      \
}
REGISTER_TEST_ALL_TYPES(NodeMerge_TransposeMaxPool3DTranspose_Negative);
#undef REGISTER_TEST

/////////////////////////////////////////////////////////////////////
//  Unit tests related to rewriting node to Mkl node
/////////////////////////////////////////////////////////////////////

// Single Conv2D Op; No Mkl layer on the input and on the output.
// We will generate dummy Mkl tensor as 2nd input of Conv2D.
#define REGISTER_TEST(NAME, T, INPUT)                                         \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                     \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                            \
              "node { name: 'B' op: '" #INPUT "'}"                            \
              "node { name: 'C' op: 'Conv2D'"                                 \
              " attr { key: 'T'                value { type: " #T " } }"      \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"         \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"          \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'padding'          value { s: 'SAME' } }"         \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " input: ['A', 'B']}"                                           \
              "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: " #T \
              " } }"                                                          \
              " input: ['B', 'C'] }");                                        \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                  \
              "A(" #INPUT ");B(" #INPUT ");C(_MklConv2D);D(Zeta);"            \
              "DMT/_0(Const);DMT/_1(Const)|A->C;A:control->DMT/_0:control;"   \
              "A:control->DMT/_1:control;B->C:1;B->D;C->D:1;DMT/_0->C:2;"     \
              "DMT/_1->C:3");                                                 \
  }
REGISTER_TEST_ALL_TYPES(NodeRewrite_Conv2D_Basic);
#undef REGISTER_TEST

// Test case for the Depthwise FWD pass
#define REGISTER_TEST(NAME, T, INPUT)                                         \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                     \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                            \
              "node { name: 'B' op: '" #INPUT "'}"                            \
              "node { name: 'C' op: 'DepthwiseConv2dNative'"                  \
              " attr { key: 'T'                value { type: " #T " } }"      \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'padding'          value { s: 'SAME' } }"         \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " input: ['A', 'B']}"                                           \
              "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: " #T \
              " } }"                                                          \
              " input: ['B', 'C'] }");                                        \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                  \
              "A(" #INPUT ");B(" #INPUT ");C(_MklDepthwiseConv2dNative);"     \
              "D(Zeta);DMT/_0(Const);DMT/_1(Const)|A->C;A:control->"          \
              "DMT/_0:control;A:control->DMT/_1:control;B->C:1;B->D;C->D:1;"  \
              "DMT/_0->C:2;DMT/_1->C:3");                                     \
  }
REGISTER_TEST_ALL_TYPES(NodeRewrite_DepthwiseConv2dNative_Basic);
#undef REGISTER_TEST

// 2 Conv2D Ops in sequence. Both should get transformed and 1st Conv2D will
// have 2 outputs, both of which will be inputs to next Conv2D.
#define REGISTER_TEST(NAME, T, INPUT)                                         \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                     \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                            \
              "node { name: 'B' op: '" #INPUT "'}"                            \
              "node { name: 'C' op: 'Conv2D'"                                 \
              " attr { key: 'T'                value { type: " #T " } }"      \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"         \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"          \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'padding'          value { s: 'SAME' } }"         \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " input: ['A', 'B']}"                                           \
              "node { name: 'D' op: 'Conv2D'"                                 \
              " attr { key: 'T'                value { type: " #T " } }"      \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"         \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"          \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'padding'          value { s: 'SAME' } }"         \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " input: ['A', 'C']}"                                           \
              "node { name: 'E' op: 'Zeta' "                                  \
              " attr { key: 'T' value { type: " #T "} }"                      \
              " input: ['C', 'D'] }");                                        \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                  \
              "A(" #INPUT ");B(" #INPUT ");C(_MklConv2D);D(_MklConv2D);"      \
              "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);E(Zeta)|A->C;A->D;"  \
              "A:control->DMT/_0:control;A:control->DMT/_1:control;"          \
              "A:control->DMT/_2:control;B->C:1;C->D:1;C->E;"                 \
              "C:2->D:3;D->E:1;DMT/_0->C:2;DMT/_1->C:3;DMT/_2->D:2");         \
  }
REGISTER_TEST_ALL_TYPES(NodeRewrite_Conv2D_Positive1);
#undef REGISTER_TEST

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
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_HALF } }"
      " input: ['B', 'C'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(HalfInput);B(HalfInput);C(Conv2D);D(Zeta)|"
            "A->C;B->C:1;B->D;C->D:1");
}
TEST_F(MklLayoutPassTest, NodeRewrite_QuantizeV2Op_Negative_ConstInp) {
  InitGraph(
      "node { name: 'A' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'B' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'C' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'D' op: 'QuantizeV2'"
      " attr { key: 'T'                 value { type: DT_QUINT8 } }"
      " attr { key: 'mode'              value { s: 'SCALED' } }"
      " attr { key: 'round_mode'        value { s: 'HALF_TO_EVEN' } }"
      " input: ['A', 'B', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_QUINT8 } }"
      " input: ['D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Const);B(Const);C(Const);D(QuantizeV2);E(Zeta)|"
            "A->D;B->D:1;C->D:2;D->E");
}

TEST_F(MklLayoutPassTest, NodeRewrite_QuantizeV2Op_MinFirst) {
  InitGraph(
      "node { name: 'A' op: 'Input' } "
      "node { name: 'B' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'C' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'D' op: 'QuantizeV2'"
      " attr { key: 'T'                 value { type: DT_QUINT8 } }"
      " attr { key: 'mode'              value { s: 'MIN_FIRST' } }"
      " attr { key: 'round_mode'        value { s: 'HALF_TO_EVEN' } }"
      " input: ['A', 'B', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_QUINT8 } }"
      " input: ['D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Const);C(Const);D(_MklQuantizeV2);DMT/_0(Const);DMT/"
            "_1(Const);DMT/_2(Const);E(Zeta)|"
            "A->D;A:control->DMT/_0:control;A:control->DMT/"
            "_1:control;A:control->DMT/_2:control;B->D:1;C->D:2;D->E;DMT/"
            "_0->D:3;DMT/_1->D:4;DMT/_2->D:5");
}

TEST_F(MklLayoutPassTest, NodeRewrite_QuantizeV2Op_Negative_NarrowRange_True) {
  InitGraph(
      "node { name: 'A' op: 'Input' } "
      "node { name: 'B' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'C' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'D' op: 'QuantizeV2'"
      " attr { key: 'T'                 value { type: DT_QUINT8 } }"
      " attr { key: 'mode'              value { s: 'SCALED' } }"
      " attr { key: 'round_mode'        value { s: 'HALF_TO_EVEN' } }"
      " attr { key: 'narrow_range'      value { b: true } }"
      " attr { key: 'axis'              value { i: -1 } }"
      " input: ['A', 'B', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_QUINT8 } }"
      " input: ['D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Const);C(Const);D(QuantizeV2);E(Zeta)|"
            "A->D;B->D:1;C->D:2;D->E");
}

TEST_F(MklLayoutPassTest, NodeRewrite_QuantizeV2Op_Negative_PerSlice_Enabled) {
  InitGraph(
      "node { name: 'A' op: 'Input' } "
      "node { name: 'B' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'C' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'D' op: 'QuantizeV2'"
      " attr { key: 'T'                 value { type: DT_QUINT8 } }"
      " attr { key: 'mode'              value { s: 'SCALED' } }"
      " attr { key: 'round_mode'        value { s: 'HALF_TO_EVEN' } }"
      " attr { key: 'narrow_range'      value { b: false } }"
      " attr { key: 'axis'              value { i: 2 } }"
      " input: ['A', 'B', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_QUINT8 } }"
      " input: ['D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Const);C(Const);D(QuantizeV2);E(Zeta)|"
            "A->D;B->D:1;C->D:2;D->E");
}

TEST_F(MklLayoutPassTest, NodeRewrite_QuantizeV2Op_Negative_HalfFromZero) {
  InitGraph(
      "node { name: 'A' op: 'Input' } "
      "node { name: 'B' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'C' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'D' op: 'QuantizeV2'"
      " attr { key: 'T'                 value { type: DT_QUINT8 } }"
      " attr { key: 'mode'              value { s: 'SCALED' } }"
      " attr { key: 'round_mode'        value { s: 'HALF_FROM_ZERO' } }"
      " input: ['A', 'B', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_QUINT8 } }"
      " input: ['D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Const);C(Const);D(QuantizeV2);E(Zeta)|"
            "A->D;B->D:1;C->D:2;D->E");
}
TEST_F(MklLayoutPassTest, NodeRewrite_QuantizeV2Op_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input' } "
      "node { name: 'B' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'C' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'D' op: 'QuantizeV2'"
      " attr { key: 'T'                 value { type: DT_QUINT8 } }"
      " attr { key: 'mode'              value { s: 'SCALED' } }"
      " attr { key: 'round_mode'        value { s: 'HALF_TO_EVEN' } }"
      " input: ['A', 'B', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_QUINT8 } }"
      " input: ['D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Const);C(Const);D(_MklQuantizeV2);DMT/_0(Const);DMT/"
            "_1(Const);DMT/_2(Const);E(Zeta)|"
            "A->D;A:control->DMT/_0:control;A:control->DMT/"
            "_1:control;A:control->DMT/_2:control;B->D:1;C->D:2;D->E;DMT/"
            "_0->D:3;DMT/_1->D:4;DMT/_2->D:5");
}

TEST_F(MklLayoutPassTest, NodeRewrite_Dequantize_Negative_Const_Input) {
  InitGraph(
      "node { name: 'A' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_QUINT8 } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_QUINT8 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Dequantize'"
      " attr { key: 'T'             value { type: DT_QUINT8 } }"
      " attr { key: 'mode'          value { s: 'SCALED' } }"
      " input: ['A', 'B', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Const);B(Input);C(Input);D(Dequantize);"
            "E(Zeta)|A->D;B->D:1;C->D:2;D->E");
}

TEST_F(MklLayoutPassTest, NodeRewrite_Dequantize_Negative_Non_SCALED_Mode) {
  InitGraph(
      "node { name: 'A' op: 'QuantizedInput'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Dequantize'"
      " attr { key: 'T'             value { type: DT_QUINT8 } }"
      " attr { key: 'mode'          value { s: 'MIN_FIRST' } }"
      " input: ['A', 'B', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(QuantizedInput);B(Input);C(Input);D(Dequantize);"
            "E(Zeta)|A->D;B->D:1;C->D:2;D->E");
}

// Rewrite test for _FusedConv2D Op with BiasAdd fusion
#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
    InitGraph(                                                                 \
        "node { name: 'A' op: '" #INPUT "'}"                                   \
        "node { name: 'B' op: '" #INPUT "'}"                                   \
        "node { name: 'C' op: '" #INPUT "'}"                                   \
        "node { name: 'D' op: '_FusedConv2D'"                                  \
        " attr { key: 'T'                value { type: " #T " } }"             \
        " attr { key: 'num_args'         value { i: 1 } }"                     \
        " attr { key: 'data_format'      value { s: 'NCHW' } }"                \
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'padding'          value { s: 'SAME' } }"                \
        " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'fused_ops'        value { list: {s: 'BiasAdd'} } }"     \
        " attr { key: 'epsilon'          value { f: 0.001 }}"                  \
        " input: ['A', 'B', 'C']}"                                             \
        "node { name: 'E' op: 'Zeta'"                                          \
        "attr { key: 'T' value { type: " #T " } }"                             \
        " input: ['D', 'C'] }");                                               \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                   \
              "A(" #INPUT ");B(" #INPUT ");C(" #INPUT ");D(_MklFusedConv2D);"  \
              "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);E(Zeta)|A->D;"        \
              "A:control->DMT/_0:control;A:control->DMT/_1:control;"           \
              "A:control->DMT/_2:control;B->D:1;C->D:2;C->E:1;D->E;"           \
              "DMT/_0->D:3;DMT/_1->D:4;DMT/_2->D:5");                          \
  }
REGISTER_TEST_ALL_TYPES(NodeRewrite_FusedConv2D_Positive1);
#undef REGISTER_TEST

// Rewrite test for _FusedConv2D Op with Relu fusion
#define REGISTER_TEST(NAME, T, INPUT)                                         \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                     \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                            \
              "node { name: 'B' op: '" #INPUT "'}"                            \
              "node { name: 'C' op: '" #INPUT "'}"                            \
              "node { name: 'D' op: '_FusedConv2D'"                           \
              " attr { key: 'T'                value { type: " #T " } }"      \
              " attr { key: 'num_args'         value { i: 1 } }"              \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'padding'          value { s: 'SAME' } }"         \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'fused_ops'        value { list: {s: 'Relu'} } }" \
              " attr { key: 'epsilon'          value { f: 0.001 }}"           \
              " input: ['A', 'B', 'C']}"                                      \
              "node { name: 'E' op: 'Zeta'"                                   \
              "attr { key: 'T' value { type: " #T " } }"                      \
              " input: ['D', 'C'] }");                                        \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                  \
              "A(" #INPUT ");B(" #INPUT ");C(" #INPUT ");D(_MklFusedConv2D);" \
              "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);E(Zeta)|A->D;"       \
              "A:control->DMT/_0:control;A:control->DMT/_1:control;"          \
              "A:control->DMT/_2:control;B->D:1;C->D:2;C->E:1;D->E;"          \
              "DMT/_0->D:3;DMT/_1->D:4;DMT/_2->D:5");                         \
  }
REGISTER_TEST_ALL_TYPES(NodeRewrite_FusedConv2D_Positive2);
#undef REGISTER_TEST

// Rewrite test for _FusedConv2D Op with BiasAdd+Relu fusion
#define REGISTER_TEST(NAME, T, INPUT)                                         \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                     \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                            \
              "node { name: 'B' op: '" #INPUT "'}"                            \
              "node { name: 'C' op: '" #INPUT "'}"                            \
              "node { name: 'D' op: '_FusedConv2D'"                           \
              " attr { key: 'T'                value { type: " #T " } }"      \
              " attr { key: 'num_args'         value { i: 1 } }"              \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'padding'          value { s: 'SAME' } }"         \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'fused_ops'"                                      \
              "             value { list: {s: 'BiasAdd', s: 'Relu'} } }"      \
              " attr { key: 'epsilon'          value { f: 0.001 }}"           \
              " input: ['A', 'B', 'C']}"                                      \
              "node { name: 'E' op: 'Zeta'"                                   \
              "attr { key: 'T' value { type: " #T " } }"                      \
              " input: ['D', 'C'] }");                                        \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                  \
              "A(" #INPUT ");B(" #INPUT ");C(" #INPUT ");D(_MklFusedConv2D);" \
              "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);E(Zeta)|A->D;"       \
              "A:control->DMT/_0:control;A:control->DMT/_1:control;"          \
              "A:control->DMT/_2:control;B->D:1;C->D:2;C->E:1;D->E;"          \
              "DMT/_0->D:3;DMT/_1->D:4;DMT/_2->D:5");                         \
  }
REGISTER_TEST_ALL_TYPES(NodeRewrite_FusedConv2D_Positive3);
#undef REGISTER_TEST

// Rewrite test for _FusedConv2D Op with BiasAdd+Relu6 fusion
#define REGISTER_TEST(NAME, T, INPUT)                                         \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                     \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                            \
              "node { name: 'B' op: '" #INPUT "'}"                            \
              "node { name: 'C' op: '" #INPUT "'}"                            \
              "node { name: 'D' op: '_FusedConv2D'"                           \
              " attr { key: 'T'                value { type: " #T " } }"      \
              " attr { key: 'num_args'         value { i: 1 } }"              \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'padding'          value { s: 'SAME' } }"         \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'fused_ops'"                                      \
              "             value { list: {s: 'BiasAdd', s: 'Relu6'} } }"     \
              " attr { key: 'epsilon'          value { f: 0.001 }}"           \
              " input: ['A', 'B', 'C']}"                                      \
              "node { name: 'E' op: 'Zeta'"                                   \
              "attr { key: 'T' value { type: " #T " } }"                      \
              " input: ['D', 'C'] }");                                        \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                  \
              "A(" #INPUT ");B(" #INPUT ");C(" #INPUT ");D(_MklFusedConv2D);" \
              "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);E(Zeta)|A->D;"       \
              "A:control->DMT/_0:control;A:control->DMT/_1:control;"          \
              "A:control->DMT/_2:control;B->D:1;C->D:2;C->E:1;D->E;"          \
              "DMT/_0->D:3;DMT/_1->D:4;DMT/_2->D:5");                         \
  }
REGISTER_TEST_ALL_TYPES(NodeRewrite_FusedConv2D_Positive4);
#undef REGISTER_TEST

// Rewrite test for _FusedConv2D Op with BiasAdd+Elu fusion
#define REGISTER_TEST(NAME, T, INPUT)                                         \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                     \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                            \
              "node { name: 'B' op: '" #INPUT "'}"                            \
              "node { name: 'C' op: '" #INPUT "'}"                            \
              "node { name: 'D' op: '_FusedConv2D'"                           \
              " attr { key: 'T'                value { type: " #T " } }"      \
              " attr { key: 'num_args'         value { i: 1 } }"              \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'padding'          value { s: 'SAME' } }"         \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'fused_ops'"                                      \
              "             value { list: {s: 'BiasAdd', s: 'Elu'} } }"       \
              " attr { key: 'epsilon'          value { f: 0.001 }}"           \
              " input: ['A', 'B', 'C']}"                                      \
              "node { name: 'E' op: 'Zeta'"                                   \
              "attr { key: 'T' value { type: " #T " } }"                      \
              " input: ['D', 'C'] }");                                        \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                  \
              "A(" #INPUT ");B(" #INPUT ");C(" #INPUT ");D(_MklFusedConv2D);" \
              "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);E(Zeta)|A->D;"       \
              "A:control->DMT/_0:control;A:control->DMT/_1:control;"          \
              "A:control->DMT/_2:control;B->D:1;C->D:2;C->E:1;D->E;"          \
              "DMT/_0->D:3;DMT/_1->D:4;DMT/_2->D:5");                         \
  }
REGISTER_TEST_ALL_TYPES(NodeRewrite_FusedConv2D_Positive5);
#undef REGISTER_TEST

// Rewrite test for _FusedConv2D Op with BiasAdd+Add fusion
#define REGISTER_TEST(NAME, T, INPUT)                                         \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                     \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                            \
              "node { name: 'B' op: '" #INPUT "'}"                            \
              "node { name: 'C' op: '" #INPUT "'}"                            \
              "node { name: 'D' op: '" #INPUT "'}"                            \
              "node { name: 'E' op: '_FusedConv2D'"                           \
              " attr { key: 'T'                value { type: " #T " } }"      \
              " attr { key: 'num_args'         value { i: 2 } }"              \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'padding'          value { s: 'SAME' } }"         \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'fused_ops'"                                      \
              "             value { list: {s: 'BiasAdd', s: 'Add'} } }"       \
              " attr { key: 'epsilon'          value { f: 0.001 }}"           \
              " input: ['A', 'B', 'C', 'D']}"                                 \
              "node { name: 'F' op: 'Zeta'"                                   \
              "attr { key: 'T' value { type: " #T " } }"                      \
              " input: ['E', 'D'] }");                                        \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                  \
              "A(" #INPUT ");B(" #INPUT ");C(" #INPUT ");D(" #INPUT ");"      \
              "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);DMT/_3(Const);"      \
              "E(_MklFusedConv2D);F(Zeta)|A->E;A:control->DMT/_0:control;"    \
              "A:control->DMT/_1:control;A:control->DMT/_2:control;"          \
              "A:control->DMT/_3:control;B->E:1;C->E:2;D->E:3;D->F:1;"        \
              "DMT/_0->E:4;DMT/_1->E:5;DMT/_2->E:6;DMT/_3->E:7;E->F");        \
  }
REGISTER_TEST_ALL_TYPES(NodeRewrite_FusedConv2D_Positive6);
#undef REGISTER_TEST

// Rewrite test for _FusedConv2D Op with BiasAdd+Add+Relu fusion
#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
    InitGraph(                                                                 \
        "node { name: 'A' op: '" #INPUT "'}"                                   \
        "node { name: 'B' op: '" #INPUT "'}"                                   \
        "node { name: 'C' op: '" #INPUT "'}"                                   \
        "node { name: 'D' op: '" #INPUT "'}"                                   \
        "node { name: 'E' op: '_FusedConv2D'"                                  \
        " attr { key: 'T'                value { type: " #T " } }"             \
        " attr { key: 'num_args'         value { i: 2 } }"                     \
        " attr { key: 'data_format'      value { s: 'NCHW' } }"                \
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'padding'          value { s: 'SAME' } }"                \
        " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'fused_ops'"                                             \
        "             value { list: {s: 'BiasAdd', s: 'Add', s: 'Relu'} } }"   \
        " attr { key: 'epsilon'          value { f: 0.001 }}"                  \
        " input: ['A', 'B', 'C', 'D']}"                                        \
        "node { name: 'F' op: 'Zeta'"                                          \
        "attr { key: 'T' value { type: " #T " } }"                             \
        " input: ['E', 'D'] }");                                               \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                   \
              "A(" #INPUT ");B(" #INPUT ");C(" #INPUT ");D(" #INPUT ");"       \
              "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);DMT/_3(Const);"       \
              "E(_MklFusedConv2D);F(Zeta)|A->E;A:control->DMT/_0:control;"     \
              "A:control->DMT/_1:control;A:control->DMT/_2:control;"           \
              "A:control->DMT/_3:control;B->E:1;C->E:2;D->E:3;D->F:1;"         \
              "DMT/_0->E:4;DMT/_1->E:5;DMT/_2->E:6;DMT/_3->E:7;E->F");         \
  }
REGISTER_TEST_ALL_TYPES(NodeRewrite_FusedConv2D_Positive7);
#undef REGISTER_TEST

// Rewrite test for _FusedDepthwiseConv2dNative Op fusion
#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
    InitGraph(                                                                 \
        "node { name: 'A' op: '" #INPUT "'}"                                   \
        "node { name: 'B' op: '" #INPUT "'}"                                   \
        "node { name: 'C' op: '" #INPUT "'}"                                   \
        "node { name: 'D' op: '_FusedDepthwiseConv2dNative'"                   \
        " attr { key: 'T'                value { type: " #T " } }"             \
        " attr { key: 'num_args'         value { i: 1 } }"                     \
        " attr { key: 'data_format'      value { s: 'NCHW' } }"                \
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'padding'          value { s: 'SAME' } }"                \
        " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'fused_ops'        value { list: " FUSED_OPS " } }"      \
        " attr { key: 'epsilon'          value { f: 0.001 }}"                  \
        " input: ['A', 'B', 'C']}"                                             \
        "node { name: 'E' op: 'Zeta'"                                          \
        "attr { key: 'T' value { type: " #T " } }"                             \
        " input: ['D', 'C'] }");                                               \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                   \
              "A(" #INPUT ");B(" #INPUT ");C(" #INPUT ");"                     \
              "D(_MklFusedDepthwiseConv2dNative);"                             \
              "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);E(Zeta)|A->D;"        \
              "A:control->DMT/_0:control;A:control->DMT/_1:control;"           \
              "A:control->DMT/_2:control;B->D:1;C->D:2;C->E:1;D->E;"           \
              "DMT/_0->D:3;DMT/_1->D:4;DMT/_2->D:5");                          \
  }

// BiasAdd fusion
#define FUSED_OPS "{s: 'BiasAdd'}"
REGISTER_TEST_ALL_TYPES(NodeRewrite_FusedDepthwiseConv2dNative_Positive1);

// BiasAdd + Relu fusion
#define FUSED_OPS "{s: 'BiasAdd', s: 'Relu'}"
REGISTER_TEST_ALL_TYPES(NodeRewrite_FusedDepthwiseConv2dNative_Positive2);

// BiasAdd + Relu6 fusion
#define FUSED_OPS "{s: 'BiasAdd', s: 'Relu6'}"
REGISTER_TEST_ALL_TYPES(NodeRewrite_FusedDepthwiseConv2dNative_Positive3);

// BiasAdd + Elu fusion
#define FUSED_OPS "{s: 'BiasAdd', s: 'Elu'}"
REGISTER_TEST_ALL_TYPES(NodeRewrite_FusedDepthwiseConv2dNative_Positive4);

#undef FUSED_OPS
#undef REGISTER_TEST

// Rewrite test for _FusedConv2D Op with unsupported fusion
#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
    InitGraph(                                                                 \
        "node { name: 'A' op: '" #INPUT "'}"                                   \
        "node { name: 'B' op: '" #INPUT "'}"                                   \
        "node { name: 'C' op: '" #INPUT "'}"                                   \
        "node { name: 'D' op: '_FusedConv2D'"                                  \
        " attr { key: 'T'                value { type: " #T " } }"             \
        " attr { key: 'num_args'         value { i: 1 } }"                     \
        " attr { key: 'data_format'      value { s: 'NCHW' } }"                \
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'padding'          value { s: 'SAME' } }"                \
        " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'fused_ops'        value { list: {s: 'Unsupported'} } }" \
        " attr { key: 'epsilon'          value { f: 0.001 }}"                  \
        " input: ['A', 'B', 'C']}"                                             \
        "node { name: 'E' op: 'Zeta'"                                          \
        "attr { key: 'T' value { type: " #T " } }"                             \
        " input: ['D', 'C'] }");                                               \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                   \
              "A(" #INPUT ");B(" #INPUT ");C(" #INPUT ");D(_FusedConv2D);"     \
              "E(Zeta)|A->D;B->D:1;C->D:2;C->E:1;D->E");                       \
  }
REGISTER_TEST_ALL_TYPES(NodeRewrite_FusedConv2D_Negative1);
#undef REGISTER_TEST

// Rewrite test for _FusedDepthwiseConv2dNative with unsupported fusion
#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
    InitGraph(                                                                 \
        "node { name: 'A' op: '" #INPUT "'}"                                   \
        "node { name: 'B' op: '" #INPUT "'}"                                   \
        "node { name: 'C' op: '" #INPUT "'}"                                   \
        "node { name: 'D' op: '_FusedDepthwiseConv2dNative'"                   \
        " attr { key: 'T'                value { type: " #T " } }"             \
        " attr { key: 'num_args'         value { i: 1 } }"                     \
        " attr { key: 'data_format'      value { s: 'NCHW' } }"                \
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'padding'          value { s: 'SAME' } }"                \
        " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'fused_ops'        value { list: {s: 'Unsupported'} } }" \
        " attr { key: 'epsilon'          value { f: 0.001 }}"                  \
        " input: ['A', 'B', 'C']}"                                             \
        "node { name: 'E' op: 'Zeta'"                                          \
        "attr { key: 'T' value { type: " #T " } }"                             \
        " input: ['D', 'C'] }");                                               \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                   \
              "A(" #INPUT ");B(" #INPUT ");C(" #INPUT ");"                     \
              "D(_FusedDepthwiseConv2dNative);"                                \
              "E(Zeta)|A->D;B->D:1;C->D:2;C->E:1;D->E");                       \
  }
REGISTER_TEST_ALL_TYPES(NodeRewrite_FusedDepthwiseConv2dNative_Negative1);
#undef REGISTER_TEST

// Rewrite test for _FusedConv2D Op with unsupported type
#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
    InitGraph(                                                                 \
        "node { name: 'A' op: '" #INPUT "'}"                                   \
        "node { name: 'B' op: '" #INPUT "'}"                                   \
        "node { name: 'C' op: '" #INPUT "'}"                                   \
        "node { name: 'D' op: '_FusedConv2D'"                                  \
        " attr { key: 'T'                value { type:" #T  "} }"              \
        " attr { key: 'num_args'         value { i: 1 } }"                     \
        " attr { key: 'data_format'      value { s: 'NCHW' } }"                \
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'padding'          value { s: 'SAME' } }"                \
        " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'fused_ops'        value { list: {s: 'BiasAdd'} } }"     \
        " attr { key: 'epsilon'          value { f: 0.001 }}"                  \
        " input: ['A', 'B', 'C']}"                                             \
        "node { name: 'E' op: 'Zeta'"                                          \
        "attr { key: 'T' value { type: " #T "} }"                              \
        " input: ['D', 'C'] }");                                               \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                   \
              "A(" #INPUT ");B(" #INPUT ");C(" #INPUT ");"                     \
              "D(_FusedConv2D);E(Zeta)|A->D;B->D:1;C->D:2;C->E:1;D->E");       \
}
REGISTER_TEST(NodeRewrite_FusedConv2D_Negative2, DT_DOUBLE, DoubleInput);
#undef REGISTER_TEST

// Rewrite test for _FusedDepthwiseConv2dNativeOp with unsupported type
#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
    InitGraph(                                                                 \
        "node { name: 'A' op: '" #INPUT "'}"                                   \
        "node { name: 'B' op: '" #INPUT "'}"                                   \
        "node { name: 'C' op: '" #INPUT "'}"                                   \
        "node { name: 'D' op: '_FusedDepthwiseConv2dNative'"                   \
        " attr { key: 'T'                value { type:" #T  "} }"              \
        " attr { key: 'num_args'         value { i: 1 } }"                     \
        " attr { key: 'data_format'      value { s: 'NCHW' } }"                \
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'padding'          value { s: 'SAME' } }"                \
        " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'fused_ops'        value { list: {s: 'BiasAdd'} } }"     \
        " attr { key: 'epsilon'          value { f: 0.001 }}"                  \
        " input: ['A', 'B', 'C']}"                                             \
        "node { name: 'E' op: 'Zeta'"                                          \
        "attr { key: 'T' value { type: " #T "} }"                              \
        " input: ['D', 'C'] }");                                               \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                   \
              "A(" #INPUT ");B(" #INPUT ");C(" #INPUT ");"                     \
              "D(_FusedDepthwiseConv2dNative);"                                \
              "E(Zeta)|A->D;B->D:1;C->D:2;C->E:1;D->E");                       \
}
REGISTER_TEST(NodeRewrite_FusedDepthwiseConv2dNative_Negative2,
              DT_DOUBLE, DoubleInput);
#undef REGISTER_TEST

// Test set: _FusedMatMul -> MklFusedMatMul rewrite tests
#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
  InitGraph(                                                                   \
      "node { name: 'A' op: '" #INPUT "'}"                                     \
      "node { name: 'B' op: '" #INPUT "'}"                                     \
      "node { name: 'C' op: '" #INPUT "'}"                                     \
      "node { name: 'D' op: '_FusedMatMul'"                                    \
      " attr { key: 'T'                value { type:" #T  "} }"                \
      " attr { key: 'transpose_a'      value { b: false } }"                   \
      " attr { key: 'transpose_b'      value { b: false } }"                   \
      " attr { key: 'num_args'         value { i: 1 } }"                       \
      " attr { key: 'fused_ops'        value { list: {s: 'BiasAdd'} } }"       \
      " attr { key: 'epsilon'          value { f: 0.001 }}"                    \
      " input: ['A', 'B', 'C']}"                                               \
      "node { name: 'Z' op: 'Zeta'"                                            \
      " attr {key: 'T'                 value { type: " #T " } }"               \
      " input: ['D', 'C']}");                                                  \
  EXPECT_EQ(DoMklLayoutOptimizationPass(),                                     \
            "A(" #INPUT ");B(" #INPUT ");C(" #INPUT ");D(_MklFusedMatMul);"    \
            "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);Z(Zeta)"                \
            "|A->D;A:control->DMT/_0:control;A:control->DMT/_1:control;"       \
            "A:control->DMT/_2:control;B->D:1;C->D:2;C->Z:1;D->Z;DMT/_0->D:3;" \
            "DMT/_1->D:4;DMT/_2->D:5");                                        \
}
REGISTER_TEST_ALL_TYPES(NodeRewrite_FusedMatMul_Positive)
#undef REGISTER_TEST

// Test set: _FusedMatMul -> MklFusedMatMul rewrite tests
#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
  InitGraph(                                                                   \
      "node { name: 'A' op: '" #INPUT "'}"                                     \
      "node { name: 'B' op: '" #INPUT "'}"                                     \
      "node { name: 'C' op: '" #INPUT "'}"                                     \
      "node { name: 'D' op: '_FusedMatMul'"                                    \
      " attr { key: 'T'                value { type: " #T "} }"                \
      " attr { key: 'transpose_a'      value { b: true } }"                    \
      " attr { key: 'transpose_b'      value { b: false } }"                   \
      " attr { key: 'num_args'         value { i: 1 } }"                       \
      " attr { key: 'fused_ops'        value { list: {s: 'BiasAdd'} } }"       \
      " attr { key: 'epsilon'          value { f: 0.001 }}"                    \
      " input: ['A', 'B', 'C']}"                                               \
      "node { name: 'Z' op: 'Zeta'"                                            \
      " attr {key: 'T'                 value { type: " #T " } }"               \
      " input: ['D', 'C']}");                                                  \
  EXPECT_EQ(DoMklLayoutOptimizationPass(),                                     \
            "A(" #INPUT ");B(" #INPUT ");C(" #INPUT ");D(_FusedMatMul);Z(Zeta)"\
            "|A->D;B->D:1;C->D:2;C->Z:1;D->Z");                                \
}
REGISTER_TEST_ALL_TYPES(NodeRewrite_FusedMatMul_Negative);
#undef REGISTER_TEST

// Merge test for PadWithFusedConv2D Op with BiasAdd fusion
// padding is VALID type
// A = input(image), B = input(paddings), C = Pad(A, B) = input of conv2D,
// D = input(filter), E = input(bias), F = _FusedConv2D(C, D, E)
// G = Zeta(F, E)
// After layout pass
// _MklPadWithFusedConv2D(A, D, E, B, DMT/_0, DMT/_1, DMT/_2, DMT/_3)
#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
    InitGraph(                                                                 \
        "node { name: 'A' op: '" #INPUT "'}"                                   \
        "node { name: 'B' op: 'Int32Input'}"                                   \
        "node { name: 'C' op: 'Pad'"                                           \
        " attr { key: 'T'                value { type: " #T " } }"             \
        " attr { key: 'Tpaddings'        value { type: DT_INT32 } }"           \
        " input: ['A', 'B']}"                                                  \
        "node { name: 'D' op: '" #INPUT "'}"                                   \
        "node { name: 'E' op: '" #INPUT "'}"                                   \
        "node { name: 'F' op: '_FusedConv2D'"                                  \
        " attr { key: 'T'                value { type: " #T " } }"             \
        " attr { key: 'num_args'         value { i: 1 } }"                     \
        " attr { key: 'data_format'      value { s: 'NCHW' } }"                \
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'padding'          value { s: 'VALID' } }"               \
        " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'fused_ops'        value { list: {s: 'BiasAdd'} } }"     \
        " attr { key: 'epsilon'          value { f: 0.001 }}"                  \
        " input: ['C', 'D', 'E']}"                                             \
        "node { name: 'G' op: 'Zeta'"                                          \
        " attr { key: 'T' value { type: " #T " } }"                            \
        " input: ['F', 'E'] }");                                               \
    EXPECT_EQ(                                                                 \
        DoMklLayoutOptimizationPass(),                                         \
        "A(" #INPUT ");B(Int32Input);D(" #INPUT ");DMT/_0(Const);DMT/_1(Const);"\
        "DMT/_2(Const);DMT/_3(Const);E(" #INPUT ");F(_MklPadWithFusedConv2D);" \
        "G(Zeta)|A->F;A:control->DMT/_0:control;A:control->DMT/_1:control;"    \
        "A:control->DMT/_2:control;A:control->DMT/_3:control;B->F:3;D->F:1;"   \
        "DMT/_0->F:4;DMT/_1->F:5;DMT/_2->F:6;DMT/_3->F:7;E->F:2;E->G:1;F->G"); \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_PadWithFusedConv2D_Positive1);
#undef REGISTER_TEST

// Merge test for PadWithFusedConv2D Op with BiasAdd+Relu fusion
// padding is VALID type
// A = input(image), B = input(paddings), C = Pad(A, B) = input of conv2D,
// D = input(filter), E = input(bias), F = _FusedConv2D(C, D, E) (With relu)
// G = Zeta(F, E)
// After layout pass
// _MklPadWithFusedConv2D(A, D, E, B, DMT/_0, DMT/_1, DMT/_2, DMT/_3)
#define REGISTER_TEST(NAME, T, INPUT)                                         \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                     \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                            \
              "node { name: 'B' op: 'Int32Input'}"                            \
              "node { name: 'C' op: 'Pad'"                                    \
              " attr { key: 'T'                value { type: " #T " } }"      \
              " attr { key: 'Tpaddings'        value { type: DT_INT32 } }"    \
              " input: ['A', 'B']}"                                           \
              "node { name: 'D' op: '" #INPUT "'}"                            \
              "node { name: 'E' op: '" #INPUT "'}"                            \
              "node { name: 'F' op: '_FusedConv2D'"                           \
              " attr { key: 'T'                value { type: " #T " } }"      \
              " attr { key: 'num_args'         value { i: 1 } }"              \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'padding'          value { s: 'VALID' } }"        \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'fused_ops'"                                      \
              "             value { list: {s: 'BiasAdd', s: 'Relu'} } }"      \
              " attr { key: 'epsilon'          value { f: 0.001 }}"           \
              " input: ['C', 'D', 'E']}"                                      \
              "node { name: 'G' op: 'Zeta'"                                   \
              "attr { key: 'T' value { type: " #T "} }"                       \
              " input: ['F', 'E'] }");                                        \
    EXPECT_EQ(                                                                \
        DoMklLayoutOptimizationPass(),                                        \
        "A(" #INPUT ");B(Int32Input);D(" #INPUT ");DMT/_0(Const);DMT/_1(Const);"\
        "DMT/_2(Const);DMT/_3(Const);E(" #INPUT ");F(_MklPadWithFusedConv2D);"\
        "G(Zeta)|A->F;A:control->DMT/_0:control;A:control->DMT/_1:control;"   \
        "A:control->DMT/_2:control;A:control->DMT/_3:control;B->F:3;"         \
        "D->F:1;DMT/_0->F:4;DMT/_1->F:5;DMT/_2->F:6;DMT/"                     \
        "_3->F:7;E->F:2;E->G:1;F->G");                                        \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_PadWithFusedConv2D_Positive2);
#undef REGISTER_TEST

// Merge test for PadWithFusedConv2D Op with unsupported fusion
// padding is VALID type
// A = input(image), B = input(paddings), C = Pad(A, B) = input of conv2D,
// D = input(filter), E = input(bias),
// F = _FusedConv2D(C, D, E) (With Unsupported), G = Zeta(F, E)
// After layout pass - No merging
#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
    InitGraph(                                                                 \
        "node { name: 'A' op: '" #INPUT "'}"                                   \
        "node { name: 'B' op: 'Int32Input'}"                                   \
        "node { name: 'C' op: 'Pad'"                                           \
        " attr { key: 'T'                value { type: " #T " } }"             \
        " attr { key: 'Tpaddings'        value { type: DT_INT32 } }"           \
        " input: ['A', 'B']}"                                                  \
        "node { name: 'D' op: '" #INPUT "'}"                                   \
        "node { name: 'E' op: '" #INPUT "'}"                                   \
        "node { name: 'F' op: '_FusedConv2D'"                                  \
        " attr { key: 'T'                value { type: " #T " } }"             \
        " attr { key: 'num_args'         value { i: 1 } }"                     \
        " attr { key: 'data_format'      value { s: 'NCHW' } }"                \
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'padding'          value { s: 'VALID' } }"               \
        " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'fused_ops'        value { list: {s: 'Unsupported'} } }" \
        " attr { key: 'epsilon'          value { f: 0.001 }}"                  \
        " input: ['C', 'D', 'E']}"                                             \
        "node { name: 'G' op: 'Zeta'"                                          \
        " attr { key: 'T' value { type: " #T " } }"                            \
        " input: ['F', 'E'] }");                                               \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                   \
              "A(" #INPUT ");B(Int32Input);C(Pad);D(" #INPUT ");E(" #INPUT ");"\
              "F(_FusedConv2D);G(Zeta)|A->C;B->C:1;C->F;D->F:1;E->F:2;E->G:1;" \
              "F->G");                                                         \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_PadWithFusedConv2D_Negative1);
#undef REGISTER_TEST

// Merge test for PadWithFusedConv2D Op with BiasAdd fusion
// padding is SAME type
// A = input(image), B = input(paddings), C = Pad(A, B) = input of conv2D,
// D = input(filter), E = input(bias), F = _FusedConv2D(C,D,E)
// G = Zeta(F,E)
// After layout pass - No merging
#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
    InitGraph(                                                                 \
        "node { name: 'A' op: '" #INPUT "'}"                                   \
        "node { name: 'B' op: 'Int32Input'}"                                   \
        "node { name: 'C' op: 'Pad'"                                           \
        " attr { key: 'T'                value { type: " #T " } }"             \
        " attr { key: 'Tpaddings'        value { type: DT_INT32 } }"           \
        " input: ['A', 'B']}"                                                  \
        "node { name: 'D' op: '" #INPUT "'}"                                   \
        "node { name: 'E' op: '" #INPUT "'}"                                   \
        "node { name: 'F' op: '_FusedConv2D'"                                  \
        " attr { key: 'T'                value { type: " #T " } }"             \
        " attr { key: 'num_args'         value { i: 1 } }"                     \
        " attr { key: 'data_format'      value { s: 'NCHW' } }"                \
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'padding'          value { s: 'SAME' } }"                \
        " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'fused_ops'        value { list: {s: 'BiasAdd'} } }"     \
        " attr { key: 'epsilon'          value { f: 0.001 }}"                  \
        " input: ['C', 'D', 'E']}"                                             \
        "node { name: 'G' op: 'Zeta'"                                          \
        " attr { key: 'T' value { type: " #T " } }"                            \
        " input: ['F', 'E'] }");                                               \
    EXPECT_EQ(                                                                 \
        DoMklLayoutOptimizationPass(),                                         \
        "A(" #INPUT ");B(Int32Input);C(Pad);D(" #INPUT ");DMT/_0(Const);DMT/"  \
        "_1(Const);DMT/_2(Const);E(" #INPUT ");F(_MklFusedConv2D);G(Zeta)|A->C;"\
        "B->C:1;C->F;C:control->DMT/_0:control;C:control->DMT/_1:control;"     \
        "C:control->DMT/_2:control;D->F:1;DMT/_0->F:3;DMT/_1->F:4;DMT/"        \
        "_2->F:5;E->F:2;E->G:1;F->G");                                         \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_PadWithFusedConv2D_Negative2);
#undef REGISTER_TEST

// Merge test for PadWithFusedConv2D Op with BiasAdd+Relu fusion
// padding is SAME type
// A = input(image), B = input(paddings), C = Pad(A, B) = input of conv2D,
// D = input(filter), E = input(bias), F = _FusedConv2D(C,D,E)(With relu)
// G = Zeta(F,E)
// After layout pass - No merging
#define REGISTER_TEST(NAME, T, INPUT)                                         \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                     \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                            \
              "node { name: 'B' op: 'Int32Input'}"                            \
              "node { name: 'C' op: 'Pad'"                                    \
              " attr { key: 'T'                value { type: " #T " } }"      \
              " attr { key: 'Tpaddings'        value { type: DT_INT32 } }"    \
              " input: ['A', 'B']}"                                           \
              "node { name: 'D' op: '" #INPUT "'}"                            \
              "node { name: 'E' op: '" #INPUT "'}"                            \
              "node { name: 'F' op: '_FusedConv2D'"                           \
              " attr { key: 'T'                value { type: " #T " } }"      \
              " attr { key: 'num_args'         value { i: 1 } }"              \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'padding'          value { s: 'SAME' } }"         \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "    \
              "i:1, i:1} } }"                                                 \
              " attr { key: 'fused_ops'"                                      \
              "             value { list: {s: 'BiasAdd', s: 'Relu'} } }"      \
              " attr { key: 'epsilon'          value { f: 0.001 }}"           \
              " input: ['C', 'D', 'E']}"                                      \
              "node { name: 'G' op: 'Zeta'"                                   \
              " attr { key: 'T' value { type: " #T " } }"                     \
              " input: ['F', 'E'] }");                                        \
    EXPECT_EQ(                                                                \
        DoMklLayoutOptimizationPass(),                                        \
        "A(" #INPUT ");B(Int32Input);C(Pad);D(" #INPUT ");DMT/_0(Const);DMT/" \
        "_1(Const);DMT/_2(Const);E(" #INPUT ");F(_MklFusedConv2D);G(Zeta)|"   \
        "A->C;B->C:1;C->F;C:control->DMT/_0:control;C:control->DMT/_1:control;"\
        "C:control->DMT/_2:control;D->F:1;DMT/_0->F:3;DMT/_1->F:4;DMT/"       \
        "_2->F:5;E->F:2;E->G:1;F->G");                                        \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_PadWithFusedConv2D_Negative3);
#undef REGISTER_TEST

// Tests that there are no duplicate input control edges after merge.
// If both the merging ops have input control edges from a common op
// then, the merged op will have only one control edge from that
// common op. This test only add additional input control edge check
// based on the previous test NodeMerge_PadWithFusedConv2D_Positive1
// padding is VALID type
// A = input(image), X = input, B = input(paddings),
// C = Pad(A, B) = input of conv2D,
// D = input(filter), E = input(bias), F = _FusedConv2D(C, D, E)
// G = Zeta(F, E)
// X:control->C:control
// X:control->F:control
// After layout pass:
// _MklPadWithFusedConv2D(A, D, B, F, DMT/_0, DMT/_1, DMT/_2, DMT/_3)
// X:control->E:control (only one control edge)
#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
    DCHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);       \
    InitGraph(                                                                 \
        "node { name: 'X' op: '" #INPUT "'}"                                   \
        "node { name: 'A' op: '" #INPUT "'}"                                   \
        "node { name: 'B' op: 'Int32Input'}"                                   \
        "node { name: 'C' op: 'Pad'"                                           \
        " attr { key: 'T'                value { type: " #T " } }"             \
        " attr { key: 'Tpaddings'        value { type: DT_INT32 } }"           \
        " input: ['A', 'B']}"                                                  \
        "node { name: 'D' op: '" #INPUT "'}"                                   \
        "node { name: 'E' op: '" #INPUT "'}"                                   \
        "node { name: 'F' op: '_FusedConv2D'"                                  \
        " attr { key: 'T'                value { type: " #T " } }"             \
        " attr { key: 'num_args'         value { i: 1 } }"                     \
        " attr { key: 'data_format'      value { s: 'NCHW' } }"                \
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'padding'          value { s: 'VALID' } }"               \
        " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'fused_ops'        value { list: {s: 'BiasAdd'} } }"     \
        " attr { key: 'epsilon'          value { f: 0.001 }}"                  \
        " input: ['C', 'D', 'E']}"                                             \
        "node { name: 'G' op: 'Zeta'"                                          \
        " attr {key: 'T'                 value { type: " #T " } }"             \
        " input: ['F', 'E']}");                                                \
    Node* x = FindNode("X");                                                   \
    Node* c = FindNode("C");                                                   \
    Node* f = FindNode("F");                                                   \
    const Edge* edge = graph_.AddControlEdge(x, c);                            \
    const Edge* edge_1 = graph_.AddControlEdge(x, f);                          \
    ASSERT_NE(edge, nullptr);                                                  \
    ASSERT_NE(edge_1, nullptr);                                                \
    EXPECT_EQ(DoMklLayoutOptimizationPass(),                                   \
              "A(" #INPUT ");B(Int32Input);D(" #INPUT ");DMT/_0(Const);"       \
              "DMT/_1(Const);DMT/_2(Const);DMT/_3(Const);E(" #INPUT ");"       \
              "F(_MklPadWithFusedConv2D);G(Zeta);X(" #INPUT ")|A->F;A:control->"\
              "DMT/_0:control;A:control->DMT/_1:control;A:control->"           \
              "DMT/_2:control;A:control->DMT/_3:control;B->F:3;D->F:1;"        \
              "DMT/_0->F:4;DMT/_1->F:5;DMT/_2->F:6;DMT/_3->F:7;E->F:2;E->G:1;" \
              "F->G;X:control->F:control");                                    \
  }
REGISTER_TEST_ALL_TYPES(Input_ControlEdge_PadWithFusedConv2D_Positive);
#undef REGISTER_TEST

// ts that there are no duplicate output control edges after merge.
// If both the merging ops have output control edge to a common op,
// then after merge, the merged op will have only one control edge
// to that common op. This test only add additional output control edge check
// based on the previous test NodeMerge_PadWithFusedConv2D_Positive1
// padding is VALID type
// A = input(image), B = input(paddings), C = Pad(A, B) = input of conv2D,
// D = input(filter), E = input(bias), F = _FusedConv2D(C, D, E)
// G = Zeta(F, E), X = input
// C:control->X:control
// F:control->X:control
// After layout pass:
// _MklPadWithFusedConv2D(A, D, B, F, DMT/_0, DMT/_1, DMT/_2, DMT/_2)
// F:control->X:control (only one control edge)
#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
    DCHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);       \
    InitGraph(                                                                 \
        "node { name: 'X' op: '" #INPUT "'}"                                   \
        "node { name: 'A' op: '" #INPUT "'}"                                   \
        "node { name: 'B' op: 'Int32Input'}"                                   \
        "node { name: 'C' op: 'Pad'"                                           \
        " attr { key: 'T'                value { type: " #T " } }"             \
        " attr { key: 'Tpaddings'        value { type: DT_INT32 } }"           \
        " input: ['A', 'B']}"                                                  \
        "node { name: 'D' op: '" #INPUT "'}"                                   \
        "node { name: 'E' op: '" #INPUT "'}"                                   \
        "node { name: 'F' op: '_FusedConv2D'"                                  \
        " attr { key: 'T'                value { type: " #T " } }"             \
        " attr { key: 'num_args'         value { i: 1 } }"                     \
        " attr { key: 'data_format'      value { s: 'NCHW' } }"                \
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'padding'          value { s: 'VALID' } }"               \
        " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'fused_ops'        value { list: {s: 'BiasAdd'} } }"     \
        " attr { key: 'epsilon'          value { f: 0.001 }}"                  \
        " input: ['C', 'D', 'E']}"                                             \
        "node { name: 'G' op: 'Zeta'"                                          \
        " attr {key: 'T'                 value { type: " #T " } }"             \
        " input: ['F', 'E']}");                                                \
    Node* x = FindNode("X");                                                   \
    Node* c = FindNode("C");                                                   \
    Node* f = FindNode("F");                                                   \
    const Edge* edge = graph_.AddControlEdge(c, x);                            \
    const Edge* edge_1 = graph_.AddControlEdge(f, x);                          \
    ASSERT_NE(edge, nullptr);                                                  \
    ASSERT_NE(edge_1, nullptr);                                                \
    EXPECT_EQ(                                                                 \
        DoMklLayoutOptimizationPass(),                                         \
        "A(" #INPUT ");B(Int32Input);D(" #INPUT ");DMT/_0(Const);DMT/_1(Const);"\
        "DMT/_2(Const);DMT/_3(Const);E(" #INPUT ");F(_MklPadWithFusedConv2D);" \
        "G(Zeta);X(" #INPUT ")|A->F;A:control->DMT/_0:control;A:control->DMT/" \
        "_1:control;A:control->DMT/_2:control;A:control->DMT/_3:control;B->F:3;"\
        "D->F:1;DMT/_0->F:4;DMT/_1->F:5;DMT/_2->F:6;DMT/_3->F:7;E->F:2;E->G:1;"\
        "F->G;F:control->X:control");                                          \
  }
REGISTER_TEST_ALL_TYPES(Output_ControlEdge_PadWithFusedConv2D_Positive);
#undef REGISTER_TEST

// Pad + _FusedConv2D with padding is VALID,
// Input node pointing to both Pad and _FusedConv2D
// Output of both Pad and _FusedConv2D feeds one node (G as Output2)
// A = input(as image), B = input(as paddings), C = Pad(A, B)
// E = input(as bias), F = _FusedConv2D(C, A, E), G = Output(C, F)
// After layout pass - No merging, since Pad and _FusedConv2D both
// feed to the same node (Z)
#define REGISTER_TEST(NAME, T, INPUT, OUTPUT)                                  \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
    DCHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);       \
    InitGraph(                                                                 \
        "node { name: 'A' op: '" #INPUT "'}"                                   \
        "node { name: 'B' op: 'Int32Input'}"                                   \
        "node { name: 'C' op: 'Pad'"                                           \
        " attr { key: 'T'                value { type: " #T " } }"             \
        " attr { key: 'Tpaddings'        value { type: DT_INT32 } }"           \
        " input: ['A', 'B']}"                                                  \
        "node { name: 'E' op: '" #INPUT "'}"                                   \
        "node { name: 'F' op: '_FusedConv2D'"                                  \
        " attr { key: 'T'                value { type: " #T " } }"             \
        " attr { key: 'num_args'         value { i: 1 } }"                     \
        " attr { key: 'data_format'      value { s: 'NCHW' } }"                \
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'padding'          value { s: 'VALID' } }"               \
        " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} " \
        "} }"                                                                  \
        " attr { key: 'fused_ops'        value { list: {s: 'BiasAdd'} } }"     \
        " attr { key: 'epsilon'          value { f: 0.001 }}"                  \
        " input: ['C', 'A', 'E']}"                                             \
        "node { name: 'G' op: '" #OUTPUT "'"                                   \
        " input: ['C', 'F']}");                                                \
    EXPECT_EQ(                                                                 \
        DoMklLayoutOptimizationPass(),                                         \
        "A(" #INPUT ");B(Int32Input);C(Pad);DMT/_0(Const);DMT/_1(Const);DMT/"  \
        "_2(Const);E(" #INPUT ");F(_MklFusedConv2D);G(" #OUTPUT                \
        ")|A->C;A->F:1;B->C:1;C->F;C->G;C:control->DMT/_0:control;"            \
        "C:control->DMT/_1:control;C:control->DMT/_2:control;DMT/_0->F:3;"     \
        "DMT/_1->F:4;DMT/_2->F:5;E->F:2;F->G:1");                              \
  }
REGISTER_TEST(NodeMerge_PadWithFusedConv2D_Common_InOutput, DT_FLOAT,
              Float32Input, Float32Output2);
#ifdef ENABLE_INTEL_MKL_BFLOAT16
// TODO(nhasabni): Enable bfloat16 test when we enable the operator.
REGISTER_TEST(NodeMerge_PadWithFusedConv2D_Common_InOutput, DT_BFLOAT16,
              BFloat16Input, BFloat16Output2);
#endif
#undef REGISTER_TEST
// clang-format on

TEST_F(MklLayoutPassTest, NodeRewrite_Conv2DGradFilter_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Int32Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Conv2DBackpropFilter'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Int32Input);C(Input);D(_MklConv2DBackpropFilter);"
            "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);E(Zeta)|"
            "A->D;A->E;A:control->DMT/_0:control;A:control->DMT/_1:control;"
            "A:control->DMT/_2:control;B->D:1;C->D:2;D->E:1;DMT/_0->D:3;"
            "DMT/_1->D:4;DMT/_2->D:5");
}

TEST_F(MklLayoutPassTest, NodeRewrite_Conv2DGradInput_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Int32Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Conv2DBackpropInput'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['B', 'A', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Int32Input);C(Input);D(_MklConv2DBackpropInput);"
            "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);E(Zeta)|"
            "A->D:1;A->E;B->D;B:control->DMT/_0:control;"
            "B:control->DMT/_1:control;B:control->DMT/_2:control;C->D:2;"
            "D->E:1;DMT/_0->D:3;DMT/_1->D:4;DMT/_2->D:5");
}

TEST_F(MklLayoutPassTest,
       NodeRewrite_DepthwiseConv2dNativeGradFilter_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Int32Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'DepthwiseConv2dNativeBackpropFilter'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Int32Input);C(Input);D(_"
            "MklDepthwiseConv2dNativeBackpropFilter);"
            "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);E(Zeta)|"
            "A->D;A->E;A:control->DMT/_0:control;A:control->DMT/_1:control;"
            "A:control->DMT/_2:control;B->D:1;C->D:2;D->E:1;DMT/_0->D:3;"
            "DMT/_1->D:4;DMT/_2->D:5");
}

TEST_F(MklLayoutPassTest, NodeRewrite_DepthwiseConv2dNativeGradInput_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Int32Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'DepthwiseConv2dNativeBackpropInput'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['B', 'A', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Int32Input);C(Input);D(_"
            "MklDepthwiseConv2dNativeBackpropInput);"
            "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);E(Zeta)|"
            "A->D:1;A->E;B->D;B:control->DMT/_0:control;"
            "B:control->DMT/_1:control;B:control->DMT/_2:control;C->D:2;"
            "D->E:1;DMT/_0->D:3;DMT/_1->D:4;DMT/_2->D:5");
}

// Check that we never rewrite BiasAddGrad.
TEST_F(MklLayoutPassTest, NodeRewrite_BiasAddGrad_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Polygamma'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Zeta'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['C', 'A']}"
      "node { name: 'E' op: 'BiasAddGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Polygamma);D(Zeta);E(BiasAddGrad)|"
            "A->C;A->D:1;B->C:1;C->D;D->E");
}

// Check that we never rewrite BiasAddGrad.
TEST_F(MklLayoutPassTest, NodeRewrite_BiasAddGrad_Positive1) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'MatMul'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'transpose_a'      value { b: false } }"
      " attr { key: 'transpose_b'      value { b: false } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Zeta'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['C', 'A']}"
      "node { name: 'E' op: 'BiasAddGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklMatMul);D(Zeta);E(BiasAddGrad)"
            "|A->C;A->D:1;B->C:1;C->D;D->E");
}

// Check that we never rewrite BiasAddGrad.
TEST_F(MklLayoutPassTest, NodeRewrite_BiasAddGrad_Positive2) {
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
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B', 'M', 'N']}"
      "node { name: 'D' op: 'Zeta'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['C', 'A']}"
      "node { name: 'E' op: 'BiasAddGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklConv2D);D(Zeta);E(BiasAddGrad);"
            "M(_MklInput);N(_MklInput)|A->C;A->D:1;B->C:1;C->D;D->E;"
            "M->C:2;N->C:3");
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
      " input: ['A', 'B:0', 'B:1']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'D'] }");
  EXPECT_EQ(
      DoMklLayoutOptimizationPass(),
      "A(Const);B(InputList);C(Input);D(_MklConcat);DMT/_0(Const);"
      "DMT/_1(Const);DMT/_2(Const);E(Zeta)|A->D;A:control->DMT/_0:control;"
      "A:control->DMT/_1:control;A:control->DMT/_2:control;B->D:1;"
      "B:1->D:2;C->E;D->E:1;DMT/_0->D:3;DMT/_1->D:4;DMT/_2->D:5");
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
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B']}"
      "node { name: 'F' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
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
      "node { name: 'I' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'H'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);DMT/_3(Const);DMT/_4(Const);E(_MklConv2D);"
            "F(_MklConv2D);G(Const);H(_MklConcat);I(Zeta)|A->E;A->I;"
            "A:control->DMT/_0:control;A:control->DMT/_1:control;"
            "B->E:1;C->F;C:control->DMT/_2:control;C:control->DMT/_3:control;"
            "D->F:1;DMT/_0->E:2;DMT/_1->E:3;DMT/_2->F:2;DMT/_3->F:3;"
            "DMT/_4->H:3;E->H:1;E:2->H:4;F->H:2;F:2->H:5;G->H;"
            "G:control->DMT/_4:control;H->I:1");
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
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B']}"
      "node { name: 'F' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
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
      "node { name: 'I' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'H'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);DMT/_3(Const);E(_MklConv2D);F(Zeta);G(Const);"
            "H(_MklConcat);I(Zeta)|A->E;A->I;A:control->DMT/_0:control;"
            "A:control->DMT/_1:control;B->E:1;C->F;D->F:1;DMT/_0->E:2;"
            "DMT/_1->E:3;DMT/_2->H:3;DMT/_3->H:5;E->H:1;E:2->H:4;F->H:2;"
            "G->H;G:control->DMT/_2:control;G:control->DMT/_3:control;H->I:1");
}

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
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Const);B(InputList);C(Input);D(_MklConcatV2);DMT/_0(Const);"
            "DMT/_1(Const);DMT/_2(Const);E(Zeta)|A->D:2;B->D;B:1->D:1;"
            "B:control->DMT/_0:control;B:control->DMT/_1:control;"
            "B:control->DMT/_2:control;C->E;D->E:1;DMT/_0->D:3;"
            "DMT/_1->D:4;DMT/_2->D:5");
}

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
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B']}"
      "node { name: 'F' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
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
      "node { name: 'I' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'H'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);DMT/_3(Const);DMT/_4(Const);E(_MklConv2D);"
            "F(_MklConv2D);G(Const);H(_MklConcatV2);I(Zeta)|A->E;A->I;"
            "A:control->DMT/_0:control;A:control->DMT/_1:control;B->E:1;C->F;"
            "C:control->DMT/_2:control;C:control->DMT/_3:control;"
            "D->F:1;DMT/_0->E:2;DMT/_1->E:3;DMT/_2->F:2;DMT/_3->F:3;"
            "DMT/_4->H:5;E->H;E:2->H:3;E:control->DMT/_4:control;F->H:1;"
            "F:2->H:4;G->H:2;H->I:1");
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
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B']}"
      "node { name: 'F' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
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
      "node { name: 'I' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'H'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);DMT/_3(Const);E(_MklConv2D);F(Zeta);G(Const);"
            "H(_MklConcatV2);I(Zeta)|A->E;A->I;A:control->DMT/_0:control;"
            "A:control->DMT/_1:control;B->E:1;C->F;D->F:1;DMT/_0->E:2;"
            "DMT/_1->E:3;DMT/_2->H:4;DMT/_3->H:5;E->H;E:2->H:3;"
            "E:control->DMT/_2:control;E:control->DMT/_3:control;F->H:1;"
            "G->H:2;H->I:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_Relu_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Relu'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(_MklRelu);C(Zeta);DMT/_0(Const)|A->B;A->C;"
            "A:control->DMT/_0:control;B->C:1;DMT/_0->B:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_ReluGrad_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'ReluGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklReluGrad);D(Zeta);DMT/_0(Const);"
            "DMT/_1(Const)|A->C;A->D;A:control->DMT/_0:control;"
            "A:control->DMT/_1:control;B->C:1;C->D:1;DMT/_0->C:2;DMT/_1->C:3");
}

TEST_F(MklLayoutPassTest, NodeRewrite_ReluReluGrad_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Relu'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'ReluGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(_MklRelu);C(_MklReluGrad);D(Zeta);DMT/_0(Const);"
            "DMT/_1(Const)|A->B;A->C;A->D;A:control->DMT/_0:control;"
            "A:control->DMT/_1:control;B->C:1;B:1->C:3;C->D:1;DMT/_0->B:1;"
            "DMT/_1->C:2");
}

TEST_F(MklLayoutPassTest, NodeRewrite_Relu6_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Relu6'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(_MklRelu6);C(Zeta);DMT/_0(Const)|A->B;A->C;"
            "A:control->DMT/_0:control;B->C:1;DMT/_0->B:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_Relu6Grad_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Relu6Grad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklRelu6Grad);D(Zeta);DMT/_0(Const);"
            "DMT/_1(Const)|A->C;A->D;A:control->DMT/_0:control;"
            "A:control->DMT/_1:control;B->C:1;C->D:1;DMT/_0->C:2;DMT/_1->C:3");
}

TEST_F(MklLayoutPassTest, NodeRewrite_Relu6Relu6Grad_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Relu6'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Relu6Grad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(_MklRelu6);C(_MklRelu6Grad);D(Zeta);DMT/_0(Const);"
            "DMT/_1(Const)|A->B;A->C;A->D;A:control->DMT/_0:control;"
            "A:control->DMT/_1:control;B->C:1;B:1->C:3;C->D:1;DMT/_0->B:1;"
            "DMT/_1->C:2");
}

TEST_F(MklLayoutPassTest, NodeRewrite_LeakyRelu_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'LeakyRelu'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'alpha'            value { f: 0.1 } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(_MklLeakyRelu);C(Zeta);DMT/_0(Const)|A->B;A->C;"
            "A:control->DMT/_0:control;B->C:1;DMT/_0->B:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_LeakyRelu_Negative) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'LeakyRelu'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'alpha'            value { f: 2.0 } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(LeakyRelu);C(Zeta)|A->B;A->C;B->C:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_LeakyReluGrad_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'LeakyReluGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'alpha'            value { f: 0.1 } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklLeakyReluGrad);D(Zeta);DMT/_0(Const);"
            "DMT/_1(Const)|A->C;A->D;A:control->DMT/_0:control;"
            "A:control->DMT/_1:control;B->C:1;C->D:1;DMT/_0->C:2;DMT/_1->C:3");
}

TEST_F(MklLayoutPassTest, NodeRewrite_LeakyReluGrad_Negative) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'LeakyReluGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'alpha'            value { f: 2.0 } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }");
  EXPECT_EQ(
      DoMklLayoutOptimizationPass(),
      "A(Input);B(Input);C(LeakyReluGrad);D(Zeta)|A->C;A->D;B->C:1;C->D:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_LeakyReluLeakyReluGrad_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'LeakyRelu'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'alpha'            value { f: 0.1 } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'LeakyReluGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'alpha'            value { f: 0.1 } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }");
  EXPECT_EQ(
      DoMklLayoutOptimizationPass(),
      "A(Input);B(_MklLeakyRelu);C(_MklLeakyReluGrad);D(Zeta);DMT/_0(Const);"
      "DMT/_1(Const)|A->B;A->C;A->D;A:control->DMT/_0:control;"
      "A:control->DMT/_1:control;B->C:1;B:1->C:3;C->D:1;DMT/_0->B:1;"
      "DMT/_1->C:2");
}

TEST_F(MklLayoutPassTest, NodeRewrite_AvgPool_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'AvgPool'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:1, i:3, i:3} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:1, i:2, i:2} } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(_MklAvgPool);C(Zeta);DMT/_0(Const)|A->B;A->C;"
            "A:control->DMT/_0:control;B->C:1;DMT/_0->B:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_AvgPoolGrad_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Int32Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'AvgPoolGrad' "
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:1, i:3, i:3} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:1, i:2, i:2} } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'C'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Int32Input);B(Input);C(_MklAvgPoolGrad);D(Zeta);DMT/_0(Const);"
            "DMT/_1(Const)|A->C;A:control->DMT/_0:control;"
            "A:control->DMT/_1:control;B->C:1;B->D;C->D:1;DMT/_0->C:2;"
            "DMT/_1->C:3");
}

TEST_F(MklLayoutPassTest, NodeRewrite_AvgPoolAvgPoolGrad_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'I' op: 'Int32Input'}"
      "node { name: 'B' op: 'AvgPool'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:1, i:3, i:3} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:1, i:2, i:2} } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'AvgPoolGrad' "
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:1, i:3, i:3} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:1, i:2, i:2} } }"
      " input: ['I', 'B'] }"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(_MklAvgPool);C(_MklAvgPoolGrad);D(Zeta);DMT/_0(Const);"
            "DMT/_1(Const);I(Int32Input)|A->B;A->D;A:control->DMT/_0:control;"
            "B->C:1;B:1->C:3;C->D:1;DMT/_0->B:1;DMT/_1->C:2;I->C;"
            "I:control->DMT/_1:control");
}

TEST_F(MklLayoutPassTest, NodeRewrite_FusedBatchNormGrad_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'FusedBatchNormGrad'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'epsilon'      value { f: 0.0001 } }"
      " attr { key: 'is_training'  value { b: true } }"
      " input: ['A', 'B', 'C', 'D', 'E'] }"
      "node { name: 'G' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'F'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);DMT/_3(Const);DMT/_4(Const);E(Input);"
            "F(_MklFusedBatchNormGrad);G(Zeta)|A->F;A->G;"
            "A:control->DMT/_0:control;A:control->DMT/_1:control;"
            "A:control->DMT/_2:control;A:control->DMT/_3:control;"
            "A:control->DMT/_4:control;B->F:1;C->F:2;D->F:3;"
            "DMT/_0->F:5;DMT/_1->F:6;DMT/_2->F:7;DMT/_3->F:8;DMT/_4->F:9;"
            "E->F:4;F->G:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_FusedBatchNormGradV2_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'FusedBatchNormGradV2'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'U'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'epsilon'      value { f: 0.0001 } }"
      " attr { key: 'is_training'  value { b: true } }"
      " input: ['A', 'B', 'C', 'D', 'E'] }"
      "node { name: 'G' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'F'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);DMT/_3(Const);DMT/_4(Const);E(Input);"
            "F(_MklFusedBatchNormGradV2);G(Zeta)|A->F;A->G;"
            "A:control->DMT/_0:control;A:control->DMT/_1:control;"
            "A:control->DMT/_2:control;A:control->DMT/_3:control;"
            "A:control->DMT/_4:control;B->F:1;C->F:2;D->F:3;"
            "DMT/_0->F:5;DMT/_1->F:6;DMT/_2->F:7;DMT/_3->F:8;DMT/_4->F:9;"
            "E->F:4;F->G:1");
}

// T, U combination is not supported by MKL. Node will not be rewritten
// into MKL node.
TEST_F(MklLayoutPassTest, NodeRewrite_FusedBatchNormGradV2_Negative) {
  InitGraph(
      "node { name: 'A' op: 'HalfInput'}"
      "node { name: 'B' op: 'HalfInput'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'FusedBatchNormGradV2'"
      " attr { key: 'T'            value { type: DT_HALF } }"
      " attr { key: 'U'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'epsilon'      value { f: 0.0001 } }"
      " attr { key: 'is_training'  value { b: true } }"
      " input: ['A', 'B', 'C', 'D', 'E'] }"
      "node { name: 'G' op: 'Zeta' attr { key: 'T' value { type: DT_HALF } }"
      " input: ['A', 'F'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(HalfInput);B(HalfInput);C(Input);D(Input);E(Input);"
            "F(FusedBatchNormGradV2);G(Zeta)|A->F;A->G;"
            "B->F:1;C->F:2;D->F:3;E->F:4;F->G:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_FusedBatchNorm_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'FusedBatchNorm'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'epsilon'      value { f: 0.0001 } }"
      " attr { key: 'is_training'  value { b: true } }"
      " input: ['A', 'B', 'C', 'D', 'E'] }"
      "node { name: 'G' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'F'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);DMT/_3(Const);DMT/_4(Const);E(Input);"
            "F(_MklFusedBatchNorm);G(Zeta)|A->F;A->G;"
            "A:control->DMT/_0:control;A:control->DMT/_1:control;"
            "A:control->DMT/_2:control;A:control->DMT/_3:control;"
            "A:control->DMT/_4:control;B->F:1;C->F:2;D->F:3;"
            "DMT/_0->F:5;DMT/_1->F:6;DMT/_2->F:7;DMT/_3->F:8;DMT/_4->F:9;"
            "E->F:4;F->G:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_FusedBatchNormV2_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'FusedBatchNormV2'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'U'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'epsilon'      value { f: 0.0001 } }"
      " attr { key: 'is_training'  value { b: true } }"
      " input: ['A', 'B', 'C', 'D', 'E'] }"
      "node { name: 'G' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'F'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);DMT/_3(Const);DMT/_4(Const);E(Input);"
            "F(_MklFusedBatchNormV2);G(Zeta)|A->F;A->G;"
            "A:control->DMT/_0:control;A:control->DMT/_1:control;"
            "A:control->DMT/_2:control;A:control->DMT/_3:control;"
            "A:control->DMT/_4:control;B->F:1;C->F:2;D->F:3;"
            "DMT/_0->F:5;DMT/_1->F:6;DMT/_2->F:7;DMT/_3->F:8;DMT/_4->F:9;"
            "E->F:4;F->G:1");
}

// T, U combination is not supported by MKL. Node will not be rewritten
// into MKL node.
TEST_F(MklLayoutPassTest, NodeRewrite_FusedBatchNormV2_Negative) {
  InitGraph(
      "node { name: 'A' op: 'HalfInput'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'FusedBatchNormV2'"
      " attr { key: 'T'            value { type: DT_HALF } }"
      " attr { key: 'U'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'epsilon'      value { f: 0.0001 } }"
      " attr { key: 'is_training'  value { b: true } }"
      " input: ['A', 'B', 'C', 'D', 'E'] }"
      "node { name: 'G' op: 'Zeta' attr { key: 'T' value { type: DT_HALF } }"
      " input: ['A', 'F'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(HalfInput);B(Input);C(Input);D(Input);E(Input);"
            "F(FusedBatchNormV2);G(Zeta)|A->F;A->G;"
            "B->F:1;C->F:2;D->F:3;E->F:4;F->G:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_FusedBatchNormV3_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'FusedBatchNormV3'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'U'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'epsilon'      value { f: 0.0001 } }"
      " attr { key: 'is_training'  value { b: true } }"
      " input: ['A', 'B', 'C', 'D', 'E'] }"
      "node { name: 'G' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'F'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);DMT/_3(Const);DMT/_4(Const);E(Input);"
            "F(_MklFusedBatchNormV3);G(Zeta)|A->F;A->G;"
            "A:control->DMT/_0:control;A:control->DMT/_1:control;"
            "A:control->DMT/_2:control;A:control->DMT/_3:control;"
            "A:control->DMT/_4:control;B->F:1;C->F:2;D->F:3;"
            "DMT/_0->F:5;DMT/_1->F:6;DMT/_2->F:7;DMT/_3->F:8;DMT/_4->F:9;"
            "E->F:4;F->G:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_FusedBatchNormV3_Negative) {
  InitGraph(
      "node { name: 'A' op: 'HalfInput'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'FusedBatchNormV3'"
      " attr { key: 'T'            value { type: DT_HALF } }"
      " attr { key: 'U'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'epsilon'      value { f: 0.0001 } }"
      " attr { key: 'is_training'  value { b: true } }"
      " input: ['A', 'B', 'C', 'D', 'E'] }"
      "node { name: 'G' op: 'Zeta' attr { key: 'T' value { type: DT_HALF } }"
      " input: ['A', 'F'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(HalfInput);B(Input);C(Input);D(Input);E(Input);"
            "F(FusedBatchNormV3);G(Zeta)|A->F;A->G;"
            "B->F:1;C->F:2;D->F:3;E->F:4;F->G:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_QuantizedDepthwiseConv2D_Positive) {
  InitGraph(
      "node { name: 'A' op: 'QuantizedUnsignedInt8Input'}"
      "node { name: 'B' op: 'QuantizedSignedInt8Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'Input'}"
      "node { name: 'G' op: 'QuantizedSignedInt32Input'}"
      "node { name: 'H' op: 'QuantizedDepthwiseConv2D'"
      " attr { key: 'Tinput'           value { type: DT_QUINT8 } }"
      " attr { key: 'Tfilter'          value { type: DT_QINT8 } }"
      " attr { key: 'out_type'         value { type: DT_QINT32 } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B', 'C', 'D', 'E', 'F'] }"
      "node { name: 'I' op: 'Zeta' attr { key: 'T' value { type: DT_QINT32 } }"
      " input: ['G', 'H'] }");
  EXPECT_EQ(
      DoMklLayoutOptimizationPass(),
      "A(QuantizedUnsignedInt8Input);B(QuantizedSignedInt8Input);C(Input);"
      "D(Input);DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);DMT/_3(Const);"
      "DMT/_4(Const);DMT/_5(Const);E(Input);F(Input);"
      "G(QuantizedSignedInt32Input);H(_MklQuantizedDepthwiseConv2D);I(Zeta)"
      "|A->H;A:control->DMT/_0:control;A:control->DMT/_1:control;"
      "A:control->DMT/_2:control;A:control->DMT/_3:control;"
      "A:control->DMT/_4:control;A:control->DMT/_5:control;B->H:1;C->H:2;"
      "D->H:3;DMT/_0->H:6;DMT/_1->H:7;DMT/_2->H:8;DMT/_3->H:9;DMT/_4->H:10;"
      "DMT/_5->H:11;E->H:4;F->H:5;G->I;H->I:1");
}

/////////////////////////////////////////////////////////////////////
//  Unit tests related to context-based node rewrite
/////////////////////////////////////////////////////////////////////

// If any of the inputs is an MKL op, then rewrite Slice to Mkl op.
TEST_F(MklLayoutPassTest, NodeRewrite_Ctxbased_Slice_Positive) {
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
      "node { name: 'D' op: 'Int32Input'}"
      "node { name: 'E' op: 'Int32Input'}"
      "node { name: 'F' op: 'Slice'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'Index'        value { type: DT_INT32 } }"
      " input: ['C', 'D', 'E'] }"
      "node { name: 'G' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklConv2D);D(Int32Input);"
            "DMT/_0(Const);DMT/_1(Const);"
            "E(Int32Input);F(_MklSlice);G(Zeta);M(_MklInput);N(_MklInput)|"
            "A->C;A->G;B->C:1;C->F;C->G:1;C:2->F:3;"
            "C:control->DMT/_0:control;C:control->DMT/"
            "_1:control;"
            "D->F:1;DMT/_0->F:4;DMT/_1->F:5;"
            "E->F:2;M->C:2;N->C:3");
}

// If none of the inputs is an MKL op, then Slice should not be rewritten.
TEST_F(MklLayoutPassTest, NodeRewrite_Ctxbased_Slice_Negative) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Int32Input'}"
      "node { name: 'C' op: 'Int32Input'}"
      "node { name: 'D' op: 'Slice'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'Index'        value { type: DT_INT32 } }"
      " input: ['A', 'B', 'C'] }"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Int32Input);C(Int32Input);"
            "D(Slice);E(Zeta)|A->D;A->E;B->D:1;C->D:2;D->E:1");
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
      " attr { key: 'depth_radius' value { i: 2 } }"
      " input: ['E', 'F', 'B'] }"
      "node { name: 'H' op: 'Input'}"
      "node { name: 'I' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['H', 'G'] }");
  EXPECT_EQ(
      DoMklLayoutOptimizationPass(),
      "A(Input);B(_MklLRN);C(_MklMaxPool);D(Input);DMT/_0(Const);DMT/_1(Const);"
      "DMT/_2(Const);E(_MklMaxPoolGrad);F(Input);G(_MklLRNGrad);H(Input);"
      "I(Zeta)|A->B;A:control->DMT/_0:control;B->C;B->E;B->G:2;B:1->G:3;"
      "B:2->C:1;B:2->E:4;B:2->G:6;B:3->G:7;B:control->DMT/_1:control;C->E:1;"
      "C:1->E:3;C:2->E:5;C:3->E:7;D->E:2;DMT/_0->B:1;DMT/_1->E:6;DMT/_2->G:5;"
      "E->G;E:1->G:4;E:control->DMT/_2:control;F->G:1;G->I:1;H->I");
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
      " attr { key: 'depth_radius' value { i: 2 } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'LRNGrad'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'alpha'        value { f: 0.001 } }"
      " attr { key: 'beta'         value { f: 0.75 } }"
      " attr { key: 'bias'         value { f: 1.0 } }"
      " attr { key: 'depth_radius' value { i: 2 } }"
      " input: ['C', 'D', 'B'] }"
      "node { name: 'F' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'E'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(_MklLRN);C(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);E(_MklLRNGrad);F(Zeta)|"
            "A->B;A:control->DMT/_0:control;B->E:2;B:1->E:3;B:2->E:6;B:3->E:7;"
            "C->E;C->F;C:control->DMT/_1:control;C:control->DMT/_2:control;"
            "D->E:1;DMT/_0->B:1;DMT/_1->E:4;DMT/_2->E:5;E->F:1");
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
      " attr { key: 'depth_radius' value { i: 2 } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(_MklLRN);C(Zeta);DMT/_0(Const)|"
            "A->B;A->C;A:control->DMT/_0:control;B->C:1;DMT/_0->B:1");
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
      " attr { key: 'depth_radius' value { i: 2 } }"
      " input: ['A', 'B', 'C'] }"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(LRNGrad);"
            "E(Zeta)|A->D;A->E;B->D:1;C->D:2;D->E:1");
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
      " attr { key: 'depth_radius' value { i: 2 } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'LRNGrad'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'alpha'        value { f: 0.001 } }"
      " attr { key: 'beta'         value { f: 0.75 } }"
      " attr { key: 'bias'         value { f: 1.0 } }"
      " attr { key: 'depth_radius' value { i: 2 } }"
      " input: ['C', 'D', 'B'] }"
      "node { name: 'F' op: 'LRNGrad'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'alpha'        value { f: 0.001 } }"
      " attr { key: 'beta'         value { f: 0.75 } }"
      " attr { key: 'bias'         value { f: 1.0 } }"
      " attr { key: 'depth_radius' value { i: 2 } }"
      " input: ['C', 'B', 'D'] }"
      "node { name: 'G' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['E', 'F'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(_MklLRN);C(Input);D(Input);DMT/_0(Const);DMT/_1(Const);"
            "DMT/_2(Const);E(_MklLRNGrad);F(LRNGrad);G(Zeta)|A->B;"
            "A:control->DMT/_0:control;B->E:2;B->F:1;B:1->E:3;B:2->E:6;"
            "B:3->E:7;C->E;C->F;C:control->DMT/_1:control;"
            "C:control->DMT/_2:control;D->E:1;D->F:2;DMT/_0->B:1;"
            "DMT/_1->E:4;DMT/_2->E:5;E->G;F->G:1");
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
      "node { name: 'F' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'E'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(_MklMaxPool);C(Input);D(Input);DMT/_0(Const);"
            "DMT/_1(Const);DMT/_2(Const);E(_MklMaxPoolGrad);F(Zeta)|"
            "A->B;A:control->DMT/_0:control;B->E:1;B:1->E:3;B:2->E:5;B:3->E:7;"
            "C->E;C->F;C:control->DMT/_1:control;C:control->DMT/_2:control;"
            "D->E:2;DMT/_0->B:1;DMT/_1->E:4;DMT/_2->E:6;E->F:1");
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
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(_MklMaxPool);C(Zeta);DMT/_0(Const)|"
            "A->B;A->C;A:control->DMT/_0:control;B->C:1;DMT/_0->B:1");
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
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'D'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(MaxPoolGrad);"
            "E(Zeta)|A->D;A->E;B->D:1;C->D:2;D->E:1");
}

// Test MaxPool handling for batch-wise pooling (NCHW)
// No rewrite should take place in such case
TEST_F(MklLayoutPassTest, NodeWorkspace_MaxPool_Negative3) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'MaxPool'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'ksize'        value { list: {i: 2, i:1, i:1, i:1} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(MaxPool);C(Zeta)|A->B;A->C;B->C:1");
}

// Test MaxPool handling for batch-wise pooling (NCHW)
// No rewrite should take place in such case
TEST_F(MklLayoutPassTest, NodeWorkspace_MaxPool_Negative4) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'MaxPool'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 2, i:1, i:1, i:1} } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(MaxPool);C(Zeta)|A->B;A->C;B->C:1");
}

// Test MaxPool handling for depth-wise pooling (NHWC)
// No rewrite should take place in such case
TEST_F(MklLayoutPassTest, NodeWorkspace_MaxPool_Negative5) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'MaxPool'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:2, i:1, i:1} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(MaxPool);C(Zeta)|A->B;A->C;B->C:1");
}

// Test MaxPool handling for depth-wise pooling (NCHW)
// No rewrite should take place in such case
TEST_F(MklLayoutPassTest, NodeWorkspace_MaxPool_Negative6) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'MaxPool'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:2, i:1, i:1} } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(MaxPool);C(Zeta)|A->B;A->C;B->C:1");
}

// Test MaxPool handling for batch-wise pooling (NHWC)
// No rewrite should take place in such case
TEST_F(MklLayoutPassTest, NodeWorkspace_MaxPool_Negative7) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'MaxPool'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NHWC' } }"
      " attr { key: 'ksize'        value { list: {i: 2, i:1, i:1, i:1} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(MaxPool);C(Zeta)|A->B;A->C;B->C:1");
}

// Test MaxPool handling for batch-wise pooling (NHWC)
// No rewrite should take place in such case
TEST_F(MklLayoutPassTest, NodeWorkspace_MaxPool_Negative8) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'MaxPool'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NHWC' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 2, i:1, i:1, i:1} } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(MaxPool);C(Zeta)|A->B;A->C;B->C:1");
}

// Test MaxPool handling for depth-wise pooling (NHWC)
// No rewrite should take place in such case
TEST_F(MklLayoutPassTest, NodeWorkspace_MaxPool_Negative9) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'MaxPool'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NHWC' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:1, i:1, i:2} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(MaxPool);C(Zeta)|A->B;A->C;B->C:1");
}

// Test MaxPool handling for depth-wise pooling (NHWC)
// No rewrite should take place in such case
TEST_F(MklLayoutPassTest, NodeWorkspace_MaxPool_Negative10) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'MaxPool'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NHWC' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:1, i:1, i:2} } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(MaxPool);C(Zeta)|A->B;A->C;B->C:1");
}

/////////////////////////////////////////////////////////////////////

// Single Conv2D Op on GPU device
// No rewrite should happen
TEST_F(MklLayoutPassTest, NodeRewrite_Conv2D_DeviceTest) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'C'] }",
      kGPUDevice);
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Conv2D);D(Zeta)|A->C;B->C:1;B->D;C->D:1");
}

TEST_F(MklLayoutPassTest, NodeMerge_Conv2DBackprop_DeviceTest) {
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
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B', 'C', 'M', 'N', 'O']}"
      "node { name: 'E' op: 'Zeta'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['D', 'A']}"
      "node { name: 'F' op: 'BiasAddGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['E'] }",
      kGPUDevice);
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(_MklConv2DWithBias);"
            "E(Zeta);F(BiasAddGrad);M(_MklInput);N(_MklInput);"
            "O(_MklInput)|A->D;A->E:1;B->D:1;C->D:2;D->E;E->F;"
            "M->D:3;N->D:4;O->D:5");
}

TEST_F(MklLayoutPassTest, NodeRewrite_Conv2DGradFilter_DeviceTest) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Int32Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Conv2DBackpropFilter'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'D'] }",
      kGPUDevice);
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Int32Input);C(Input);D(Conv2DBackpropFilter);E(Zeta)|"
            "A->D;A->E;B->D:1;C->D:2;D->E:1");
}

TEST_F(MklLayoutPassTest,
       NodeRewrite_DepthwiseConv2dNativeGradFilter_DeviceTest) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Int32Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'DepthwiseConv2dNativeBackpropFilter'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'D'] }",
      kGPUDevice);
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Int32Input);C(Input);D("
            "DepthwiseConv2dNativeBackpropFilter);E(Zeta)|"
            "A->D;A->E;B->D:1;C->D:2;D->E:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_Relu_DeviceTest) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Relu'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }",
      kGPUDevice);
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Relu);C(Zeta)|A->B;A->C;B->C:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_ReluGrad_DeviceTest) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'ReluGrad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }",
      kGPUDevice);
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(ReluGrad);D(Zeta)|A->C;A->D;B->C:1;C->D:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_Relu6_DeviceTest) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Relu6'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }",
      kGPUDevice);
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Relu6);C(Zeta)|A->B;A->C;B->C:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_Relu6Grad_DeviceTest) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Relu6Grad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }",
      kGPUDevice);
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Relu6Grad);D(Zeta)|A->C;A->D;B->C:1;C->D:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_MaxPool_DeviceTest) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'MaxPool'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NHWC' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }",
      kGPUDevice);
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(MaxPool);C(Zeta)|A->B;A->C;B->C:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_AvgPool_DeviceTest) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'AvgPool'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NHWC' } }"
      " attr { key: 'ksize'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'      value { s: 'VALID' } }"
      " attr { key: 'strides'      value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }",
      kGPUDevice);
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(AvgPool);C(Zeta)|A->B;A->C;B->C:1");
}

// Concat Op test: Concat with no Mkl layer feeding it
TEST_F(MklLayoutPassTest, NodeRewrite_Concat_DeviceTest) {
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
      " input: ['A', 'B:0', 'B:1']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'D'] }",
      kGPUDevice);
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Const);B(InputList);C(Input);D(Concat);E(Zeta)|A->D;"
            "B->D:1;B:1->D:2;C->E;D->E:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_ConcatV2_DeviceTest) {
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
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'D'] }",
      kGPUDevice);
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Const);B(InputList);C(Input);D(ConcatV2);E(Zeta)|"
            "A->D:2;B->D;B:1->D:1;C->E;D->E:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_FusedBatchNorm_DeviceTest) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'FusedBatchNorm'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'epsilon'      value { f: 0.0001 } }"
      " attr { key: 'is_training'  value { b: true } }"
      " input: ['A', 'B', 'C', 'D', 'E'] }"
      "node { name: 'G' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'F'] }",
      kGPUDevice);
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(Input);E(Input);"
            "F(FusedBatchNorm);G(Zeta)|A->F;A->G;B->F:1;C->F:2;D->F:3;"
            "E->F:4;F->G:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_FusedBatchNormV2_DeviceTest) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'FusedBatchNorm'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'U'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'epsilon'      value { f: 0.0001 } }"
      " attr { key: 'is_training'  value { b: true } }"
      " input: ['A', 'B', 'C', 'D', 'E'] }"
      "node { name: 'G' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'F'] }",
      kGPUDevice);
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(Input);E(Input);"
            "F(FusedBatchNorm);G(Zeta)|A->F;A->G;B->F:1;C->F:2;D->F:3;"
            "E->F:4;F->G:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_FusedBatchNormV3_DeviceTest) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'FusedBatchNorm'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'U'            value { type: DT_FLOAT } }"
      " attr { key: 'data_format'  value { s: 'NCHW' } }"
      " attr { key: 'epsilon'      value { f: 0.0001 } }"
      " attr { key: 'is_training'  value { b: true } }"
      " input: ['A', 'B', 'C', 'D', 'E'] }"
      "node { name: 'G' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'F'] }",
      kGPUDevice);
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(Input);D(Input);E(Input);"
            "F(FusedBatchNorm);G(Zeta)|A->F;A->G;B->F:1;C->F:2;D->F:3;"
            "E->F:4;F->G:1");
}

TEST_F(MklLayoutPassTest, NodeMerge_Conv2DWithBias_DeviceTest) {
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
      "node { name: 'Z' op: 'Zeta'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['E', 'Y']}",
      kGPUDevice);
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklConv2D);D(Input);E(BiasAdd);"
            "M(_MklInput);N(_MklInput);Y(Input);Z(Zeta)|A->C;"
            "B->C:1;C->E;D->E:1;E->Z;M->C:2;N->C:3;Y->Z:1");
}

TEST_F(MklLayoutPassTest, NodeRewrite_Slice_DeviceTest) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Int32Input'}"
      "node { name: 'C' op: 'Int32Input'}"
      "node { name: 'D' op: 'Slice'"
      " attr { key: 'T'            value { type: DT_FLOAT } }"
      " attr { key: 'Index'        value { type: DT_INT32 } }"
      " input: ['A', 'B', 'C'] }"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'D'] }",
      kGPUDevice);
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Int32Input);C(Int32Input);D(Slice);E(Zeta)|A->D;A->E;"
            "B->D:1;C->D:2;D->E:1");
}

// The following positive and negative tests test the rewrite of Add and AddV2
// to MKL versions. The operators will be rewritten only if one of the inputs
// comes from another MKL operator.
TEST_F(MklLayoutPassTest, PositiveRewriteAdd) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'M' op: 'Relu'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A']}"
      "node { name: 'N' op: 'Add'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['M', 'B']}");
  EXPECT_EQ(
      DoMklLayoutOptimizationPass(),
      "A(Input);B(Input);DMT/_0(Const);DMT/_1(Const);M(_MklRelu);N(_MklAdd)"
      "|A->M;A:control->DMT/_0:control;B->N:1;DMT/_0->M:1;DMT/_1->N:3;M->N;"
      "M:1->N:2;M:control->DMT/_1:control");
}

TEST_F(MklLayoutPassTest, NegativeRewriteAdd) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'N' op: 'Add'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A', 'B']}");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);N(Add)|A->N;B->N:1");
}

TEST_F(MklLayoutPassTest, PositiveRewriteAddV2) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'M' op: 'Relu'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A']}"
      "node { name: 'N' op: 'AddV2'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['M', 'B']}");
  EXPECT_EQ(
      DoMklLayoutOptimizationPass(),
      "A(Input);B(Input);DMT/_0(Const);DMT/_1(Const);M(_MklRelu);N(_MklAddV2)"
      "|A->M;A:control->DMT/_0:control;B->N:1;DMT/_0->M:1;DMT/_1->N:3;M->N;"
      "M:1->N:2;M:control->DMT/_1:control");
}

TEST_F(MklLayoutPassTest, NegativeRewriteAddV2) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'N' op: 'AddV2'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " input: ['A', 'B']}");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);N(AddV2)|A->N;B->N:1");
}

/////////////////////////////////////////////////////////////////////
//         Post-rewrite fixup pass test
/////////////////////////////////////////////////////////////////////

TEST_F(MklLayoutPassTest, PostRewriteFixUpPass) {
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
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B', 'M', 'N']}"
      "node { name: 'D' op: 'Const' "
      " attr { key: 'dtype' value { type: DT_UINT8 } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_UINT8 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'E' op: '_MklAdd'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['C', 'A', 'D', 'D']}");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklConv2D);D(Const);E(_MklAdd);"
            "M(_MklInput);N(_MklInput)|A->C;A->E:1;B->C:1;C->E;C:2->E:2;"
            "D->E:3;M->C:2;N->C:3");
}

/////////////////////////////////////////////////////////////////////
//         Unit tests related to filter caching.
//
// These tests check if the attribute `is_filter_const` is set to true
// when filter is a constant and false otherwise for various operators
// such as Conv2D, Conv2DWithBias, Conv3D etc.
/////////////////////////////////////////////////////////////////////

// Conv2D op where filter is a constant.
TEST_F(MklLayoutPassTest, Conv2D_FilterCaching_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Const' "  // Filter
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_FLOAT tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'C' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'C'] }");
  EXPECT_TRUE(DoMklLayoutOptimizationPassGetAttrVal<bool>("is_filter_const",
                                                          "_MklConv2D"));
}

// Conv2D op where filter is NOT a constant.
TEST_F(MklLayoutPassTest, Conv2D_FilterCaching_Negative) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"  // Filter
      "node { name: 'C' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'C'] }");
  EXPECT_FALSE(DoMklLayoutOptimizationPassGetAttrVal<bool>("is_filter_const",
                                                           "_MklConv2D"));
}

// Conv2D + BiasAdd fusion where filter is a constant.
TEST_F(MklLayoutPassTest, Conv2DWithBias_FilterCaching_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Const'"  // Filter
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_FLOAT tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'C' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'BiasAdd'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['C', 'D'] }"
      "node { name: 'Y' op: 'Input'}"
      "node { name: 'Z' op: 'Zeta'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['E', 'Y']}");
  EXPECT_TRUE(DoMklLayoutOptimizationPassGetAttrVal<bool>(
      "is_filter_const", "_MklConv2DWithBias"));
}

// Conv2D + BiasAdd fusion where filter is NOT a constant.
TEST_F(MklLayoutPassTest, Conv2DWithBias_FilterCaching_Negative) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"  // Filter
      "node { name: 'C' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'BiasAdd'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['C', 'D'] }"
      "node { name: 'Y' op: 'Input'}"
      "node { name: 'Z' op: 'Zeta'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['E', 'Y']}");
  EXPECT_FALSE(DoMklLayoutOptimizationPassGetAttrVal<bool>(
      "is_filter_const", "_MklConv2DWithBias"));
}

// Conv3D op where filter is a constant.
TEST_F(MklLayoutPassTest, Conv3D_FilterCaching_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Const' "  // Filter
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_FLOAT tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'C' op: 'Conv3D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCDHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1, "
      "i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1, "
      "i:1} } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'C'] }");
  EXPECT_TRUE(DoMklLayoutOptimizationPassGetAttrVal<bool>("is_filter_const",
                                                          "_MklConv3D"));
}

// Conv3D op where filter is NOT a constant.
TEST_F(MklLayoutPassTest, Conv3D_FilterCaching_Negative) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"  // Filter
      "node { name: 'C' op: 'Conv3D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCDHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1, "
      "i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1, "
      "i:1} } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'C'] }");
  EXPECT_FALSE(DoMklLayoutOptimizationPassGetAttrVal<bool>("is_filter_const",
                                                           "_MklConv3D"));
}

// Pad + Conv2D fusion where filter is a constant.
TEST_F(MklLayoutPassTest, PadWithConv2D_FilterCaching_Positive) {
  DCHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Int32Input'}"
      "node { name: 'C' op: 'Pad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'Tpaddings'        value { type: DT_INT32 } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Const'"  // Filter
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_FLOAT tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'E' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NHWC' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'VALID' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['C', 'D'] }"
      "node { name: 'Y' op: 'Input'}"
      "node { name: 'Z' op: 'Zeta'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['E', 'Y']}");
  EXPECT_TRUE(DoMklLayoutOptimizationPassGetAttrVal<bool>("is_filter_const",
                                                          "_MklPadWithConv2D"));
}

// Pad + Conv2D fusion where filter is NOT a constant.
TEST_F(MklLayoutPassTest, PadWithConv2D_FilterCaching_Negative) {
  DCHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Int32Input'}"
      "node { name: 'C' op: 'Pad'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'Tpaddings'        value { type: DT_INT32 } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Input'}"  // Filter
      "node { name: 'E' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NHWC' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'VALID' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['C', 'D'] }"
      "node { name: 'Y' op: 'Input'}"
      "node { name: 'Z' op: 'Zeta'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['E', 'Y']}");
  EXPECT_FALSE(DoMklLayoutOptimizationPassGetAttrVal<bool>(
      "is_filter_const", "_MklPadWithConv2D"));
}

// _FusedConv2D + BiasAdd fusion where filter is a constant.
TEST_F(MklLayoutPassTest, FusedConv2DWithBias_FilterCaching_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Const'"  // Filter
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_FLOAT tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: '_FusedConv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'num_args'         value { i: 1 } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'fused_ops'        value { list: {s: 'BiasAdd'} } }"
      " attr { key: 'epsilon'          value { f: 0.001 }}"
      " input: ['A', 'B', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['D', 'C'] }");
  EXPECT_TRUE(DoMklLayoutOptimizationPassGetAttrVal<bool>("is_filter_const",
                                                          "_MklFusedConv2D"));
}

// _FusedDepthwiseConv2dNative + BiasAdd fusion where filter is a constant.
TEST_F(MklLayoutPassTest,
       FusedDepthwiseConv2dNativeWithBias_FilterCaching_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Const'"  // Filter
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_FLOAT tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: '_FusedDepthwiseConv2dNative'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'num_args'         value { i: 1 } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'fused_ops'        value { list: {s: 'BiasAdd'} } }"
      " attr { key: 'epsilon'          value { f: 0.001 }}"
      " input: ['A', 'B', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['D', 'C'] }");
  EXPECT_TRUE(DoMklLayoutOptimizationPassGetAttrVal<bool>(
      "is_filter_const", "_MklFusedDepthwiseConv2dNative"));
}

// _FusedConv2D + BiasAdd fusion where filter is NOT a constant.
TEST_F(MklLayoutPassTest, FusedConv2DWithBias_FilterCaching_Negative) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"  // Filter
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: '_FusedConv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'num_args'         value { i: 1 } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'fused_ops'        value { list: {s: 'BiasAdd'} } }"
      " attr { key: 'epsilon'          value { f: 0.001 }}"
      " input: ['A', 'B', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['D', 'C'] }");
  EXPECT_FALSE(DoMklLayoutOptimizationPassGetAttrVal<bool>("is_filter_const",
                                                           "_MklFusedConv2D"));
}

// _FusedDepthwiseConv2dNative + BiasAdd fusion where filter is NOT a constant.
TEST_F(MklLayoutPassTest,
       FusedDepthwiseConv2dNativeWithBias_FilterCaching_Negative) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"  // Filter
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: '_FusedDepthwiseConv2dNative'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'num_args'         value { i: 1 } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'fused_ops'        value { list: {s: 'BiasAdd'} } }"
      " attr { key: 'epsilon'          value { f: 0.001 }}"
      " input: ['A', 'B', 'C']}"
      "node { name: 'E' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['D', 'C'] }");
  EXPECT_FALSE(DoMklLayoutOptimizationPassGetAttrVal<bool>(
      "is_filter_const", "_MklFusedDepthwiseConv2dNative"));
}
// Depthwise Conv2D op where filter is a constant.
TEST_F(MklLayoutPassTest, DepthwiseConv2dNative_FilterCaching_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Const'"  // Filter
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " attr { key: 'value' value { "
      "    tensor { dtype: DT_FLOAT tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'C' op: 'DepthwiseConv2dNative'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'C'] }");
  EXPECT_TRUE(DoMklLayoutOptimizationPassGetAttrVal<bool>(
      "is_filter_const", "_MklDepthwiseConv2dNative"));
}

// Depthwise Conv2D op where filter is NOT a constant.
TEST_F(MklLayoutPassTest, DepthwiseConv2dNative_FilterCaching_Negative) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"  // Filter
      "node { name: 'C' op: 'DepthwiseConv2dNative'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " attr { key: 'dilations'        value { list: {i: 1, i:1, i:1, i:1} } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Zeta' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'C'] }");
  EXPECT_FALSE(DoMklLayoutOptimizationPassGetAttrVal<bool>(
      "is_filter_const", "_MklDepthwiseConv2dNative"));
}

// Fused QuantizedMatMulWithBias Op Rewrite test
// Rewrite the QuantizedMatMulWithBias with _MklQuantizedMatMulWithBias
TEST_F(MklLayoutPassTest, NodeRewrite_QuantizedMatMulWithBias_Positive) {
  InitGraph(
      "node { name: 'A' op: 'QUInt8Input' }"
      "node { name: 'B' op: 'QInt8Input' }"
      "node { name: 'C' op: 'QInt32Input' }"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'Input'}"
      "node { name: 'G' op: 'Input'}"
      "node { name: 'H' op: 'QInt32Input'}"
      "node { name: 'I' op: 'QuantizedMatMulWithBias'"
      " attr { key: 'T1'    value { type: DT_QUINT8 } }"
      " attr { key: 'T2'    value { type: DT_QINT8 } }"
      " attr { key: 'Tbias'    value { type: DT_QINT32 } }"
      " attr { key: 'Toutput' value { type: DT_QINT32 } }"
      " input: ['A', 'B', 'C', 'D', 'E', 'F', 'G']}"
      "node { name: 'J' op: 'Zeta' attr { key: 'T' value { type: DT_QINT32 } }"
      " input: ['I', 'H'] }");

  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(QUInt8Input);B(QInt8Input);C(QInt32Input);D(Input);"
            "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);DMT/_3(Const);"
            "DMT/_4(Const);DMT/_5(Const);DMT/_6(Const);E(Input);F(Input);"
            "G(Input);H(QInt32Input);I(_MklQuantizedMatMulWithBias);"
            "J(Zeta)|A->I;"
            "A:control->DMT/_0:control;A:control->DMT/_1:control;"
            "A:control->DMT/_2:control;A:control->DMT/_3:control;"
            "A:control->DMT/_4:control;A:control->DMT/_5:control;"
            "A:control->DMT/_6:control;B->I:1;C->I:2;D->I:3;DMT/_0->I:7;"
            "DMT/_1->I:8;DMT/_2->I:9;DMT/_3->I:10;DMT/_4->I:11;DMT/_5->I:12;"
            "DMT/_6->I:13;E->I:4;F->I:5;G->I:6;H->J:1;I->J");
}

// Rewrite test for QuantizedMatMulWithBias Op with unsupported input
// Rewrite should not happen
TEST_F(MklLayoutPassTest, NodeRewrite_QuantizedMatMulWithBias_Negative) {
  InitGraph(
      "node { name: 'A' op: 'QUInt8Input' }"
      "node { name: 'B' op: 'QUInt8Input' }"
      "node { name: 'C' op: 'QInt32Input' }"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'Input'}"
      "node { name: 'G' op: 'Input'}"
      "node { name: 'H' op: 'QInt32Input'}"
      "node { name: 'I' op: 'QuantizedMatMulWithBias'"
      " attr { key: 'T1'    value { type: DT_QUINT8 } }"
      " attr { key: 'T2'    value { type: DT_QUINT8 } }"
      " attr { key: 'Tbias'    value { type: DT_QINT32 } }"
      " attr { key: 'Toutput' value { type: DT_QINT32 } }"
      " input: ['A', 'B', 'C', 'D', 'E', 'F', 'G']}"
      "node { name: 'J' op: 'Zeta' attr { key: 'T' value { type: DT_QINT32 } }"
      " input: ['I', 'H'] }");

  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(QUInt8Input);B(QUInt8Input);C(QInt32Input);D(Input);E(Input);"
            "F(Input);G(Input);H(QInt32Input);I(QuantizedMatMulWithBias);"
            "J(Zeta)|A->I;B->I:1;C->I:2;D->I:3;"
            "E->I:4;F->I:5;G->I:6;H->J:1;I->J");
}

// Fused QuantizedMatMulWithBiasAndRelu Op Rewrite test
// Rewrite the QuantizedMatMulWithBiasAndRelu with
// _MklQuantizedMatMulWithBiasAndRelu
TEST_F(MklLayoutPassTest, NodeRewrite_QuantizedMatMulWithBiasAndRelu_Positive) {
  InitGraph(
      "node { name: 'A' op: 'QUInt8Input' }"
      "node { name: 'B' op: 'QInt8Input' }"
      "node { name: 'C' op: 'Input' }"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'Input'}"
      "node { name: 'G' op: 'Input'}"
      "node { name: 'H' op: 'QInt32Input'}"
      "node { name: 'I' op: 'QuantizedMatMulWithBiasAndRelu'"
      " attr { key: 'T1'    value { type: DT_QUINT8 } }"
      " attr { key: 'T2'    value { type: DT_QINT8 } }"
      " attr { key: 'Toutput' value { type: DT_QINT32 } }"
      " input: ['A', 'B', 'C', 'D', 'E', 'F', 'G']}"
      "node { name: 'J' op: 'Zeta' attr { key: 'T' value { type: DT_QINT32 } }"
      " input: ['I', 'H'] }");

  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(QUInt8Input);B(QInt8Input);C(Input);D(Input);"
            "DMT/_0(Const);DMT/_1(Const);DMT/_2(Const);DMT/_3(Const);"
            "DMT/_4(Const);DMT/_5(Const);DMT/_6(Const);E(Input);F(Input);"
            "G(Input);H(QInt32Input);I(_MklQuantizedMatMulWithBiasAndRelu);"
            "J(Zeta)|A->I;"
            "A:control->DMT/_0:control;A:control->DMT/_1:control;"
            "A:control->DMT/_2:control;A:control->DMT/_3:control;"
            "A:control->DMT/_4:control;A:control->DMT/_5:control;"
            "A:control->DMT/_6:control;B->I:1;C->I:2;D->I:3;DMT/_0->I:7;"
            "DMT/_1->I:8;DMT/_2->I:9;DMT/_3->I:10;DMT/_4->I:11;DMT/_5->I:12;"
            "DMT/_6->I:13;E->I:4;F->I:5;G->I:6;H->J:1;I->J");
}

// Rewrite test for QuantizedMatMulWithBiasAndRelu Op with unsupported input
// Rewrite should not happen
TEST_F(MklLayoutPassTest, NodeRewrite_QuantizedMatMulWithBiasAndRelu_Negative) {
  InitGraph(
      "node { name: 'A' op: 'QUInt8Input' }"
      "node { name: 'B' op: 'QUInt8Input' }"
      "node { name: 'C' op: 'Input' }"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'Input'}"
      "node { name: 'G' op: 'Input'}"
      "node { name: 'H' op: 'QInt32Input'}"
      "node { name: 'I' op: 'QuantizedMatMulWithBiasAndRelu'"
      " attr { key: 'T1'    value { type: DT_QUINT8 } }"
      " attr { key: 'T2'    value { type: DT_QUINT8 } }"
      " attr { key: 'Toutput' value { type: DT_QINT32 } }"
      " input: ['A', 'B', 'C', 'D', 'E', 'F', 'G']}"
      "node { name: 'J' op: 'Zeta' attr { key: 'T' value { type: DT_QINT32 } }"
      " input: ['I', 'H'] }");

  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(QUInt8Input);B(QUInt8Input);C(Input);D(Input);"
            "E(Input);F(Input);G(Input);H(QInt32Input);"
            "I(QuantizedMatMulWithBiasAndRelu);J(Zeta)|A->I;"
            "B->I:1;C->I:2;D->I:3;E->I:4;F->I:5;"
            "G->I:6;H->J:1;I->J");
}

// Fused QuantizedMatMulWithBiasAndReluAndRequantize Op Rewrite test
// Rewrite the QuantizedMatMulWithBiasAndReluAndRequantize with
// _MklQuantizedMatMulWithBiasAndReluAndRequantize
TEST_F(MklLayoutPassTest,
       NodeRewrite_QuantizedMatMulWithBiasAndReluAndRequantize_Positive) {
  InitGraph(
      "node { name: 'A' op: 'QUInt8Input' }"
      "node { name: 'B' op: 'QInt8Input' }"
      "node { name: 'C' op: 'QInt32Input' }"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'Input'}"
      "node { name: 'G' op: 'Input'}"
      "node { name: 'H' op: 'Input'}"
      "node { name: 'I' op: 'Input'}"
      "node { name: 'J' op: 'QUInt8Input'}"
      "node { name: 'K' op: 'QuantizedMatMulWithBiasAndReluAndRequantize'"
      " attr { key: 'T1'      value { type: DT_QUINT8 } }"
      " attr { key: 'T2'      value { type: DT_QINT8 } }"
      " attr { key: 'Tbias'   value { type: DT_QINT32 } }"
      " attr { key: 'Toutput' value { type: DT_QUINT8 } }"
      " input: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']}"
      "node { name: 'L' op: 'Zeta' attr { key: 'T' value { type: DT_QUINT8 } }"
      " input: ['K', 'J'] }");

  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(QUInt8Input);B(QInt8Input);C(QInt32Input);"
            "D(Input);DMT/_0(Const);"
            "DMT/_1(Const);DMT/_2(Const);DMT/_3(Const);DMT/_4(Const);"
            "DMT/_5(Const);DMT/_6(Const);DMT/_7(Const);DMT/_8(Const);E(Input);"
            "F(Input);G(Input);H(Input);I(Input);J(QUInt8Input);"
            "K(_MklQuantizedMatMulWithBiasAndReluAndRequantize);L(Zeta)|A->K;"
            "A:control->DMT/_0:control;A:control->DMT/_1:control;"
            "A:control->DMT/_2:control;A:control->DMT/_3:control;"
            "A:control->DMT/_4:control;A:control->DMT/_5:control;"
            "A:control->DMT/_6:control;A:control->DMT/_7:control;"
            "A:control->DMT/_8:control;B->K:1;C->K:2;D->K:3;DMT/_0->K:9;"
            "DMT/_1->K:10;DMT/_2->K:11;DMT/_3->K:12;DMT/_4->K:13;DMT/_5->K:14;"
            "DMT/_6->K:15;DMT/_7->K:16;DMT/_8->K:17;E->K:4;F->K:5;G->K:6;"
            "H->K:7;I->K:8;J->L:1;K->L");
}

// Rewrite test for QuantizedMatMulWithBiasAndRelu Op with unsupported input
// Rewrite should not happen
TEST_F(MklLayoutPassTest,
       NodeRewrite_QuantizedMatMulWithBiasAndReluAndRequantize_Negative) {
  InitGraph(
      "node { name: 'A' op: 'QUInt8Input' }"
      "node { name: 'B' op: 'QUInt8Input' }"
      "node { name: 'C' op: 'QInt32Input' }"
      "node { name: 'D' op: 'Input'}"
      "node { name: 'E' op: 'Input'}"
      "node { name: 'F' op: 'Input'}"
      "node { name: 'G' op: 'Input'}"
      "node { name: 'H' op: 'Input'}"
      "node { name: 'I' op: 'Input'}"
      "node { name: 'J' op: 'QUInt8Input'}"
      "node { name: 'K' op: 'QuantizedMatMulWithBiasAndReluAndRequantize'"
      " attr { key: 'T1'      value { type: DT_QUINT8 } }"
      " attr { key: 'T2'      value { type: DT_QUINT8 } }"
      " attr { key: 'Tbias'   value { type: DT_QINT32 } }"
      " attr { key: 'Toutput' value { type: DT_QUINT8 } }"
      " input: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']}"
      "node { name: 'L' op: 'Zeta' attr { key: 'T' value { type: DT_QUINT8 } }"
      " input: ['K', 'J'] }");

  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(QUInt8Input);B(QUInt8Input);C(QInt32Input);"
            "D(Input);E(Input);F(Input);G(Input);H(Input);I(Input);"
            "J(QUInt8Input);"
            "K(QuantizedMatMulWithBiasAndReluAndRequantize);L(Zeta)|A->K;"
            "B->K:1;C->K:2;D->K:3;E->K:4;F->K:5;G->K:6;"
            "H->K:7;I->K:8;J->L:1;K->L");
}

TEST_F(MklLayoutPassTest, MatMul_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'MatMul'"
      " attr { key: 'T'      value { type: DT_FLOAT } }"
      " input: ['A', 'B']}");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklMatMul)|A->C;B->C:1");
}

TEST_F(MklLayoutPassTest, BatchMatMul_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'BatchMatMul'"
      " attr { key: 'T'      value { type: DT_FLOAT } }"
      " input: ['A', 'B']}");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklBatchMatMul)|A->C;B->C:1");
}

TEST_F(MklLayoutPassTest, BatchMatMulV2_Positive) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'BatchMatMulV2'"
      " attr { key: 'T'      value { type: DT_FLOAT } }"
      " input: ['A', 'B']}");
  EXPECT_EQ(DoMklLayoutOptimizationPass(),
            "A(Input);B(Input);C(_MklBatchMatMulV2)|A->C;B->C:1");
}

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
        "node { name: 'op%04d' op: 'Zeta' attr { key: 'T' value { "
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

#endif  // INTEL_MKL && ENABLE_MKL
