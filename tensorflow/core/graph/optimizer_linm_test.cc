/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/optimizer_linm.h"

#include <utility>
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

class OptimizerLINMTest : public ::testing::Test {
 public:
  OptimizerLINMTest() : graph_(OpRegistry::Global()) {}

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

  string DoLINM() {
    string before = CanonicalGraphString(&graph_);
    LOG(ERROR) << "Before rewrites: " << before;

    OptimizeLINM(&graph_);

    string result = CanonicalGraphString(&graph_);
    LOG(ERROR) << "After rewrites:  " << result;
    return result;
  }

  const string& OriginalGraph() const { return original_; }

  Graph graph_;
  string original_;
};

REGISTER_OP("Input").Output("o: float").SetIsStateful();

// Note that the "rules" in these tests are not meant to be logically correct
TEST_F(OptimizerLINMTest, None) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoLINM(),
            "A(Input);B(Input);C(Mul);D(Mul)|A->C;A->D;B->C:1;B->D:1");
}

TEST_F(OptimizerLINMTest, Enter) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Enter'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'frame_name' value { s: 'while' } }"
      " attr { key: 'is_constant' value { b: true } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'C'] }");
  // A(Input);B(Enter);C(Mul);D(Mul)|A->B;B->C;B->C:1;B->D;C->D:1
  EXPECT_EQ(DoLINM(),
      "A(Input);B(Enter);C(Mul);D(Mul);linm_new_enter_0(Enter)|"
      "A->B;A->C;A->C:1;B->D;C->linm_new_enter_0;linm_new_enter_0->D:1");
}

TEST_F(OptimizerLINMTest, Const) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Enter'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'frame_name' value { s: 'while' } }"
      " attr { key: 'is_constant' value { b: true } }"
      " input: ['A'] }"
      "node { name: 'C' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } }"
      " input: ['^B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'C'] }"
      "node { name: 'E' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'D'] }");
  // A(Input);B(Enter);C(Const);D(Mul);E(Mul)|
  // A->B;B->D;B:control->C:control;C->D:1;C->E;D->E:1
  EXPECT_EQ(DoLINM(),
      "A(Input);B(Enter);C(Const);D(Mul);E(Mul);"
      "linm_new_const_0(Const);linm_new_enter_1(Enter)|"
      "A->B;A->D;B:control->C:control;C->E;D->linm_new_enter_1;"
      "linm_new_const_0->D:1;linm_new_enter_1->E:1");
}

}  // namespace
}  // namespace tensorflow
