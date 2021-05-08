/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_MKL_GRAPH_TESTING_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_MKL_GRAPH_TESTING_UTIL_H_

#ifdef INTEL_MKL

#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/mkl_layout_pass.h"
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
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

namespace {

const char kCPUDevice[] = "/job:a/replica:0/task:0/device:CPU:0";
const char kGPUDevice[] = "/job:a/replica:0/task:0/device:GPU:0";

// Common helper functions that are used in mkl_layout_pass tests.

static void InitGraph(const string& s, Graph* graph,
                      const string& device = kCPUDevice) {
  GraphDef graph_def;

  auto parser = protobuf::TextFormat::Parser();
  EXPECT_TRUE(parser.MergeFromString(s, &graph_def)) << s;
  GraphConstructorOptions opts;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, graph));

  for (Node* node : graph->nodes()) {
    node->set_assigned_device_name(device);
  }
}

class MklLayoutPassTest : public ::testing::Test {
 public:
  MklLayoutPassTest() : graph_(OpRegistry::Global()) {}
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
    T attr_val = T();
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

REGISTER_OP("BFloat16Input").Output("o: bfloat16").SetIsStateful();
REGISTER_OP("BFloat16InputList")
    .Output("o: N * bfloat16")
    .Attr("N: int")
    .SetIsStateful();
REGISTER_OP("BFloat16Output2")
    .Input("i: bfloat16")
    .Input("i1: bfloat16")
    .SetIsStateful();
}  // namespace
}  // namespace tensorflow

#endif  // INTEL_MKL

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_MKL_GRAPH_TESTING_UTIL_H_
