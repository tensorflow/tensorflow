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

#include "tensorflow/compiler/jit/graph_to_functiondef.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {
namespace {

bool EqualFunctionDef(const FunctionDef& a, const FunctionDef& b,
                      string* diff) {
  // TODO(phawkins) use a more sophisticated equality test.
  if (a.DebugString() != b.DebugString()) {
    if (diff) {
      *diff = strings::StrCat("Definition mismatch for function ",
                              a.signature().name(), ":\n", a.DebugString(),
                              "\n ---- vs. ----\n", b.DebugString());
    }
    return false;
  }
  return true;
}

TEST(GraphToFunctionDefTest, Basics) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(root.WithOpName("A"), DT_FLOAT, 0);
  auto b = ops::_Arg(root.WithOpName("B"), DT_FLOAT, 1);
  auto c = ops::_Arg(root.WithOpName("C"), DT_FLOAT, 2);
  auto d = ops::Add(root.WithOpName("D"), a, b);
  auto e = ops::Add(root.WithOpName("b"), d, c);
  auto f = ops::Neg(root.WithOpName("h"), e);
  auto g = ops::AddN(root.WithOpName("G"), std::initializer_list<Output>{e, f});
  auto h = ops::_Retval(root.WithOpName("H"), g, 0);

  GraphDef graph_def;
  TF_EXPECT_OK(root.ToGraphDef(&graph_def));

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphConstructorOptions options;
  TF_EXPECT_OK(ConvertGraphDefToGraph(options, graph_def, graph.get()));

  FunctionDef fdef;
  TF_EXPECT_OK(GraphToFunctionDef(*graph, "test_fn", &fdef));

  FunctionDef fdef_expected = FunctionDefHelper::Create(
      "test_fn",                             // function name
      {"a: float", "b: float", "c: float"},  // inputs
      {"h_0: float"},                        // outputs
      {},                                    // attrs
      {
          // nodes in the function body
          {{"D"}, "Add", {"a", "b"}, {{"T", DT_FLOAT}}},
          {{"b_0"}, "Add", {"D:z:0", "c"}, {{"T", DT_FLOAT}}},
          {{"h"}, "Neg", {"b_0:z:0"}, {{"T", DT_FLOAT}}},
          {{"G"}, "AddN", {"b_0:z:0", "h:y:0"}, {{"N", 2}, {"T", DT_FLOAT}}},
      },
      {{"h_0", "G:sum:0"}});  // return values

  string diff;
  bool fdefs_equal = EqualFunctionDef(fdef_expected, fdef, &diff);
  EXPECT_TRUE(fdefs_equal) << diff;
}

}  // namespace
}  // namespace tensorflow
