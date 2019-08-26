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

#include "tensorflow/core/kernels/data/rewrite_utils.h"

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace data {
namespace {

TEST(DatasetUtilsTest, HashSubgraphFunctionSameFunctionDifferentNames) {
  FunctionDefLibrary fl1;

  FunctionDef* f1 = fl1.add_function();
  *f1 = FunctionDefHelper::Create(
      "AddAndMul", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  FunctionDef* f2 = fl1.add_function();
  *f2 = FunctionDefHelper::Create(
      "AddAndMul2", {"input: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"input", "input"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"input", "input"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  EXPECT_EQ(HashSubgraphFunction(fl1, f1), HashSubgraphFunction(fl1, f2));
}

TEST(DatasetUtilsTest, HashSubgraphFunctionDifferentFunctions) {
  FunctionDefLibrary fl1;

  FunctionDef* f1 = fl1.add_function();
  *f1 = FunctionDefHelper::Create(
      "AddAndMul", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  FunctionDef* f2 = fl1.add_function();
  *f2 = FunctionDefHelper::Create(
      "AddAndAdd", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  // The second op in `f2` is changed to "Add"
  EXPECT_NE(HashSubgraphFunction(fl1, f1), HashSubgraphFunction(fl1, f2));
}

TEST(DatasetUtilsTest, HashSubgraphFunctionDifferentInternalNodeNames) {
  FunctionDefLibrary fl1;

  FunctionDef* f1 = fl1.add_function();
  *f1 = FunctionDefHelper::Create(
      "AddAndMul", {"i: float", "j: float", "k: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "j"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"add", "k"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "ret"}});

  FunctionDef* f2 = fl1.add_function();
  *f2 = FunctionDefHelper::Create(
      "AddAndMul", {"a: float", "b: float", "c: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"a", "b"}, {{"T", DT_FLOAT}}},
       {{"mul"}, "Mul", {"add", "c"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "mul:z:0"}},
      /*control_ret_def=*/{{"must_execute", "mul"}});

  EXPECT_EQ(HashSubgraphFunction(fl1, f1), HashSubgraphFunction(fl1, f2));
}

TEST(DatasetUtilsTest, HashSubgraphSameGraphDifferentNames) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n3));

  uint64 hash1 = HashSubgraph(gd, n3);

  n1->Clear();
  TF_CHECK_OK(NodeDefBuilder("graph_3/node_7", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  n2->Clear();
  TF_CHECK_OK(NodeDefBuilder("graph_4/node_9", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n2));

  n3->Clear();
  TF_CHECK_OK(NodeDefBuilder("graph_5/node_11", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n3));

  uint64 hash2 = HashSubgraph(gd, n3);

  EXPECT_EQ(hash1, hash2);
}

TEST(DatasetUtilsTest, HashSubgraphDifferentGraphs) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n3));

  uint64 hash1 = HashSubgraph(gd, n3);

  n3->Clear();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Mul")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n3));

  uint64 hash2 = HashSubgraph(gd, n3);

  // We expect different hashes because the op of n3 has changed.
  EXPECT_NE(hash1, hash2);
}

TEST(DatasetUtilsTest, HashSubgraphReversedOrder) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n3));

  uint64 hash1 = HashSubgraph(gd, n3);

  n3->Clear();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n2->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Finalize(n3));

  uint64 hash2 = HashSubgraph(gd, n3);

  // We expect different hashes because the inputs of n3 are swapped.
  EXPECT_NE(hash1, hash2);
}

TEST(DatasetUtilsTest, HashSubgraphInputPortChanged) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n3));

  uint64 hash1 = HashSubgraph(gd, n3);

  n3->Clear();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 1, DT_INT32)
                  .Input(n2->name(), 2, DT_INT32)
                  .Finalize(n3));

  uint64 hash2 = HashSubgraph(gd, n3);

  // We expect different hashes because the input ports for nodes used by n3
  // has changed.
  EXPECT_NE(hash1, hash2);
}

TEST(DatasetUtilsTest, HashSubgraphSameFunctionDifferentNames) {
  GraphDef gd;
  FunctionDefLibrary* fl1 = gd.mutable_library();

  FunctionDef* f1 = fl1->add_function();
  *f1 = FunctionDefHelper::Create(
      "AddAndMul", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  FunctionDef* f2 = fl1->add_function();
  *f2 = FunctionDefHelper::Create(
      "AddAndMul2", {"input: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"input", "input"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"input", "input"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  AttrValue a1;
  NameAttrList* nal1 = a1.mutable_func();
  nal1->set_name("AddAndMul");

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  std::vector<NodeDefBuilder::NodeOut> func_inputs;
  func_inputs.emplace_back(n1->name(), 0, DT_FLOAT);
  func_inputs.emplace_back(n1->name(), 0, DT_FLOAT);

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "For")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(func_inputs)
                  .Attr("body", a1)
                  .Device("CPU:0")
                  .Finalize(n2));

  uint64 hash1 = HashSubgraph(gd, n2);

  n2->Clear();
  AttrValue a2;
  NameAttrList* nal2 = a2.mutable_func();
  nal2->set_name("AddAndMul2");

  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "For")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(func_inputs)
                  .Attr("body", a2)
                  .Device("CPU:0")
                  .Finalize(n2));

  uint64 hash2 = HashSubgraph(gd, n2);

  EXPECT_EQ(hash1, hash2);
}

TEST(DatasetUtilsTest, HashSubgraphDifferentFunctions) {
  GraphDef gd;

  FunctionDefLibrary* fl1 = gd.mutable_library();
  FunctionDef* f1 = fl1->add_function();

  FunctionDef func = FunctionDefHelper::Create(
      "AddAndMul", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});
  *f1 = func;

  FunctionDef* f2 = fl1->add_function();
  func = FunctionDefHelper::Create(
      "AddAndMul2", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "ret"}});
  *f2 = func;

  AttrValue a1;
  NameAttrList* nal1 = a1.mutable_func();
  nal1->set_name("AddAndMul");

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  std::vector<NodeDefBuilder::NodeOut> func_inputs;
  func_inputs.emplace_back(n1->name(), 0, DT_FLOAT);
  func_inputs.emplace_back(n1->name(), 0, DT_FLOAT);

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "For")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(func_inputs)
                  .Attr("body", a1)
                  .Device("CPU:0")
                  .Finalize(n2));

  uint64 hash1 = HashSubgraph(gd, n2);

  n2->Clear();
  AttrValue a2;
  NameAttrList* nal2 = a2.mutable_func();
  nal2->set_name("AddAndMul2");

  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "For")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(func_inputs)
                  .Attr("body", a2)
                  .Device("CPU:0")
                  .Finalize(n2));

  uint64 hash2 = HashSubgraph(gd, n2);

  EXPECT_NE(hash1, hash2);
}

TEST(DatasetUtilsTest, HashSubgraphDifferentControlInputs) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Const")
                  .Attr("value", 10)
                  .Device("CPU:0")
                  .Finalize(n3));

  NodeDef* n4 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_4", "Identity")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .ControlInput(n2->name())
                  .Finalize(n4));

  uint64 hash1 = HashSubgraph(gd, n4);

  n4->Clear();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_4", "Identity")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .ControlInput(n3->name())
                  .Finalize(n4));

  uint64 hash2 = HashSubgraph(gd, n4);

  // Control inputs are different between these two graphs.
  EXPECT_NE(hash1, hash2);
}

TEST(DatasetUtilsTest, HashSubgraphControlInputDifferentOrdering) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Const")
                  .Attr("value", 10)
                  .Device("CPU:0")
                  .Finalize(n3));

  NodeDef* n4 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_4", "Identity")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .ControlInput(n2->name())
                  .ControlInput(n3->name())
                  .Finalize(n4));

  uint64 hash1 = HashSubgraph(gd, n4);

  n4->Clear();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_4", "Identity")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .ControlInput(n3->name())
                  .ControlInput(n2->name())
                  .Finalize(n4));

  uint64 hash2 = HashSubgraph(gd, n4);

  EXPECT_EQ(hash1, hash2);
}

TEST(DatasetUtilsTest, HashSubgraphDifferentGraphSamePartialGraph) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();

  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n3));

  uint64 hash1 = HashSubgraph(gd, n1);

  n3->Clear();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Mul")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n3));

  uint64 hash2 = HashSubgraph(gd, n1);

  EXPECT_EQ(hash1, hash2);
}

TEST(DatasetUtilsTest, HashSubgraphWithManyControlDependencies) {
  GraphDef gd;
  NodeDef* n;

  for (int i = 0; i < 1000; ++i) {
    n = gd.add_node();
    NodeDefBuilder ndb(absl::StrCat("graph_1/node_", i), "Const");
    ndb.Attr("value", 1);
    ndb.Device("CPU:0");
    for (int j = 0; j < i; ++j) {
      ndb.ControlInput(absl::StrCat("graph_1/node_", j));
    }
    TF_CHECK_OK(ndb.Finalize(n));
  }

  // No checks here, because so long as this does not time out, we are OK.
  HashSubgraph(gd, n);
}

TEST(DatasetUtilsTest, HashSubgraphFunctionsWithControlDependencyLoop) {
  GraphDef gd;

  FunctionDefLibrary* fl1 = gd.mutable_library();
  FunctionDef* f1 = fl1->add_function();

  AttrValue a1;
  NameAttrList* nal1 = a1.mutable_func();
  nal1->set_name("AddAndMul");

  std::pair<string, FunctionDefHelper::AttrValueWrapper> func_attr = {
      "body", FunctionDefHelper::AttrValueWrapper(*nal1)};

  FunctionDef func = FunctionDefHelper::Create(
      /*function_name=*/"AddAndMul",
      /*in_def=*/{"i: float"},
      /*out_def=*/{"o: float"},
      /*attr_def=*/{},
      /*node_def=*/
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}, {"ret"}},
       // This creates a dependency on the same function.
       {{"for"}, "For", {"i", "i", "i"}, {func_attr}, {"ret"}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "for:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});
  *f1 = func;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  std::vector<NodeDefBuilder::NodeOut> func_inputs;
  func_inputs.emplace_back(n1->name(), 0, DT_FLOAT);
  func_inputs.emplace_back(n1->name(), 0, DT_FLOAT);

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "For")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(func_inputs)
                  .ControlInput("graph_1/node_2")
                  .Attr("body", a1)
                  .Device("CPU:0")
                  .Finalize(n2));

  // No checks in the test, the fact that it runs and doesn't timeout or exhaust
  // the stack means it is successful.
  HashSubgraph(gd, n2);
}

TEST(DatasetUtilsTest, HashSubgraphWithControlDependencyLoop) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .ControlInput("graph_1/node_2")
                  .Finalize(n1));

  NodeDef* n2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .ControlInput("graph_1/node_1")
                  .Finalize(n2));

  NodeDef* n3 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .ControlInput("graph_1/node_1")
                  .ControlInput("graph_1/node_2")
                  .Finalize(n3));

  // No checks in the test, the fact that it runs and doesn't timeout or exhaust
  // the stack means it is successful.
  HashSubgraph(gd, n3);
}

TEST(DatasetUtilsTest, HashSubgraphWithControlDependencyLoopDifferentNames) {
  GraphDef gd1;

  NodeDef* n1 = gd1.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .ControlInput("graph_1/node_2")
                  .Finalize(n1));

  NodeDef* n2 = gd1.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_2", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .ControlInput("graph_1/node_1")
                  .Finalize(n2));

  NodeDef* n3 = gd1.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .ControlInput("graph_1/node_1")
                  .ControlInput("graph_1/node_2")
                  .Finalize(n3));

  GraphDef gd2;

  NodeDef* n4 = gd2.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_4", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .ControlInput("graph_1/node_5")
                  .Finalize(n4));

  NodeDef* n5 = gd2.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_5", "Const")
                  .Attr("value", 2)
                  .Device("CPU:0")
                  .ControlInput("graph_1/node_4")
                  .Finalize(n5));

  NodeDef* n6 = gd2.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_6", "Add")
                  .Device("CPU:0")
                  .Input(n4->name(), 0, DT_INT32)
                  .Input(n5->name(), 0, DT_INT32)
                  .ControlInput("graph_1/node_4")
                  .ControlInput("graph_1/node_5")
                  .Finalize(n6));

  EXPECT_EQ(HashSubgraph(gd1, n3), HashSubgraph(gd2, n6));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
