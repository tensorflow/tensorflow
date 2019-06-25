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

#include "tensorflow/core/kernels/data/dataset_utils.h"

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

TEST(DatasetUtilsTest, VariantTensorDataRoundtrip) {
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(writer.WriteScalar("Int64", 24));
  Tensor input_tensor(DT_FLOAT, {1});
  input_tensor.flat<float>()(0) = 2.0f;
  TF_ASSERT_OK(writer.WriteTensor("Tensor", input_tensor));
  TF_ASSERT_OK(writer.Flush());

  VariantTensorDataReader reader(&data);
  int64 val_int64;
  TF_ASSERT_OK(reader.ReadScalar("Int64", &val_int64));
  EXPECT_EQ(val_int64, 24);
  Tensor val_tensor;
  TF_ASSERT_OK(reader.ReadTensor("Tensor", &val_tensor));
  EXPECT_EQ(input_tensor.NumElements(), val_tensor.NumElements());
  EXPECT_EQ(input_tensor.flat<float>()(0), val_tensor.flat<float>()(0));
}

TEST(DatasetUtilsTest, VariantTensorDataNonExistentKey) {
  VariantTensorData data;
  strings::StrAppend(&data.metadata_, "key1", "@@");
  data.tensors_.push_back(Tensor(DT_INT64, {1}));
  VariantTensorDataReader reader(&data);
  int64 val_int64;
  string val_string;
  Tensor val_tensor;
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadScalar("NonExistentKey", &val_int64).code());
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadScalar("NonExistentKey", &val_string).code());
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadTensor("NonExistentKey", &val_tensor).code());
}

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

TEST(DatasetUtilsTest, AddToFunctionLibrary) {
  auto make_fn_a = [](const string& fn_name) {
    return FunctionDefHelper::Create(
        /*function_name=*/fn_name,
        /*in_def=*/{"arg: int64"},
        /*out_def=*/{"ret: int64"},
        /*attr_def=*/{},
        /*node_def=*/{{{"node"}, "Identity", {"arg"}, {{"T", DT_INT64}}}},
        /*ret_def=*/{{"ret", "node:output:0"}});
  };

  auto make_fn_b = [](const string& fn_name) {
    return FunctionDefHelper::Create(
        /*function_name=*/fn_name,
        /*in_def=*/{"arg: int64"},
        /*out_def=*/{"ret: int64"},
        /*attr_def=*/{},
        /*node_def=*/
        {{{"node"}, "Identity", {"arg"}, {{"T", DT_INT64}}},
         {{"node2"}, "Identity", {"node:output:0"}, {{"T", DT_INT64}}}},
        /*ret_def=*/{{"ret", "node2:output:0"}});
  };

  FunctionDefLibrary fdef_base;
  *fdef_base.add_function() = make_fn_a("0");
  *fdef_base.add_function() = make_fn_a("1");
  *fdef_base.add_function() = make_fn_a("2");

  FunctionDefLibrary fdef_to_add;
  *fdef_to_add.add_function() = make_fn_b("0");  // Override
  *fdef_to_add.add_function() = make_fn_a("1");  // Do nothing
  *fdef_to_add.add_function() = make_fn_b("3");  // Add new function

  FunctionLibraryDefinition flib_0(OpRegistry::Global(), fdef_base);
  TF_ASSERT_OK(AddToFunctionLibrary(&flib_0, fdef_to_add));

  FunctionLibraryDefinition flib_1(OpRegistry::Global(), fdef_base);
  FunctionLibraryDefinition flib_to_add(OpRegistry::Global(), fdef_to_add);
  TF_ASSERT_OK(AddToFunctionLibrary(&flib_1, flib_to_add));

  for (const auto& flib : {flib_0, flib_1}) {
    EXPECT_TRUE(FunctionDefsEqual(*flib.Find("0"), make_fn_b("0")));
    EXPECT_TRUE(FunctionDefsEqual(*flib.Find("1"), make_fn_a("1")));
    EXPECT_TRUE(FunctionDefsEqual(*flib.Find("2"), make_fn_a("2")));
    EXPECT_TRUE(FunctionDefsEqual(*flib.Find("3"), make_fn_b("3")));
  }
}

TEST(DatasetUtilsTest, AddToFunctionLibraryWithConflictingSignatures) {
  FunctionDefLibrary fdef_base;
  *fdef_base.add_function() = FunctionDefHelper::Create(
      /*function_name=*/"0",
      /*in_def=*/{"arg: int64"},
      /*out_def=*/{"ret: int64"},
      /*attr_def=*/{},
      /*node_def=*/{},
      /*ret_def=*/{{"ret", "arg"}});

  FunctionDefLibrary fdef_to_add;
  *fdef_to_add.add_function() = FunctionDefHelper::Create(
      /*function_name=*/"0",
      /*in_def=*/{"arg: int64"},
      /*out_def=*/{"ret: int64", "ret2: int64"},
      /*attr_def=*/{},
      /*node_def=*/{},
      /*ret_def=*/{{"ret", "arg"}, {"ret2", "arg"}});

  FunctionLibraryDefinition flib_0(OpRegistry::Global(), fdef_base);
  Status s = AddToFunctionLibrary(&flib_0, fdef_to_add);
  EXPECT_EQ(error::Code::INVALID_ARGUMENT, s.code());
  EXPECT_EQ(
      "Cannot add function '0' because a different function with the same "
      "signature already exists.",
      s.error_message());

  FunctionLibraryDefinition flib_1(OpRegistry::Global(), fdef_base);
  FunctionLibraryDefinition flib_to_add(OpRegistry::Global(), fdef_to_add);
  s = AddToFunctionLibrary(&flib_1, flib_to_add);
  EXPECT_EQ(error::Code::INVALID_ARGUMENT, s.code());
  EXPECT_EQ(
      "Cannot add function '0' because a different function with the same "
      "signature already exists.",
      s.error_message());
}

TEST(DatasetUtilsTest, RunnerWithMaxParallelism) {
  auto runner =
      RunnerWithMaxParallelism([](const std::function<void()> fn) { fn(); }, 2);
  auto fn = []() { ASSERT_EQ(GetPerThreadMaxParallelism(), 2); };
  runner(fn);
}
}  // namespace
}  // namespace data
}  // namespace tensorflow
