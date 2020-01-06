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
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace data {
namespace {

class DatasetHashUtilsTest : public ::testing::Test {
 protected:
  uint64 GetHash(const FunctionDefLibrary& library, const FunctionDef& fn) {
    uint64 hash = 0;
    TF_CHECK_OK(HashFunction(library, fn, &hash));
    return hash;
  }

  uint64 GetHash(const GraphDef& graph, const NodeDef& node) {
    uint64 hash = 0;
    TF_CHECK_OK(HashNode(graph, node, &hash));
    return hash;
  }

  uint64 GetHash(const Tensor& tensor) {
    uint64 hash = 0;
    TF_CHECK_OK(HashTensor(tensor, &hash));
    return hash;
  }
};

string full_name(string key) {
  return strings::StrCat(kFullNameRandomHex, kPipe, "Iterator:", key);
}

TEST(DatasetUtilsTest, VariantTensorDataRoundtrip) {
  VariantTensorDataWriter writer;
  TF_ASSERT_OK(writer.WriteScalar(full_name("Int64"), 24));
  Tensor input_tensor(DT_FLOAT, {1});
  input_tensor.flat<float>()(0) = 2.0f;
  TF_ASSERT_OK(writer.WriteTensor(full_name("Tensor"), input_tensor));
  std::vector<const VariantTensorData*> data;
  writer.GetData(&data);

  VariantTensorDataReader reader(data);
  int64 val_int64;
  TF_ASSERT_OK(reader.ReadScalar(full_name("Int64"), &val_int64));
  EXPECT_EQ(val_int64, 24);
  Tensor val_tensor;
  TF_ASSERT_OK(reader.ReadTensor(full_name("Tensor"), &val_tensor));
  EXPECT_EQ(input_tensor.NumElements(), val_tensor.NumElements());
  EXPECT_EQ(input_tensor.flat<float>()(0), val_tensor.flat<float>()(0));
}

TEST(DatasetUtilsTest, VariantTensorDataNonExistentKey) {
  VariantTensorData data;
  strings::StrAppend(&data.metadata_, "key1", "@@");
  data.tensors_.push_back(Tensor(DT_INT64, {1}));
  std::vector<const VariantTensorData*> reader_data;
  reader_data.push_back(&data);
  VariantTensorDataReader reader(reader_data);
  int64 val_int64;
  tstring val_string;
  Tensor val_tensor;
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadScalar(full_name("NonExistentKey"), &val_int64).code());
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadScalar(full_name("NonExistentKey"), &val_string).code());
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadTensor(full_name("NonExistentKey"), &val_tensor).code());
}

TEST(DatasetUtilsTest, VariantTensorDataRoundtripIteratorName) {
  VariantTensorDataWriter writer;
  TF_ASSERT_OK(writer.WriteScalar("Iterator", "Int64", 24));
  Tensor input_tensor(DT_FLOAT, {1});
  input_tensor.flat<float>()(0) = 2.0f;
  TF_ASSERT_OK(writer.WriteTensor("Iterator", "Tensor", input_tensor));
  std::vector<const VariantTensorData*> data;
  writer.GetData(&data);

  VariantTensorDataReader reader(data);
  int64 val_int64;
  TF_ASSERT_OK(reader.ReadScalar("Iterator", "Int64", &val_int64));
  EXPECT_EQ(val_int64, 24);
  Tensor val_tensor;
  TF_ASSERT_OK(reader.ReadTensor("Iterator", "Tensor", &val_tensor));
  EXPECT_EQ(input_tensor.NumElements(), val_tensor.NumElements());
  EXPECT_EQ(input_tensor.flat<float>()(0), val_tensor.flat<float>()(0));
}

TEST(DatasetUtilsTest, VariantTensorDataNonExistentKeyIteratorName) {
  VariantTensorData data;
  strings::StrAppend(&data.metadata_, "key1", "@@");
  data.tensors_.push_back(Tensor(DT_INT64, {1}));
  std::vector<const VariantTensorData*> reader_data;
  reader_data.push_back(&data);
  VariantTensorDataReader reader(reader_data);
  int64 val_int64;
  tstring val_string;
  Tensor val_tensor;
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadScalar("Iterator", "NonExistentKey", &val_int64).code());
  EXPECT_EQ(
      error::NOT_FOUND,
      reader.ReadScalar("Iterator", "NonExistentKey", &val_string).code());
  EXPECT_EQ(
      error::NOT_FOUND,
      reader.ReadTensor("Iterator", "NonExistentKey", &val_tensor).code());
}

TEST(DatasetUtilsTest, VariantTensorDataWriteAfterFlushing) {
  VariantTensorDataWriter writer;
  TF_ASSERT_OK(writer.WriteScalar(full_name("Int64"), 24));
  std::vector<const VariantTensorData*> data;
  writer.GetData(&data);
  Tensor input_tensor(DT_FLOAT, {1});
  input_tensor.flat<float>()(0) = 2.0f;
  EXPECT_EQ(error::FAILED_PRECONDITION,
            writer.WriteTensor(full_name("Tensor"), input_tensor).code());
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

TEST_F(DatasetHashUtilsTest, HashFunctionSameFunctionDifferentNames) {
  FunctionDefLibrary fl;

  FunctionDef* f1 = fl.add_function();
  *f1 = FunctionDefHelper::Create(
      "AddAndMul", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  FunctionDef* f2 = fl.add_function();
  *f2 = FunctionDefHelper::Create(
      "AddAndMul2", {"input: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"input", "input"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"input", "input"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  EXPECT_EQ(GetHash(fl, *f1), GetHash(fl, *f2));
}

TEST_F(DatasetHashUtilsTest, HashFunctionDifferentFunctions) {
  FunctionDefLibrary fl;

  FunctionDef* f1 = fl.add_function();
  *f1 = FunctionDefHelper::Create(
      "AddAndMul", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  FunctionDef* f2 = fl.add_function();
  *f2 = FunctionDefHelper::Create(
      "AddAndAdd", {"i: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Add", {"i", "i"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  // The second op in `f2` is changed to "Add"
  EXPECT_NE(GetHash(fl, *f1), GetHash(fl, *f2));
}

TEST_F(DatasetHashUtilsTest, HashFunctionDifferentInternalNodeNames) {
  FunctionDefLibrary fl;

  FunctionDef* f1 = fl.add_function();
  *f1 = FunctionDefHelper::Create(
      "AddAndMul", {"i: float", "j: float", "k: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"i", "j"}, {{"T", DT_FLOAT}}},
       {{"ret"}, "Mul", {"add", "k"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "ret"}});

  FunctionDef* f2 = fl.add_function();
  *f2 = FunctionDefHelper::Create(
      "AddAndMul", {"a: float", "b: float", "c: float"}, {"o: float"}, {},
      {{{"add"}, "Add", {"a", "b"}, {{"T", DT_FLOAT}}},
       {{"mul"}, "Mul", {"add", "c"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/{{"o", "mul:z:0"}},
      /*control_ret_def=*/{{"must_execute", "mul"}});

  EXPECT_EQ(GetHash(fl, *f1), GetHash(fl, *f2));
}

TEST_F(DatasetHashUtilsTest, HashNodeSameGraphDifferentNames) {
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

  uint64 hash1 = GetHash(gd, *n3);

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

  uint64 hash2 = GetHash(gd, *n3);

  EXPECT_EQ(hash1, hash2);
}

TEST_F(DatasetHashUtilsTest, HashNodeDifferentGraphs) {
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

  uint64 hash1 = GetHash(gd, *n3);

  n3->Clear();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Mul")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n3));

  uint64 hash2 = GetHash(gd, *n3);

  // We expect different hashes because the op of n3 has changed.
  EXPECT_NE(hash1, hash2);
}

TEST_F(DatasetHashUtilsTest, HashSameGraphDifferentSeeds) {
  GraphDef gd;

  NodeDef* n1 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_1", "Const")
                  .Attr("value", 1)
                  .Device("CPU:0")
                  .Finalize(n1));

  NodeDef* seed = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/seed", "Const")
                  .Attr("value", 123)
                  .Device("CPU:0")
                  .Finalize(seed));

  NodeDef* seed2 = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/seed2", "Const")
                  .Attr("value", 456)
                  .Device("CPU:0")
                  .Finalize(seed2));

  NodeDef* range_ds = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/range", "RangeDataset")
                  .Input(n1->name(), 0, DT_INT64)
                  .Input(n1->name(), 0, DT_INT64)
                  .Input(n1->name(), 0, DT_INT64)
                  .Device("CPU:0")
                  .Finalize(range_ds));

  NodeDef* shuffle_ds = gd.add_node();
  TF_CHECK_OK(NodeDefBuilder("graph_1/shuffle", "ShuffleDataset")
                  .Input(range_ds->name(), 0, DT_VARIANT)
                  .Input(n1->name(), 0, DT_INT64)
                  .Input(seed->name(), 0, DT_INT64)
                  .Input(seed2->name(), 0, DT_INT64)
                  .Device("CPU:0")
                  .Finalize(shuffle_ds));

  uint64 hash1 = GetHash(gd, *shuffle_ds);

  seed->Clear();
  seed2->Clear();

  TF_CHECK_OK(NodeDefBuilder("graph_1/seed", "Const")
                  .Attr("value", 789)
                  .Device("CPU:0")
                  .Finalize(seed));
  TF_CHECK_OK(NodeDefBuilder("graph_1/seed2", "Const")
                  .Attr("value", 654)
                  .Device("CPU:0")
                  .Finalize(seed2));

  uint64 hash2 = GetHash(gd, *shuffle_ds);

  EXPECT_EQ(hash1, hash2);
}

TEST_F(DatasetHashUtilsTest, HashNodeReversedOrder) {
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

  uint64 hash1 = GetHash(gd, *n3);

  n3->Clear();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n2->name(), 0, DT_INT32)
                  .Input(n1->name(), 0, DT_INT32)
                  .Finalize(n3));

  uint64 hash2 = GetHash(gd, *n3);

  // We expect different hashes because the inputs of n3 are swapped.
  EXPECT_NE(hash1, hash2);
}

TEST_F(DatasetHashUtilsTest, HashNodeInputPortChanged) {
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

  uint64 hash1 = GetHash(gd, *n3);

  n3->Clear();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Add")
                  .Device("CPU:0")
                  .Input(n1->name(), 1, DT_INT32)
                  .Input(n2->name(), 2, DT_INT32)
                  .Finalize(n3));

  uint64 hash2 = GetHash(gd, *n3);

  // We expect different hashes because the input ports for nodes used by n3
  // has changed.
  EXPECT_NE(hash1, hash2);
}

TEST_F(DatasetHashUtilsTest, HashNodeSameFunctionDifferentNames) {
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

  uint64 hash1 = GetHash(gd, *n2);

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

  uint64 hash2 = GetHash(gd, *n2);

  EXPECT_EQ(hash1, hash2);
}

TEST_F(DatasetHashUtilsTest, HashNodeDifferentFunctions) {
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

  uint64 hash1 = GetHash(gd, *n2);

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

  uint64 hash2 = GetHash(gd, *n2);

  EXPECT_NE(hash1, hash2);
}

TEST_F(DatasetHashUtilsTest, HashNodeDifferentControlInputs) {
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

  uint64 hash1 = GetHash(gd, *n4);

  n4->Clear();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_4", "Identity")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .ControlInput(n3->name())
                  .Finalize(n4));

  uint64 hash2 = GetHash(gd, *n4);

  // Control inputs are different between these two graphs.
  EXPECT_NE(hash1, hash2);
}

TEST_F(DatasetHashUtilsTest, HashNodeControlInputDifferentOrdering) {
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

  uint64 hash1 = GetHash(gd, *n4);

  n4->Clear();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_4", "Identity")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .ControlInput(n3->name())
                  .ControlInput(n2->name())
                  .Finalize(n4));

  uint64 hash2 = GetHash(gd, *n4);

  EXPECT_EQ(hash1, hash2);
}

TEST_F(DatasetHashUtilsTest, HashNodeDifferentGraphSamePartialGraph) {
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

  uint64 hash1 = GetHash(gd, *n1);

  n3->Clear();
  TF_CHECK_OK(NodeDefBuilder("graph_1/node_3", "Mul")
                  .Device("CPU:0")
                  .Input(n1->name(), 0, DT_INT32)
                  .Input(n2->name(), 0, DT_INT32)
                  .Finalize(n3));

  uint64 hash2 = GetHash(gd, *n1);

  EXPECT_EQ(hash1, hash2);
}

TEST_F(DatasetHashUtilsTest, HashNodeWithManyControlDependencies) {
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
  GetHash(gd, *n);
}

TEST_F(DatasetHashUtilsTest, HashFunctionsWithControlDependencyLoop) {
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
  GetHash(gd, *n2);
}

TEST_F(DatasetHashUtilsTest, HashNodeWithControlDependencyLoop) {
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
  GetHash(gd, *n3);
}

TEST_F(DatasetHashUtilsTest, HashNodeWithControlDependencyLoopDifferentNames) {
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

  EXPECT_EQ(GetHash(gd1, *n3), GetHash(gd2, *n6));
}

TEST_F(DatasetHashUtilsTest, HashInt32Tensor) {
  Tensor s1(42);
  Tensor s2(42);
  Tensor s3(43);

  EXPECT_EQ(GetHash(s1), GetHash(s2));
  EXPECT_NE(GetHash(s1), GetHash(s3));

  Tensor v1(DT_INT32, TensorShape({2}));
  v1.vec<int32>()(0) = 0;
  v1.vec<int32>()(1) = 1;
  Tensor v2(DT_INT32, TensorShape({2}));
  v2.vec<int32>()(0) = 0;
  v2.vec<int32>()(1) = 1;
  Tensor v3(DT_INT32, TensorShape({2}));
  v3.vec<int32>()(0) = 0;
  v3.vec<int32>()(1) = 2;

  EXPECT_EQ(GetHash(v1), GetHash(v2));
  EXPECT_NE(GetHash(v1), GetHash(v3));
}

TEST_F(DatasetHashUtilsTest, HashStringTensor) {
  Tensor s1("hello");
  Tensor s2("hello");
  Tensor s3("world");

  EXPECT_EQ(GetHash(s1), GetHash(s2));
  EXPECT_NE(GetHash(s1), GetHash(s3));

  Tensor v1(DT_STRING, TensorShape({2}));
  v1.vec<tstring>()(0) = "hello";
  v1.vec<tstring>()(1) = "world";
  Tensor v2(DT_STRING, TensorShape({2}));
  v2.vec<tstring>()(0) = "hello";
  v2.vec<tstring>()(1) = "world";
  Tensor v3(DT_STRING, TensorShape({2}));
  v3.vec<tstring>()(0) = "hello";
  v3.vec<tstring>()(1) = "universe";

  EXPECT_EQ(GetHash(v1), GetHash(v2));
  EXPECT_NE(GetHash(v1), GetHash(v3));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
