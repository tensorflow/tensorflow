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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

TEST(DatasetUtilsTest, ComputeMoveVector) {
  struct TestCase {
    std::vector<int> indices;
    std::vector<bool> expected;
  };

  TestCase test_cases[] = {
      TestCase{{}, {}},
      TestCase{{1}, {true}},
      TestCase{{1, 1}, {false, true}},
      TestCase{{1, 2}, {true, true}},
      TestCase{{1, 1, 2}, {false, true, true}},
      TestCase{{1, 2, 2}, {true, false, true}},
  };

  for (auto& test_case : test_cases) {
    EXPECT_EQ(test_case.expected, ComputeMoveVector(test_case.indices));
  }
}

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
}  // namespace
}  // namespace data
}  // namespace tensorflow
