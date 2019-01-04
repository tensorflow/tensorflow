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

#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
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

TEST(DatasetUtilsTest, VariantTensorData_Writer_Reader) {
  VariantTensorData data;

  // Basic test cases.
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(writer.WriteScalar("Int64", 24));
  TF_ASSERT_OK(writer.WriteScalar("", "Empty_Key"));
  Tensor input_tensor(DT_FLOAT, {1});
  input_tensor.flat<float>()(0) = 2.0f;
  TF_ASSERT_OK(writer.WriteTensor("Tensor", input_tensor));
  TF_ASSERT_OK(writer.Flush());

  VariantTensorDataReader reader(&data);
  EXPECT_OK(reader.status());
  int64 val_int64;
  TF_ASSERT_OK(reader.ReadScalar("Int64", &val_int64));
  EXPECT_EQ(val_int64, 24);
  string val_string;
  TF_ASSERT_OK(reader.ReadScalar("", &val_string));
  EXPECT_EQ(val_string, "Empty_Key");
  Tensor val_tensor;
  TF_ASSERT_OK(reader.ReadTensor("Tensor", &val_tensor));
  EXPECT_EQ(input_tensor.NumElements(), val_tensor.NumElements());
  EXPECT_EQ(input_tensor.flat<float>()(0), val_tensor.flat<float>()(0));

  // Test the non-existing key for VariantTensorDataReader.
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadScalar("Non_Existing_Key", &val_int64).code());
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadScalar("Non_Existing_Key", &val_string).code());
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadTensor("Non_Existing_Key", &val_tensor).code());

  // Test the invalid parameter for the constructor of VariantTensorDataReader.
  data.metadata_ = "Invalid Metadata";
  VariantTensorDataReader reader2(&data);
  EXPECT_EQ(error::INTERNAL, reader2.status().code());
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
