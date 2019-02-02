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

}  // namespace
}  // namespace data
}  // namespace tensorflow
