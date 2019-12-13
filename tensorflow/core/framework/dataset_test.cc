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

#include "tensorflow/core/framework/dataset.h"

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

REGISTER_DATASET_OP_NAME("DummyDatasetOp");

TEST(DatasetTest, RegisterDatasetOp) {
  EXPECT_TRUE(data::DatasetOpRegistry::IsRegistered("DummyDatasetOp"));
  EXPECT_FALSE(data::DatasetOpRegistry::IsRegistered("InvalidDatasetOp"));
}

enum DataTypeTest { _int32, _int64, _float, _double, _string };

struct DatasetTestParam {
  const DataTypeTest type;
  const std::vector<Tensor> tensor;
  const int64 expected_bytes;
};

class DatasetTestTotalBytes
    : public ::testing::TestWithParam<DatasetTestParam> {};

TEST_P(DatasetTestTotalBytes, TestTotalBytes) {
  const DatasetTestParam& test_case = GetParam();
  if (test_case.type == _string) {
    // TotalBytes() is approximate and gives an upper bound for strings
    EXPECT_LE(data::GetTotalBytes(test_case.tensor), test_case.expected_bytes);
  } else {
    EXPECT_EQ(data::GetTotalBytes(test_case.tensor), test_case.expected_bytes);
  }
}

std::vector<Tensor> tensor_int32s{test::AsTensor<int32>({1, 2, 3, 4, 5}),
                                  test::AsTensor<int32>({1, 2, 3, 4})};

std::vector<Tensor> tensor_int64s{test::AsTensor<int64>({1, 2, 3, 4, 5}),
                                  test::AsTensor<int64>({10, 12})};

std::vector<Tensor> tensor_floats{test::AsTensor<float>({1.0, 2.0, 3.0, 4.0})};

std::vector<Tensor> tensor_doubles{
    test::AsTensor<double>({100.0}), test::AsTensor<double>({200.0}),
    test::AsTensor<double>({400.0}), test::AsTensor<double>({800.0})};

const string str = "test string";  // NOLINT
std::vector<Tensor> tensor_strs{test::AsTensor<string>({str})};

const DatasetTestParam test_cases[] = {
    {_int32, tensor_int32s, 4 /*bytes*/ * 9 /*elements*/},
    {_int64, tensor_int64s, 8 /*bytes*/ * 7 /*elements*/},
    {_float, tensor_floats, 4 /*bytes*/ * 4 /*elements*/},
    {_double, tensor_doubles, 8 /*bytes*/ * 4 /*elements*/},
    {_string, tensor_strs,
     static_cast<int64>(sizeof(str) + str.size()) /*bytes*/},
};

INSTANTIATE_TEST_SUITE_P(DatasetTestTotalBytes, DatasetTestTotalBytes,
                         ::testing::ValuesIn(test_cases));

}  // namespace tensorflow
