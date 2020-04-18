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

enum DataTypeTest {
  _tf_int_32,
  _tf_int_64,
  _tf_float_,
  _tf_double_,
  _tf_string_
};

struct DatasetTestParam {
  const DataTypeTest type;
  // This has to be a function pointer, to make sure the tensors we use as
  // parameters of the test case do not become globals. Ordering of static
  // initializers and globals can cause errors in the test.
  std::function<std::vector<Tensor>()> tensor_factory;
  const int64 expected_bytes;
};

class DatasetTestTotalBytes
    : public ::testing::TestWithParam<DatasetTestParam> {};

TEST_P(DatasetTestTotalBytes, TestTotalBytes) {
  const DatasetTestParam& test_case = GetParam();
  if (test_case.type == _tf_string_) {
    // TotalBytes() is approximate and gives an upper bound for strings
    EXPECT_LE(data::GetTotalBytes(test_case.tensor_factory()),
              test_case.expected_bytes);
  } else {
    EXPECT_EQ(data::GetTotalBytes(test_case.tensor_factory()),
              test_case.expected_bytes);
  }
}

std::vector<Tensor> tensor_tf_int_32s() {
  return {test::AsTensor<int32>({1, 2, 3, 4, 5}),
          test::AsTensor<int32>({1, 2, 3, 4})};
}

std::vector<Tensor> tensor_tf_int_64s() {
  return {test::AsTensor<int64>({1, 2, 3, 4, 5}),
          test::AsTensor<int64>({10, 12})};
}

std::vector<Tensor> tensor_tf_float_s() {
  return {test::AsTensor<float>({1.0, 2.0, 3.0, 4.0})};
}

std::vector<Tensor> tensor_tf_double_s() {
  return {test::AsTensor<double>({100.0}), test::AsTensor<double>({200.0}),
          test::AsTensor<double>({400.0}), test::AsTensor<double>({800.0})};
}

const tstring str = "test string";  // NOLINT
std::vector<Tensor> tensor_strs() { return {test::AsTensor<tstring>({str})}; }

INSTANTIATE_TEST_SUITE_P(
    DatasetTestTotalBytes, DatasetTestTotalBytes,
    ::testing::ValuesIn(std::vector<DatasetTestParam>{
        {_tf_int_32, tensor_tf_int_32s, 4 /*bytes*/ * 9 /*elements*/},
        {_tf_int_64, tensor_tf_int_64s, 8 /*bytes*/ * 7 /*elements*/},
        {_tf_float_, tensor_tf_float_s, 4 /*bytes*/ * 4 /*elements*/},
        {_tf_double_, tensor_tf_double_s, 8 /*bytes*/ * 4 /*elements*/},
        {_tf_string_, tensor_strs,
         static_cast<int64>(sizeof(str) + str.size()) /*bytes*/}}));

TEST(DatasetTest, JobServiceTokenIsEmpty) {
  data::JobToken token;
  EXPECT_TRUE(token.is_empty());
}

TEST(DatasetTest, JobTokenHoldsJobId) {
  int64 job_id = 5;
  data::JobToken token(job_id);
  EXPECT_EQ(job_id, token.job_id());
  EXPECT_FALSE(token.is_empty());
}

TEST(DatasetTest, JobTokenEncodeDecode) {
  int64 job_id = 5;
  data::JobToken token(job_id);
  VariantTensorData data;
  token.Encode(&data);
  data::JobToken decoded;
  decoded.Decode(data);
  EXPECT_FALSE(token.is_empty());
  EXPECT_EQ(job_id, token.job_id());
}

}  // namespace tensorflow
