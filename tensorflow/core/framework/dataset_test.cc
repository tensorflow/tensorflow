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
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {

TEST(DatasetTest, FullName) {
  EXPECT_EQ(FullName("prefix", "name"),
            "60d899aa0d8ce4351e7c3b419e92d25b|prefix:name");
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
  const int64_t expected_bytes;
};

class DatasetTestTotalBytes
    : public ::testing::TestWithParam<DatasetTestParam> {};

TEST_P(DatasetTestTotalBytes, TestTotalBytes) {
  const DatasetTestParam& test_case = GetParam();
  if (test_case.type == _tf_string_) {
    // TotalBytes() is approximate and gives an upper bound for strings
    EXPECT_LE(GetTotalBytes(test_case.tensor_factory()),
              test_case.expected_bytes);
  } else {
    EXPECT_EQ(GetTotalBytes(test_case.tensor_factory()),
              test_case.expected_bytes);
  }
}

std::vector<Tensor> tensor_tf_int_32s() {
  return {test::AsTensor<int32>({1, 2, 3, 4, 5}),
          test::AsTensor<int32>({1, 2, 3, 4})};
}

std::vector<Tensor> tensor_tf_int_64s() {
  return {test::AsTensor<int64_t>({1, 2, 3, 4, 5}),
          test::AsTensor<int64_t>({10, 12})};
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
         static_cast<int64_t>(sizeof(str) + str.size()) /*bytes*/}}));

struct MergeOptionsTestParam {
  const std::string source;
  const std::string destination;
  const std::string expected;
};

class MergeOptionsTest
    : public ::testing::TestWithParam<MergeOptionsTestParam> {};

TEST_P(MergeOptionsTest, MergeOptions) {
  const MergeOptionsTestParam& test_case = GetParam();
  Options source;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(test_case.source,
                                                          &source));
  Options destination;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(test_case.destination,
                                                          &destination));
  Options expected;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(test_case.expected,
                                                          &expected));
  internal::MergeOptions(source, &destination);
  EXPECT_EQ(expected.SerializeAsString(), destination.SerializeAsString());
}

INSTANTIATE_TEST_SUITE_P(
    MergeOptionsTest, MergeOptionsTest,
    ::testing::ValuesIn(std::vector<MergeOptionsTestParam>{
        // Destination is empty.
        {/*source=*/"deterministic: false", /*destination=*/"",
         /*expected=*/"deterministic: false"},
        // Source and destination have the same values.
        {/*source=*/"deterministic: false",
         /*destination=*/"deterministic: false",
         /*expected=*/"deterministic: false"},
        // Source values override destination values.
        {/*source=*/"deterministic: false",
         /*destination=*/"deterministic: true",
         /*expected=*/"deterministic: false"},
        // Values are enums.
        {/*source=*/"external_state_policy: POLICY_IGNORE",
         /*destination=*/"external_state_policy: POLICY_FAIL",
         /*expected=*/"external_state_policy: POLICY_IGNORE"}}));

TEST(DatasetTest, IsDatasetOp) {
  OpDef op_def;
  // Test zero outputs.
  EXPECT_FALSE(DatasetOpKernel::IsDatasetOp(op_def));

  // Test invalid output type.
  op_def.add_output_arg()->set_type(DT_STRING);
  EXPECT_FALSE(DatasetOpKernel::IsDatasetOp(op_def));

  // Test invalid op name.
  op_def.mutable_output_arg(0)->set_type(DT_VARIANT);
  op_def.set_name("Identity");
  EXPECT_FALSE(DatasetOpKernel::IsDatasetOp(op_def));

  // Test valid op names.
  for (const auto& name : {"Dataset", "RangeDataset", "MapDatasetV1",
                           "ParallelInterleaveDatasetV42",
                           "DataServiceDatasetV1000", "DatasetFromGraph"}) {
    op_def.set_name(name);
    EXPECT_TRUE(DatasetOpKernel::IsDatasetOp(op_def));
  }
}

}  // namespace data
}  // namespace tensorflow
