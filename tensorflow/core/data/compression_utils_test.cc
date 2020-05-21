/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/compression_utils.h"

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {

class ParameterizedCompressionUtilsTest
    : public DatasetOpsTestBase,
      public ::testing::WithParamInterface<std::vector<Tensor>> {};

TEST_P(ParameterizedCompressionUtilsTest, RoundTrip) {
  std::vector<Tensor> element = GetParam();
  CompressedElement compressed;
  TF_ASSERT_OK(CompressElement(element, &compressed));
  std::vector<Tensor> round_trip_element;
  TF_ASSERT_OK(UncompressElement(compressed, &round_trip_element));
  TF_EXPECT_OK(
      ExpectEqual(element, round_trip_element, /*compare_order=*/true));
}

std::vector<std::vector<Tensor>> TestCases() {
  return {
      CreateTensors<int64>(TensorShape{1}, {{1}}),             // int64
      CreateTensors<int64>(TensorShape{1}, {{1}, {2}}),        // multiple int64
      CreateTensors<tstring>(TensorShape{1}, {{"a"}, {"b"}}),  // tstring
      {CreateTensor<tstring>(TensorShape{1}, {"a"}),
       CreateTensor<int64>(TensorShape{1}, {1})},  // mixed tstring/int64
      {},                                          // empty
  };
}

INSTANTIATE_TEST_SUITE_P(Instantiation, ParameterizedCompressionUtilsTest,
                         ::testing::ValuesIn(TestCases()));

}  // namespace data
}  // namespace tensorflow
