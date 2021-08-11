/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/convert/fixtures/op_converter_param_fixture.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorflow/core/platform/status_matchers.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

// Builds and runs the converted network. Checks output tensor shape. Tests
// output values using a matcher. The network can have multiple input and
// output tensors. The inputs are defined by the input_data_ member
// variable.
void ParameterizedOpConverterTestBase::BuildAndRun(
    const string& name,
    const std::vector<std::vector<int>>& expected_output_dims,
    const Status& expected_runtime_status,
    const std::vector<::testing::Matcher<std::vector<float>>>& matcher,
    const std::vector<DataType>& out_tf_types) {
  TensorShape shape;
  const int n_output = expected_output_dims.size();
  ASSERT_EQ(n_output, matcher.size());
  DataVec output_data;
  for (int i = 0; i < n_output; i++) {
    TF_EXPECT_OK(TensorShapeUtils::MakeShape(expected_output_dims[i], &shape));
    string out_name = (i == 0) ? name : StrCat(name, ":", i);
    DataType out_tf_type = out_tf_types.size() > i ? out_tf_types[i] : tf_type_;
    InputOutputData data{out_name, tensor_factory_.ConstructTensor(
                                       shape.num_elements(), 0, out_tf_type)};
    output_data.push_back(data);
  }
  const int batch_size =
      input_data_.empty() ||
              TensorShapeUtils::IsScalar(input_data_[0].tensor.shape())
          ? 1
          : input_data_[0].tensor.shape().dim_size(0);
  Status stat =
      OpConverterTest::BuildAndRun(input_data_, &output_data, batch_size);
  ASSERT_EQ(expected_runtime_status.ok(), stat.ok())
      << "expected status: " << expected_runtime_status
      << ", actual status: " << stat;
  if (expected_runtime_status.ok() && stat.ok()) {
    for (int i = 0; i < n_output; i++) {
      // Check the shape of the actual output tensors
      TF_EXPECT_OK(
          TensorShapeUtils::MakeShape(expected_output_dims[i], &shape));
      EXPECT_TRUE(output_data[i].tensor.shape() == shape)
          << "Expected shape: " << shape.DebugString()
          << ", actual shape: " << output_data[i].tensor.shape().DebugString();
      EXPECT_THAT(GetDataAsFloat(output_data[i]), matcher[i]);
    }
  }
}

// Runs validation and conversion. If conversion is successfull then builds
// the TRT network, executes it and checks the output. Handles multiple
// output tensors.
void ParameterizedOpConverterTestBase::TestOpConverterMultiOut(
    const string& name, const NodeDef node_def,
    const std::vector<std::vector<int>>& expected_output_dims,
    const Status& expected_conversion_status,
    const Status& expected_runtime_status,
    const std::vector<::testing::Matcher<std::vector<float>>>& matcher,
    const std::vector<DataType>& out_tf_type) {
  RunValidationAndConversion(node_def, expected_conversion_status, name,
                             expected_output_dims);
  if (expected_conversion_status.ok()) {
    BuildAndRun(name, expected_output_dims, expected_runtime_status, matcher,
                out_tf_type);
  }
}

// Runs validation and conversion. If conversion is successfull then builds
// the TRT network, executes it and checks the output.
void ParameterizedOpConverterTestBase::TestOpConverter(
    const string& name, const NodeDef node_def,
    const std::vector<int>& expected_output_dims,
    const Status& expected_conversion_status,
    const Status& expected_runtime_status,
    const ::testing::Matcher<std::vector<float>>& matcher,
    const std::vector<DataType>& out_tf_types) {
  RunValidationAndConversion(
      node_def, expected_conversion_status, name,
      std::vector<std::vector<int>>({expected_output_dims}));
  if (expected_conversion_status.ok()) {
    BuildAndRun(name, std::vector<std::vector<int>>({expected_output_dims}),
                expected_runtime_status,
                std::vector<::testing::Matcher<std::vector<float>>>({matcher}),
                out_tf_types);
  }
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
