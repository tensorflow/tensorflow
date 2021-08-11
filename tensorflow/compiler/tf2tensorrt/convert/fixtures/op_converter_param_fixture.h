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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_FIXTURES_OP_CONVERTER_PARAM_FIXTURE_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_FIXTURES_OP_CONVERTER_PARAM_FIXTURE_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/compiler/tf2tensorrt/common/datavec.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/fixtures/op_converter_fixture.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_testutils.h"

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

// Parameterized version of OpConverterTest. We have the following parameters:
// 1. TrtTestMode: implicit batch, explicit batch, dynamic shape modes
// 2. DataType of the input TF tensors: DT_FLOAT, DT_HALF, DT_INT32
// 3. TrtPrecisionMode argument for the Converter: FP32, FP16, INT8
// We will introduce subclasses that will be instantiated using different
// combinations of the DataType and TrtPrecisionMode parameters.
class ParameterizedOpConverterTestBase
    : public OpConverterTest,
      public ::testing::WithParamInterface<
          std::tuple<TrtTestMode, DataType, TrtPrecisionMode>> {
 public:
  ParameterizedOpConverterTestBase()
      : trt_mode_(std::get<0>(GetParam())),
        tf_type_(std::get<1>(GetParam())),
        converter_precision_(std::get<2>(GetParam())) {
    LOG(INFO) << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
    LOG(INFO) << "tf_type_: " << DebugString(tf_type_);
    LOG(INFO) << "trt_mode_: " << DebugString(trt_mode_);
    LOG(INFO) << "converter_precision_: " << DebugString(converter_precision_);
    LOG(INFO) << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
  }

  void Reset() {
    OpConverterTest::Reset(converter_precision_, trt_mode_);
    input_data_.clear();
  }

  void Reset(TrtPrecisionMode precision) {
    OpConverterTest::Reset(precision, trt_mode_);
    input_data_.clear();
  }

  // Getters of protected attributes
  DataType get_tf_type() { return tf_type_; }
  TrtTestMode get_trt_mode() { return trt_mode_; }
  TrtPrecisionMode get_converter_precision() { return converter_precision_; }

  // Adds an input ITensor for TRT network. Also creates the corresponding TF
  // tensor, and stores it in the list of inputs (input_data_).
  //
  // The TF tensor is always created with concrete static input shape given by
  // dims. The ITensor can have static or dynamic shape based on the trt_mode
  // attribute. The ITensor shape is set automatically according to the
  // trt_mode parameter, unless the user overrides it with an explicit
  // partial_input_shape_dims argument.
  //
  // Parameters:
  // - name of the input node
  // - dims actual dimensions of the tensor that we will use during the test
  //   (including explicit batch dim)
  // - values initial values for the TF tensor
  // - dtype data type of the tensor
  // - partial_input_shape dimensions which can include unknown shapes. This
  // can
  //   be empty, in that case the partial_input_shape will be set
  //   automatically depending on the trt_mode argument. (This argument also
  //   includes explicit batch dim).
  // - add_input_status adding ITensor to the network can fail in implicit
  // batch
  //   mode if the batch size is inconsistent. Using the add_input_status arg
  //   we can test such errors.
  //
  template <typename T = int>
  void AddTestTensor(const string& name, const std::vector<int32>& dims,
                     DataType tf_type, const std::vector<T>& values,
                     const std::vector<int32>& partial_input_shape_dims = {},
                     Status add_input_status = Status::OK()) {
    if (!dims.empty()) {
      const auto num_elements = std::accumulate(
          std::begin(dims), std::end(dims), 1, std::multiplies<double>());
      if (!values.empty() && num_elements != values.size()) {
        // Note: for conversion only tests, it is valid to have empty values,
        // otherwise the number of elements should match.
        LOG(WARNING) << "Expected Test Tensor Shape: " << DebugString(dims)
                     << ", Received Input Tensor: " << DebugString(values);
      }
    }

    std::vector<int32> partial_shape;
    if (!partial_input_shape_dims.empty()) {
      partial_shape = partial_input_shape_dims;
    } else {
      if (trt_mode_ == TrtTestMode::kDynamicShape) {
        // In dynamic shape mode we make all dims unknown.
        partial_shape = std::vector<int32>(dims.size(), -1);
      } else {
        // Use static (known) input shapes.
        partial_shape = dims;
      }
    }
    nvinfer1::DataType trt_type;
    TF_ASSERT_OK(TfTypeToTrtType(tf_type, &trt_type));
    AddTestTensorWithTFDims(name, partial_shape, trt_type, add_input_status);
    if (!values.empty()) {
      VLOG(2) << "Adding test tensor: " << name << " "
              << DataTypeString(tf_type);
      InputOutputData data{name,
                           tensor_factory_.AsTensor(values, dims, tf_type)};
      VLOG(2) << "Added tensor: " << data.name << " with dtype "
              << DataTypeString(data.tensor.dtype());
      input_data_.push_back(data);
    }
  }

  // Adds test tensor (same as above) but with the default tf_type defined by
  // the test params.
  template <typename T = int>
  void AddTestTensor(const string& name, const std::vector<int32>& dims,
                     const std::vector<T>& values = {},
                     const std::vector<int32>& partial_input_shape_dims = {}) {
    AddTestTensor<T>(name, dims, tf_type_, values, partial_input_shape_dims);
  }

  // Builds and runs the converted network. Checks output tensor shape. Tests
  // output values using a matcher. The network can have multiple input and
  // output tensors. The inputs are defined by the input_data_ member
  // variable.
  void BuildAndRun(
      const string& name,
      const std::vector<std::vector<int>>& expected_output_dims,
      const Status& expected_runtime_status,
      const std::vector<::testing::Matcher<std::vector<float>>>& matcher,
      const std::vector<DataType>& out_tf_types = {});

  // Runs validation and conversion. If conversion is successfull then builds
  // the TRT network, executes it and checks the output. Handles multiple
  // output tensors.
  void TestOpConverterMultiOut(
      const string& name, const NodeDef node_def,
      const std::vector<std::vector<int>>& expected_output_dims,
      const Status& expected_conversion_status,
      const Status& expected_runtime_status,
      const std::vector<::testing::Matcher<std::vector<float>>>& matcher,
      const std::vector<DataType>& out_tf_type = {});

  // Runs validation and conversion. If conversion is successfull then builds
  // the TRT network, executes it and checks the output.
  void TestOpConverter(const string& name, const NodeDef node_def,
                       const std::vector<int>& expected_output_dims,
                       const Status& expected_conversion_status,
                       const Status& expected_runtime_status,
                       const ::testing::Matcher<std::vector<float>>& matcher,
                       const std::vector<DataType>& out_tf_types = {});

 protected:
  const TrtTestMode trt_mode_;
  const DataType tf_type_;
  const TrtPrecisionMode converter_precision_;
  DataVec input_data_;
};

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_FIXTURES_OP_CONVERTER_PARAM_FIXTURE_H_
