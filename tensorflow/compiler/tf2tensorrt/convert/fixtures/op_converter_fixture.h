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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_TEST_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_TEST_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/compiler/tf2tensorrt/common/datavec.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/utils/tensor_testutils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_testutils.h"

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tensorrt {

// TensorRT modes for testing. We define the following three modes:
// 1. Implicit batch mode: The tensors have static (known) input shape and the
//    the batch dimension (first dim) is removed from the TRT tensor shape. In
//    a loose notation: trt_shape = tf_shape[1:].
// 2. Explicit batch mode: static (known) input shape, but the batch dimension
//    is part of the trt tensor shape. (trt_shape = tf_shape)
// 3. Dynamic shape mode allows unknown input shapes, and requires explicit
//    batch size definition (trt_shape = tf_shape).
//
// Note that the Converter only distinguishes between two modes:
// - use_implicit_batch == true, this corresponds to kImplicitBatch,
// - use_implicit_batch == false which includes both kExplicitBatch and
//   kDynamicShape.
//
// For the converter, the distinction between explicit batch or dynamic shape
// mode follows from the input tensors of the network: dynamic shape input
// implies dynamic shape mode, while static shape input tensors imply explicit
// batch mode. We want to test all these modes, therefore we define the
// TrtTestMode with the following three options.
enum class TrtTestMode {
  kImplicitBatch = 0,
  kExplicitBatch = 1,
  kDynamicShape = 2
};

static constexpr std::array<TrtTestMode, 3> ValidTrtModes = {
    TrtTestMode::kImplicitBatch, TrtTestMode::kExplicitBatch,
    TrtTestMode::kDynamicShape};

inline string DebugString(const TrtTestMode mode) {
  switch (mode) {
    case TrtTestMode::kImplicitBatch:
      return "kImplicitBatch";
    case TrtTestMode::kExplicitBatch:
      return "kExplicitBatch";
    case TrtTestMode::kDynamicShape:
      return "kDynamicShape";
    default:
      return "Invalid TrtTestMode";
  }
}

namespace convert {

// Class to test various op converters, using both a TrtNodeValidator and
// Converter.
class OpConverterTest : public ::testing::Test {
 public:
  OpConverterTest() : scope_(Scope::NewRootScope()) {
    QCHECK_EQ(0, cudaStreamCreate(&stream_));
    Reset();
  }

  ~OpConverterTest() noexcept { QCHECK_EQ(0, cudaStreamDestroy(stream_)); }

  Status GetTensorOrWeights(const string& name, TRT_TensorOrWeights* output) {
    return converter_->GetTensorOrWeights(name, output);
  }

  void Reset(TrtPrecisionMode precision_mode_to_test = TrtPrecisionMode::FP32,
             TrtTestMode trt_mode = TrtTestMode::kImplicitBatch);

  void CheckDataTypeMatches(const DataVec& datas);

  Status BuildAndRun(const DataVec& input_data, DataVec* output_data,
                     const int batch_size = 1);

  // Adds ITensor for both validation and conversion, assuming explicit batch
  // dimension is included in dims (ie for an NCHW tensor dims = {N, C, H,
  // W}).
  void AddTestTensorWithTFDims(
      const string& name, const std::vector<int32>& dims,
      nvinfer1::DataType trt_type = nvinfer1::DataType::kFLOAT,
      Status add_input_status = Status::OK());

  // Adds ITensor for both validation and conversion. The difference compared
  // to AddTestTensorWithTFDims is in the meaning of the dims parameter. To
  // define a tensor with NCHW shape, here we set dims = {C,H,W} and
  // batch_size = N.
  // TODO(tfeher) remove this function once all test are updated to use the
  // other version of AddTestTensor (defined by
  // ParameterizedOpConverterTestBase).
  void AddTestTensor(const string& name, const std::vector<int32>& dims,
                     int batch_size = 1,
                     nvinfer1::DataType trt_dtype = nvinfer1::DataType::kFLOAT);

  // Add weights for both validation and conversion.
  template <typename T>
  void AddTestWeights(const string& name, const std::vector<int>& dims,
                      const std::vector<T>& values) {
    // Add weights for validation.
    TensorShape shape;
    TF_EXPECT_OK(TensorShapeUtils::MakeShape(dims, &shape));
    Tensor t = tensor_factory_.AsTensor<T>(values, shape);
    node_inputs_[name] = ops::Const(scope_.WithOpName(name), t);

    // Add weights for conversion.
    nvinfer1::DataType dtype;
    TF_ASSERT_OK(TfTypeToTrtType(DataTypeToEnum<T>::v(), &dtype));
    const nvinfer1::Dims trt_dims = nvinfer_factory::dims::Create(dims);
    const int64_t num_elements = TRT_ShapedWeights::count(trt_dims);
    QCHECK_EQ(num_elements, values.size())
        << num_elements << " vs " << values.size();
    TRT_ShapedWeights weights(dtype);
    if (num_elements) {
      weights = converter_->weight_store_.GetTempWeights(dtype, trt_dims);
      QCHECK_EQ(weights.size_bytes(), sizeof(T) * values.size())
          << weights.size_bytes() << " vs " << sizeof(T) * values.size();
      memcpy(weights.GetValues(), values.data(), weights.size_bytes());
    }
    TF_EXPECT_OK(
        converter_->AddTensorOrWeights(name, TRT_TensorOrWeights{weights}));
  }

  template <typename T = int32>
  void AddTestWeights(const string& name, const std::vector<int>& dims,
                      const std::vector<T>& values, DataType tf_type) {
    if (tf_type == DT_FLOAT) {
      AddTestWeights(name, dims, CastVector<T, float>(values));
    } else if (tf_type == DT_HALF) {
      AddTestWeights(name, dims, CastVector<T, Eigen::half>(values));
    } else if (tf_type == DT_INT32) {
      AddTestWeights(name, dims, CastVector<T, int32>(values));
    } else {
      FAIL() << "Cannot create test weights with type "
             << DataTypeString(tf_type);
    }
  }

  // Test validation in validation-only mode.
  Status RunValidation(const Node* node);

  void RunConversion(const Node* node, error::Code expected_code = error::OK,
                     const std::string& expected_msg_substr = "");

  // Helper method to run both validation and conversion, when the expected
  // output are same.
  void RunValidationAndConversion(const NodeDef& node_def,
                                  error::Code expected_code = error::OK,
                                  const std::string& expected_msg_substr = "",
                                  bool should_run_conversion = true);

  // Helper method to run both validation and conversion, and check the output
  // shapes.
  void RunValidationAndConversion(
      const NodeDef& node_def, const Status& status,
      const std::string& output_name,
      const std::vector<std::vector<int>>& exp_out_dims);

  // Expose quantization_ranges_ for tests
  std::unordered_map<ITensorProxyPtr*, float>& quantization_ranges_proxy() {
    return converter_->quantization_ranges_proxy_;
  }

  // Expose quantization_ranges_ for tests
  std::unordered_map<nvinfer1::ITensor*, float>& quantization_ranges() {
    return converter_->quantization_ranges_;
  }

  std::unique_ptr<Converter> converter_;

  TensorFactory<GpuManagedAllocator>& GetTensorFactory() {
    return tensor_factory_;
  }

  const TensorFactory<GpuManagedAllocator>& GetTensorFactory() const {
    return tensor_factory_;
  }

 protected:
  Logger& logger_ = *Logger::GetLogger();
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine_;
  cudaStream_t stream_;

  TensorFactory<GpuManagedAllocator> tensor_factory_;

  // The scope that contains the graph being converted. Because
  // allocator_ provides the storage for tensor contents that
  // are represented as attributes for graph nodes within scope_,
  // allocator_ needs to be available when destructing scope_.
  // Therefore, scope_ comes after allocator_ in the class
  // member field list.
  Scope scope_;
  std::unordered_map<string, Output> node_inputs_;
};

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_TEST_H_
