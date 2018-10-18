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

#include "tensorflow/contrib/tensorrt/convert/convert_nodes.h"

#include <memory>
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/contrib/tensorrt/log/trt_logger.h"
#include "tensorflow/contrib/tensorrt/plugin/trt_plugin_factory.h"
#include "tensorflow/core/framework/node_def.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "cuda/include/cuda.h"
#include "cuda/include/cuda_runtime_api.h"
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

using ::testing::ElementsAre;

void ExpectStatus(Status status, error::Code code, const char* substr) {
  EXPECT_EQ(code, status.code()) << status;
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr(substr)) << status;
}

nvinfer1::Dims GetTestDims(const std::vector<int>& d) {
  nvinfer1::Dims dims;
  dims.nbDims = d.size();
  for (int i = 0; i < d.size(); ++i) {
    dims.d[i] = d[i];
  }
  return dims;
}

// Fake ITensor implementation for testing purposes.
class FakeITensor : public nvinfer1::ITensor {
 public:
  FakeITensor() {}

  FakeITensor(const nvinfer1::Dims& dims, const string& name = "")
      : name_(name), dims_(dims) {}

  FakeITensor(const string& name, const std::vector<int>& dims)
      : name_(name), dims_(GetTestDims(dims)) {}

  void setName(const char* name) override { name_ = name; }

  const char* getName() const override { return name_.c_str(); }

  void setDimensions(nvinfer1::Dims dimensions) override { dims_ = dimensions; }

  nvinfer1::Dims getDimensions() const override { return dims_; }

  void setType(nvinfer1::DataType type) override { type_ = type; }

  nvinfer1::DataType getType() const override { return type_; }

  bool isNetworkInput() const override { return false; }

  bool isNetworkOutput() const override { return false; }

  void setBroadcastAcrossBatch(bool broadcastAcrossBatch) override {}

  bool getBroadcastAcrossBatch() const override { return false; }

  nvinfer1::TensorLocation getLocation() const override { return location_; }

  void setLocation(nvinfer1::TensorLocation location) override {
    location_ = location;
  }

#if NV_TENSORRT_MAJOR >= 5
  bool setDynamicRange(float min, float max) override {}
#endif

 private:
  string name_;
  nvinfer1::Dims dims_;
  nvinfer1::DataType type_;
  nvinfer1::TensorLocation location_;
};

bool Equals(const nvinfer1::Dims& lhs, const nvinfer1::Dims& rhs) {
  if (lhs.nbDims != rhs.nbDims) return false;
  for (int i = 0; i < lhs.nbDims; ++i) {
    if (lhs.d[i] != rhs.d[i]) return false;
    // We don't check the types in the tests.
  }
  return true;
}

bool operator==(const TRT_ShapedWeights& lhs, const TRT_ShapedWeights& rhs) {
  return Equals(lhs.shape_, rhs.shape_) && lhs.type_ == rhs.type_ &&
         lhs.values_ == rhs.values_;
}

TEST(TRT_ShapedWeights_Test, Basic) {
  {
    float raw_weights[10];
    TRT_ShapedWeights weights(DT_FLOAT, raw_weights, GetTestDims({2, 5}));

    nvinfer1::Weights trt_weights = weights.GetWeightsForTRT();
    EXPECT_EQ(nvinfer1::DataType::kFLOAT, trt_weights.type);
    EXPECT_EQ(static_cast<void*>(raw_weights), trt_weights.values);
    EXPECT_EQ(10, trt_weights.count);

    EXPECT_EQ(static_cast<void*>(raw_weights), weights.GetValues());
    EXPECT_EQ(10, weights.count());
    EXPECT_EQ(40, weights.size_bytes());
  }
  {
    int32 raw_weights = 0;
    TRT_ShapedWeights weights(DT_INT32, &raw_weights, GetTestDims({1, 1, 1}));

    nvinfer1::Weights trt_weights = weights.GetWeightsForTRT();
    EXPECT_EQ(nvinfer1::DataType::kINT32, trt_weights.type);
    EXPECT_EQ(static_cast<void*>(&raw_weights), trt_weights.values);
    EXPECT_EQ(1, trt_weights.count);

    EXPECT_EQ(static_cast<void*>(&raw_weights), weights.GetValues());
    EXPECT_EQ(1, weights.count());
    EXPECT_EQ(4, weights.size_bytes());
  }
  {
    TRT_ShapedWeights weights(DT_FLOAT);

    nvinfer1::Weights trt_weights = weights.GetWeightsForTRT();
    EXPECT_EQ(nvinfer1::DataType::kFLOAT, trt_weights.type);
    EXPECT_EQ(nullptr, trt_weights.values);
    EXPECT_EQ(0, trt_weights.count);

    EXPECT_EQ(nullptr, weights.GetValues());
    EXPECT_EQ(0, weights.count());
    EXPECT_EQ(0, weights.size_bytes());
  }
}

TEST(TRT_TensorOrWeights_Test, Basic) {
  {
    nvinfer1::Dims dims;
    dims.nbDims = 1;
    dims.d[0] = 1;
    FakeITensor itensor(dims);

    TRT_TensorOrWeights tw(&itensor);
    EXPECT_EQ(true, tw.is_tensor());
    EXPECT_EQ(false, tw.is_weights());
    EXPECT_EQ(&itensor, tw.tensor());
    EXPECT_TRUE(Equals(dims, tw.shape()))
        << "- expected: " << DebugString(dims)
        << "\n        vs\n-   actual: " << DebugString(tw.shape());
  }
  {
    TRT_ShapedWeights weights(DT_FLOAT);
    TRT_TensorOrWeights tw(weights);
    EXPECT_EQ(false, tw.is_tensor());
    EXPECT_EQ(true, tw.is_weights());
    EXPECT_EQ(weights, tw.weights());

    nvinfer1::Dims dims;
    dims.nbDims = 0;
    EXPECT_TRUE(Equals(dims, tw.shape()))
        << "- expected: " << DebugString(dims)
        << "\n        vs\n-   actual: " << DebugString(tw.shape());
  }
}

class ConverterForTest : public Converter {
 public:
  ConverterForTest()
      : Converter(nullptr, /*fp16=*/false, /*max_batch_size=*/1) {
    QCHECK_EQ(0, cudaStreamCreate(&stream_));
    Reset();
  }

  ~ConverterForTest() override { QCHECK_EQ(0, cudaStreamDestroy(stream_)); }

  // Helper methods for testing purposes.

  void AddOpConverter(const string& op_name, OpConverter op_converter) {
    op_registry_[op_name] = op_converter;
  }

  void AddTensorOrWeights(const string& name, TRT_TensorOrWeights tw) {
    ASSERT_TRUE(trt_tensors_.insert({name, tw}).second);
  }

  void Reset() {
    // Clear the tensor map.
    trt_tensors_.clear();
    // Reset the INetworkDefinition.
    engine_.reset(nullptr);
    network_.reset(nullptr);
    builder_.reset(nullptr);
    builder_.reset(nvinfer1::createInferBuilder(logger_));
    network_.reset(builder_->createNetwork());
    trt_network_ = network_.get();
  }

  void BuildAndRun(const char* input_name, const std::vector<float>& input_data,
                   const char* output_name, std::vector<float>* output_data) {
    // Mark the output tensor as TRT engine output.
    TRT_TensorOrWeights tensor = GetTensorOrWeights(output_name);
    tensor.tensor()->setName(output_name);
    network()->markOutput(*tensor.tensor());

    // Build the TRT engine.
    QCHECK_EQ(nullptr, engine_.get());
    engine_.reset(builder_->buildCudaEngine(*network()));
    CHECK_NOTNULL(engine_.get());

    // Execute the TRT engine.
    const int input_size = input_data.size() * sizeof(float);
    const int output_size = output_data->size() * sizeof(float);
    const int input_index = engine_->getBindingIndex(input_name);
    const int output_index = engine_->getBindingIndex(output_name);

    ASSERT_EQ(engine_->getNbBindings(), 2);
    void* buffers[2];
    ASSERT_EQ(0, cudaMalloc(&buffers[input_index], input_size));
    ASSERT_EQ(0, cudaMalloc(&buffers[output_index], output_size));
    ASSERT_EQ(0, cudaMemcpyAsync(buffers[input_index], input_data.data(),
                                 input_size, cudaMemcpyHostToDevice, stream_));
    TrtUniquePtrType<nvinfer1::IExecutionContext> execution_context(
        engine_->createExecutionContext());
    execution_context->enqueue(1, buffers, stream_, nullptr);
    ASSERT_EQ(0, cudaMemcpyAsync(output_data->data(), buffers[output_index],
                                 output_size, cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);
    ASSERT_EQ(0, cudaFree(buffers[input_index]));
    ASSERT_EQ(0, cudaFree(buffers[output_index]));
  }

 private:
  Logger logger_;
  TrtUniquePtrType<nvinfer1::IBuilder> builder_;
  TrtUniquePtrType<nvinfer1::INetworkDefinition> network_;
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine_;
  cudaStream_t stream_;
};

class ConverterTest : public ::testing::Test {
 protected:
  nvinfer1::ITensor* AddTestTensor(const char* name,
                                   const std::vector<int>& dims) {
    nvinfer1::ITensor* tensor = converter_.network()->addInput(
        name, nvinfer1::DataType::kFLOAT, GetTestDims(dims));
    converter_.AddTensorOrWeights(name, TRT_TensorOrWeights{tensor});
    return tensor;
  }

  template <typename CType>
  TRT_ShapedWeights AddTestWeights(const char* name, const DataType dtype,
                                   const std::vector<int>& dims,
                                   const std::vector<CType>& values) {
    const nvinfer1::Dims trt_dims = GetTestDims(dims);
    const int64_t num_elements = TrtDimsNumElements(trt_dims);
    QCHECK_EQ(num_elements, values.size())
        << num_elements << " vs " << values.size();
    TRT_ShapedWeights weights(dtype);
    if (num_elements) {
      const int64_t size_bytes = DataTypeSize(dtype) * num_elements;
      QCHECK_EQ(size_bytes, sizeof(CType) * values.size())
          << size_bytes << " vs " << sizeof(CType) * values.size();
      converter_.weight_store()->store_.push_back(
          std::vector<uint8_t>(size_bytes));
      void* dst =
          static_cast<void*>(converter_.weight_store()->store_.back().data());
      memcpy(dst, values.data(), size_bytes);
      weights = TRT_ShapedWeights(dtype, dst, trt_dims);
    }
    converter_.AddTensorOrWeights(name, TRT_TensorOrWeights{weights});
    return weights;
  }

  NodeDef MakeNodeDef(const string& name, const string& op,
                      const std::vector<string>& inputs) {
    NodeDef node_def;
    node_def.set_name(name);
    node_def.set_op(op);
    for (const string& input : inputs) {
      node_def.add_input(input);
    }
    return node_def;
  }

  ConverterForTest converter_;
};

TEST_F(ConverterTest, GetTempWeights) {
  TRT_ShapedWeights weights =
      converter_.GetTempWeights(DT_FLOAT, GetTestDims({2, 3}));

  nvinfer1::Weights trt_weights = weights.GetWeightsForTRT();
  EXPECT_EQ(nvinfer1::DataType::kFLOAT, trt_weights.type);
  EXPECT_NE(nullptr, trt_weights.values);
  EXPECT_EQ(6, trt_weights.count);

  EXPECT_NE(nullptr, weights.GetValues());
  EXPECT_EQ(6, weights.count());
  EXPECT_EQ(24, weights.size_bytes());

  // TODO(aaroey): test the case where shape element count is 0.
}

TEST_F(ConverterTest, GetInputs) {
  NodeDef node_def;
  node_def.add_input("^control_input");
  node_def.add_input("input");
  node_def.add_input("input:0");
  node_def.add_input("input:1");
  node_def.add_input("weird_input:2:3:4:0");

  FakeITensor input, input_1, input_2;
  TF_EXPECT_OK(converter_.AddInputTensor("input", &input));
  TF_EXPECT_OK(converter_.AddInputTensor("input:1", &input_1));
  TF_EXPECT_OK(converter_.AddInputTensor("weird_input:2:3:4", &input_2));

  std::vector<TRT_TensorOrWeights> inputs;
  TF_EXPECT_OK(converter_.GetInputs(node_def, &inputs));
  EXPECT_EQ(4, inputs.size());
  EXPECT_EQ(&input, inputs[0].tensor());
  EXPECT_EQ(&input, inputs[1].tensor());
  EXPECT_EQ(&input_1, inputs[2].tensor());
  EXPECT_EQ(&input_2, inputs[3].tensor());
}

TEST_F(ConverterTest, ConvertNode) {
  FakeITensor output_tensors[2];
  auto op_converter = [&output_tensors](
                          Converter& ctx, const NodeDef& node_def,
                          const std::vector<TRT_TensorOrWeights>& inputs,
                          std::vector<TRT_TensorOrWeights>* outputs) -> Status {
    nvinfer1::Dims dims = inputs[0].tensor()->getDimensions();
    for (int i = 0; i < 2; ++i) {
      dims.d[0] += 1;
      output_tensors[i].setDimensions(dims);
      outputs->push_back(TRT_TensorOrWeights(&output_tensors[i]));
    }
    return Status::OK();
  };
  converter_.AddOpConverter("MyOp", op_converter);

  FakeITensor input_tensor("my_input", {12345});
  TF_EXPECT_OK(converter_.AddInputTensor("my_input", &input_tensor));

  NodeDef node_def = MakeNodeDef("my_op", "MyOp", {"my_input"});
  TF_EXPECT_OK(converter_.ConvertNode(node_def));

  TRT_TensorOrWeights actual_output_1 = converter_.GetTensorOrWeights("my_op");
  EXPECT_EQ(&output_tensors[0], actual_output_1.tensor());
  EXPECT_EQ(12346, actual_output_1.tensor()->getDimensions().d[0]);

  TRT_TensorOrWeights actual_output_2 =
      converter_.GetTensorOrWeights("my_op:1");
  EXPECT_EQ(&output_tensors[1], actual_output_2.tensor());
  EXPECT_EQ(12347, actual_output_2.tensor()->getDimensions().d[0]);
}

TEST_F(ConverterTest, TransposeTensor) {
  nvinfer1::ITensor* input_tensor = AddTestTensor("", {2, 3, 5});
  const nvinfer1::ITensor* output_tensor = nullptr;

  // Rank doesn't match.
  ExpectStatus(
      converter_.TransposeTensor(input_tensor, {0, 1}, &output_tensor),
      error::INVALID_ARGUMENT,
      "Rank of perm for transpose does not match with that of the input");

  // Transpose at batch dimension.
  ExpectStatus(
      converter_.TransposeTensor(input_tensor, {1, 0, 2, 3}, &output_tensor),
      error::UNIMPLEMENTED, "Transpose at batch dimension is not supported.");

  // OK.
  TF_EXPECT_OK(
      converter_.TransposeTensor(input_tensor, {0, 3, 1, 2}, &output_tensor));
  EXPECT_TRUE(Equals(GetTestDims({5, 2, 3}), output_tensor->getDimensions()))
      << DebugString(*output_tensor);
}

TEST_F(ConverterTest, PrepareTensorForShape_Tensor) {
  nvinfer1::ITensor* input_tensor = AddTestTensor("", {2, 3, 5});
  TRT_TensorOrWeights tw(input_tensor);
  const nvinfer1::ITensor* output_tensor = nullptr;

  // Shape size doesn't match.
  ExpectStatus(converter_.PrepareTensorForShape(tw, GetTestDims({2, 3, 6}),
                                                &output_tensor),
               error::INVALID_ARGUMENT, "Reshape shapes are not compatible.");

  // TODO(aaroey): we should check the case where uninferred dimensions are not
  // an exact divisor of input dim ensions, e.g. for dims {-1, 7}.

  // Infer shape, ok.
  TF_EXPECT_OK(converter_.PrepareTensorForShape(tw, GetTestDims({-1, 2}),
                                                &output_tensor));
  EXPECT_TRUE(Equals(GetTestDims({15, 2}), output_tensor->getDimensions()))
      << DebugString(*output_tensor);

  // Regular shape.
  TF_EXPECT_OK(converter_.PrepareTensorForShape(tw, GetTestDims({10, 3}),
                                                &output_tensor));
  EXPECT_TRUE(Equals(GetTestDims({10, 3}), output_tensor->getDimensions()))
      << DebugString(*output_tensor);
}

#if NV_TENSORRT_MAJOR > 3
TEST_F(ConverterTest, PrepareTensorForShape_Weights) {
  TRT_ShapedWeights weights =
      converter_.GetTempWeights(DT_FLOAT, GetTestDims({2, 3, 5}));
  TRT_TensorOrWeights tw(weights);
  const nvinfer1::ITensor* output_tensor = nullptr;
  TF_EXPECT_OK(converter_.PrepareTensorForShape(tw, GetTestDims({10, 3}),
                                                &output_tensor));
  EXPECT_TRUE(Equals(GetTestDims({10, 3}), output_tensor->getDimensions()))
      << DebugString(*output_tensor);
}
#endif

template <DataType dtype, typename InputCType, typename OutputCType>
void TestConvertConst(ConverterForTest* converter) {
  NodeDef node_def;
  node_def.set_name("my_const");
  node_def.set_op("Const");

  auto reset_and_test = [&node_def, converter](
                            const Tensor& tensor, const bool as_tensor_content,
                            const std::vector<int>& expected_dims,
                            const std::vector<OutputCType>& expected_value) {
    converter->Reset();

    auto& attr = *node_def.mutable_attr();
    if (as_tensor_content) {
      tensor.AsProtoTensorContent(attr["value"].mutable_tensor());
    } else {
      tensor.AsProtoField(attr["value"].mutable_tensor());
    }
    TF_EXPECT_OK(converter->ConvertNode(node_def));
    TRT_TensorOrWeights output = converter->GetTensorOrWeights("my_const");
    EXPECT_TRUE(Equals(GetTestDims(expected_dims), output.weights().shape_))
        << output.DebugString();
    ASSERT_EQ(expected_value.size(), output.weights().count())
        << output.DebugString();
    const OutputCType* actual_values =
        static_cast<const OutputCType*>(output.weights().GetValues());
    for (int i = 0; i < expected_value.size(); ++i) {
      EXPECT_EQ(expected_value[i], actual_values[i]);
    }
  };

  auto& attr = *node_def.mutable_attr();
  attr["dtype"].set_type(dtype);
  {
    // By default empty tensor will pick DT_FLOAT as data type and we fix it
    // here.
    attr["value"].mutable_tensor()->set_dtype(dtype);
    Tensor t;  // Empty tensor.
    reset_and_test(t, false, {}, {});
  }
  {
    Tensor t = ::tensorflow::test::AsScalar<InputCType>(12);
    reset_and_test(t, false, {1}, {12});
    reset_and_test(t, true, {1}, {12});
  }
  {
    Tensor t = ::tensorflow::test::AsTensor<InputCType>({1, 2});
    reset_and_test(t, false, {2}, {1, 2});
    reset_and_test(t, true, {2}, {1, 2});
  }
  {
    Tensor t = ::tensorflow::test::AsTensor<InputCType>({1, 2, 3, 4, 5, 6},
                                                        TensorShape({2, 3}));
    reset_and_test(t, false, {2, 3}, {1, 2, 3, 4, 5, 6});
    reset_and_test(t, true, {2, 3}, {1, 2, 3, 4, 5, 6});
  }
}

TEST_F(ConverterTest, ConvertConst) {
  {
    converter_.Reset();
    NodeDef node_def = MakeNodeDef("my_const", "Const", {"input"});
    AddTestTensor("input", {1});
    ExpectStatus(
        converter_.ConvertNode(node_def), error::INVALID_ARGUMENT,
        "Constant node is expected to have empty input list: my_const");
  }
  {
    converter_.Reset();
    NodeDef node_def = MakeNodeDef("my_const", "Const", {});
    (*node_def.mutable_attr())["dtype"].set_type(DT_DOUBLE);
    ExpectStatus(converter_.ConvertNode(node_def), error::INVALID_ARGUMENT,
                 "Unsupported data type");
  }

  TestConvertConst<DT_FLOAT, float, float>(&converter_);
  TestConvertConst<DT_INT8, int8, int32>(&converter_);
#if NV_TENSORRT_MAJOR > 3
  TestConvertConst<DT_INT32, int32, int32>(&converter_);
#endif
}

TEST_F(ConverterTest, ConvertTranspose) {
  {
    // Input list is empty, should fail.
    NodeDef node_def = MakeNodeDef("my_transpose", "Transpose", {});
    ExpectStatus(converter_.ConvertNode(node_def), error::INVALID_ARGUMENT,
                 "Input expects tensor and weights, at my_transpose");
  }
  NodeDef node_def =
      MakeNodeDef("my_transpose", "Transpose", {"input", "weights"});
  {
    // Permutation is a tensor, should fail.
    converter_.Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestTensor("weights", {3});
    ExpectStatus(converter_.ConvertNode(node_def), error::INVALID_ARGUMENT,
                 "Input expects tensor and weights, at my_transpose");
  }
  {
    // Transpose at batch dimension, should fail.
    converter_.Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("weights", DT_INT32, {4}, {1, 0, 2, 3});
    ExpectStatus(converter_.ConvertNode(node_def), error::UNIMPLEMENTED,
                 "Transpose at batch dimension is not supported");
  }
  {
    // Permutation rank doesn't match, should fail.
    converter_.Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("weights", DT_INT32, {3}, {0, 1, 2});
    ExpectStatus(
        converter_.ConvertNode(node_def), error::INVALID_ARGUMENT,
        "Rank of perm for transpose does not match with that of the input.");
  }
  {
    // Ok.
    converter_.Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("weights", DT_INT32, {4}, {0, 3, 1, 2});
    TF_EXPECT_OK(converter_.ConvertNode(node_def));
    TRT_TensorOrWeights output = converter_.GetTensorOrWeights("my_transpose");
    EXPECT_TRUE(output.is_tensor());
    EXPECT_TRUE(
        Equals(GetTestDims({3, 1, 2}), output.tensor()->getDimensions()))
        << output.DebugString();

    std::vector<float> output_data(6);
    converter_.BuildAndRun("input", {1, 2, 3, 4, 5, 6}, "my_transpose",
                           &output_data);
    EXPECT_THAT(output_data, ElementsAre(1, 4, 2, 5, 3, 6));
  }
}

TEST_F(ConverterTest, ConvertReshape) {
  {
    // Input list is empty, should fail.
    NodeDef node_def = MakeNodeDef("my_reshape", "Reshape", {});
    ExpectStatus(converter_.ConvertNode(node_def), error::INVALID_ARGUMENT,
                 "Input expects weights for shape, at my_reshape");
  }
  NodeDef node_def = MakeNodeDef("my_reshape", "Reshape", {"input", "weights"});
  {
    // Shape is a tensor, should fail.
    converter_.Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestTensor("weights", {3});
    ExpectStatus(converter_.ConvertNode(node_def), error::INVALID_ARGUMENT,
                 "Input expects weights for shape, at my_reshape");
  }
  {
    // Reshape to scalar, should fail.
    converter_.Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("weights", DT_INT32, {}, {});
    ExpectStatus(converter_.ConvertNode(node_def), error::UNIMPLEMENTED,
                 "Reshape to shape=[] is not supported, at my_reshape");
  }
  {
    // Reshape at batch dimension, should fail.
    converter_.Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("weights", DT_INT32, {4}, {-1, 1, 1, 2});
    ExpectStatus(converter_.ConvertNode(node_def), error::UNIMPLEMENTED,
                 "Reshape on batch dimension is not supported, at my_reshape");
  }
  {
    // Reshape at batch dimension, should fail.
    converter_.Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("weights", DT_INT32, {4}, {3, 1, 1, 2});
    ExpectStatus(converter_.ConvertNode(node_def), error::UNIMPLEMENTED,
                 "Reshape on batch dimension is not supported, at my_reshape");
  }
  // Reshape on non batch dimensions, ok.
  for (int batch_dim : {-1, 1}) {
    converter_.Reset();
    AddTestTensor("input", {1, 2, 3});
    AddTestWeights<int32>("weights", DT_INT32, {4}, {batch_dim, 1, 3, 2});
    TF_EXPECT_OK(converter_.ConvertNode(node_def));
    TRT_TensorOrWeights output = converter_.GetTensorOrWeights("my_reshape");
    EXPECT_TRUE(output.is_tensor());
    EXPECT_TRUE(
        Equals(GetTestDims({1, 3, 2}), output.tensor()->getDimensions()))
        << output.DebugString();

    std::vector<float> output_data(6);
    converter_.BuildAndRun("input", {1, 2, 3, 4, 5, 6}, "my_reshape",
                           &output_data);
    EXPECT_THAT(output_data, ElementsAre(1, 2, 3, 4, 5, 6));
  }
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
