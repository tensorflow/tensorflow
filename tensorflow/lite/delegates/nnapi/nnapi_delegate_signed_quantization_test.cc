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
#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/fully_connected.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {

namespace ops {
namespace builtin {

TfLiteRegistration* Register_CONVOLUTION_REF();
TfLiteRegistration* Register_DEQUANTIZE();

}  // namespace builtin
}  // namespace ops

namespace {

class SingleOpModelWithNNAPI : public SingleOpModel {
 public:
  SingleOpModelWithNNAPI() = default;
  void Init(const NnApi* nnapi) {
    stateful_delegate_.reset(new StatefulNnApiDelegate(nnapi));
    SetDelegate(stateful_delegate_.get());
  }

  StatefulNnApiDelegate* GetDelegate() { return stateful_delegate_.get(); }

  void SetBufferHandle(int index, TfLiteBufferHandle handle) {
    interpreter_->SetBufferHandle(index, handle, stateful_delegate_.get());
  }
  TfLiteStatus GetCompilationStatus() { return compilation_status_; }

 protected:
  std::unique_ptr<StatefulNnApiDelegate> stateful_delegate_;
  TfLiteStatus compilation_status_;
};

class HybridFullyConnectedOpModel : public SingleOpModelWithNNAPI {
 public:
  HybridFullyConnectedOpModel(const NnApi* nnapi, int units, int batches,
                              const TensorData& input,
                              const TensorData& weights,
                              const TensorData& output = {TensorType_FLOAT32},
                              bool asymmetric_inputs = false)
      : batches_(batches), units_(units) {
    SingleOpModelWithNNAPI::Init(nnapi);
    int total_input_size = 1;
    for (size_t i = 0; i < input.shape.size(); ++i) {
      total_input_size *= input.shape[i];
    }
    input_size_ = total_input_size / batches_;

    input_ = AddInput(input);
    weights_ = AddInput(weights);

    TensorData bias{TensorType_FLOAT32, {units_}};
    bias_ = AddInput(bias);

    output_ = AddOutput(output);

    auto options = CreateFullyConnectedOptions(
                       builder_, ActivationFunctionType_RELU,
                       tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                       false, asymmetric_inputs)
                       .Union();
    SetBuiltinOp(BuiltinOperator_FULLY_CONNECTED,
                 BuiltinOptions_FullyConnectedOptions, options);
    resolver_ = absl::make_unique<SingleOpResolver>(
        BuiltinOperator_FULLY_CONNECTED,
        ops::builtin::Register_FULLY_CONNECTED_PIE());
    BuildInterpreter({GetShape(input_), GetShape(weights_), GetShape(bias_)},
                     /*num_threads=*/-1,
                     /* allow_fp32_relax_to_fp16 */ false,
                     /*apply_delegate=*/false);
    compilation_status_ = ApplyDelegate();
  }
  void SetBias(const std::vector<float>& f) { PopulateTensor(bias_, f); }
  void SetWeights(const std::vector<float>& data) {
    SymmetricQuantizeAndPopulate(weights_, data);
  }
  void SetSignedWeights(std::initializer_list<float> f) {
    SignedSymmetricQuantizeAndPopulate(weights_, f);
  }

  void SetInput(const std::vector<float>& f) { PopulateTensor(input_, f); }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  int input_size() { return input_size_; }
  int num_units() { return units_; }
  int num_batches() { return batches_; }

 protected:
  int input_;
  int weights_;
  int bias_;
  int output_;

  int batches_;
  int units_;
  int input_size_;
};

struct NnApiSignedQuantizationTest
    : ::tflite::delegate::nnapi::NnApiDelegateMockTest {
  static void SetUpTestSuite() { tensors_count = new std::map<int, int>(); }
  void SetUp() override {
    ::tflite::delegate::nnapi::NnApiDelegateMockTest::SetUp();
    nnapi_mock_->StubAddOperandWith(
        [](ANeuralNetworksModel* model,
           const ANeuralNetworksOperandType* type) -> int {
          const auto nn_tensor_type = type->type;
          if (tensors_count->find(nn_tensor_type) == tensors_count->end()) {
            tensors_count->insert({nn_tensor_type, 0});
          }
          tensors_count->at(nn_tensor_type)++;
          return ANEURALNETWORKS_NO_ERROR;
        });
  }
  void TearDown() override { tensors_count->clear(); }
  static void TearDownTestSuite() {
    delete tensors_count;
    tensors_count = nullptr;
  }
  static std::map<int, int>* tensors_count;
};
std::map<int, int>* NnApiSignedQuantizationTest::tensors_count = nullptr;

TEST_F(NnApiSignedQuantizationTest,
       HybridFullyConnectedMapsToSignedSymmOnSdk29) {
  nnapi_mock_->SetAndroidSdkVersion(29);

  HybridFullyConnectedOpModel m(
      nnapi_mock_->GetNnApi(), /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_FLOAT32, {2, 10}},
      /*weights=*/{TensorType_INT8, {3, 10}, 0, 0, 10.0 / 127.0, 0});
  m.SetSignedWeights({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});
  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);

  ASSERT_EQ(tensors_count->size(), 3);
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_FLOAT32),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_INT32), tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_QUANT8_SYMM),
            tensors_count->end());

  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_FLOAT32),
            4);  // fc_input, fc_weights, fc_bias, fc_output
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_INT32), 1);  // activation
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_QUANT8_SYMM),
            1);  // dequantize_weights_input
}

TEST_F(NnApiSignedQuantizationTest,
       HybridFullyConnectedMapsToSignedSymmOnSdk30) {
  nnapi_mock_->SetAndroidSdkVersion(30);

  HybridFullyConnectedOpModel m(
      nnapi_mock_->GetNnApi(), /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_FLOAT32, {2, 10}},
      /*weights=*/{TensorType_INT8, {3, 10}, 0, 0, 10.0 / 127.0, 0});
  m.SetSignedWeights({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});
  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);

  ASSERT_EQ(tensors_count->size(), 3);
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_FLOAT32),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_INT32), tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_QUANT8_SYMM),
            tensors_count->end());

  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_FLOAT32),
            4);  // fc_input, fc_weights, fc_bias, fc_output
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_INT32), 1);  // activation
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_QUANT8_SYMM),
            1);  // dequantize_weights_input
}

template <typename FilterType>
class BaseConvolutionOpModel : public SingleOpModelWithNNAPI {
 public:
  BaseConvolutionOpModel(
      const NnApi* nnapi, TfLiteRegistration* registration,
      const TensorData& input, const TensorData& filter,
      const TensorData& output, int stride_width = 2, int stride_height = 2,
      enum Padding padding = Padding_VALID,
      enum ActivationFunctionType activation = ActivationFunctionType_NONE,
      int dilation_width_factor = 1, int dilation_height_factor = 1,
      std::initializer_list<FilterType> filter_data = {}) {
    SingleOpModelWithNNAPI::Init(nnapi);

    input_ = AddInput(input);

    if (filter_data.size()) {
      filter_ = AddConstInput(filter, filter_data);
    } else {
      filter_ = AddInput(filter);
    }

    int bias_size = GetShape(filter_)[0];
    if (input.type == TensorType_FLOAT32) {
      bias_ = AddInput({TensorType_FLOAT32, {bias_size}});
    } else {
      // This is a quantized version. The scale of 'bias' depends on the scales
      // of input and filter. Supposedly this is correctly set during quantized
      // training.
      if (filter.per_channel_quantization) {
        // per channel quantization.
        std::vector<float> bias_scale(
            filter.per_channel_quantization_scales.size());
        std::vector<int64_t> bias_zero_points(
            filter.per_channel_quantization_scales.size());
        for (size_t i = 0; i < filter.per_channel_quantization_scales.size();
             ++i) {
          bias_scale[i] =
              input.scale * filter.per_channel_quantization_scales[i];
          bias_zero_points[i] = 0;
        }
        tflite::TensorType bias_type = TensorType_INT32;
        if (input.type == TensorType_INT16) {
          // In case of 16-bit, the bias type is set to be int 64.
          bias_type = TensorType_INT64;
        }
        TensorData bias{bias_type,
                        {bias_size},
                        /*min=*/0,
                        /*max=*/0,
                        /*scale=*/0,
                        /*zero_point=*/0,
                        true,
                        /*per_channel_quantization_scales=*/bias_scale,
                        /*per_channel_quantization_offsets=*/bias_zero_points,
                        /*channel_index==*/0};
        bias_ = AddInput(bias);
      } else {
        // per tensor quantization.
        auto bias_scale = GetScale(input_) * GetScale(filter_);
        TensorData bias{TensorType_INT32, {bias_size}, 0, 0, bias_scale};
        bias_ = AddInput(bias);
      }
    }

    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_CONV_2D, BuiltinOptions_Conv2DOptions,
                 CreateConv2DOptions(
                     builder_, padding, stride_width, stride_height, activation,
                     dilation_width_factor, dilation_height_factor)
                     .Union());

    resolver_ = absl::make_unique<SingleOpResolver>(BuiltinOperator_CONV_2D,
                                                    registration);
    BuildInterpreter({GetShape(input_), GetShape(filter_), GetShape(bias_)},
                     /*num_threads=*/-1,
                     /* allow_fp32_relax_to_fp16 */ false,
                     /*apply_delegate=*/false);
    compilation_status_ = ApplyDelegate();
  }

 protected:
  int input_;
  int filter_;
  int bias_;
  int output_;
};

class QuantizedConvolutionOpModel : public BaseConvolutionOpModel<uint8_t> {
 public:
  using BaseConvolutionOpModel::BaseConvolutionOpModel;

  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(input_, data);
  }

  void SetFilter(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(filter_, data);
  }

  void SetBias(std::initializer_list<float> data) {
    QuantizeAndPopulate<int32_t>(bias_, data);
  }

  std::vector<uint8_t> GetOutput() { return ExtractVector<uint8_t>(output_); }
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

TEST_F(NnApiSignedQuantizationTest,
       Conv2DUnsignedPerTensorMapsToUnsignedOnSdk29) {
  QuantizedConvolutionOpModel m(nnapi_mock_->GetNnApi(),
                                ops::builtin::Register_CONVOLUTION_REF(),
                                {TensorType_UINT8, {2, 2, 4, 1}, -63.5, 64},
                                {TensorType_UINT8, {3, 2, 2, 1}, -63.5, 64},
                                {TensorType_UINT8, {}, -127, 128});
  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  });
  m.SetFilter({
      1, 2, 3, 4,    // first 2x2 filter
      -1, 1, -1, 1,  // second 2x2 filter
      -1, -1, 1, 1,  // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);

  ASSERT_EQ(tensors_count->size(), 3);
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_INT32),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_INT32), tensors_count->end());

  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM),
            3);  // input, filter, output
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_INT32), 1);  // bias
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_INT32),
            4);  //  padding, stride_width, stride_height, activation
}

TEST_F(NnApiSignedQuantizationTest,
       Conv2dUnsignedPerTensorMapsToUnsignedOnSdk30) {
  nnapi_mock_->SetAndroidSdkVersion(30);
  QuantizedConvolutionOpModel m(nnapi_mock_->GetNnApi(),
                                ops::builtin::Register_CONVOLUTION_REF(),
                                {TensorType_UINT8, {2, 2, 4, 1}, -63.5, 64},
                                {TensorType_UINT8, {3, 2, 2, 1}, -63.5, 64},
                                {TensorType_UINT8, {}, -127, 128});
  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  });
  m.SetFilter({
      1, 2, 3, 4,    // first 2x2 filter
      -1, 1, -1, 1,  // second 2x2 filter
      -1, -1, 1, 1,  // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);

  ASSERT_EQ(tensors_count->size(), 3);
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_INT32),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_INT32), tensors_count->end());

  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM),
            3);  // input, filter, output
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_INT32), 1);  // bias
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_INT32),
            4);  //  padding, stride_width, stride_height, activation
}

class PerChannelQuantizedConvolutionOpModel
    : public BaseConvolutionOpModel<int8_t> {
 public:
  using BaseConvolutionOpModel::BaseConvolutionOpModel;

  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<int8_t>(input_, data);
  }

  void SetFilter(std::initializer_list<float> data) {
    PerChannelSymmetricQuantizeAndPopulate(filter_, data);
  }

  void SetBias(std::initializer_list<float> data) {
    PerChannelQuantizeBias(bias_, data);
  }

  std::vector<int8_t> GetOutput() { return ExtractVector<int8_t>(output_); }
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<int8_t>(ExtractVector<int8_t>(output_), GetScale(output_),
                              GetZeroPoint(output_));
  }
};

TEST_F(NnApiSignedQuantizationTest,
       Conv2dSignedPerTensorMapsToUnsignedOnSdk29) {
  nnapi_mock_->SetAndroidSdkVersion(29);
  PerChannelQuantizedConvolutionOpModel m(
      nnapi_mock_->GetNnApi(), ops::builtin::Register_CONVOLUTION_REF(),
      {TensorType_INT8, {1, 2, 3, 2}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
       {2, 2, 2, 2},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{1},
       /*per_channel_quantization_offsets=*/{0},
       /*channel_index=*/0},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1},
      /*stride_width=*/1, /*stride_height=*/1);
  m.SetInput({
      // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
      3, 2,    // batch = 0, y = 0, x = 0
      1, -1,   // batch = 0, y = 0, x = 1
      -2, -3,  // batch = 0, y = 0, x = 2
      4, 3,    // batch = 0, y = 1, x = 0
      2, -2,   // batch = 0, y = 1, x = 1
      -3, -4,  // batch = 0, y = 1, x = 2
  });
  m.SetFilter(
      // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
      {
          1, 2,  // out channel = 0, y = 0, x = 0
          3, 4,  // out channel = 0, y = 0, x = 1
          3, 4,  // out channel = 0, y = 1, x = 0
          5, 6,  // out channel = 0, y = 1, x = 1
          7, 8,  // out channel = 1, y = 0, x = 0
          5, 6,  // out channel = 1, y = 0, x = 1
          3, 4,  // out channel = 1, y = 1, x = 0
          1, 2,  // out channel = 1, y = 1, x = 1
      });
  m.SetBias({3, -2});

  // Invoke and verify output.
  // output has dimension [1 * 1 * 2 * 2] as [batch, y, x, output_channel]
  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);

  ASSERT_EQ(tensors_count->size(), 3);
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_INT32),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_INT32), tensors_count->end());

  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM),
            3);  // input, filter, output
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_INT32), 1);  // bias
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_INT32),
            4);  //  padding, stride_width, stride_height, activation
}

TEST_F(NnApiSignedQuantizationTest,
       Conv2dSignedPerTensorMapsToUnsignedOnSdk30) {
  nnapi_mock_->SetAndroidSdkVersion(30);
  PerChannelQuantizedConvolutionOpModel m(
      nnapi_mock_->GetNnApi(), ops::builtin::Register_CONVOLUTION_REF(),
      {TensorType_INT8, {1, 2, 3, 2}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
       {2, 2, 2, 2},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{1},
       /*per_channel_quantization_offsets=*/{0},
       /*channel_index=*/0},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1},
      /*stride_width=*/1, /*stride_height=*/1);
  m.SetInput({
      // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
      3, 2,    // batch = 0, y = 0, x = 0
      1, -1,   // batch = 0, y = 0, x = 1
      -2, -3,  // batch = 0, y = 0, x = 2
      4, 3,    // batch = 0, y = 1, x = 0
      2, -2,   // batch = 0, y = 1, x = 1
      -3, -4,  // batch = 0, y = 1, x = 2
  });
  m.SetFilter(
      // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
      {
          1, 2,  // out channel = 0, y = 0, x = 0
          3, 4,  // out channel = 0, y = 0, x = 1
          3, 4,  // out channel = 0, y = 1, x = 0
          5, 6,  // out channel = 0, y = 1, x = 1
          7, 8,  // out channel = 1, y = 0, x = 0
          5, 6,  // out channel = 1, y = 0, x = 1
          3, 4,  // out channel = 1, y = 1, x = 0
          1, 2,  // out channel = 1, y = 1, x = 1
      });
  m.SetBias({3, -2});

  // Invoke and verify output.
  // output has dimension [1 * 1 * 2 * 2] as [batch, y, x, output_channel]
  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);

  ASSERT_EQ(tensors_count->size(), 3);
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_INT32),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_INT32), tensors_count->end());

  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED),
            3);  // input, filter, output
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_INT32), 1);  // bias
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_INT32),
            4);  //  padding, stride_width, stride_height, activation
}

TEST_F(NnApiSignedQuantizationTest,
       Conv2dSignedPerChannelMapsToUnsignedOnSdk29) {
  PerChannelQuantizedConvolutionOpModel m(
      nnapi_mock_->GetNnApi(), ops::builtin::Register_CONVOLUTION_REF(),
      {TensorType_INT8, {1, 2, 3, 2}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
       {2, 2, 2, 2},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{1, 2},
       /*per_channel_quantization_offsets=*/{0, 0},
       /*channel_index=*/0},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1},
      /*stride_width=*/1, /*stride_height=*/1);
  m.SetInput({
      // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
      3, 2,    // batch = 0, y = 0, x = 0
      1, -1,   // batch = 0, y = 0, x = 1
      -2, -3,  // batch = 0, y = 0, x = 2
      4, 3,    // batch = 0, y = 1, x = 0
      2, -2,   // batch = 0, y = 1, x = 1
      -3, -4,  // batch = 0, y = 1, x = 2
  });
  m.SetFilter(
      // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
      {
          1, 2,  // out channel = 0, y = 0, x = 0
          3, 4,  // out channel = 0, y = 0, x = 1
          3, 4,  // out channel = 0, y = 1, x = 0
          5, 6,  // out channel = 0, y = 1, x = 1
          7, 8,  // out channel = 1, y = 0, x = 0
          5, 6,  // out channel = 1, y = 0, x = 1
          3, 4,  // out channel = 1, y = 1, x = 0
          1, 2,  // out channel = 1, y = 1, x = 1
      });
  m.SetBias({3, -2});

  // Invoke and verify output.
  // output has dimension [1 * 1 * 2 * 2] as [batch, y, x, output_channel]
  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);

  ASSERT_EQ(tensors_count->size(), 4);
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_INT32),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_INT32), tensors_count->end());

  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM),
            2);  // input, output
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL),
            1);                                                   // filter
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_INT32), 1);  // bias
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_INT32),
            4);  //  padding, stride_width, stride_height, activation
}

TEST_F(NnApiSignedQuantizationTest, Conv2dSignedPerChannelMapsToSignedOnSdk30) {
  nnapi_mock_->SetAndroidSdkVersion(30);
  PerChannelQuantizedConvolutionOpModel m(
      nnapi_mock_->GetNnApi(), ops::builtin::Register_CONVOLUTION_REF(),
      {TensorType_INT8, {1, 2, 3, 2}, -63.5, 64, 0.5, -1},
      {TensorType_INT8,
       // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
       {2, 2, 2, 2},
       0,
       0,
       0,
       0,
       /*per_channel_quantization=*/true,
       /*per_channel_quantization_scales=*/{1, 2},
       /*per_channel_quantization_offsets=*/{0, 0},
       /*channel_index=*/0},
      {TensorType_INT8, {}, -63.5, 64, 0.5, -1},
      /*stride_width=*/1, /*stride_height=*/1);
  m.SetInput({
      // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
      3, 2,    // batch = 0, y = 0, x = 0
      1, -1,   // batch = 0, y = 0, x = 1
      -2, -3,  // batch = 0, y = 0, x = 2
      4, 3,    // batch = 0, y = 1, x = 0
      2, -2,   // batch = 0, y = 1, x = 1
      -3, -4,  // batch = 0, y = 1, x = 2
  });
  m.SetFilter(
      // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
      {
          1, 2,  // out channel = 0, y = 0, x = 0
          3, 4,  // out channel = 0, y = 0, x = 1
          3, 4,  // out channel = 0, y = 1, x = 0
          5, 6,  // out channel = 0, y = 1, x = 1
          7, 8,  // out channel = 1, y = 0, x = 0
          5, 6,  // out channel = 1, y = 0, x = 1
          3, 4,  // out channel = 1, y = 1, x = 0
          1, 2,  // out channel = 1, y = 1, x = 1
      });
  m.SetBias({3, -2});

  // Invoke and verify output.
  // output has dimension [1 * 1 * 2 * 2] as [batch, y, x, output_channel]
  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);

  ASSERT_EQ(tensors_count->size(), 4);
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_INT32),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_INT32), tensors_count->end());

  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED),
            2);  // input, output
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL),
            1);                                                   // filter
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_INT32), 1);  // bias
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_INT32),
            4);  //  padding, stride_width, stride_height, activation
}

class QuantizeOpModel : public SingleOpModelWithNNAPI {
 public:
  QuantizeOpModel(const NnApi* nnapi, const TensorData& input,
                  const TensorData& output) {
    SingleOpModelWithNNAPI::Init(nnapi);
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_QUANTIZE, BuiltinOptions_QuantizeOptions,
                 CreateQuantizeOptions(builder_).Union());

    BuildInterpreter({GetShape(input_)}, /*num_threads=*/-1,
                     /* allow_fp32_relax_to_fp16 */ false,
                     /*apply_delegate=*/false);
    compilation_status_ = ApplyDelegate();
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  template <typename T>
  void SetInputAndQuantize(std::initializer_list<float> data) {
    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 private:
  int input_;
  int output_;
};

TEST_F(NnApiSignedQuantizationTest, QuantizeUint8MapsToUint8OnSdk29) {
  // [-63.5, 64] -> scale=0.5 zero_point=127 for UINT8
  QuantizeOpModel m(nnapi_mock_->GetNnApi(), {TensorType_FLOAT32, {2, 5}},
                    {TensorType_UINT8, {2, 5}, 0, 0, 0.5, 127});

  m.SetInput({-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64});
  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);

  ASSERT_EQ(tensors_count->size(), 2);
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_FLOAT32),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM),
            tensors_count->end());

  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_FLOAT32),
            1);  // input
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM),
            1);  // output
}

TEST_F(NnApiSignedQuantizationTest, QuantizeUint8MapsToUint8OnSdk30) {
  nnapi_mock_->SetAndroidSdkVersion(30);
  // [-63.5, 64] -> scale=0.5 zero_point=127 for UINT8
  QuantizeOpModel m(nnapi_mock_->GetNnApi(), {TensorType_FLOAT32, {2, 5}},
                    {TensorType_UINT8, {2, 5}, 0, 0, 0.5, 127});

  m.SetInput({-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64});
  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);

  ASSERT_EQ(tensors_count->size(), 2);
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_FLOAT32),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM),
            tensors_count->end());

  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_FLOAT32),
            1);  // input
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM),
            1);  // output
}

// Quantize with Int8 output is only supported since SDK level 30.
TEST_F(NnApiSignedQuantizationTest, QuantizeInt8MapsToInt8OnSdk30) {
  nnapi_mock_->SetAndroidSdkVersion(30);
  // [-63.5, 64] -> scale=0.5 zero_point=1 for INT8
  QuantizeOpModel m(nnapi_mock_->GetNnApi(), {TensorType_FLOAT32, {2, 5}},
                    {TensorType_INT8, {2, 5}, 0, 0, 0.5, -1});

  m.SetInput({-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64});
  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);

  ASSERT_EQ(tensors_count->size(), 2);
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_FLOAT32),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED),
            tensors_count->end());

  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_FLOAT32),
            1);  // input
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED),
            1);  // output
}

class DequantizeOpModel : public SingleOpModelWithNNAPI {
 public:
  DequantizeOpModel(const NnApi* nnapi, TensorType type,
                    std::initializer_list<int> shape, float scale,
                    int32_t zero_point, int version) {
    SingleOpModelWithNNAPI::Init(nnapi);
    const TensorData input_tensor_data = {type, shape, 0, 0, scale, zero_point};
    input_ = AddInput(input_tensor_data);
    output_ = AddOutput({TensorType_FLOAT32, shape});
    SetBuiltinOp(BuiltinOperator_DEQUANTIZE, BuiltinOptions_DequantizeOptions,
                 CreateDequantizeOptions(builder_).Union());

    resolver_ = absl::make_unique<SingleOpResolver>(
        BuiltinOperator_DEQUANTIZE, ops::builtin::Register_DEQUANTIZE(),
        version);

    BuildInterpreter({GetShape(input_)}, /*num_threads=*/-1,
                     /* allow_fp32_relax_to_fp16 */ false,
                     /*apply_delegate=*/false);
    compilation_status_ = ApplyDelegate();
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;
};

TEST_F(NnApiSignedQuantizationTest, DequantizeUint8MapsToUint8OnSdk29) {
  // [-63.5, 64] -> scale=0.5 zero_point=127 for UINT8
  DequantizeOpModel m(nnapi_mock_->GetNnApi(), TensorType_UINT8, {2, 5}, 0.5,
                      127, 1);

  m.SetInput<uint8_t>({0, 1, 2, 3, 4, 251, 252, 253, 254, 255});
  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);

  ASSERT_EQ(tensors_count->size(), 2);
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_FLOAT32),
            tensors_count->end());

  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM),
            1);  // input
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_FLOAT32),
            1);  // output
}

TEST_F(NnApiSignedQuantizationTest, DequantizeUint8MapsToUint8OnSdk30) {
  nnapi_mock_->SetAndroidSdkVersion(30);
  // [-63.5, 64] -> scale=0.5 zero_point=127 for UINT8
  DequantizeOpModel m(nnapi_mock_->GetNnApi(), TensorType_UINT8, {2, 5}, 0.5,
                      127, 1);

  m.SetInput<uint8_t>({0, 1, 2, 3, 4, 251, 252, 253, 254, 255});
  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);

  ASSERT_EQ(tensors_count->size(), 2);
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_FLOAT32),
            tensors_count->end());

  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM),
            1);  // input
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_FLOAT32),
            1);  // output
}

// Dequantize with Int8 input is only supported for symmetric quantization on
// SDK level 29
TEST_F(NnApiSignedQuantizationTest,
       DequantizeTestInt8SymmMapsToInt8SymmOnSdk29) {
  // [-63.5, 64] -> scale=0.5, zero_point=0 for INT8
  DequantizeOpModel m(nnapi_mock_->GetNnApi(), TensorType_INT8, {2, 5}, 0.5, 0,
                      2);

  m.SetInput<int8_t>({-128, -127, -126, -125, -124, 123, 124, 125, 126, 127});
  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);

  ASSERT_EQ(tensors_count->size(), 2);
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_QUANT8_SYMM),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_FLOAT32),
            tensors_count->end());

  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_QUANT8_SYMM),
            1);  // input
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_FLOAT32),
            1);  // output
}

// Dequantize with Int8 input is only supported since SDK level 30.
TEST_F(NnApiSignedQuantizationTest, DequantizeTestInt8MapsToInt8OnSdk30) {
  nnapi_mock_->SetAndroidSdkVersion(30);
  // [-63.5, 64] -> scale=0.5, zero_point=1 for INT8
  DequantizeOpModel m(nnapi_mock_->GetNnApi(), TensorType_INT8, {2, 5}, 0.5, -1,
                      2);

  m.SetInput<int8_t>({-128, -127, -126, -125, -124, 123, 124, 125, 126, 127});
  m.Invoke();
  EXPECT_EQ(m.GetCompilationStatus(), kTfLiteOk);

  ASSERT_EQ(tensors_count->size(), 2);
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED),
            tensors_count->end());
  ASSERT_NE(tensors_count->find(ANEURALNETWORKS_TENSOR_FLOAT32),
            tensors_count->end());

  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED),
            1);  // input
  EXPECT_EQ(tensors_count->at(ANEURALNETWORKS_TENSOR_FLOAT32),
            1);  // output
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
