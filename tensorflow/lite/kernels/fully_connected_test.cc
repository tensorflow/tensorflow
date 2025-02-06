/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
// Unit test for TFLite FULLY_CONNECTED op.

#include "tensorflow/lite/kernels/fully_connected.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

static float fully_connected_input[] = {
    0.503691, 0.196961, 0.521017, 0.554248, 0.288678, 0.792476, 0.561653,
    0.462230, 0.650736, 0.163132, 0.029658, 0.411544, 0.470539, 0.572390,
    0.538755, 0.212030, 0.264309, 0.193908, 0.777480, 0.745661, 0.423314,
    0.470804, 0.175501, 0.492225, 0.192743, 0.540183, 0.372514, 0.446550,
    0.498173, 0.126472, 0.132706, 0.001864, 0.323433, 0.653723, 0.556112,
    0.612111, 0.446199, 0.117765, 0.074341, 0.096935, 0.280897, 0.103999,
    0.508479, 0.751437, 0.676389, 0.047234, 0.963467, 0.940698, 0.241142,
    0.740947, 0.686359, 0.664456, 0.211751, 0.861860, 0.156681, 0.404494,
    0.402043, 0.529195, 0.851044, 0.900216, 0.655667, 0.983750, 0.902081,
    0.979100, 0.637473, 0.458193, 0.591211, 0.083671, 0.575958, 0.665552,
    0.180606, 0.856856, 0.769551, 0.689086, 0.608293, 0.445940, 0.736320,
    0.571760, 0.386637, 0.977461, 0.312707, 0.072996, 0.641918, 0.524458,
    0.934856, 0.798598, 0.928951, 0.336899, 0.327793, 0.779995, 0.237115,
    0.983460, 0.763746, 0.139196, 0.962560, 0.401218, 0.597389, 0.553771,
    0.484890, 0.173347, 0.219322, 0.665496, 0.030203, 0.988873, 0.354582,
    0.638496, 0.434813, 0.090902, 0.210256, 0.821450, 0.068363, 0.522962,
    0.894446, 0.710280, 0.047420, 0.829302, 0.508879, 0.976371, 0.166202,
    0.836672, 0.756367, 0.403317, 0.820132, 0.520112, 0.542513, 0.782691,
    0.921330, 0.139902};

static float fully_connected_golden_output[] = {
    0,        0.0732134,   0,        0,          0,         0.280859,
    0,        0.128927,    0,        0.0777251,  0,         0.270268,
    0.271435, 0.0173503,   0.335465, 0.235562,

    0,        0.0745866,   0,        0.051611,   0,         0.253876,
    0,        0.0814873,   0,        0.104104,   0,         0.248529,
    0.264194, 0,           0.302973, 0.166252,

    0,        0.0170409,   0,        0.0509851,  0,         0.212834,
    0,        0.0208326,   0,        0.129932,   0.203978,  0.103428,
    0.298051, 0,           0.332233, 0.00445903,

    0,        0.125246,    0,        0.0735336,  0,         0.0910256,
    0,        0,           0,        0.18933,    0.378111,  0.0712443,
    0.277298, 0.0123414,   0.267454, 0,

    0,        0.14687,     0,        0.155495,   0.0300215, 0.147256,
    0,        0,           0,        0.156412,   0.434914,  0.0461529,
    0.246508, 0,           0.363138, 0,

    0,        0,           0,        0.0212949,  0,         0.301708,
    0,        0.35497,     0,        0.406223,   0.0260211, 0.049195,
    0.197161, 0,           0.37316,  0,

    0,        0.221783,    0,        0,          0.0116515, 0.281945,
    0,        0,           0,        0,          0.285626,  0.181773,
    0.296401, 0.170452,    0.367135, 0.142597,

    0,        0,           0,        0,          0,         0.418886,
    0,        0.291063,    0,        0.227541,   0.0424759, 0.27589,
    0.398286, 0.177146,    0.40359,  0.121452,

    0,        0.0834884,   0,        0,          0,         0.287441,
    0,        0.0046838,   0,        0.0122087,  0,         0.217376,
    0.140183, 0.0948412,   0.436677, 0.0589876,

    0,        0.0289969,   0,        0.0921397,  0,         0.396802,
    0,        0.0126157,   0,        0.0968433,  0,         0.172271,
    0.173295, 0.0664741,   0.53645,  0.00915603,

    0,        0,           0,        0,          0,         0.147942,
    0,        0.263795,    0,        0.39782,    0,         0.382435,
    0.561072, 0.0579847,   0.145712, 0.13508,

    0,        0,           0,        0.16382,    0,         0.322294,
    0,        0.163798,    0,        0.405211,   0.367953,  0.076852,
    0.342473, 0.0834118,   0.377537, 0,

    0,        0.206,       0,        0,          0,         0.375769,
    0,        0,           0,        0,          0,         0.125165,
    0,        0.105591,    0.52055,  0.0536445,

    0,        0.259261,    0,        0,          0,         0.247707,
    0,        0,           0,        0,          0,         0.215862,
    0.149153, 0.224678,    0.359519, 0.129419,

    0,        0.17611,     0,        0.280895,   0,         0.576484,
    0,        0.000418848, 0,        0,          0,         0.151112,
    0.211902, 0,           0.566341, 0.106305,

    0,        0.0246284,   0,        0,          0,         0.196267,
    0,        0.0248624,   0,        0.265635,   0,         0.436199,
    0.408079, 0.134514,    0.328489, 0.411368};

class BaseFullyConnectedOpModel : public SingleOpModel {
 public:
  // TODO(ahentz): test different activation types too.
  BaseFullyConnectedOpModel(
      TfLiteRegistration* registration, int units, int batches,
      const TensorData& input, const TensorData& output = {TensorType_FLOAT32},
      const TensorType& bias_type = TensorType_FLOAT32,
      bool keep_num_dims = false, bool bias_tensor_optional = false,
      ActivationFunctionType activation_func = ActivationFunctionType_RELU,
      FullyConnectedOptionsWeightsFormat weights_format =
          FullyConnectedOptionsWeightsFormat_DEFAULT,
      int input_size = -1, bool weights_per_channel_quantized = false,
      std::vector<float> per_channel_quantization_scales = {},
      const TensorType& filter_type = TensorType_FLOAT32)
      : batches_(batches),
        units_(units),
        input_size_(input_size),
        bias_type_(bias_type) {
    if (input_size_ == -1) {
      // Calculate input_size_ from batch and input shape.
      int total_input_size = 1;
      for (size_t i = 0; i < input.shape.size(); ++i) {
        total_input_size *= input.shape[i];
      }
      input_size_ = total_input_size / batches_;
    }

    input_ = AddInput(input);
    if (weights_per_channel_quantized) {
      std::vector<int64_t> per_channel_quantization_offsets(
          per_channel_quantization_scales.size(), 0);
      weights_ = AddInput({filter_type,
                           {units_, input_size_},
                           0,
                           0,
                           0,
                           0,
                           true,
                           per_channel_quantization_scales,
                           per_channel_quantization_offsets,
                           0});
    } else {
      // per-tensor
      float min = input.min;
      float max = input.max;
      if (filter_type == TensorType_INT4 || filter_type == TensorType_INT8) {
        min = filter_type == TensorType_INT4 ? -7.f : -63.5f;
        max = filter_type == TensorType_INT4 ? 7.f : 64.f;
      }
      weights_ = AddInput({filter_type, {units_, input_size_}, min, max});
    }

    if (bias_tensor_optional) {
      bias_ = AddNullInput();
    } else if (bias_type == TensorType_FLOAT32) {
      bias_ = AddInput({TensorType_FLOAT32, {units_}});
    } else {
      // This is a quantized version. The scale of 'bias' depends on the scales
      // of input and filter. Supposedly this is correctly set during quantized
      // training.
      if (weights_per_channel_quantized) {
        std::vector<float> bias_scales = per_channel_quantization_scales;
        const float input_scale = GetScale(input_);
        for (float& bias_scale : bias_scales) {
          bias_scale *= input_scale;
        }
        std::vector<int64_t> bias_zero_points(
            per_channel_quantization_scales.size(), 0);
        TensorData bias{bias_type,   {units_},         0, 0, 0, 0, true,
                        bias_scales, bias_zero_points, 0};
        bias_ = AddInput(bias);
      } else {
        auto bias_scale = GetScale(input_) * GetScale(weights_);
        TensorData bias{bias_type, {units_}, 0, 0, bias_scale};
        bias_ = AddInput(bias);
      }
    }

    output_ = AddOutput(output);
    if (weights_format != FullyConnectedOptionsWeightsFormat_DEFAULT) {
      AddOutput({TensorType_UINT8, input.shape});
    }

    SetBuiltinOp(BuiltinOperator_FULLY_CONNECTED,
                 BuiltinOptions_FullyConnectedOptions,
                 CreateFullyConnectedOptions(
                     builder_, activation_func, weights_format, keep_num_dims,
                     /*asymmetric_quantize_inputs=*/true, bias_type)
                     .Union());
    resolver_ = std::make_unique<SingleOpResolver>(
        BuiltinOperator_FULLY_CONNECTED, registration);
    BuildInterpreter({GetShape(input_), GetShape(weights_),
                      (bias_ == kTfLiteOptionalTensor) ? std::vector<int>()
                                                       : GetShape(bias_)});
  }

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
  TensorType bias_type_;
};

class FloatFullyConnectedOpModel : public BaseFullyConnectedOpModel {
 public:
  using BaseFullyConnectedOpModel::BaseFullyConnectedOpModel;

  void SetBias(const std::vector<float>& f) { PopulateTensor(bias_, f); }

  void SetWeights(const std::vector<float>& f) { PopulateTensor(weights_, f); }

  void SetInput(const std::vector<float>& data) {
    PopulateTensor(input_, data);
  }
  void SetInput(int offset, float* begin, float* end) {
    PopulateTensor(input_, offset, begin, end);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }
};

class QuantizedFullyConnectedOpModel : public BaseFullyConnectedOpModel {
 public:
  using BaseFullyConnectedOpModel::BaseFullyConnectedOpModel;
  QuantizedFullyConnectedOpModel(
      TfLiteRegistration* registration, int units, int batches,
      const TensorData& input, const TensorData& output = {TensorType_INT8},
      const TensorType& bias_type = TensorType_INT32,
      bool keep_num_dims = false, bool bias_tensor_optional = false,
      ActivationFunctionType activation_func = ActivationFunctionType_RELU,
      FullyConnectedOptionsWeightsFormat weights_format =
          FullyConnectedOptionsWeightsFormat_DEFAULT,
      int input_size = -1, const TensorType& filter_type = TensorType_INT8)
      : BaseFullyConnectedOpModel(
            registration, units, batches, input, output, bias_type,
            keep_num_dims, bias_tensor_optional, activation_func,
            weights_format, input_size, false, {}, filter_type) {}

  void SetBias(const std::vector<float>& data) {
    if (bias_type_ == TensorType_INT32) {
      QuantizeAndPopulate<int32_t>(bias_, data);
    } else {
      QuantizeAndPopulate<int64_t>(bias_, data);
    }
  }

  template <typename T>
  void SetWeights(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(weights_, data);
  }

  void SetWeights4bit(const std::vector<float>& data) {
    QuantizeAndPopulate4bit(weights_, data);
  }

  template <typename T>
  void ShuffleAndSetWeights(const std::vector<float>& data, int input_depth,
                            int output_depth) {
    std::vector<float> shuffled_data(data.size());
    CHECK_EQ(input_depth % 16, 0);
    CHECK_EQ(output_depth % 4, 0);
    float* shuffled_data_ptr = shuffled_data.data();
    for (int block_o = 0; block_o < output_depth; block_o += 4) {
      for (int block_i = 0; block_i < input_depth; block_i += 16) {
        for (int o = 0; o < 4; o++) {
          for (int i = 0; i < 16; i++) {
            *shuffled_data_ptr++ =
                data[(block_o + o) * input_depth + block_i + i];
          }
        }
      }
    }
    TfLiteTensor* t = interpreter_->tensor(weights_);
    auto quantized_data =
        Quantize<T>(shuffled_data, t->params.scale, t->params.zero_point);
    for (T& q : quantized_data) {
      q ^= 0x80;
    }
    PopulateTensor(weights_, 0, quantized_data.data(),
                   quantized_data.data() + quantized_data.size());
  }

  template <typename T>
  void SetInput(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }
};

class PerChannelQuantizedFullyConnectedOpModel
    : public BaseFullyConnectedOpModel {
 public:
  using BaseFullyConnectedOpModel::BaseFullyConnectedOpModel;
  PerChannelQuantizedFullyConnectedOpModel(
      TfLiteRegistration* registration, int units, int batches,
      const TensorData& input,
      const std::vector<float>& per_channel_quantization_scales,
      const TensorData& output = {TensorType_INT8},
      const TensorType& bias_type = TensorType_INT32,
      bool keep_num_dims = false, bool bias_tensor_optional = false,
      ActivationFunctionType activation_func = ActivationFunctionType_RELU,
      FullyConnectedOptionsWeightsFormat weights_format =
          FullyConnectedOptionsWeightsFormat_DEFAULT,
      int input_size = -1, const TensorType& filter_type = TensorType_INT8)
      : BaseFullyConnectedOpModel(
            registration, units, batches, input, output, bias_type,
            keep_num_dims, bias_tensor_optional, activation_func,
            weights_format, input_size, true, per_channel_quantization_scales,
            filter_type) {}

  void SetBias(const std::vector<float>& data) {
    PerChannelQuantizeBias(bias_, data);
  }

  template <typename T>
  void SetWeights(const std::vector<float>& data) {
    PerChannelSymmetricQuantizeAndPopulate(weights_, data);
  }

  void SetWeights4bit(const std::vector<float>& data) {
    // 4 bit logic handled in PerChannelSymmetricQuantizeAndPopulate.
    CHECK_EQ(interpreter_->tensor(weights_)->type, kTfLiteInt4);
    PerChannelSymmetricQuantizeAndPopulate(weights_, data);
  }

  template <typename T>
  void SetInput(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }
};

// In the hybrid model the weights are quantized (to uint8). But the bias,
// input (and output) are expected to be in float precision.
class HybridFullyConnectedOpModel : public SingleOpModel {
 public:
  HybridFullyConnectedOpModel(int units, int batches, const TensorData& input,
                              const TensorData& weights,
                              const TensorData& output = {TensorType_FLOAT32},
                              bool asymmetric_inputs = false,
                              int num_threads = 1)
      : batches_(batches), units_(units) {
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
    resolver_ = std::make_unique<SingleOpResolver>(
        BuiltinOperator_FULLY_CONNECTED,
        ops::builtin::Register_FULLY_CONNECTED_PIE());
    BuildInterpreter({GetShape(input_), GetShape(weights_), GetShape(bias_)},
                     num_threads, /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true);
  }
  void SetBias(const std::vector<float>& f) { PopulateTensor(bias_, f); }
  void SetWeights(const std::vector<float>& data) {
    SymmetricQuantizeAndPopulate(weights_, data);
  }

  void SetSignedWeights(std::initializer_list<float> f) {
    SignedSymmetricQuantizeAndPopulate(weights_, f);
  }

  void SetSignedPerChannelWeights(std::initializer_list<float> f) {
    PerChannelSymmetricQuantizeAndPopulate(weights_, f);
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

const auto kKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_FULLY_CONNECTED_REF()},
    {"GenericOptimized", ops::builtin::Register_FULLY_CONNECTED_GENERIC_OPT()},
    {"Pie", ops::builtin::Register_FULLY_CONNECTED_PIE()},
});

class FloatFullyConnectedOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMap;
  }
};

const auto kKernelMapNoPie = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_FULLY_CONNECTED_REF()},
    {"GenericOptimized", ops::builtin::Register_FULLY_CONNECTED_GENERIC_OPT()},
});

class QuantizedFullyConnectedOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMapNoPie;
  }
};

const auto kKernelMapHybrid = new std::map<string, TfLiteRegistration*>({
    {"Pie", ops::builtin::Register_FULLY_CONNECTED_PIE()},
    // Only Pie supports the hybrid path, so the optimized kernel should fall
    // back to the Pie path in such cases.
    {"GenericOptimized", ops::builtin::Register_FULLY_CONNECTED_GENERIC_OPT()},
});

// Hybrid mode is used by the Pie quantized kernel.
class HybridFullyConnectedOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMapHybrid;
  }
};

// TODO(ahentz): add more small tests like this one, focused on making sure the
// calculations are correct.
TEST_P(FloatFullyConnectedOpTest, SimpleTest) {
  FloatFullyConnectedOpModel m(GetRegistration(), /*units=*/3, /*batches=*/2,
                               /*input=*/{TensorType_FLOAT32, {2, 10}});
  m.SetWeights({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(m.GetOutput(), ElementsAre(24, 25, 26, 58, 59, 60));
}

TEST_P(FloatFullyConnectedOpTest, SimpleTest2) {
  FloatFullyConnectedOpModel m(GetRegistration(), /*units=*/1, /*batches=*/2,
                               /*input=*/{TensorType_FLOAT32, {2, 2}});
  m.SetWeights({
      2, 4,  // u = 0
  });
  m.SetBias({1});

  m.SetInput({
      1, 2,  // b = 0
      2, 1,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 1));
  EXPECT_THAT(m.GetOutput(), ElementsAre(11, 9));
}

TEST_P(FloatFullyConnectedOpTest, FilterWithZeroSecondDimension1) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }

  FloatFullyConnectedOpModel m(GetRegistration(), /*units=*/2, /*batches=*/2,
                               /*input=*/{TensorType_FLOAT32, {2, 0}});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 2));
  EXPECT_THAT(m.GetOutput(), ElementsAre(0, 0, 0, 0));
}

TEST_P(FloatFullyConnectedOpTest, FilterWithZeroSecondDimension2) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }

  FloatFullyConnectedOpModel m(GetRegistration(), /*units=*/2, /*batches=*/2,
                               /*input=*/{TensorType_FLOAT32, {2, 2, 0}},
                               /*output=*/{TensorType_FLOAT32},
                               /*bias_type=*/TensorType_FLOAT32,
                               /*keep_num_dims=*/true);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 2, 2));
  EXPECT_THAT(m.GetOutput(), ElementsAre(0, 0, 0, 0, 0, 0, 0, 0));
}

TEST_P(FloatFullyConnectedOpTest, FilterWithZeroSecondDimension3) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }

  FloatFullyConnectedOpModel m(GetRegistration(), /*units=*/2, /*batches=*/2,
                               /*input=*/{TensorType_FLOAT32, {2, 2, 0}});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(4, 2));
  EXPECT_THAT(m.GetOutput(), ElementsAre(0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(FloatFullyConnectedOpTest, SimpleTestNoBias) {
  // The optimized kernel assumes that the bias is specified.
  FloatFullyConnectedOpModel m(ops::builtin::Register_FULLY_CONNECTED_PIE(),
                               /*units=*/1, /*batches=*/2,
                               /*input=*/{TensorType_FLOAT32, {2, 2}},
                               /*output=*/{TensorType_FLOAT32},
                               /*bias_type=*/TensorType_FLOAT32,
                               /*keep_num_dims=*/false,
                               /*bias_tensor_optional=*/true);
  m.SetWeights({
      2, 4,  // u = 0
  });

  m.SetInput({
      1, 2,  // b = 0
      2, 1,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 1));
  EXPECT_THAT(m.GetOutput(), ElementsAre(10, 8));
}

TEST(FloatFullyConnectedOpTest, SimpleTestEmptyOutput) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }

  FloatFullyConnectedOpModel m(ops::builtin::Register_FULLY_CONNECTED_PIE(),
                               /*units=*/1, /*batches=*/2,
                               /*input=*/{TensorType_FLOAT32, {0, 2}},
                               /*output=*/{TensorType_FLOAT32},
                               /*bias_type=*/TensorType_FLOAT32,
                               /*keep_num_dims=*/false,
                               /*bias_tensor_optional=*/true,
                               /*activation_func=*/ActivationFunctionType_RELU,
                               /*weights_format=*/
                               FullyConnectedOptionsWeightsFormat_DEFAULT,
                               /*input_size=*/2);
  m.SetWeights({
      2, 4,  // u = 0
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(0, 1));
}

TEST_P(QuantizedFullyConnectedOpTest, SimpleTestQuantizedUint8) {
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_UINT8, {2, 10}, -63.5, 64},
      /*output=*/{TensorType_UINT8, {}, -127, 128},
      /*bias_type=*/TensorType_INT32,
      /*keep_num_dims =*/false, /*bool bias_tensor_optional =*/false,
      /*ActivationFunctionType activation_func =*/ActivationFunctionType_RELU,
      /*FullyConnectedOptionsWeightsFormat weights_format =*/
      FullyConnectedOptionsWeightsFormat_DEFAULT,
      /*input_size=*/-1, /*filter_type=*/TensorType_UINT8);

  // input_product_scale < output_scale was not true.
  m.SetWeights<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  24, 25, 26,  //
                  58, 59, 60,  //
              })));
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAre(151, 152, 153, 185, 186, 187));
}

TEST_P(QuantizedFullyConnectedOpTest, SimpleTestQuantizedUint8NoBias) {
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_UINT8, {2, 10}, -63.5, 64},
      /*output=*/{TensorType_UINT8, {}, -127, 128},
      /*bias_type=*/TensorType_INT32,
      /*keep_num_dims =*/false, /*bool bias_tensor_optional =*/true,
      /*ActivationFunctionType activation_func =*/ActivationFunctionType_RELU,
      /*FullyConnectedOptionsWeightsFormat weights_format =*/
      FullyConnectedOptionsWeightsFormat_DEFAULT,
      /*input_size=*/-1, /*filter_type=*/TensorType_UINT8);

  // input_product_scale < output_scale was not true.
  m.SetWeights<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });

  m.SetInput<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  23, 23, 23,  //
                  57, 57, 57,  //
              })));
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAre(150, 150, 150, 184, 184, 184));
}

// The expected values for this test were obtained by running the test with the
// same parameters but by setting filter_type == TensorType_INT8 and
// m.SetWeights<int8_t>.
TEST_P(QuantizedFullyConnectedOpTest, SimpleTestQuantizedInt4) {
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_INT8, {2, 10}, -63.5, 64},
      /*output=*/{TensorType_INT8, {}, -127, 128}, TensorType_INT32, false,
      false, ActivationFunctionType_RELU,
      FullyConnectedOptionsWeightsFormat_DEFAULT, -1, TensorType_INT4);

  // Scale is set to 1.f by QuantizationParams() so don't exceed [-7,7]
  m.SetWeights4bit({
      1, 2, 3, 4, 5, 6, 7, 6, 5, 4,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 6, 5, 4,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 6, 5, 4,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      testing::Pointwise(testing::FloatEq(), {104, 105, 106, 98, 99, 100}));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAre(103, 104, 105, 97, 98, 99));
}

TEST_P(QuantizedFullyConnectedOpTest, SimpleTestQuantizedInt8) {
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_INT8, {2, 10}, -63.5, 64},
      /*output=*/{TensorType_INT8, {}, -127, 128});

  // input_product_scale < output_scale was not true.
  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({24, 25, 26, 58, 59, 60})));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAre(23, 24, 25, 57, 58, 59));
}

TEST_P(QuantizedFullyConnectedOpTest, SimpleTestPerChannelQuantizedInt8) {
  PerChannelQuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_INT8, {2, 10}, -63.5, 64},
      /*per_channel_quantization_scales=*/{0.2, 0.25, 0.5},
      /*output=*/{TensorType_INT8, {}, -127, 128});

  // input_product_scale < output_scale was not true.
  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({24, 25, 26, 58, 59, 60})));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAre(23, 24, 25, 57, 58, 59));
}

TEST_P(QuantizedFullyConnectedOpTest,
       SimpleTestPerChannelQuantizedOutputShape3DInt8) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }

  PerChannelQuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_INT8, {2, 2, 5}, -63.5, 64},
      /*per_channel_quantization_scales=*/{0.2, 0.25, 0.5},
      /*output=*/{TensorType_INT8, {}, -127, 128},
      /*bias_type=*/TensorType_INT32,
      /*keep_num_dims=*/true, /*bias_tensor_optional=*/false,
      /*activation_func=*/ActivationFunctionType_RELU,
      /*weights_format=*/FullyConnectedOptionsWeightsFormat_DEFAULT,
      /*input_size=*/5);

  // input_product_scale < output_scale was not true.
  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5,  // u = 0
      1, 2, 3, 4, 5,  // u = 1
      1, 2, 3, 4, 5,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<int8_t>({
      1, 2,  3,  4,  -5,  // b = 0, i = 0
      1, 2,  3,  -4, 5,   // b = 0, i = 1
      1, 2,  -3, 4,  5,   // b = 1, i = 0
      1, -2, 3,  4,  5,   // b = 1, i = 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  6, 7, 8,     // b = 0, i = 0
                  24, 25, 26,  // b = 0, i = 1
                  38, 39, 40,  // b = 1, i = 0
                  48, 49, 50   // b = 1, i = 1
              })));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAre(5, 6, 7,     // b = 0, i = 0
                                                 23, 24, 25,  // b = 0, i = 1
                                                 37, 38, 39,  // b = 1, i = 0
                                                 47, 48, 49   // b = 1, i = 1
                                                 ));
}

TEST_P(QuantizedFullyConnectedOpTest, SimpleTestPerChannelQuantizedInt4) {
  PerChannelQuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_INT8, {2, 10}, -63.5, 64},
      /*per_channel_quantization_scales=*/{1.0, 1.0, 1.0},
      /*output=*/{TensorType_INT8, {}, -127, 128},
      /*bias_type=*/TensorType_INT32, false, false, ActivationFunctionType_RELU,
      FullyConnectedOptionsWeightsFormat_DEFAULT, -1, TensorType_INT4);

  m.SetWeights4bit({
      1, 2, 3, 4, 5, 6, 7, 6, 5, 4,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 6, 5, 4,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 6, 5, 4,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({104, 105, 106, 98, 99, 100})));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAre(103, 104, 105, 97, 98, 99));
}

TEST_P(QuantizedFullyConnectedOpTest, SimpleTestQuantizedInt16NoBias) {
  const float scale = 128.0 / 65536;
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_INT16, {2, 10}, 0, 0, scale, 0},
      /*output=*/{TensorType_INT16, {}, 0, 0, scale, 0},
      /*bias_type=*/TensorType_INT64,
      /*keep_num_dims=*/false, /*bool bias_tensor_optional=*/true,
      /*ActivationFunctionType activation_func=*/ActivationFunctionType_RELU,
      /*FullyConnectedOptionsWeightsFormat weights_format=*/
      FullyConnectedOptionsWeightsFormat_DEFAULT);

  // input_product_scale < output_scale was not true.
  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });

  m.SetInput<int16_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear({23, 23, 23, 57, 57, 57})));
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAre(11776, 11776, 11776, 29184, 29184, 29184));
}

TEST_P(QuantizedFullyConnectedOpTest, SimpleTestQuantizedInt16Bias32) {
  const float scale = 128.0 / 65536;
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_INT16, {2, 10}, 0, 0, scale, 0},
      /*output=*/{TensorType_INT16, {}, 0, 0, scale, 0},
      /*bias_type=*/TensorType_INT32);

  // input_product_scale < output_scale was not true.
  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<int16_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear({24, 25, 26, 58, 59, 60})));
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAre(12288, 12800, 13312, 29696, 30208, 30720));
}

TEST_P(QuantizedFullyConnectedOpTest, SimpleTestQuantizedInt16Bias32Weight4) {
  const float scale = 128.0 / 65536;
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_INT16, {2, 10}, 0, 0, scale, 0},
      /*output=*/{TensorType_INT16, {}, 0, 0, scale, 0},
      /*bias_type=*/TensorType_INT32, /*keep_num_dims=*/false,
      /*bias_tensor_optional=*/false,
      /*activation_func*/ ActivationFunctionType_RELU,
      /*weights_format=*/FullyConnectedOptionsWeightsFormat_DEFAULT,
      /*input_size=*/-1, /*filter_type=*/TensorType_INT4);

  m.SetWeights4bit({
      1, 2, 3, 4, 5, 6, -7, 1, 2, 3,  // u = 0
      1, 2, 3, 4, 5, 6, -7, 1, 2, 3,  // u = 1
      1, 2, 3, 4, 5, 6, -7, 1, 2, 3,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<int16_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear({3, 4, 5, 23, 24, 25})));
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAre(1536, 2048, 2560, 11776, 12288, 12800));
}

TEST_P(QuantizedFullyConnectedOpTest,
       SimpleTestPerChannelQuantizedInt16Bias32) {
  const float scale = 128.0 / 65536;
  PerChannelQuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_INT16, {2, 10}, 0, 0, scale, 0},
      /*per_channel_quantization_scales=*/{0.2, 0.25, 0.5},
      /*output=*/{TensorType_INT16, {}, 0, 0, scale, 0},
      /*bias_type=*/TensorType_INT32);

  // input_product_scale < output_scale was not true.
  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<int16_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear({24, 25, 26, 58, 59, 60})));
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAre(12288, 12800, 13312, 29696, 30208, 30720));
}

TEST_P(QuantizedFullyConnectedOpTest,
       SimpleTestPerChannelQuantizedInt16Bias32Weight4) {
  const float scale = 128.0 / 65536;
  PerChannelQuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_INT16, {2, 10}, 0, 0, scale, 0},
      /*per_channel_quantization_scales=*/{1.0, 1.0, 1.0},
      /*output=*/{TensorType_INT16, {}, 0, 0, scale, 0},
      /*bias_type=*/TensorType_INT32, false, false, ActivationFunctionType_RELU,
      FullyConnectedOptionsWeightsFormat_DEFAULT, -1, TensorType_INT4);

  m.SetWeights4bit({
      1, 2, 3, 4, 5, 6, -7, 1, 2, 3,  // u = 0
      1, 2, 3, 4, 5, 6, -7, 1, 2, 3,  // u = 1
      1, 2, 3, 4, 5, 6, -7, 1, 2, 3,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<int16_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear({3, 4, 5, 23, 24, 25})));
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAre(1536, 2048, 2560, 11776, 12288, 12800));
}

TEST_P(QuantizedFullyConnectedOpTest, SimpleTestQuantizedInt16Bias64) {
  const float scale = 128.0 / 65536;
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_INT16, {2, 10}, 0, 0, scale, 0},
      /*output=*/{TensorType_INT16, {}, 0, 0, scale, 0},
      /*bias_type=*/TensorType_INT64);

  // input_product_scale < output_scale was not true.
  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<int16_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear({24, 25, 26, 58, 59, 60})));
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAre(12288, 12800, 13312, 29696, 30208, 30720));
}

TEST_P(QuantizedFullyConnectedOpTest, SimpleTestQuantizedInt8NoBias) {
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_INT8, {2, 10}, -63.5, 64},
      /*output=*/{TensorType_INT8, {}, -127, 128},
      /*bias_type=*/TensorType_INT32,
      /*keep_num_dims =*/false, /*bool bias_tensor_optional =*/true,
      /*ActivationFunctionType activation_func =*/ActivationFunctionType_RELU,
      /*FullyConnectedOptionsWeightsFormat weights_format =*/
      FullyConnectedOptionsWeightsFormat_DEFAULT);

  // input_product_scale < output_scale was not true.
  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });

  m.SetInput<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({23, 23, 23, 57, 57, 57})));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAre(22, 22, 22, 56, 56, 56));
}

TEST_P(QuantizedFullyConnectedOpTest, SimpleTestQuantizedOutputShape3DInt8) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }

  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_INT8, {2, 2, 5}, -63.5, 64},
      /*output=*/{TensorType_INT8, {}, -127, 128},
      /*bias_type=*/TensorType_INT32,
      /*keep_num_dims=*/true, /*bias_tensor_optional=*/false,
      /*activation_func=*/ActivationFunctionType_RELU,
      /*weights_format=*/FullyConnectedOptionsWeightsFormat_DEFAULT,
      /*input_size=*/5);

  // input_product_scale < output_scale was not true.
  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5,  // u = 0
      1, 2, 3, 4, 5,  // u = 1
      1, 2, 3, 4, 5,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<int8_t>({
      1, 2,  3,  4,  -5,  // b = 0, i = 0
      1, 2,  3,  -4, 5,   // b = 0, i = 1
      1, 2,  -3, 4,  5,   // b = 1, i = 0
      1, -2, 3,  4,  5,   // b = 1, i = 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  6, 7, 8,     // b = 0, i = 0
                  24, 25, 26,  // b = 0, i = 1
                  38, 39, 40,  // b = 1, i = 0
                  48, 49, 50   // b = 1, i = 1
              })));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAre(5, 6, 7,     // b = 0, i = 0
                                                 23, 24, 25,  // b = 0, i = 1
                                                 37, 38, 39,  // b = 1, i = 0
                                                 47, 48, 49   // b = 1, i = 1
                                                 ));
}

TEST_P(QuantizedFullyConnectedOpTest, SimpleTestQuantizedOutputShape3DInt16) {
  const float scale = 128.0 / 65536;
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_INT16, {2, 2, 5}, 0, 0, scale, 0},
      /*output=*/{TensorType_INT16, {}, 0, 0, scale, 0},
      /*bias_type=*/TensorType_INT64,
      /*keep_num_dims=*/true, /*bias_tensor_optional=*/false,
      /*activation_func=*/ActivationFunctionType_RELU,
      /*weights_format=*/FullyConnectedOptionsWeightsFormat_DEFAULT,
      /*input_size=*/5);

  // input_product_scale < output_scale was not true.
  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5,  // u = 0
      1, 2, 3, 4, 5,  // u = 1
      1, 2, 3, 4, 5,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<int16_t>({
      1, 2,  3,  4,  -5,  // b = 0, i = 0
      1, 2,  3,  -4, 5,   // b = 0, i = 1
      1, 2,  -3, 4,  5,   // b = 1, i = 0
      1, -2, 3,  4,  5,   // b = 1, i = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear({
                  6, 7, 8,     // b = 0, i = 0
                  24, 25, 26,  // b = 0, i = 1
                  38, 39, 40,  // b = 1, i = 0
                  48, 49, 50   // b = 1, i = 1
              })));
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAre(3072, 3584, 4096,     // b = 0, i = 0
                          12288, 12800, 13312,  // b = 0, i = 1
                          19456, 19968, 20480,  // b = 1, i = 0
                          24576, 25088, 25600   // b = 1, i = 1
                          ));
}
// Test the GEMV path.
TEST_P(QuantizedFullyConnectedOpTest, SimpleTestSingleBatchQuantizedInt8) {
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/4, /*batches*/ 1,
      /*input=*/{TensorType_INT8, {1, 10}, -63.5, 64},
      /*output=*/{TensorType_INT8, {}, -127, 128});

  // input_product_scale < output_scale was not true.
  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 3
  });
  m.SetBias({1, 2, 3, 4});

  m.SetInput<int8_t>({
      1, 2, 3, 4, 5, 6, 7, -8, 9, -10  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({58, 59, 60, 61})));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAre(57, 58, 59, 60));
}

TEST_P(QuantizedFullyConnectedOpTest,
       SimpleTestQuantizedOutputMultiplierGreaterThan1Uint8) {
  // real_multiplier = 2.
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_UINT8, {2, 10}, -127, 128},
      /*output=*/{TensorType_UINT8, {}, -63.5, 64},
      /*bias_type=*/TensorType_INT32,
      /*keep_num_dims =*/false, /*bool bias_tensor_optional =*/false,
      /*ActivationFunctionType activation_func =*/ActivationFunctionType_RELU,
      /*FullyConnectedOptionsWeightsFormat weights_format =*/
      FullyConnectedOptionsWeightsFormat_DEFAULT,
      /*input_size=*/-1, /*filter_type=*/TensorType_UINT8);

  m.SetWeights<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  24, 25, 26,  // first batch
                  58, 59, 60,  // second batch
              })));
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAre(175, 177, 179, 243, 245, 247));
}

TEST_P(QuantizedFullyConnectedOpTest,
       SimpleTestQuantizedOutputMultiplierGreaterThan1Int8) {
  // real_multiplier = 2.
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_INT8, {2, 10}, -127, 128},
      /*output=*/{TensorType_INT8, {}, -63.5, 64});

  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  24, 25, 26,  // first batch
                  58, 59, 60,  // second batch
              })));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAre(47, 49, 51, 115, 117, 119));
}

void SimpleTestQuantizedInt16OutputCase(
    TfLiteRegistration* registration, int input_depth, int output_depth,
    int batches, FullyConnectedOptionsWeightsFormat weights_format) {
  const uint8_t kWeightsZeroPoint = 128;
  const float kWeightsScale = 1.f / 128.f;
  const uint8_t kInputZeroPoint = 128;
  const float kInputScale = 1.f / 128.f;
  const float kInputMin = (0 - kInputZeroPoint) * kInputScale;
  const float kInputMax = (255 - kInputZeroPoint) * kInputScale;
  // Output ranges in [-8..8] encoded as int16
  const float kOutputScale = 8.f / 32768.f;
  const float kOutputMin = -32768 * kOutputScale;
  const float kOutputMax = 32767 * kOutputScale;

  QuantizedFullyConnectedOpModel m(
      registration, output_depth, batches,
      /*input=*/
      {TensorType_UINT8, {batches, input_depth}, kInputMin, kInputMax},
      /*output=*/{TensorType_INT16, {}, kOutputMin, kOutputMax},
      /*bias_type=*/TensorType_INT32,
      /*keep_num_dims=*/false,
      /*bias_tensor_optional=*/false,
      /*activation_func=*/ActivationFunctionType_NONE, weights_format,
      /*input_size=*/-1, /*filter_type=*/TensorType_UINT8);

  std::mt19937 random_engine;
  // Some compilers don't support uint8_t for uniform_distribution.
  std::uniform_int_distribution<uint32_t> weights_dist(
      0, std::numeric_limits<uint8_t>::max());

  std::vector<float> weights_data(input_depth * output_depth);
  for (auto& w : weights_data) {
    uint8_t q = static_cast<uint8_t>(weights_dist(random_engine));
    w = (q - kWeightsZeroPoint) * kWeightsScale;
  }

  // Based on weights_format, enforce any shape requirement for that format/path
  // and set the (possibly shuffled) weights.
  switch (weights_format) {
    case FullyConnectedOptionsWeightsFormat_DEFAULT:
      m.SetWeights<uint8_t>(weights_data);
      break;
    case FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8:
      // The shuffled path currently supports only a restrictive subset of
      // shapes, described by the following assertions:
      CHECK_EQ(input_depth % 16, 0);
      CHECK_EQ(output_depth % 4, 0);
      CHECK(batches == 1 || batches == 4);
      m.ShuffleAndSetWeights<uint8_t>(weights_data, input_depth, output_depth);
      break;
    default:
      LOG(FATAL) << "Unhandled weights format";
  }

  // Some compilers don't support uint8_t for uniform_distribution.
  std::uniform_int_distribution<uint32_t> input_dist(
      0, std::numeric_limits<uint8_t>::max());
  std::vector<float> input_data(input_depth * batches);
  for (auto& i : input_data) {
    uint8_t q = static_cast<uint8_t>(input_dist(random_engine));
    i = (q - kInputZeroPoint) * kInputScale;
  }

  std::vector<float> bias_data(output_depth);
  // As the output ranges in [-8, 8], it's reasonable to have bias values
  // in [-1, 1], this won't result in too much saturation.
  std::uniform_real_distribution<float> bias_dist(-1.f, 1.f);
  for (auto& b : bias_data) {
    b = bias_dist(random_engine);
  }

  m.SetBias(bias_data);
  m.SetInput<uint8_t>(input_data);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  std::vector<float> expected_output_data(output_depth * batches);
  for (int b = 0; b < batches; b++) {
    for (int o = 0; o < output_depth; o++) {
      float accum = bias_data[o];
      for (int i = 0; i < input_depth; i++) {
        accum +=
            input_data[b * input_depth + i] * weights_data[o * input_depth + i];
      }
      accum = std::min(accum, kOutputMax);
      accum = std::max(accum, kOutputMin);
      expected_output_data[b * output_depth + o] = accum;
    }
  }

  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear(expected_output_data, 3e-4f)));
}

TEST_P(QuantizedFullyConnectedOpTest,
       SimpleTestQuantizedInt16OutputDefaultWeights) {
  for (int input_depth : {1, 3, 10, 100}) {
    for (int output_depth : {1, 3, 10, 100}) {
      for (int batch : {1, 3, 10, 100}) {
        SimpleTestQuantizedInt16OutputCase(
            GetRegistration(), input_depth, output_depth, batch,
            FullyConnectedOptionsWeightsFormat_DEFAULT);
      }
    }
  }
}

TEST_P(QuantizedFullyConnectedOpTest,
       SimpleTestQuantizedInt16OutputShuffled4x16Int8Weights) {
  // The shuffled weights block shape is 4x16. The shape of the weights matrix
  // is: rows = output_depth, cols = input_depth. It must be a multiple of 4x16.
  // This means that output_depth must be a multiple of 4, and input_depth must
  // be a multiple of 16.
  for (int input_depth_numblocks : {1, 3}) {
    for (int output_depth_numblocks : {1, 3}) {
      int input_depth = 16 * input_depth_numblocks;
      int output_depth = 4 * output_depth_numblocks;
      // The fast shuffled path is currently supporting only batch sizes of 1
      // and 4. The idea is that the whole point of that path is to go as fast
      // as possible for small batch size, which requires fully specializing
      // it for each batch size, and for larger batch sizes the generic
      // gemmlowp-based implementation is fast enough.
      for (int batch : {1, 4}) {
        SimpleTestQuantizedInt16OutputCase(
            GetRegistration(), input_depth, output_depth, batch,
            FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8);
      }
    }
  }
}

TEST(HybridFullyConnectedOpTest, SimpleTestQuantizedUint8) {
  HybridFullyConnectedOpModel m(
      /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_FLOAT32, {2, 10}},
      /*weights=*/
      {TensorType_UINT8, {3, 10}, 0, 0, 10.0 / 127.0, 0});  // Hybrid

  m.SetWeights({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     24, 25, 26,  //
                                     58, 59, 60,  //
                                 },
                                 /*max_abs_err=*/1.3f)));
}

TEST(HybridFullyConnectedOpTest, SimpleTestQuantizedInt8) {
  HybridFullyConnectedOpModel m(
      /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_FLOAT32, {2, 10}},
      /*weights=*/{TensorType_INT8, {3, 10}, 0, 0, 10.0 / 127.0, 0});  // Hybrid

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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     24, 25, 26,  //
                                     58, 59, 60,  //
                                 },
                                 /*max_abs_err=*/1.3f)));
}

TEST(HybridFullyConnectedOpTest, SimpleTestQuantizedInt4) {
  HybridFullyConnectedOpModel m(
      /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_FLOAT32, {2, 10}},
      /*weights=*/{TensorType_INT4, {3, 10}, 0, 0, 1.0, 0});

  m.SetSignedWeights({
      1, 2, 3, 4, 5, 6, 7, 6, 5, 4,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 6, 5, 4,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 6, 5, 4,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     104, 105, 106,  //
                                     98, 99, 100,    //
                                 },
                                 /*max_abs_err=*/0.5f)));
}

TEST(HybridFullyConnectedOpTest, SimpleTestQuantizedInt8MultiThreaded) {
  for (int num_threads = 1; num_threads <= 4; ++num_threads) {
    HybridFullyConnectedOpModel m(
        /*units=*/3, /*batches=*/4,
        /*input=*/{TensorType_FLOAT32, {4, 10}},
        /*weights=*/
        {TensorType_INT8, {3, 10}, 0, 0, 10.0 / 127.0, 0},
        /*output=*/{TensorType_FLOAT32}, /*asymmetric_inputs=*/false,
        /*num_threads=*/num_threads);  // Hybrid

    m.SetSignedWeights({
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
    });
    m.SetBias({1, 2, 3});

    m.SetInput({
        1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
        1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
        1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 2
        1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 3
    });

    ASSERT_EQ(m.Invoke(), kTfLiteOk);

    EXPECT_THAT(m.GetOutputShape(), ElementsAre(4, 3));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                   {
                                       24, 25, 26,  //
                                       58, 59, 60,  //
                                       24, 25, 26,  //
                                       58, 59, 60,  //
                                   },
                                   /*max_abs_err=*/1.3f)));
  }
}

TEST(HybridAsymmetricInputFullyConnectedOpTest, SimpleTestQuantizedUint8) {
  HybridFullyConnectedOpModel m(
      /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_FLOAT32, {2, 10}},
      /*weights=*/
      {TensorType_UINT8, {3, 10}, 0, 0, 10.0 / 127.0, 0}, {TensorType_FLOAT32},
      /*asymmetric_quantize_input*/ true);  // Hybrid asymmetric

  m.SetWeights({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     24, 25, 26,  //
                                     58, 59, 60,  //
                                 },
                                 /*max_abs_err=*/0.64f)));
}

TEST(HybridAsymmetricInputFullyConnectedOpTest, SimpleTestQuantizedInt8) {
  HybridFullyConnectedOpModel m(
      /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_FLOAT32, {2, 10}},
      /*weights=*/{TensorType_INT8, {3, 10}, 0, 0, 10.0 / 127.0, 0},
      {TensorType_FLOAT32},
      /*asymmetric_quantize_input*/ true);

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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     24, 25, 26,  //
                                     58, 59, 60,  //
                                 },
                                 /*max_abs_err=*/1.3f)));
}

TEST(HybridAsymmetricInputPerChannelWeightsFullyConnectedOpTest,
     SimpleTestQuantizedPerChannelInt8) {
  HybridFullyConnectedOpModel m(
      /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_FLOAT32, {2, 10}},
      /*weights=*/
      {TensorType_INT8,
       {3, 10},
       0,
       0,
       0.0f,
       0,
       true,
       {10.0 / 127.0, 20.0 / 127.0, 30.0 / 127.0},
       {0, 0, 0}},
      {TensorType_FLOAT32},
      /*asymmetric_quantize_input*/ true);

  m.SetSignedPerChannelWeights({
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  // u = 0
      11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  // u = 1
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     24, 195, 366,  //
                                     58, 251, 441,  //
                                 },
                                 /*max_abs_err=*/1.3f)));
}

TEST(HybridAsymmetricInputPerChannelWeightsFullyConnectedOpTest,
     SimpleTestQuantizedPerChannelInt4) {
  HybridFullyConnectedOpModel m(
      /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_FLOAT32, {2, 10}},
      /*weights=*/
      {TensorType_INT4,
       {3, 10},
       0,
       0,
       0.0f,
       0,
       true,
       {10.0 / 7.0, 20.0 / 7.0, 30.0 / 7.0},
       {0, 0, 0}},
      {TensorType_FLOAT32},
      /*asymmetric_quantize_input*/ true);

  m.SetSignedPerChannelWeights({
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  // u = 0
      11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  // u = 1
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     35, 188, 368,  //
                                     53, 275, 430,  //
                                 },
                                 /*max_abs_err=*/0.5f)));
}

TEST_P(FloatFullyConnectedOpTest, SimpleTest4DInput) {
  // Note that it is not required that the first dimension be the number of
  // batches. All we care is that the input can be evenly distributed in
  // batches. In this case, we need the input to have multiples of '2'.
  FloatFullyConnectedOpModel m(GetRegistration(),
                               /*units=*/3, /*batches=*/2,
                               /*input=*/{TensorType_FLOAT32, {4, 1, 5, 1}});
  m.SetWeights({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // first batch
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // second batch
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 24, 25, 26,  // first batch
                                 58, 59, 60,  // second batch
                             }));
}

TEST_P(FloatFullyConnectedOpTest, SimpleTest4DInput4DOutput) {
  // Note that it is not required that the first dimension be the number of
  // batches. All we care is that the input can be evenly distributed in
  // batches. In this case, we need the input to have multiples of '2'.
  FloatFullyConnectedOpModel m(GetRegistration(),
                               /*units=*/3, /*batches=*/2,
                               /*input=*/{TensorType_FLOAT32, {1, 2, 1, 10}},
                               /*output=*/{TensorType_FLOAT32},
                               /*bias_type=*/TensorType_FLOAT32,
                               /*keep_num_dims=*/true);
  m.SetWeights({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // first batch
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // second batch
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 2, 1, 3));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 24, 25, 26,  // first batch
                                 58, 59, 60,  // second batch
                             }));
}

#if GTEST_HAS_DEATH_TEST
TEST_P(FloatFullyConnectedOpTest, SimpleTest4DInputInvalidShape) {
  // Note that it is not required that the first dimension be the number of
  // batches. But it is required that the last dimension is the 'input_dim'.
  //
  // For this particular test, it is required for the output to be reformattable
  // into a shape of form {4, 1, 5, ?} but since the output size (the product of
  // output dimensions: units times batches) is 6, this is not possible.
  EXPECT_DEATH(FloatFullyConnectedOpModel m(
                   GetRegistration(), /*units=*/3, /*batches=*/2,
                   /*input=*/{TensorType_FLOAT32, {4, 1, 5, 1}},
                   /*output=*/{TensorType_FLOAT32},
                   /*bias_type=*/TensorType_FLOAT32,
                   /*keep_num_dims=*/true),
               "Cannot allocate tensors");
}
#endif

TEST_P(QuantizedFullyConnectedOpTest, SimpleTest4dInputQuantizedUint8) {
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_UINT8, {4, 1, 5, 1}, -63.5, 64},
      /*output=*/{TensorType_UINT8, {}, -127, 128},
      /*bias_type=*/TensorType_INT32, /*keep_num_dims =*/false,
      /*bool bias_tensor_optional =*/false,
      /*ActivationFunctionType activation_func =*/ActivationFunctionType_RELU,
      /*FullyConnectedOptionsWeightsFormat weights_format =*/
      FullyConnectedOptionsWeightsFormat_DEFAULT,
      /*input_size=*/-1, /*filter_type=*/TensorType_UINT8);

  // input_product_scale < output_scale was not true.
  m.SetWeights<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  24, 25, 26,  //
                  58, 59, 60,  //
              })));
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAre(151, 152, 153, 185, 186, 187));
}

TEST_P(QuantizedFullyConnectedOpTest,
       SimpleTest4dInputQuantizedOutputMultiplierGreaterThan1Uint8) {
  // real_multiplier = 2.
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_UINT8, {4, 1, 5, 1}, -127, 128},
      /*output=*/{TensorType_UINT8, {}, -63.5, 64},
      /*bias_type=*/TensorType_INT32, /*keep_num_dims =*/false,
      /*bool bias_tensor_optional =*/false,
      /*ActivationFunctionType activation_func =*/ActivationFunctionType_RELU,
      /*FullyConnectedOptionsWeightsFormat weights_format =*/
      FullyConnectedOptionsWeightsFormat_DEFAULT,
      /*input_size=*/-1, /*filter_type=*/TensorType_UINT8);

  m.SetWeights<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  24, 25, 26,  // first batch
                  58, 59, 60,  // second batch
              })));
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAre(175, 177, 179, 243, 245, 247));
}

INSTANTIATE_TEST_SUITE_P(
    FloatFullyConnectedOpTest, FloatFullyConnectedOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)));

INSTANTIATE_TEST_SUITE_P(
    QuantizedFullyConnectedOpTest, QuantizedFullyConnectedOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMapNoPie)));

// TODO(ahentz): Reconsider this test. Having arbitrary weights makes it hard
// to debug errors and doesn't necessarily test all the important details.
TEST_P(FloatFullyConnectedOpTest, BlackBoxTest) {
  FloatFullyConnectedOpModel m(GetRegistration(), /*units=*/16, /*batches=*/2,
                               /*input=*/{TensorType_FLOAT32, {2, 8}});
  m.SetWeights(
      {0.091327,  0.103366,  -0.316505, -0.083120, 0.149366,  -0.196636,
       -0.123672, 0.062800,  0.063031,  0.191670,  -0.062001, -0.061504,
       -0.275581, 0.059388,  -0.118497, -0.079224, 0.109758,  0.008307,
       -0.062657, -0.060962, -0.049782, -0.106719, -0.319482, -0.103650,
       0.266455,  0.051517,  -0.123448, 0.322464,  0.043282,  -0.173782,
       -0.190381, 0.002013,  0.096086,  0.131157,  0.031164,  0.100638,
       -0.312191, -0.080923, -0.101318, -0.116614, 0.142238,  0.086540,
       -0.139154, 0.174268,  -0.073161, 0.080072,  0.006874,  0.229382,
       -0.104321, -0.176035, -0.208587, -0.001019, -0.162032, 0.080824,
       -0.025021, 0.074460,  -0.252595, -0.161750, -0.136403, 0.008308,
       0.005710,  0.096600,  0.289839,  0.218816,  -0.304651, -0.070958,
       0.054598,  0.147113,  -0.139112, -0.072798, -0.163335, -0.167863,
       -0.128762, -0.035780, 0.117262,  0.017177,  0.263335,  -0.176612,
       0.262961,  -0.093654, -0.339283, 0.333071,  0.180827,  0.287583,
       0.066350,  -0.197947, -0.114449, -0.236035, 0.103532,  -0.034284,
       0.093299,  -0.145361, 0.054001,  0.250570,  0.157010,  -0.143480,
       -0.139061, -0.048873, 0.067557,  0.139038,  0.324106,  0.227041,
       0.037793,  -0.225747, -0.241619, 0.357835,  0.135762,  -0.306764,
       -0.125982, 0.091916,  0.266587,  0.030135,  0.265148,  0.141627,
       0.020120,  0.083815,  -0.124556, -0.100124, -0.048159, 0.181172,
       0.302309,  -0.041084, 0.146334,  -0.061511, -0.232605, 0.281324,
       0.145408,  -0.221897});
  m.SetBias({-0.160594, 0.205770, -0.078307, -0.077984, 0.001937, 0.015860,
             0.036810, 0.012346, 0.001028, 0.038551, 0.075415, 0.020804,
             0.048478, -0.032270, 0.175688, -0.085662});

  const int input_sequence_size = sizeof(fully_connected_input) /
                                  sizeof(float) /
                                  (m.input_size() * m.num_batches());
  for (int i = 0; i < input_sequence_size; i++) {
    // TODO(ahentz): This is what the original test was doing: two equal
    // batches per invocation. We could instead use two different batches.
    float* batch_start = fully_connected_input + i * m.input_size();
    float* batch_end = batch_start + m.input_size();
    m.SetInput(0, batch_start, batch_end);
    m.SetInput(m.input_size(), batch_start, batch_end);

    ASSERT_EQ(m.Invoke(), kTfLiteOk);

    float* golden_start = fully_connected_golden_output + i * m.num_units();
    float* golden_end = golden_start + m.num_units();
    std::vector<float> expected;
    expected.insert(expected.end(), golden_start, golden_end);
    expected.insert(expected.end(), golden_start, golden_end);

    EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(expected)));
  }
}

template <typename T>
class SparseFullyConnectedOpModel : public SingleOpModel {
 public:
  SparseFullyConnectedOpModel(TfLiteRegistration* registration, int units,
                              int batches, const TensorData& input,
                              const TensorData& weights,
                              const std::vector<T>& weights_data,
                              const TensorData& output = {TensorType_FLOAT32},
                              bool bias_tensor_optional = false,
                              int num_threads = 1,
                              bool symmetric_quantize_weights = false,
                              bool asymmetric_quantize_inputs = false)
      : batches_(batches), units_(units) {
    int total_input_size = 1;
    for (size_t i = 0; i < input.shape.size(); ++i) {
      total_input_size *= input.shape[i];
    }
    input_size_ = total_input_size / batches_;

    input_ = AddInput(input);
    weights_ =
        AddConstSparseInput(weights, weights_data, symmetric_quantize_weights);

    if (bias_tensor_optional) {
      bias_ = AddNullInput();
    } else if (input.type == TensorType_INT8) {
      // This is a quantized version. The scale of 'bias' depends on the scales
      // of input and filter.
      float bias_scale = GetScale(input_);
      if (weights.per_channel_quantization &&
          !weights.per_channel_quantization_scales.empty()) {
        bias_scale *= weights.per_channel_quantization_scales[0];
      } else {
        bias_scale *= GetScale(weights_);
      }
      TensorData bias = {TensorType_INT32, {units_}, 0, 0, bias_scale};
      bias_ = AddInput(bias);
    } else {
      bias_ = AddInput({input.type, {units_}});
    }

    output_ = AddOutput(output);

    SetBuiltinOp(
        BuiltinOperator_FULLY_CONNECTED, BuiltinOptions_FullyConnectedOptions,
        CreateFullyConnectedOptions(builder_, ActivationFunctionType_RELU,
                                    FullyConnectedOptionsWeightsFormat_DEFAULT,
                                    /*keep_num_dims=*/false,
                                    asymmetric_quantize_inputs)
            .Union());
    resolver_ = std::make_unique<SingleOpResolver>(
        BuiltinOperator_FULLY_CONNECTED, registration);
    std::vector<std::vector<int>> inputs = {GetShape(input_),
                                            GetShape(weights_)};
    inputs.push_back((bias_ == kTfLiteOptionalTensor) ? std::vector<int>()
                                                      : GetShape(bias_));
    BuildInterpreter(inputs, num_threads, /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false);
  }
  void SetBias(const std::vector<T>& data) { PopulateTensor(bias_, data); }
  void SetInput(const std::vector<T>& data) { PopulateTensor(input_, data); }
  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
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

class SparseFullyConnectedOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMapNoPie;
  }
};

struct SparseTestParam {
  std::string kernel_tag;
  bool asymmetric_quantize_input;
};

class SparseHybridFullyConnectedOpTest
    : public ::testing::TestWithParam<SparseTestParam> {
 public:
  static std::vector<string> GetKernelTags(
      const std::map<string, TfLiteRegistration*>& kernel_map) {
    std::vector<string> tags;
    tags.reserve(kernel_map.size());
    for (const auto& it : kernel_map) {
      tags.push_back(it.first);
    }
    return tags;
  }

 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() {
    return *kKernelMapNoPie;
  }
  TfLiteRegistration* GetRegistration() {
    return GetKernelMap().at(GetParam().kernel_tag);
  }
};

TEST_P(SparseFullyConnectedOpTest, SimpleTest) {
  std::initializer_list<float> weight_data = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  };
  TensorData weight = {};
  weight.type = TensorType_FLOAT32;
  weight.shape = {3, 10};
  weight.traversal_order = {0, 1};
  weight.format = {kTfLiteDimDense, kTfLiteDimSparseCSR};
  SparseFullyConnectedOpModel<float> m(
      GetRegistration(), /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_FLOAT32, {2, 10}}, weight, weight_data);
  m.SetBias({1, 2, 3});

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(m.GetOutput(), ElementsAre(24, 25, 26, 58, 59, 60));
}

TEST_P(SparseFullyConnectedOpTest, SimpleTestNoBias) {
  std::initializer_list<float> weight_data = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  };
  TensorData weight = {};
  weight.type = TensorType_FLOAT32;
  weight.shape = {3, 10};
  weight.traversal_order = {0, 1};
  weight.format = {kTfLiteDimDense, kTfLiteDimSparseCSR};
  SparseFullyConnectedOpModel<float> m(
      GetRegistration(), /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_FLOAT32, {2, 10}}, weight, weight_data,
      /*output=*/{TensorType_FLOAT32},
      /*bias_tensor_optional=*/true);

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(m.GetOutput(), ElementsAre(23, 23, 23, 57, 57, 57));
}

TEST_P(SparseFullyConnectedOpTest, SimpleTest2) {
  std::initializer_list<float> weight_data = {
      2, 4  // u = 0
  };
  TensorData weight = {};
  weight.type = TensorType_FLOAT32;
  weight.shape = {1, 2};
  weight.traversal_order = {0, 1};
  weight.format = {kTfLiteDimDense, kTfLiteDimSparseCSR};
  SparseFullyConnectedOpModel<float> m(
      GetRegistration(), /*units=*/1, /*batches=*/2,
      /*input=*/{TensorType_FLOAT32, {2, 2}}, weight, weight_data);
  m.SetBias({1});

  m.SetInput({
      1, 2,  // b = 0
      2, 1   // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 1));
  EXPECT_THAT(m.GetOutput(), ElementsAre(11, 9));
}

TEST_P(SparseFullyConnectedOpTest, Simple1x4Test) {
  std::initializer_list<float> weight_data = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  // u = 2
  };
  TensorData weight = {};
  weight.type = TensorType_FLOAT32;
  weight.shape = {3, 12};
  weight.traversal_order = {0, 1, 2};
  weight.format = {kTfLiteDimDense, kTfLiteDimSparseCSR};
  weight.block_map = {1};
  weight.block_size = {4};
  SparseFullyConnectedOpModel<float> m(GetRegistration(),
                                       /*units=*/3, /*batches=*/2,
                                       /*input=*/{TensorType_FLOAT32, {2, 12}},
                                       weight, weight_data);
  m.SetBias({1, 2, 3});

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 11,  12,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10, -11, 12,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(m.GetOutput(), ElementsAre(289, 290, 291, 81, 82, 83));
}

TEST_P(SparseFullyConnectedOpTest, Simple1x4TestNoBias) {
  std::initializer_list<float> weight_data = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  // u = 2
  };
  TensorData weight = {};
  weight.type = TensorType_FLOAT32;
  weight.shape = {3, 12};
  weight.traversal_order = {0, 1, 2};
  weight.format = {kTfLiteDimDense, kTfLiteDimSparseCSR};
  weight.block_map = {1};
  weight.block_size = {4};
  SparseFullyConnectedOpModel<float> m(GetRegistration(),
                                       /*units=*/3, /*batches=*/2,
                                       /*input=*/{TensorType_FLOAT32, {2, 12}},
                                       weight, weight_data,
                                       /*output=*/{TensorType_FLOAT32},
                                       /*bias_tensor_optional=*/true);
  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 11,  12,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10, -11, 12,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(m.GetOutput(), ElementsAre(288, 288, 288, 80, 80, 80));
}

TEST_P(SparseFullyConnectedOpTest, Simple1x4TestMultiThreaded) {
  std::initializer_list<float> weight_data = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  // u = 2
  };
  TensorData weight = {};
  weight.type = TensorType_FLOAT32;
  weight.shape = {3, 12};
  weight.traversal_order = {0, 1, 2};
  weight.format = {kTfLiteDimDense, kTfLiteDimSparseCSR};
  weight.block_map = {1};
  weight.block_size = {4};
  for (int num_threads = 1; num_threads <= 4; num_threads++) {
    SparseFullyConnectedOpModel<float> m(
        GetRegistration(),
        /*units=*/3, /*batches=*/2,
        /*input=*/{TensorType_FLOAT32, {2, 12}}, weight, weight_data,
        /*output=*/{TensorType_FLOAT32},
        /*bias_tensor_optional=*/false, /*num_threads=*/num_threads);
    m.SetBias({1, 2, 3});

    m.SetInput({
        1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 11,  12,  // b = 0
        1, 2, 3, 4, 5, 6, 7, -8, 9,  -10, -11, 12,  // b = 1
    });

    ASSERT_EQ(m.Invoke(), kTfLiteOk);

    EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 3));
    EXPECT_THAT(m.GetOutput(), ElementsAre(289, 290, 291, 81, 82, 83));
  }
}

TEST_P(SparseFullyConnectedOpTest, Simple1x4TestMultiThreadedMoreBatches) {
  std::initializer_list<float> weight_data = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  // u = 2
  };
  TensorData weight = {};
  weight.type = TensorType_FLOAT32;
  weight.shape = {3, 12};
  weight.traversal_order = {0, 1, 2};
  weight.format = {kTfLiteDimDense, kTfLiteDimSparseCSR};
  weight.block_map = {1};
  weight.block_size = {4};
  for (int num_threads = 1; num_threads <= 4; num_threads++) {
    SparseFullyConnectedOpModel<float> m(
        GetRegistration(),
        /*units=*/3, /*batches=*/6,
        /*input=*/{TensorType_FLOAT32, {6, 12}}, weight, weight_data,
        /*output=*/{TensorType_FLOAT32},
        /*bias_tensor_optional=*/false, /*num_threads=*/num_threads);
    m.SetBias({1, 2, 3});

    m.SetInput({
        1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 11,  12,  // b = 0
        1, 2, 3, 4, 5, 6, 7, -8, 9,  -10, -11, 12,  // b = 1
        1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 11,  12,  // b = 2
        1, 2, 3, 4, 5, 6, 7, -8, 9,  -10, -11, 12,  // b = 3
        1, 2, 3, 4, 5, 6, 7, 8,  -9, -10, 11,  12,  // b = 4
        1, 2, 3, 4, 5, 6, 7, -8, 9,  -10, -11, 12,  // b = 5
    });

    ASSERT_EQ(m.Invoke(), kTfLiteOk);

    EXPECT_THAT(m.GetOutputShape(), ElementsAre(6, 3));
    EXPECT_THAT(m.GetOutput(), ElementsAre(289, 290, 291,  // b = 0
                                           81, 82, 83,     // b = 1
                                           289, 290, 291,  // b = 2
                                           81, 82, 83,     // b = 3
                                           289, 290, 291,  // b = 4
                                           81, 82, 83      // b = 5
                                           ));
  }
}

TEST_P(SparseHybridFullyConnectedOpTest, SparseHybrid1x16Test) {
  std::initializer_list<float> weight_data = {
      /* 1st row */
      1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12, 13.13,
      14.14, 15.15, 16.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9,
      10.1, 11.11, 12.12, 13.13, 14.14, 15.15, 16.16,
      /* 2nd row */
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, -1.1, -2.2, -3.3, -4.4, -5.5, -6.6, -7.7, -8.8, -9.9, -10.1, -11.11,
      -12.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      /* 3rd row */
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 1.1, -2.2, 3.3, -4.4, 5.5, -6.6, 7.7, -8.8, 9.9, -10.1, 11.11,
      -12.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      /* 4th row */
      -1.1, 2.2, -3.3, 4.4, -5.5, 6.6, -7.7, 8.8, -9.9, 10.1, -11.11, 12.12,
      -13.13, 14.14, -15.15, 16.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1, 2.2, -3.3, 4.4, -5.5, 6.6, -7.7,
      8.8, -9.9, 10.1, -11.11, 12.12, 0.0, 0.0, 0.0, 0.0};
  TensorData weight = {};
  weight.type = TensorType_FLOAT32;
  weight.shape = {4, 48};
  weight.traversal_order = {0, 1, 2};
  weight.format = {kTfLiteDimDense, kTfLiteDimSparseCSR};
  weight.block_map = {1};
  weight.block_size = {16};
  SparseFullyConnectedOpModel<float> m(
      GetRegistration(),
      /*units=*/4, /*batches=*/2,
      /*input=*/{TensorType_FLOAT32, {2, 48}}, weight, weight_data,
      /*output=*/{TensorType_FLOAT32},
      /*bias_tensor_optional=*/false, /*num_threads)=*/1,
      /*symmetric_quantize_weights=*/true,
      /*asymmetric_quantize_inputs=*/GetParam().asymmetric_quantize_input);
  m.SetBias({1, 2, 3, 4});
  m.SetInput({
      1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0,
      1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0,
      1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0,
      1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0,
      1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0,  // b = 0
      2.5,  0.0,  -2.1, 0.0,  3.0,  0.0,  -1.3, 0.0,  1.3,  0.0,
      -1.1, 0.0,  2.0,  0.0,  -1.7, 0.0,  1.9,  0.0,  -1.5, 0.0,
      0.5,  0.0,  -0.7, 0.0,  0.8,  0.0,  -0.3, 0.0,  2.8,  0.0,
      -2.8, 0.0,  1.1,  -2.3, 1.9,  -1.9, 2.1,  -0.5, 2.4,  -0.1,
      1.0,  -2.5, 0.7,  -1.9, 0.2,  0.1,  0.2,  0.3,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 4));
  std::vector<float> expected = {0,      7.4715, 85.8359, 0,
                                 5.9655, 3.0520, 1.9480,  0};
  if (GetParam().asymmetric_quantize_input) {
    expected = {0, 7.4500, 85.5111, 0, 5.9750, 2.8856, 2.1144, 0};
  }
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(expected, 1e-3)));
}

TEST_P(SparseHybridFullyConnectedOpTest, SparseHybrid1x16TestMultiThreaded) {
  std::initializer_list<float> weight_data = {
      /* 1st row */
      1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12, 13.13,
      14.14, 15.15, 16.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9,
      10.1, 11.11, 12.12, 13.13, 14.14, 15.15, 16.16,
      /* 2nd row */
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, -1.1, -2.2, -3.3, -4.4, -5.5, -6.6, -7.7, -8.8, -9.9, -10.1, -11.11,
      -12.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      /* 3rd row */
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 1.1, -2.2, 3.3, -4.4, 5.5, -6.6, 7.7, -8.8, 9.9, -10.1, 11.11,
      -12.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      /* 4th row */
      -1.1, 2.2, -3.3, 4.4, -5.5, 6.6, -7.7, 8.8, -9.9, 10.1, -11.11, 12.12,
      -13.13, 14.14, -15.15, 16.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1, 2.2, -3.3, 4.4, -5.5, 6.6, -7.7,
      8.8, -9.9, 10.1, -11.11, 12.12, 0.0, 0.0, 0.0, 0.0};
  TensorData weight = {};
  weight.type = TensorType_FLOAT32;
  weight.shape = {4, 48};
  weight.traversal_order = {0, 1, 2};
  weight.format = {kTfLiteDimDense, kTfLiteDimSparseCSR};
  weight.block_map = {1};
  weight.block_size = {16};
  for (int num_threads = 1; num_threads <= 4; ++num_threads) {
    SparseFullyConnectedOpModel<float> m(
        GetRegistration(),
        /*units=*/4, /*batches=*/4,
        /*input=*/{TensorType_FLOAT32, {4, 48}}, weight, weight_data,
        /*output=*/{TensorType_FLOAT32},
        /*bias_tensor_optional=*/false, /*num_threads=*/num_threads,
        /*symmetric_quantize_weights=*/true,
        /*asymmetric_quantize_inputs=*/GetParam().asymmetric_quantize_input);
    m.SetBias({1, 2, 3, 4});
    m.SetInput({
        1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0,
        1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0,
        1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0,
        1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0,
        1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0,  // b = 0
        2.5,  0.0,  -2.1, 0.0,  3.0,  0.0,  -1.3, 0.0,  1.3,  0.0,
        -1.1, 0.0,  2.0,  0.0,  -1.7, 0.0,  1.9,  0.0,  -1.5, 0.0,
        0.5,  0.0,  -0.7, 0.0,  0.8,  0.0,  -0.3, 0.0,  2.8,  0.0,
        -2.8, 0.0,  1.1,  -2.3, 1.9,  -1.9, 2.1,  -0.5, 2.4,  -0.1,
        1.0,  -2.5, 0.7,  -1.9, 0.2,  0.1,  0.2,  0.3,  // b = 1
        1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0,
        1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0,
        1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0,
        1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0,
        1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0, 1.0,  -1.0,  // b = 2
        2.5,  0.0,  -2.1, 0.0,  3.0,  0.0,  -1.3, 0.0,  1.3,  0.0,
        -1.1, 0.0,  2.0,  0.0,  -1.7, 0.0,  1.9,  0.0,  -1.5, 0.0,
        0.5,  0.0,  -0.7, 0.0,  0.8,  0.0,  -0.3, 0.0,  2.8,  0.0,
        -2.8, 0.0,  1.1,  -2.3, 1.9,  -1.9, 2.1,  -0.5, 2.4,  -0.1,
        1.0,  -2.5, 0.7,  -1.9, 0.2,  0.1,  0.2,  0.3,  // b = 3
    });

    ASSERT_EQ(m.Invoke(), kTfLiteOk);

    EXPECT_THAT(m.GetOutputShape(), ElementsAre(4, 4));
    std::vector<float> expected = {
        0, 7.4715, 85.8359, 0, 5.9655, 3.0520, 1.9480, 0,
        0, 7.4715, 85.8359, 0, 5.9655, 3.0520, 1.9480, 0};
    if (GetParam().asymmetric_quantize_input) {
      expected = {
          0, 7.4500, 85.5111, 0, 5.9750, 2.8856, 2.1144, 0,
          0, 7.4500, 85.5111, 0, 5.9750, 2.8856, 2.1144, 0,
      };
    }
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray(ArrayFloatNear(expected, 1e-3)));
  }
}

TEST_P(SparseHybridFullyConnectedOpTest, SparseHybrid1x16PerChannelTest) {
  std::vector<float> weight_data = {
      1,  2,  3,  4,  -1, -2, -3, -4, 1,  2,  3,  4, -4, -3, -2, -1,  // u = 0
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,   // u = 1
      -1, -2, -3, -4, 4,  3,  2,  1,  -1, -2, -3, 4, 1,  2,  3,  4,   // u = 2
  };
  TensorData weight = {TensorType_INT8,
                       {3, 16},
                       0.0,
                       0.0,
                       0.0,
                       0,
                       true,
                       {4.0 / 127.0, 1.0 / 127.0, 4.0 / 127.0},
                       {0, 0, 0}};
  weight.traversal_order = {0, 1, 2};
  weight.format = {kTfLiteDimDense, kTfLiteDimSparseCSR};
  weight.block_map = {1};
  weight.block_size = {16};
  SparseFullyConnectedOpModel<float> m(
      GetRegistration(),
      /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_FLOAT32, {2, 16}, 0, 0, 1}, weight, weight_data,
      /*output=*/{TensorType_FLOAT32, {}, 0, 0, 1});

  m.SetBias({1, 2, 3});
  m.SetInput({
      1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,  // b = 0
      4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {10.9061, 2, 25.0938, 0, 2, 20.9691}, 1e-3)));
}
// TODO(b/148391360): Add tests for unsupported sparsity format.
// TEST_P(SparseFullyConnectedOpTest, TestUnsupportedSparsityFormat)

INSTANTIATE_TEST_SUITE_P(
    SparseFullyConnectedOpTest, SparseFullyConnectedOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMapNoPie)));

std::vector<SparseTestParam> GenerateSparseTestParam(
    std::vector<std::string> kernel_tags) {
  std::vector<SparseTestParam> test_params;
  for (const std::string& kernel_tag : kernel_tags) {
    test_params.push_back({kernel_tag, false});
    test_params.push_back({kernel_tag, true});
  }
  return test_params;
}

INSTANTIATE_TEST_SUITE_P(SparseHybridFullyConnectedOpTest,
                         SparseHybridFullyConnectedOpTest,
                         ::testing::ValuesIn(GenerateSparseTestParam(
                             SingleOpTest::GetKernelTags(*kKernelMapNoPie))));

class SparseQuantizedFullyConnectedOpModel
    : public SparseFullyConnectedOpModel<float> {
 public:
  using SparseFullyConnectedOpModel::SparseFullyConnectedOpModel;
  void SetBias(const std::vector<float>& data) {
    QuantizeAndPopulate<int32_t>(bias_, data);
  }
  void SetInput(const std::vector<float>& data) {
    QuantizeAndPopulate<int8_t>(input_, data);
  }
  std::vector<int8_t> GetOutput() { return ExtractVector<int8_t>(output_); }
};

class SparseQuantizedFullyConnectedOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMapNoPie;
  }
};

TEST_P(SparseQuantizedFullyConnectedOpTest, Simple1x16Test) {
  std::vector<float> weight_data = {
      1,  2,  3,  4,  -1, -2, -3, -4, 1,  2,  3,  4, -4, -3, -2, -1,  // u = 0
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,   // u = 1
      -1, -2, -3, -4, 4,  3,  2,  1,  -1, -2, -3, 4, 1,  2,  3,  4,   // u = 2
  };
  TensorData weight = {TensorType_INT8, {3, 16}, 0, 0, 1};
  weight.traversal_order = {0, 1, 2};
  weight.format = {kTfLiteDimDense, kTfLiteDimSparseCSR};
  weight.block_map = {1};
  weight.block_size = {16};
  SparseQuantizedFullyConnectedOpModel m(
      GetRegistration(),
      /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_INT8, {2, 16}, 0, 0, 1}, weight, weight_data,
      /*output=*/{TensorType_INT8, {}, 0, 0, 1});

  m.SetBias({1, 2, 3});
  m.SetInput({
      1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,  // b = 0
      4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(m.GetOutput(), ElementsAre(11, 2, 25, 0, 2, 21));
}

TEST_P(SparseQuantizedFullyConnectedOpTest, Simple1x16TestNoBias) {
  std::vector<float> weight_data = {
      1,  2,  3,  4,  -1, -2, -3, -4, 1,  2,  3,  4, -4, -3, -2, -1,  // u = 0
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,   // u = 1
      -1, -2, -3, -4, 4,  3,  2,  1,  -1, -2, -3, 4, 1,  2,  3,  4,   // u = 2
  };
  TensorData weight = {TensorType_INT8, {3, 16}, 0, 0, 1};
  weight.traversal_order = {0, 1, 2};
  weight.format = {kTfLiteDimDense, kTfLiteDimSparseCSR};
  weight.block_map = {1};
  weight.block_size = {16};
  SparseQuantizedFullyConnectedOpModel m(
      GetRegistration(),
      /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_INT8, {2, 16}, 0, 0, 1}, weight, weight_data,
      /*output=*/{TensorType_INT8, {}, 0, 0, 1}, /*bias_tensor_optional=*/true);

  m.SetInput({
      1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,  // b = 0
      4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(m.GetOutput(), ElementsAre(10, 0, 22, 0, 0, 18));
}

TEST_P(SparseQuantizedFullyConnectedOpTest, Simple1x16TestScaledInputOutput) {
  std::initializer_list<float> weight_data = {
      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      0,     0,     0,     0,     0,     0,     0,     0,  // u = 0
      0.28,  0.27,  0.40,  0.38,  -0.16, -0.14, -0.12, 0.03,  0.11,  0.22,
      0.02,  0.27,  0.22,  -0.39, 0.09,  -0.27, 0,     0,     0,     0,
      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      0,     0,     0,     0,     0,     0,     0,     0,  // u = 1
      0.06,  0.43,  -0.03, -0.30, -0.09, 0.49,  0.11,  0.24,  -0.21, 0.14,
      -0.18, 0.84,  0.10,  -0.20, -0.51, -0.12, 0.11,  0.02,  -0.09, -0.01,
      -0.31, 0.28,  -0.08, 0.32,  0.77,  0.69,  0.45,  -0.20, 0.21,  -0.07,
      -0.46, -0.20, 0,     0,     0,     0,     0,     0,     0,     0,
      0,     0,     0,     0,     0,     0,     0,     0,  // u = 2
  };
  TensorData weight = {TensorType_INT8, {3, 48}, 0, 0, 0.014362592250108719};
  weight.traversal_order = {0, 1, 2};
  weight.format = {kTfLiteDimDense, kTfLiteDimSparseCSR};
  weight.block_map = {1};
  weight.block_size = {16};
  SparseQuantizedFullyConnectedOpModel m(
      GetRegistration(),
      /*units=*/3, /*batches=*/1,
      /*input=*/{TensorType_INT8, {1, 48}, 0, 0, 0.01739450730383396, -128},
      weight, weight_data,
      /*output=*/{TensorType_INT8, {}, 0, 0, 0.08671142160892487, -52});
  m.SetBias({-0.21742193, -0.38303897, -0.2735016});
  m.SetInput(
      {0.15919347, 0.7385435,  0.01092399, 2.1284404,  0.39123753, 0.01069902,
       0.6752592,  0.15486322, 0.,         0.,         0.16048427, 0.33702788,
       0.,         1.1263783,  0.,         0.,         0.,         0.,
       0.5067856,  0.,         0.,         0.,         0.01031927, 0.,
       0.,         0.07268289, 0.02804407, 0.710703,   0.35505712, 0.15339729,
       0.,         0.,         0.5485122,  0.10860074, 0.01710763, 0.08116849,
       0.05225316, 0.03152719, 0.8149394,  0.6554623,  0.0311714,  0.02122466,
       0.995122,   0.06201557, 0.16699032, 0.,         0.,         0.06638951});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput(), ElementsAre(-52, -50, -52));
}

TEST_P(SparseQuantizedFullyConnectedOpTest,
       Simple1x16PerChannelQuantizationTest) {
  std::vector<float> weight_data = {
      1,  2,  3,  4,  -1, -2, -3, -4, 1,  2,  3,  4, -4, -3, -2, -1,  // u = 0
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,   // u = 1
      -1, -2, -3, -4, 4,  3,  2,  1,  -1, -2, -3, 4, 1,  2,  3,  4,   // u = 2
  };
  TensorData weight = {TensorType_INT8,
                       {3, 16},
                       0.0,
                       0.0,
                       0.0,
                       0,
                       true,
                       {4.0 / 127.0, 1.0 / 127.0, 4.0 / 127.0},
                       {0, 0, 0}};
  weight.traversal_order = {0, 1, 2};
  weight.format = {kTfLiteDimDense, kTfLiteDimSparseCSR};
  weight.block_map = {1};
  weight.block_size = {16};
  SparseQuantizedFullyConnectedOpModel m(
      GetRegistration(),
      /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_INT8, {2, 16}, 0, 0, 1}, weight, weight_data,
      /*output=*/{TensorType_INT8, {}, 0, 0, 1});

  m.SetBias({1, 2, 3});
  m.SetInput({
      1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,  // b = 0
      4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1,  // b = 1
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(m.GetOutput(), ElementsAre(11, 1, 25, 0, 1, 21));
}

INSTANTIATE_TEST_SUITE_P(
    SparseQuantizedFullyConnectedOpTest, SparseQuantizedFullyConnectedOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMapNoPie)));

}  // namespace
}  // namespace tflite
