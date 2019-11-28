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

#include <iomanip>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

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
      bool keep_num_dims = false, bool bias_tensor_optional = false,
      ActivationFunctionType activation_func = ActivationFunctionType_RELU,
      FullyConnectedOptionsWeightsFormat weights_format =
          FullyConnectedOptionsWeightsFormat_DEFAULT,
      bool add_bias_for_quantized = true)
      : batches_(batches), units_(units) {
    int total_input_size = 1;
    for (size_t i = 0; i < input.shape.size(); ++i) {
      total_input_size *= input.shape[i];
    }
    input_size_ = total_input_size / batches_;

    input_ = AddInput(input);
    weights_ =
        AddInput({input.type, {units_, input_size_}, input.min, input.max});

    if (bias_tensor_optional) {
      bias_ = AddNullInput();
    } else if (input.type == TensorType_FLOAT32) {
      bias_ = AddInput({TensorType_FLOAT32, {units_}});
    } else if (add_bias_for_quantized) {
      // This is a quantized version. The scale of 'bias' depends on the scales
      // of input and filter. Supposedly this is correctly set during quantized
      // training.
      auto bias_scale = GetScale(input_) * GetScale(weights_);
      TensorData bias{TensorType_INT32, {units_}, 0, 0, bias_scale};
      bias_ = AddInput(bias);
    }

    output_ = AddOutput(output);
    if (weights_format != FullyConnectedOptionsWeightsFormat_DEFAULT) {
      AddOutput({TensorType_UINT8, input.shape});
    }

    SetBuiltinOp(BuiltinOperator_FULLY_CONNECTED,
                 BuiltinOptions_FullyConnectedOptions,
                 CreateFullyConnectedOptions(builder_, activation_func,
                                             weights_format, keep_num_dims)
                     .Union());
    resolver_ = absl::make_unique<SingleOpResolver>(
        BuiltinOperator_FULLY_CONNECTED, registration);
    std::vector<std::vector<int>> inputs = {GetShape(input_),
                                            GetShape(weights_)};
    if (add_bias_for_quantized) {
      inputs.push_back((bias_ == kTfLiteOptionalTensor) ? std::vector<int>()
                                                        : GetShape(bias_));
    }
    BuildInterpreter(inputs);
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

  void SetBias(const std::vector<float>& data) {
    QuantizeAndPopulate<int32_t>(bias_, data);
  }
  template <typename T>
  void SetWeights(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(weights_, data);
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

// In the hybrid model the weights are quantized (to uint8). But the bias,
// input (and output) are expected to be in float precision.
class HybridFullyConnectedOpModel : public SingleOpModel {
 public:
  HybridFullyConnectedOpModel(int units, int batches, const TensorData& input,
                              const TensorData& weights,
                              const TensorData& output = {TensorType_FLOAT32})
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

    SetBuiltinOp(
        BuiltinOperator_FULLY_CONNECTED, BuiltinOptions_FullyConnectedOptions,
        CreateFullyConnectedOptions(builder_, ActivationFunctionType_RELU)
            .Union());
    resolver_ = absl::make_unique<SingleOpResolver>(
        BuiltinOperator_FULLY_CONNECTED,
        ops::builtin::Register_FULLY_CONNECTED_PIE());
    BuildInterpreter({GetShape(input_), GetShape(weights_), GetShape(bias_)});
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

  m.Invoke();

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

  m.Invoke();

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 1));
  EXPECT_THAT(m.GetOutput(), ElementsAre(11, 9));
}

TEST(FloatFullyConnectedOpTest, SimpleTestNoBias) {
  // The optimized kernel assumes that the bias is specified.
  FloatFullyConnectedOpModel m(ops::builtin::Register_FULLY_CONNECTED_PIE(),
                               /*units=*/1, /*batches=*/2,
                               /*input=*/{TensorType_FLOAT32, {2, 2}},
                               /*output=*/{TensorType_FLOAT32},
                               /*keep_num_dims=*/false,
                               /*bias_tensor_optional=*/true);
  m.SetWeights({
      2, 4,  // u = 0
  });

  m.SetInput({
      1, 2,  // b = 0
      2, 1,  // b = 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(2, 1));
  EXPECT_THAT(m.GetOutput(), ElementsAre(10, 8));
}

TEST_P(QuantizedFullyConnectedOpTest, SimpleTestQuantizedUint8) {
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_UINT8, {2, 10}, -63.5, 64},
      /*output=*/{TensorType_UINT8, {}, -127, 128});

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

  m.Invoke();

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
      /*keep_num_dims =*/false, /*bool bias_tensor_optional =*/false,
      /*ActivationFunctionType activation_func =*/ActivationFunctionType_RELU,
      /*FullyConnectedOptionsWeightsFormat weights_format =*/
      FullyConnectedOptionsWeightsFormat_DEFAULT,
      /*add_bias_for_quantized =*/false);

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

  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  23, 23, 23,  //
                  57, 57, 57,  //
              })));
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAre(150, 150, 150, 184, 184, 184));
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

  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({24, 25, 26, 58, 59, 60})));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAre(23, 24, 25, 57, 58, 59));
}

TEST_P(QuantizedFullyConnectedOpTest, SimpleTestQuantizedInt8NoBias) {
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_INT8, {2, 10}, -63.5, 64},
      /*output=*/{TensorType_INT8, {}, -127, 128},
      /*keep_num_dims =*/false, /*bool bias_tensor_optional =*/false,
      /*ActivationFunctionType activation_func =*/ActivationFunctionType_RELU,
      /*FullyConnectedOptionsWeightsFormat weights_format =*/
      FullyConnectedOptionsWeightsFormat_DEFAULT,
      /*add_bias_for_quantized =*/false);

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

  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({23, 23, 23, 57, 57, 57})));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAre(22, 22, 22, 56, 56, 56));
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

  m.Invoke();

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
      /*output=*/{TensorType_UINT8, {}, -63.5, 64});

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

  m.Invoke();

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

  m.Invoke();

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
      /*keep_num_dims=*/false,
      /*bias_tensor_optional=*/false,
      /*activation_func=*/ActivationFunctionType_NONE, weights_format);

  std::mt19937 random_engine;
  std::uniform_int_distribution<uint8_t> weights_dist;

  std::vector<float> weights_data(input_depth * output_depth);
  for (auto& w : weights_data) {
    uint8_t q = weights_dist(random_engine);
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

  std::uniform_int_distribution<uint8_t> input_dist;
  std::vector<float> input_data(input_depth * batches);
  for (auto& i : input_data) {
    uint8_t q = input_dist(random_engine);
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

  m.Invoke();

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
  // This means that output_depth must be a multiple of 4, and input_deth must
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

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     24, 25, 26,  //
                                     58, 59, 60,  //
                                 },
                                 /*max_abs_error=*/1.3f)));
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

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     24, 25, 26,  //
                                     58, 59, 60,  //
                                 },
                                 /*max_abs_error=*/1.3f)));
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

  m.Invoke();

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

  m.Invoke();

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 2, 1, 3));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 24, 25, 26,  // first batch
                                 58, 59, 60,  // second batch
                             }));
}

#ifdef GTEST_HAS_DEATH_TEST
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
                   /*keep_num_dims=*/true),
               "Cannot allocate tensors");
}
#endif

TEST_P(QuantizedFullyConnectedOpTest, SimpleTest4dInputQuantizedUint8) {
  QuantizedFullyConnectedOpModel m(
      GetRegistration(), /*units=*/3, /*batches=*/2,
      /*input=*/{TensorType_UINT8, {4, 1, 5, 1}, -63.5, 64},
      /*output=*/{TensorType_UINT8, {}, -127, 128});

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

  m.Invoke();

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
      /*output=*/{TensorType_UINT8, {}, -63.5, 64});

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

  m.Invoke();

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

    m.Invoke();

    float* golden_start = fully_connected_golden_output + i * m.num_units();
    float* golden_end = golden_start + m.num_units();
    std::vector<float> expected;
    expected.insert(expected.end(), golden_start, golden_end);
    expected.insert(expected.end(), golden_start, golden_end);

    EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(expected)));
  }
}

}  // namespace
}  // namespace tflite
