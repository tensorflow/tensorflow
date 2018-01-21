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

#include <iomanip>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

static float fully_connected_input[] = {
    0.503691f, 0.196961f, 0.521017f, 0.554248f, 0.288678f, 0.792476f, 0.561653f,
    0.462230f, 0.650736f, 0.163132f, 0.029658f, 0.411544f, 0.470539f, 0.572390f,
    0.538755f, 0.212030f, 0.264309f, 0.193908f, 0.777480f, 0.745661f, 0.423314f,
    0.470804f, 0.175501f, 0.492225f, 0.192743f, 0.540183f, 0.372514f, 0.446550f,
    0.498173f, 0.126472f, 0.132706f, 0.001864f, 0.323433f, 0.653723f, 0.556112f,
    0.612111f, 0.446199f, 0.117765f, 0.074341f, 0.096935f, 0.280897f, 0.103999f,
    0.508479f, 0.751437f, 0.676389f, 0.047234f, 0.963467f, 0.940698f, 0.241142f,
    0.740947f, 0.686359f, 0.664456f, 0.211751f, 0.861860f, 0.156681f, 0.404494f,
    0.402043f, 0.529195f, 0.851044f, 0.900216f, 0.655667f, 0.983750f, 0.902081f,
    0.979100f, 0.637473f, 0.458193f, 0.591211f, 0.083671f, 0.575958f, 0.665552f,
    0.180606f, 0.856856f, 0.769551f, 0.689086f, 0.608293f, 0.445940f, 0.736320f,
    0.571760f, 0.386637f, 0.977461f, 0.312707f, 0.072996f, 0.641918f, 0.524458f,
    0.934856f, 0.798598f, 0.928951f, 0.336899f, 0.327793f, 0.779995f, 0.237115f,
    0.983460f, 0.763746f, 0.139196f, 0.962560f, 0.401218f, 0.597389f, 0.553771f,
    0.484890f, 0.173347f, 0.219322f, 0.665496f, 0.030203f, 0.988873f, 0.354582f,
    0.638496f, 0.434813f, 0.090902f, 0.210256f, 0.821450f, 0.068363f, 0.522962f,
    0.894446f, 0.710280f, 0.047420f, 0.829302f, 0.508879f, 0.976371f, 0.166202f,
    0.836672f, 0.756367f, 0.403317f, 0.820132f, 0.520112f, 0.542513f, 0.782691f,
    0.921330f, 0.139902f};

static float fully_connected_golden_output[] = {
    0,        0.0732134f,   0,        0,          0,         0.280859f,
    0,        0.128927f,    0,        0.0777251f,  0,         0.270268f,
    0.271435f, 0.0173503f,   0.335465f, 0.235562f,

    0,        0.0745866f,   0,        0.051611f,   0,         0.253876f,
    0,        0.0814873f,   0,        0.104104f,   0,         0.248529f,
    0.264194f, 0,           0.302973f, 0.166252f,

    0,        0.0170409f,   0,        0.0509851f,  0,         0.212834f,
    0,        0.0208326f,   0,        0.129932f,   0.203978f,  0.103428f,
    0.298051f, 0,           0.332233f, 0.00445903f,

    0,        0.125246f,    0,        0.0735336f,  0,         0.0910256f,
    0,        0,           0,        0.18933f,    0.378111f,  0.0712443f,
    0.277298f, 0.0123414f,   0.267454f, 0,

    0,        0.14687f,     0,        0.155495f,   0.0300215f, 0.147256f,
    0,        0,           0,        0.156412f,   0.434914f,  0.0461529f,
    0.246508f, 0,           0.363138f, 0,

    0,        0,           0,        0.0212949f,  0,         0.301708f,
    0,        0.35497f,     0,        0.406223f,   0.0260211f, 0.049195f,
    0.197161f, 0,           0.37316f,  0,

    0,        0.221783f,    0,        0,          0.0116515f, 0.281945f,
    0,        0,           0,        0,          0.285626f,  0.181773f,
    0.296401f, 0.170452f,    0.367135f, 0.142597f,

    0,        0,           0,        0,          0,         0.418886f,
    0,        0.291063f,    0,        0.227541f,   0.0424759f, 0.27589f,
    0.398286f, 0.177146f,    0.40359f,  0.121452f,

    0,        0.0834884f,   0,        0,          0,         0.287441f,
    0,        0.0046838f,   0,        0.0122087f,  0,         0.217376f,
    0.140183f, 0.0948412f,   0.436677f, 0.0589876f,

    0,        0.0289969f,   0,        0.0921397f,  0,         0.396802f,
    0,        0.0126157f,   0,        0.0968433f,  0,         0.172271f,
    0.173295f, 0.0664741f,   0.53645f,  0.00915603f,

    0,        0,           0,        0,          0,         0.147942f,
    0,        0.263795f,    0,        0.39782f,    0,         0.382435f,
    0.561072f, 0.0579847f,   0.145712f, 0.13508f,

    0,        0,           0,        0.16382f,    0,         0.322294f,
    0,        0.163798f,    0,        0.405211f,   0.367953f,  0.076852f,
    0.342473f, 0.0834118f,   0.377537f, 0,

    0,        0.206f,       0,        0,          0,         0.375769f,
    0,        0,           0,        0,          0,         0.125165f,
    0,        0.105591f,    0.52055f,  0.0536445f,

    0,        0.259261f,    0,        0,          0,         0.247707f,
    0,        0,           0,        0,          0,         0.215862f,
    0.149153f, 0.224678f,    0.359519f, 0.129419f,

    0,        0.17611f,     0,        0.280895f,   0,         0.576484f,
    0,        0.000418848f, 0,        0,          0,         0.151112f,
    0.211902f, 0,           0.566341f, 0.106305f,

    0,        0.0246284f,   0,        0,          0,         0.196267f,
    0,        0.0248624f,   0,        0.265635f,   0,         0.436199f,
    0.408079f, 0.134514f,    0.328489f, 0.411368f};

class BaseFullyConnectedOpModel : public SingleOpModel {
 public:
  // TODO(ahentz): test different activation types too.
  BaseFullyConnectedOpModel(int units, int batches, const TensorData& input,
                            const TensorData& output = {TensorType_FLOAT32})
      : batches_(batches), units_(units) {
    int total_input_size = 1;
    for (int i = 0; i < input.shape.size(); ++i) {
      total_input_size *= input.shape[i];
    }
    input_size_ = total_input_size / batches_;

    input_ = AddInput(input);
    weights_ =
        AddInput({input.type, {units_, input_size_}, input.min, input.max});

    if (input.type == TensorType_FLOAT32) {
      bias_ = AddInput({TensorType_FLOAT32, {units_}});
    } else {
      // This is a quantized version. The scale of 'bias' depends on the scales
      // of input and filter. Supposedly this is correctly set during quantized
      // training.
      auto bias_scale = GetScale(input_) * GetScale(weights_);
      TensorData bias{TensorType_INT32, {units_}, 0, 0, bias_scale};
      bias_ = AddInput(bias);
    }

    output_ = AddOutput(output);

    SetBuiltinOp(
        BuiltinOperator_FULLY_CONNECTED, BuiltinOptions_FullyConnectedOptions,
        CreateFullyConnectedOptions(builder_, ActivationFunctionType_RELU)
            .Union());
    BuildInterpreter({GetShape(input_), GetShape(weights_), GetShape(bias_)});
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

  void SetBias(std::initializer_list<float> f) { PopulateTensor(bias_, f); }

  void SetWeights(std::initializer_list<float> f) {
    PopulateTensor(weights_, f);
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }
  void SetInput(int offset, float* begin, float* end) {
    PopulateTensor(input_, offset, begin, end);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class QuantizedFullyConnectedOpModel : public BaseFullyConnectedOpModel {
 public:
  using BaseFullyConnectedOpModel::BaseFullyConnectedOpModel;

  void SetBias(std::initializer_list<float> data) {
    QuantizeAndPopulate<int32_t>(bias_, data);
  }
  void SetWeights(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(weights_, data);
  }
  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(input_, data);
  }

  std::vector<uint8_t> GetOutput() { return ExtractVector<uint8_t>(output_); }
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

// TODO(ahentz): add more small tests like this one, focused on making sure the
// calculations are correct.
TEST(FullyConnectedOpTest, SimpleTest) {
  FloatFullyConnectedOpModel m(3, 2, {TensorType_FLOAT32, {2, 10}});
  m.SetWeights({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
  });
  m.SetBias({1, 2, 3});

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAre(24, 25, 26, 58, 59, 60));
}

TEST(FullyConnectedOpTest, SimpleTestQuantized) {
  QuantizedFullyConnectedOpModel m(
      3, 2,
      /*input=*/{TensorType_UINT8, {2, 10}, -63.5f, 64},
      /*output=*/{TensorType_UINT8, {}, -127, 128});

  // input_product_scale < output_scale was not true.
  m.SetWeights({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
  });
  m.SetBias({1, 2, 3});

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(), ElementsAreArray(ArrayFloatNear({
                                            24, 25, 26,  //
                                            58, 59, 60,  //
                                        })));
  EXPECT_THAT(m.GetOutput(), ElementsAre(151, 152, 153, 185, 186, 187));
}

TEST(FullyConnectedOpTest, SimpleTest4DInput) {
  // Note that it is not required that the first dimension be the number of
  // batches. All we care is that the input can be evenly distributed in
  // batches. In this case, we need the input to have multiples of '2'.
  FloatFullyConnectedOpModel m(/*units=*/3,
                               /*batches=*/2,
                               /*input=*/{TensorType_FLOAT32, {4, 1, 5, 1}});
  m.SetWeights({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
  });
  m.SetBias({1, 2, 3});

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // first batch
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // second batch
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 24, 25, 26,  // first batch
                                 58, 59, 60,  // second batch
                             }));
}

TEST(FullyConnectedOpTest, SimpleTest4dInputQuantized) {
  QuantizedFullyConnectedOpModel m(
      3, 2,
      /*input=*/{TensorType_UINT8, {4, 1, 5, 1}, -63.5f, 64},
      /*output=*/{TensorType_UINT8, {}, -127, 128});

  // input_product_scale < output_scale was not true.
  m.SetWeights({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
  });
  m.SetBias({1, 2, 3});

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(), ElementsAreArray(ArrayFloatNear({
                                            24, 25, 26,  //
                                            58, 59, 60,  //
                                        })));
  EXPECT_THAT(m.GetOutput(), ElementsAre(151, 152, 153, 185, 186, 187));
}

// TODO(ahentz): Reconsider this test. Having arbitrary weights makes it hard
// to debug errors and doesn't necessarily test all the important details.
TEST(FullyConnectedOpTest, BlackBoxTest) {
  FloatFullyConnectedOpModel m(16, 2, {TensorType_FLOAT32, {2, 8}});
  m.SetWeights(
      {0.091327f,  0.103366f,  -0.316505f, -0.083120f, 0.149366f,  -0.196636f,
       -0.123672f, 0.062800f,  0.063031f,  0.191670f,  -0.062001f, -0.061504f,
       -0.275581f, 0.059388f,  -0.118497f, -0.079224f, 0.109758f,  0.008307f,
       -0.062657f, -0.060962f, -0.049782f, -0.106719f, -0.319482f, -0.103650f,
       0.266455f,  0.051517f,  -0.123448f, 0.322464f,  0.043282f,  -0.173782f,
       -0.190381f, 0.002013f,  0.096086f,  0.131157f,  0.031164f,  0.100638f,
       -0.312191f, -0.080923f, -0.101318f, -0.116614f, 0.142238f,  0.086540f,
       -0.139154f, 0.174268f,  -0.073161f, 0.080072f,  0.006874f,  0.229382f,
       -0.104321f, -0.176035f, -0.208587f, -0.001019f, -0.162032f, 0.080824f,
       -0.025021f, 0.074460f,  -0.252595f, -0.161750f, -0.136403f, 0.008308f,
       0.005710f,  0.096600f,  0.289839f,  0.218816f,  -0.304651f, -0.070958f,
       0.054598f,  0.147113f,  -0.139112f, -0.072798f, -0.163335f, -0.167863f,
       -0.128762f, -0.035780f, 0.117262f,  0.017177f,  0.263335f,  -0.176612f,
       0.262961f,  -0.093654f, -0.339283f, 0.333071f,  0.180827f,  0.287583f,
       0.066350f,  -0.197947f, -0.114449f, -0.236035f, 0.103532f,  -0.034284f,
       0.093299f,  -0.145361f, 0.054001f,  0.250570f,  0.157010f,  -0.143480f,
       -0.139061f, -0.048873f, 0.067557f,  0.139038f,  0.324106f,  0.227041f,
       0.037793f,  -0.225747f, -0.241619f, 0.357835f,  0.135762f,  -0.306764f,
       -0.125982f, 0.091916f,  0.266587f,  0.030135f,  0.265148f,  0.141627f,
       0.020120f,  0.083815f,  -0.124556f, -0.100124f, -0.048159f, 0.181172f,
       0.302309f,  -0.041084f, 0.146334f,  -0.061511f, -0.232605f, 0.281324f,
       0.145408f,  -0.221897f});
  m.SetBias({-0.160594f, 0.205770f, -0.078307f, -0.077984f, 0.001937f, 0.015860f,
             0.036810f, 0.012346f, 0.001028f, 0.038551f, 0.075415f, 0.020804f,
             0.048478f, -0.032270f, 0.175688f, -0.085662f});

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

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
