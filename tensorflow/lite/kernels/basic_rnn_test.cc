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
// Unit test for TFLite RNN op.

#include <string.h>
#include <initializer_list>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

static float rnn_input[] = {
    0.23689353,   0.285385,     0.037029743, -0.19858193,  -0.27569133,
    0.43773448,   0.60379338,   0.35562468,  -0.69424844,  -0.93421471,
    -0.87287879,  0.37144363,   -0.62476718, 0.23791671,   0.40060222,
    0.1356622,    -0.99774903,  -0.98858172, -0.38952237,  -0.47685933,
    0.31073618,   0.71511042,   -0.63767755, -0.31729108,  0.33468103,
    0.75801885,   0.30660987,   -0.37354088, 0.77002847,   -0.62747043,
    -0.68572164,  0.0069220066, 0.65791464,  0.35130811,   0.80834007,
    -0.61777675,  -0.21095741,  0.41213346,  0.73784804,   0.094794154,
    0.47791874,   0.86496925,   -0.53376222, 0.85315156,   0.10288584,
    0.86684,      -0.011186242, 0.10513687,  0.87825835,   0.59929144,
    0.62827742,   0.18899453,   0.31440187,  0.99059987,   0.87170351,
    -0.35091716,  0.74861872,   0.17831337,  0.2755419,    0.51864719,
    0.55084288,   0.58982027,   -0.47443086, 0.20875752,   -0.058871567,
    -0.66609079,  0.59098077,   0.73017097,  0.74604273,   0.32882881,
    -0.17503482,  0.22396147,   0.19379807,  0.29120302,   0.077113032,
    -0.70331609,  0.15804303,   -0.93407321, 0.40182066,   0.036301374,
    0.66521823,   0.0300982,    -0.7747041,  -0.02038002,  0.020698071,
    -0.90300065,  0.62870288,   -0.23068321, 0.27531278,   -0.095755219,
    -0.712036,    -0.17384434,  -0.50593495, -0.18646687,  -0.96508682,
    0.43519354,   0.14744234,   0.62589407,  0.1653645,    -0.10651493,
    -0.045277178, 0.99032974,   -0.88255352, -0.85147917,  0.28153265,
    0.19455957,   -0.55479527,  -0.56042433, 0.26048636,   0.84702539,
    0.47587705,   -0.074295521, -0.12287641, 0.70117295,   0.90532446,
    0.89782166,   0.79817224,   0.53402734,  -0.33286154,  0.073485017,
    -0.56172788,  -0.044897556, 0.89964068,  -0.067662835, 0.76863563,
    0.93455386,   -0.6324693,   -0.083922029};

static float rnn_golden_output[] = {
    0.496726,   0,          0.965996,  0,         0.0584254, 0,
    0,          0.12315,    0,         0,         0.612266,  0.456601,
    0,          0.52286,    1.16099,   0.0291232,

    0,          0,          0.524901,  0,         0,         0,
    0,          1.02116,    0,         1.35762,   0,         0.356909,
    0.436415,   0.0355727,  0,         0,

    0,          0,          0,         0.262335,  0,         0,
    0,          1.33992,    0,         2.9739,    0,         0,
    1.31914,    2.66147,    0,         0,

    0.942568,   0,          0,         0,         0.025507,  0,
    0,          0,          0.321429,  0.569141,  1.25274,   1.57719,
    0.8158,     1.21805,    0.586239,  0.25427,

    1.04436,    0,          0.630725,  0,         0.133801,  0.210693,
    0.363026,   0,          0.533426,  0,         1.25926,   0.722707,
    0,          1.22031,    1.30117,   0.495867,

    0.222187,   0,          0.72725,   0,         0.767003,  0,
    0,          0.147835,   0,         0,         0,         0.608758,
    0.469394,   0.00720298, 0.927537,  0,

    0.856974,   0.424257,   0,         0,         0.937329,  0,
    0,          0,          0.476425,  0,         0.566017,  0.418462,
    0.141911,   0.996214,   1.13063,   0,

    0.967899,   0,          0,         0,         0.0831304, 0,
    0,          1.00378,    0,         0,         0,         1.44818,
    1.01768,    0.943891,   0.502745,  0,

    0.940135,   0,          0,         0,         0,         0,
    0,          2.13243,    0,         0.71208,   0.123918,  1.53907,
    1.30225,    1.59644,    0.70222,   0,

    0.804329,   0,          0.430576,  0,         0.505872,  0.509603,
    0.343448,   0,          0.107756,  0.614544,  1.44549,   1.52311,
    0.0454298,  0.300267,   0.562784,  0.395095,

    0.228154,   0,          0.675323,  0,         1.70536,   0.766217,
    0,          0,          0,         0.735363,  0.0759267, 1.91017,
    0.941888,   0,          0,         0,

    0,          0,          1.5909,    0,         0,         0,
    0,          0.5755,     0,         0.184687,  0,         1.56296,
    0.625285,   0,          0,         0,

    0,          0,          0.0857888, 0,         0,         0,
    0,          0.488383,   0.252786,  0,         0,         0,
    1.02817,    1.85665,    0,         0,

    0.00981836, 0,          1.06371,   0,         0,         0,
    0,          0,          0,         0.290445,  0.316406,  0,
    0.304161,   1.25079,    0.0707152, 0,

    0.986264,   0.309201,   0,         0,         0,         0,
    0,          1.64896,    0.346248,  0,         0.918175,  0.78884,
    0.524981,   1.92076,    2.07013,   0.333244,

    0.415153,   0.210318,   0,         0,         0,         0,
    0,          2.02616,    0,         0.728256,  0.84183,   0.0907453,
    0.628881,   3.58099,    1.49974,   0};

static std::initializer_list<float> rnn_weights = {
    0.461459,    0.153381,   0.529743,    -0.00371218, 0.676267,   -0.211346,
    0.317493,    0.969689,   -0.343251,   0.186423,    0.398151,   0.152399,
    0.448504,    0.317662,   0.523556,    -0.323514,   0.480877,   0.333113,
    -0.757714,   -0.674487,  -0.643585,   0.217766,    -0.0251462, 0.79512,
    -0.595574,   -0.422444,  0.371572,    -0.452178,   -0.556069,  -0.482188,
    -0.685456,   -0.727851,  0.841829,    0.551535,    -0.232336,  0.729158,
    -0.00294906, -0.69754,   0.766073,    -0.178424,   0.369513,   -0.423241,
    0.548547,    -0.0152023, -0.757482,   -0.85491,    0.251331,   -0.989183,
    0.306261,    -0.340716,  0.886103,    -0.0726757,  -0.723523,  -0.784303,
    0.0354295,   0.566564,   -0.485469,   -0.620498,   0.832546,   0.697884,
    -0.279115,   0.294415,   -0.584313,   0.548772,    0.0648819,  0.968726,
    0.723834,    -0.0080452, -0.350386,   -0.272803,   0.115121,   -0.412644,
    -0.824713,   -0.992843,  -0.592904,   -0.417893,   0.863791,   -0.423461,
    -0.147601,   -0.770664,  -0.479006,   0.654782,    0.587314,   -0.639158,
    0.816969,    -0.337228,  0.659878,    0.73107,     0.754768,   -0.337042,
    0.0960841,   0.368357,   0.244191,    -0.817703,   -0.211223,  0.442012,
    0.37225,     -0.623598,  -0.405423,   0.455101,    0.673656,   -0.145345,
    -0.511346,   -0.901675,  -0.81252,    -0.127006,   0.809865,   -0.721884,
    0.636255,    0.868989,   -0.347973,   -0.10179,    -0.777449,  0.917274,
    0.819286,    0.206218,   -0.00785118, 0.167141,    0.45872,    0.972934,
    -0.276798,   0.837861,   0.747958,    -0.0151566,  -0.330057,  -0.469077,
    0.277308,    0.415818};

static std::initializer_list<float> rnn_recurrent_weights = {
    0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.1};

static std::initializer_list<float> rnn_bias = {
    0.065691948, -0.69055247, 0.1107955,  -0.97084129, -0.23957068, -0.23566568,
    -0.389184,   0.47481549,  -0.4791103, 0.29931796,  0.10463274,  0.83918178,
    0.37197268,  0.61957061,  0.3956964,  -0.37609905};

class RNNOpModel : public SingleOpModel {
 public:
  RNNOpModel(int batches, int units, int size,
             const TensorType& weights = TensorType_FLOAT32,
             const TensorType& recurrent_weights = TensorType_FLOAT32)
      : batches_(batches), units_(units), input_size_(size) {
    input_ = AddInput(TensorType_FLOAT32);
    weights_ = AddInput(weights);
    recurrent_weights_ = AddInput(recurrent_weights);
    bias_ = AddInput(TensorType_FLOAT32);
    hidden_state_ = AddInput(TensorType_FLOAT32, true);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(
        BuiltinOperator_RNN, BuiltinOptions_RNNOptions,
        CreateRNNOptions(builder_, ActivationFunctionType_RELU).Union());
    BuildInterpreter({{batches_, input_size_},  // input tensor
                      {units_, input_size_},    // weights tensor
                      {units_, units_},         // recurrent weights tensor
                      {units_},                 // bias tensor
                      {batches_, units_}});     // hidden state tensor
  }

  void SetBias(std::initializer_list<float> f) { PopulateTensor(bias_, f); }

  void SetWeights(std::initializer_list<float> f) {
    PopulateTensor(weights_, f);
  }

  void SetRecurrentWeights(std::initializer_list<float> f) {
    PopulateTensor(recurrent_weights_, f);
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  void SetInput(int offset, float* begin, float* end) {
    PopulateTensor(input_, offset, begin, end);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

  int input_size() { return input_size_; }
  int num_units() { return units_; }
  int num_batches() { return batches_; }

 protected:
  int input_;
  int weights_;
  int recurrent_weights_;
  int bias_;
  int hidden_state_;
  int output_;

  int batches_;
  int units_;
  int input_size_;
};

// The hybrid model has quantized weights and recurrent_weights.
class HybridRNNOpModel : public RNNOpModel {
 public:
  HybridRNNOpModel(int batches, int units, int size, TensorType tensor_type)
      : RNNOpModel(batches, units, size, tensor_type, tensor_type) {
    tensor_type_ = tensor_type;
  }

  TensorType tensor_type_;

  void SetWeights(int weights_idx, const std::vector<float>& f) {
    if (tensor_type_ == TensorType_UINT8) {
      SymmetricQuantizeAndPopulate(weights_idx, f);
    } else {
      SignedSymmetricQuantizeAndPopulate(weights_idx, f);
    }
  }

  void SetWeights(std::initializer_list<float> f) { SetWeights(weights_, f); }

  void SetRecurrentWeights(std::initializer_list<float> f) {
    SetWeights(recurrent_weights_, f);
  }
};

TEST(RnnOpTest, BlackBoxTest) {
  RNNOpModel rnn(2, 16, 8);
  rnn.SetWeights(rnn_weights);
  rnn.SetBias(rnn_bias);
  rnn.SetRecurrentWeights(rnn_recurrent_weights);

  const int input_sequence_size = sizeof(rnn_input) / sizeof(float) /
                                  (rnn.input_size() * rnn.num_batches());

  for (int i = 0; i < input_sequence_size; i++) {
    float* batch_start = rnn_input + i * rnn.input_size();
    float* batch_end = batch_start + rnn.input_size();
    rnn.SetInput(0, batch_start, batch_end);
    rnn.SetInput(rnn.input_size(), batch_start, batch_end);

    rnn.Invoke();

    float* golden_start = rnn_golden_output + i * rnn.num_units();
    float* golden_end = golden_start + rnn.num_units();
    std::vector<float> expected;
    expected.insert(expected.end(), golden_start, golden_end);
    expected.insert(expected.end(), golden_start, golden_end);

    EXPECT_THAT(rnn.GetOutput(), ElementsAreArray(ArrayFloatNear(expected)));
  }
}

TEST(HybridRnnOpTest, BlackBoxTestUint8) {
  HybridRNNOpModel rnn(2, 16, 8, TensorType_UINT8);
  rnn.SetWeights(rnn_weights);
  rnn.SetBias(rnn_bias);
  rnn.SetRecurrentWeights(rnn_recurrent_weights);

  const int input_sequence_size = sizeof(rnn_input) / sizeof(float) /
                                  (rnn.input_size() * rnn.num_batches());

  for (int i = 0; i < input_sequence_size; i++) {
    float* batch_start = rnn_input + i * rnn.input_size();
    float* batch_end = batch_start + rnn.input_size();
    rnn.SetInput(0, batch_start, batch_end);
    rnn.SetInput(rnn.input_size(), batch_start, batch_end);

    rnn.Invoke();

    float* golden_start = rnn_golden_output + i * rnn.num_units();
    float* golden_end = golden_start + rnn.num_units();
    std::vector<float> expected;
    expected.insert(expected.end(), golden_start, golden_end);
    expected.insert(expected.end(), golden_start, golden_end);

    EXPECT_THAT(rnn.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                     expected, /*max_abs_error=*/0.0104)));
  }
}

TEST(HybridRnnOpTest, BlackBoxTestInt8) {
  HybridRNNOpModel rnn(2, 16, 8, TensorType_INT8);
  rnn.SetWeights(rnn_weights);
  rnn.SetBias(rnn_bias);
  rnn.SetRecurrentWeights(rnn_recurrent_weights);

  const int input_sequence_size = sizeof(rnn_input) / sizeof(float) /
                                  (rnn.input_size() * rnn.num_batches());

  for (int i = 0; i < input_sequence_size; i++) {
    float* batch_start = rnn_input + i * rnn.input_size();
    float* batch_end = batch_start + rnn.input_size();
    rnn.SetInput(0, batch_start, batch_end);
    rnn.SetInput(rnn.input_size(), batch_start, batch_end);

    rnn.Invoke();

    float* golden_start = rnn_golden_output + i * rnn.num_units();
    float* golden_end = golden_start + rnn.num_units();
    std::vector<float> expected;
    expected.insert(expected.end(), golden_start, golden_end);
    expected.insert(expected.end(), golden_start, golden_end);

    EXPECT_THAT(rnn.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                     expected, /*max_abs_error=*/0.0104)));
  }
}

}  // namespace
}  // namespace tflite
