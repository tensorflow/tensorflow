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

#include <vector>
#include <iomanip>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

static float rnn_input[] = {
    0.23689353f,   0.285385f,     0.037029743f, -0.19858193f,  -0.27569133f,
    0.43773448f,   0.60379338f,   0.35562468f,  -0.69424844f,  -0.93421471f,
    -0.87287879f,  0.37144363f,   -0.62476718f, 0.23791671f,   0.40060222f,
    0.1356622f,    -0.99774903f,  -0.98858172f, -0.38952237f,  -0.47685933f,
    0.31073618f,   0.71511042f,   -0.63767755f, -0.31729108f,  0.33468103f,
    0.75801885f,   0.30660987f,   -0.37354088f, 0.77002847f,   -0.62747043f,
    -0.68572164f,  0.0069220066f, 0.65791464f,  0.35130811f,   0.80834007f,
    -0.61777675f,  -0.21095741f,  0.41213346f,  0.73784804f,   0.094794154f,
    0.47791874f,   0.86496925f,   -0.53376222f, 0.85315156f,   0.10288584f,
    0.86684f,      -0.011186242f, 0.10513687f,  0.87825835f,   0.59929144f,
    0.62827742f,   0.18899453f,   0.31440187f,  0.99059987f,   0.87170351f,
    -0.35091716f,  0.74861872f,   0.17831337f,  0.2755419f,    0.51864719f,
    0.55084288f,   0.58982027f,   -0.47443086f, 0.20875752f,   -0.058871567f,
    -0.66609079f,  0.59098077f,   0.73017097f,  0.74604273f,   0.32882881f,
    -0.17503482f,  0.22396147f,   0.19379807f,  0.29120302f,   0.077113032f,
    -0.70331609f,  0.15804303f,   -0.93407321f, 0.40182066f,   0.036301374f,
    0.66521823f,   0.0300982f,    -0.7747041f,  -0.02038002f,  0.020698071f,
    -0.90300065f,  0.62870288f,   -0.23068321f, 0.27531278f,   -0.095755219f,
    -0.712036f,    -0.17384434f,  -0.50593495f, -0.18646687f,  -0.96508682f,
    0.43519354f,   0.14744234f,   0.62589407f,  0.1653645f,    -0.10651493f,
    -0.045277178f, 0.99032974f,   -0.88255352f, -0.85147917f,  0.28153265f,
    0.19455957f,   -0.55479527f,  -0.56042433f, 0.26048636f,   0.84702539f,
    0.47587705f,   -0.074295521f, -0.12287641f, 0.70117295f,   0.90532446f,
    0.89782166f,   0.79817224f,   0.53402734f,  -0.33286154f,  0.073485017f,
    -0.56172788f,  -0.044897556f, 0.89964068f,  -0.067662835f, 0.76863563f,
    0.93455386f,   -0.6324693f,   -0.083922029f};

static float rnn_golden_output[] = {
    0.496726f,   0,          0.965996f,  0,         0.0584254f, 0,
    0,          0.12315f,    0,         0,         0.612266f,  0.456601f,
    0,          0.52286f,    1.16099f,   0.0291232f,

    0,          0,          0.524901f,  0,         0,         0,
    0,          1.02116f,    0,         1.35762f,   0,         0.356909f,
    0.436415f,   0.0355727f,  0,         0,

    0,          0,          0,         0.262335f,  0,         0,
    0,          1.33992f,    0,         2.9739f,    0,         0,
    1.31914f,    2.66147f,    0,         0,

    0.942568f,   0,          0,         0,         0.025507f,  0,
    0,          0,          0.321429f,  0.569141f,  1.25274f,   1.57719f,
    0.8158f,     1.21805f,    0.586239f,  0.25427f,

    1.04436f,    0,          0.630725f,  0,         0.133801f,  0.210693f,
    0.363026f,   0,          0.533426f,  0,         1.25926f,   0.722707f,
    0,          1.22031f,    1.30117f,   0.495867f,

    0.222187f,   0,          0.72725f,   0,         0.767003f,  0,
    0,          0.147835f,   0,         0,         0,         0.608758f,
    0.469394f,   0.00720298f, 0.927537f,  0,

    0.856974f,   0.424257f,   0,         0,         0.937329f,  0,
    0,          0,          0.476425f,  0,         0.566017f,  0.418462f,
    0.141911f,   0.996214f,   1.13063f,   0,

    0.967899f,   0,          0,         0,         0.0831304f, 0,
    0,          1.00378f,    0,         0,         0,         1.44818f,
    1.01768f,    0.943891f,   0.502745f,  0,

    0.940135f,   0,          0,         0,         0,         0,
    0,          2.13243f,    0,         0.71208f,   0.123918f,  1.53907f,
    1.30225f,    1.59644f,    0.70222f,   0,

    0.804329f,   0,          0.430576f,  0,         0.505872f,  0.509603f,
    0.343448f,   0,          0.107756f,  0.614544f,  1.44549f,   1.52311f,
    0.0454298f,  0.300267f,   0.562784f,  0.395095f,

    0.228154f,   0,          0.675323f,  0,         1.70536f,   0.766217f,
    0,          0,          0,         0.735363f,  0.0759267f, 1.91017f,
    0.941888f,   0,          0,         0,

    0,          0,          1.5909f,    0,         0,         0,
    0,          0.5755f,     0,         0.184687f,  0,         1.56296f,
    0.625285f,   0,          0,         0,

    0,          0,          0.0857888f, 0,         0,         0,
    0,          0.488383f,   0.252786f,  0,         0,         0,
    1.02817f,    1.85665f,    0,         0,

    0.00981836f, 0,          1.06371f,   0,         0,         0,
    0,          0,          0,         0.290445f,  0.316406f,  0,
    0.304161f,   1.25079f,    0.0707152f, 0,

    0.986264f,   0.309201f,   0,         0,         0,         0,
    0,          1.64896f,    0.346248f,  0,         0.918175f,  0.78884f,
    0.524981f,   1.92076f,    2.07013f,   0.333244f,

    0.415153f,   0.210318f,   0,         0,         0,         0,
    0,          2.02616f,    0,         0.728256f,  0.84183f,   0.0907453f,
    0.628881f,   3.58099f,    1.49974f,   0
};

class UnidirectionalRNNOpModel : public SingleOpModel {
 public:
  UnidirectionalRNNOpModel(int batches, int sequence_len, int units, int size)
      : batches_(batches),
        sequence_len_(sequence_len),
        units_(units),
        input_size_(size) {
    input_ = AddInput(TensorType_FLOAT32);
    weights_ = AddInput(TensorType_FLOAT32);
    recurrent_weights_ = AddInput(TensorType_FLOAT32);
    bias_ = AddInput(TensorType_FLOAT32);
    hidden_state_ = AddOutput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(
        BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN, BuiltinOptions_RNNOptions,
        CreateRNNOptions(builder_, ActivationFunctionType_RELU).Union());
    BuildInterpreter({{batches_, sequence_len_, input_size_},
                      {units_, input_size_},
                      {units_, units_},
                      {units_}});
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

  void ResetHiddenState() {
    const int zero_buffer_size = units_ * batches_;
    std::unique_ptr<float[]> zero_buffer(new float[zero_buffer_size]);
    memset(zero_buffer.get(), 0, zero_buffer_size * sizeof(float));
    PopulateTensor(hidden_state_, 0, zero_buffer.get(),
                   zero_buffer.get() + zero_buffer_size);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

  int input_size() { return input_size_; }
  int num_units() { return units_; }
  int num_batches() { return batches_; }
  int sequence_len() { return sequence_len_; }

 private:
  int input_;
  int weights_;
  int recurrent_weights_;
  int bias_;
  int hidden_state_;
  int output_;

  int batches_;
  int sequence_len_;
  int units_;
  int input_size_;
};

// TODO(mirkov): add another test which directly compares to TF once TOCO
// supports the conversion from dynamic_rnn with BasicRNNCell.
TEST(FullyConnectedOpTest, BlackBoxTest) {
  UnidirectionalRNNOpModel rnn(2, 16, 16, 8);
  rnn.SetWeights(
      {0.461459f,    0.153381f,   0.529743f,    -0.00371218f, 0.676267f,   -0.211346f,
       0.317493f,    0.969689f,   -0.343251f,   0.186423f,    0.398151f,   0.152399f,
       0.448504f,    0.317662f,   0.523556f,    -0.323514f,   0.480877f,   0.333113f,
       -0.757714f,   -0.674487f,  -0.643585f,   0.217766f,    -0.0251462f, 0.79512f,
       -0.595574f,   -0.422444f,  0.371572f,    -0.452178f,   -0.556069f,  -0.482188f,
       -0.685456f,   -0.727851f,  0.841829f,    0.551535f,    -0.232336f,  0.729158f,
       -0.00294906f, -0.69754f,   0.766073f,    -0.178424f,   0.369513f,   -0.423241f,
       0.548547f,    -0.0152023f, -0.757482f,   -0.85491f,    0.251331f,   -0.989183f,
       0.306261f,    -0.340716f,  0.886103f,    -0.0726757f,  -0.723523f,  -0.784303f,
       0.0354295f,   0.566564f,   -0.485469f,   -0.620498f,   0.832546f,   0.697884f,
       -0.279115f,   0.294415f,   -0.584313f,   0.548772f,    0.0648819f,  0.968726f,
       0.723834f,    -0.0080452f, -0.350386f,   -0.272803f,   0.115121f,   -0.412644f,
       -0.824713f,   -0.992843f,  -0.592904f,   -0.417893f,   0.863791f,   -0.423461f,
       -0.147601f,   -0.770664f,  -0.479006f,   0.654782f,    0.587314f,   -0.639158f,
       0.816969f,    -0.337228f,  0.659878f,    0.73107f,     0.754768f,   -0.337042f,
       0.0960841f,   0.368357f,   0.244191f,    -0.817703f,   -0.211223f,  0.442012f,
       0.37225f,     -0.623598f,  -0.405423f,   0.455101f,    0.673656f,   -0.145345f,
       -0.511346f,   -0.901675f,  -0.81252f,    -0.127006f,   0.809865f,   -0.721884f,
       0.636255f,    0.868989f,   -0.347973f,   -0.10179f,    -0.777449f,  0.917274f,
       0.819286f,    0.206218f,   -0.00785118f, 0.167141f,    0.45872f,    0.972934f,
       -0.276798f,   0.837861f,   0.747958f,    -0.0151566f,  -0.330057f,  -0.469077f,
       0.277308f,    0.415818f});

  rnn.SetBias({0.065691948f, -0.69055247f, 0.1107955f, -0.97084129f, -0.23957068f,
               -0.23566568f, -0.389184f, 0.47481549f, -0.4791103f, 0.29931796f,
               0.10463274f, 0.83918178f, 0.37197268f, 0.61957061f, 0.3956964f,
               -0.37609905f});

  rnn.SetRecurrentWeights({0.1f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0.1f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0.1f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0.1f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0.1f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0.1f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0.1f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0.1f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0.1f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0.1f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0.1f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0.1f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0.1f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0.1f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0.1f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0.1f});

  rnn.ResetHiddenState();
  const int input_sequence_size = rnn.input_size() * rnn.sequence_len();
  float* batch_start = rnn_input;
  float* batch_end = batch_start + input_sequence_size;
  rnn.SetInput(0, batch_start, batch_end);
  rnn.SetInput(input_sequence_size, batch_start, batch_end);

  rnn.Invoke();

  float* golden_start = rnn_golden_output;
  float* golden_end = golden_start + rnn.num_units() * rnn.sequence_len();
  std::vector<float> expected;
  expected.insert(expected.end(), golden_start, golden_end);
  expected.insert(expected.end(), golden_start, golden_end);

  EXPECT_THAT(rnn.GetOutput(), ElementsAreArray(ArrayFloatNear(expected)));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  // On Linux, add: tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
