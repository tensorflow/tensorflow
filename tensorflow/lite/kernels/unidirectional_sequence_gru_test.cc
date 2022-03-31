/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_GRU();

namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

class GRUOpModel : public SingleOpModel {
 public:
  explicit GRUOpModel(int n_batch, int n_input, int n_output,
                      const std::vector<std::vector<int>>& input_shapes,
                      const TensorType& weight_type = TensorType_FLOAT32)
      : n_batch_(n_batch), n_input_(n_input), n_output_(n_output) {
    input_ = AddInput(TensorType_FLOAT32);
    input_state_ =
        AddVariableInput(TensorData{TensorType_FLOAT32, {n_batch, n_output}});
    gate_weight_ = AddInput(TensorType_FLOAT32);
    gate_bias_ = AddInput(TensorType_FLOAT32);
    candidate_weight_ = AddInput(TensorType_FLOAT32);
    candidate_bias_ = AddInput(TensorType_FLOAT32);

    output_ = AddOutput(TensorType_FLOAT32);
    output_state_ = AddOutput(TensorType_FLOAT32);

    SetCustomOp("UNIDIRECTIONAL_SEQUENCE_GRU", {},
                Register_UNIDIRECTIONAL_SEQUENCE_GRU);
    BuildInterpreter(input_shapes);
  }

  void SetInput(const std::vector<float>& f) { PopulateTensor(input_, f); }

  void SetInputState(const std::vector<float>& f) {
    PopulateTensor(input_state_, f);
  }

  void SetGateWeight(const std::vector<float>& f) {
    PopulateTensor(gate_weight_, f);
  }

  void SetGateBias(const std::vector<float>& f) {
    PopulateTensor(gate_bias_, f);
  }

  void SetCandidateWeight(const std::vector<float>& f) {
    PopulateTensor(candidate_weight_, f);
  }

  void SetCandidateBias(const std::vector<float>& f) {
    PopulateTensor(candidate_bias_, f);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

  int num_batches() { return n_batch_; }
  int num_inputs() { return n_input_; }
  int num_outputs() { return n_output_; }

 private:
  int input_;
  int input_state_;
  int gate_weight_;
  int gate_bias_;
  int candidate_weight_;
  int candidate_bias_;

  int output_;
  int output_state_;
  int n_batch_;
  int n_input_;
  int n_output_;
};

TEST(GRUTest, SimpleTest) {
  const int n_time = 2;
  const int n_batch = 2;
  const int n_input = 2;
  const int n_output = 3;

  GRUOpModel m(n_batch, n_input, n_output,
               {{n_time, n_batch, n_input},
                {n_batch, n_output},
                {2 * n_output, n_input + n_output},
                {2 * n_output},
                {n_output, n_input + n_output},
                {n_output}});
  // All data is randomly generated.
  m.SetInput({0.89495724, 0.34482682, 0.68505806, 0.7135783, 0.3167085,
              0.93647677, 0.47361764, 0.39643127});
  m.SetInputState(
      {0.09992421, 0.3028481, 0.78305984, 0.50438094, 0.11269058, 0.10244724});
  m.SetGateWeight({0.7256918,  0.8945897,  0.03285786, 0.42637166, 0.119376324,
                   0.83035135, 0.16997327, 0.42302176, 0.77598256, 0.2660894,
                   0.9587266,  0.6218451,  0.88164485, 0.12272458, 0.2699055,
                   0.18399088, 0.21930052, 0.3374841,  0.70866305, 0.9523419,
                   0.25170696, 0.60988617, 0.79823977, 0.64477515, 0.2602957,
                   0.5053131,  0.93722224, 0.8451359,  0.97905475, 0.38669217});
  m.SetGateBias(
      {0.032708533, 0.018445263, 0.15320699, 0.8163046, 0.26683575, 0.1412022});
  m.SetCandidateWeight({0.96165305, 0.95572084, 0.11534478, 0.96965164,
                        0.33562955, 0.8680755, 0.003066936, 0.057793964,
                        0.8671354, 0.33354893, 0.7313398, 0.78492093,
                        0.19530584, 0.116550304, 0.13599132});
  m.SetCandidateBias({0.89837056, 0.54769796, 0.63364106});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(n_time, n_batch, n_output));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {0.20112592, 0.45286041, 0.80842507, 0.59567153, 0.2619998,
                   0.22922856, 0.27715868, 0.5247152, 0.82300174, 0.65812796,
                   0.38217607, 0.3401444})));
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
