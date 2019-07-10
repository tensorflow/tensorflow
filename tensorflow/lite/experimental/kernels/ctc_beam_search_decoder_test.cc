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

#include <functional>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace ops {
namespace experimental {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

TfLiteRegistration* Register_CTC_BEAM_SEARCH_DECODER();

namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

class CTCBeamSearchDecoderOpModel : public SingleOpModel {
 public:
  CTCBeamSearchDecoderOpModel(std::initializer_list<int> input_shape,
                              std::initializer_list<int> sequence_length_shape,
                              int beam_width, int top_paths,
                              bool merge_repeated) {
    inputs_ = AddInput(TensorType_FLOAT32);
    sequence_length_ = AddInput(TensorType_INT32);

    for (int i = 0; i < top_paths * 3; ++i) {
      outputs_.push_back(AddOutput(TensorType_INT32));
    }
    outputs_.push_back(AddOutput(TensorType_FLOAT32));

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("beam_width", beam_width);
      fbb.Int("top_paths", top_paths);
      fbb.Bool("merge_repeated", merge_repeated);
    });
    fbb.Finish();
    SetCustomOp("CTCBeamSearchDecoder", fbb.GetBuffer(),
                Register_CTC_BEAM_SEARCH_DECODER);
    BuildInterpreter({input_shape, sequence_length_shape});
  }

  int inputs() { return inputs_; }

  int sequence_length() { return sequence_length_; }

  std::vector<std::vector<int>> GetDecodedOutpus() {
    std::vector<std::vector<int>> outputs;
    for (int i = 0; i < outputs_.size() - 1; ++i) {
      outputs.push_back(ExtractVector<int>(outputs_[i]));
    }
    return outputs;
  }

  std::vector<float> GetLogProbabilitiesOutput() {
    return ExtractVector<float>(outputs_[outputs_.size() - 1]);
  }

  std::vector<std::vector<int>> GetOutputShapes() {
    std::vector<std::vector<int>> output_shapes;
    for (const int output : outputs_) {
      output_shapes.push_back(GetTensorShape(output));
    }
    return output_shapes;
  }

 private:
  int inputs_;
  int sequence_length_;
  std::vector<int> outputs_;
};

TEST(CTCBeamSearchTest, SimpleTest) {
  CTCBeamSearchDecoderOpModel m({2, 1, 2}, {1}, 1, 1, true);
  m.PopulateTensor<float>(m.inputs(),
                          {-0.50922557, -1.35512652, -2.55445064, -1.58419356});
  m.PopulateTensor<int>(m.sequence_length(), {2});
  m.Invoke();

  // Make sure the output shapes are right.
  const std::vector<std::vector<int>>& output_shapes = m.GetOutputShapes();
  EXPECT_EQ(output_shapes.size(), 4);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2));
  EXPECT_THAT(output_shapes[1], ElementsAre(1));
  EXPECT_THAT(output_shapes[2], ElementsAre(2));
  EXPECT_THAT(output_shapes[3], ElementsAre(1, 1));

  // Check decoded outputs.
  const std::vector<std::vector<int>>& decoded_outputs = m.GetDecodedOutpus();
  EXPECT_EQ(decoded_outputs.size(), 3);
  EXPECT_THAT(decoded_outputs[0], ElementsAre(0, 0));
  EXPECT_THAT(decoded_outputs[1], ElementsAre(0));
  EXPECT_THAT(decoded_outputs[2], ElementsAre(1, 1));
  // Check log probabilities output.
  EXPECT_THAT(m.GetLogProbabilitiesOutput(),
              ElementsAreArray(ArrayFloatNear({-0.357094})));
}

TEST(CTCBeamSearchTest, MultiBatchTest) {
  CTCBeamSearchDecoderOpModel m({3, 3, 3}, {3}, 1, 1, true);
  m.PopulateTensor<float>(
      m.inputs(),
      {-0.63649208, -0.00487571, -0.04249819, -0.67754697, -1.0341399,
       -2.14717721, -0.77686821, -3.41973774, -0.05151402, -0.21482619,
       -0.57411168, -1.45039917, -0.73769373, -2.10941739, -0.44818325,
       -0.25287673, -2.80057302, -0.54748312, -0.73334867, -0.86537719,
       -0.2065197,  -0.18725838, -1.42770405, -0.86051965, -1.61642301,
       -2.07275114, -0.9201845});
  m.PopulateTensor<int>(m.sequence_length(), {3, 3, 3});
  m.Invoke();

  // Make sure the output shapes are right.
  const std::vector<std::vector<int>>& output_shapes = m.GetOutputShapes();
  EXPECT_EQ(output_shapes.size(), 4);
  EXPECT_THAT(output_shapes[0], ElementsAre(4, 2));
  EXPECT_THAT(output_shapes[1], ElementsAre(4));
  EXPECT_THAT(output_shapes[2], ElementsAre(2));
  EXPECT_THAT(output_shapes[3], ElementsAre(3, 1));

  // Check decoded outputs.
  const std::vector<std::vector<int>>& decoded_outputs = m.GetDecodedOutpus();
  EXPECT_EQ(decoded_outputs.size(), 3);
  EXPECT_THAT(decoded_outputs[0], ElementsAre(0, 0, 0, 1, 1, 0, 2, 0));
  EXPECT_THAT(decoded_outputs[1], ElementsAre(1, 0, 0, 0));
  EXPECT_THAT(decoded_outputs[2], ElementsAre(3, 2));
  // Check log probabilities output.
  EXPECT_THAT(m.GetLogProbabilitiesOutput(),
              ElementsAreArray(ArrayFloatNear({-1.88343, -1.41188, -1.20958})));
}

TEST(CTCBeamSearchTest, MultiPathsTest) {
  CTCBeamSearchDecoderOpModel m({3, 2, 5}, {2}, 3, 2, true);
  m.PopulateTensor<float>(
      m.inputs(),
      {-2.206851,   -0.09542714, -0.2393415,  -3.81866197, -0.27241158,
       -0.20371124, -0.68236623, -1.1397166,  -0.17422639, -1.85224048,
       -0.9406037,  -0.32544678, -0.21846784, -0.38377237, -0.33498676,
       -0.10139782, -0.51886883, -0.21678554, -0.15267063, -1.91164412,
       -0.31328673, -0.27462716, -0.65975336, -1.53671973, -2.76554225,
       -0.23920634, -1.2370502,  -4.98751576, -3.12995717, -0.43129368});
  m.PopulateTensor<int>(m.sequence_length(), {3, 3});
  m.Invoke();

  // Make sure the output shapes are right.
  const std::vector<std::vector<int>>& output_shapes = m.GetOutputShapes();
  EXPECT_EQ(output_shapes.size(), 7);
  EXPECT_THAT(output_shapes[0], ElementsAre(4, 2));
  EXPECT_THAT(output_shapes[1], ElementsAre(3, 2));
  EXPECT_THAT(output_shapes[2], ElementsAre(4));
  EXPECT_THAT(output_shapes[3], ElementsAre(3));
  EXPECT_THAT(output_shapes[4], ElementsAre(2));
  EXPECT_THAT(output_shapes[5], ElementsAre(2));
  EXPECT_THAT(output_shapes[6], ElementsAre(2, 2));

  // Check decoded outputs.
  const std::vector<std::vector<int>>& decoded_outputs = m.GetDecodedOutpus();
  EXPECT_EQ(decoded_outputs.size(), 6);
  EXPECT_THAT(decoded_outputs[0], ElementsAre(0, 0, 0, 1, 1, 0, 1, 1));
  EXPECT_THAT(decoded_outputs[1], ElementsAre(0, 0, 0, 1, 1, 0));
  EXPECT_THAT(decoded_outputs[2], ElementsAre(1, 2, 3, 0));
  EXPECT_THAT(decoded_outputs[3], ElementsAre(2, 1, 0));
  EXPECT_THAT(decoded_outputs[4], ElementsAre(2, 2));
  EXPECT_THAT(decoded_outputs[5], ElementsAre(2, 2));
  // Check log probabilities output.
  EXPECT_THAT(m.GetLogProbabilitiesOutput(),
              ElementsAreArray(
                  ArrayFloatNear({-2.65148, -2.65864, -2.17914, -2.61357})));
}

TEST(CTCBeamSearchTest, NonEqualSequencesTest) {
  CTCBeamSearchDecoderOpModel m({3, 3, 4}, {3}, 3, 1, true);
  m.PopulateTensor<float>(
      m.inputs(),
      {-1.26658163, -0.25760023, -0.03917975, -0.63772235, -0.03794756,
       -0.45063099, -0.27706473, -0.01569179, -0.59940385, -0.35700127,
       -0.48920721, -1.42635476, -1.3462478,  -0.02565498, -0.30179568,
       -0.6491698,  -0.55017719, -2.92291466, -0.92522973, -0.47592022,
       -0.07099135, -0.31575624, -0.86345281, -0.36017021, -0.79208612,
       -1.75306124, -0.65089224, -0.00912786, -0.42915003, -1.72606203,
       -1.66337589, -0.70800793, -2.52272352, -0.67329562, -2.49145522,
       -0.49786342});
  m.PopulateTensor<int>(m.sequence_length(), {1, 2, 3});
  m.Invoke();

  // Make sure the output shapes are right.
  const std::vector<std::vector<int>>& output_shapes = m.GetOutputShapes();
  EXPECT_EQ(output_shapes.size(), 4);
  EXPECT_THAT(output_shapes[0], ElementsAre(3, 2));
  EXPECT_THAT(output_shapes[1], ElementsAre(3));
  EXPECT_THAT(output_shapes[2], ElementsAre(2));
  EXPECT_THAT(output_shapes[3], ElementsAre(3, 1));

  // Check decoded outputs.
  const std::vector<std::vector<int>>& decoded_outputs = m.GetDecodedOutpus();
  EXPECT_EQ(decoded_outputs.size(), 3);
  EXPECT_THAT(decoded_outputs[0], ElementsAre(0, 0, 1, 0, 2, 0));
  EXPECT_THAT(decoded_outputs[1], ElementsAre(2, 0, 1));
  EXPECT_THAT(decoded_outputs[2], ElementsAre(3, 1));
  // Check log probabilities output.
  EXPECT_THAT(m.GetLogProbabilitiesOutput(),
              ElementsAreArray(ArrayFloatNear({-0.97322, -1.16334, -2.15553})));
}

}  // namespace
}  // namespace experimental
}  // namespace ops
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
