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
// Unit test for TFLite SVDF op.

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

static float svdf_input[] = {
    0.12609188f,  -0.46347019f, -0.89598465f,
    0.35867718f,  0.36897406f,  0.73463392f,

    0.14278367f,  -1.64410412f, -0.75222826f,
    -0.57290924f, 0.12729003f,  0.7567004f,

    0.49837467f,  0.19278903f,  0.26584083f,
    0.17660543f,  0.52949083f,  -0.77931279f,

    -0.11186574f, 0.13164264f,  -0.05349274f,
    -0.72674477f, -0.5683046f,  0.55900657f,

    -0.68892461f, 0.37783599f,  0.18263303f,
    -0.63690937f, 0.44483393f,  -0.71817774f,

    -0.81299269f, -0.86831826f, 1.43940818f,
    -0.95760226f, 1.82078898f,  0.71135032f,

    -1.45006323f, -0.82251364f, -1.69082689f,
    -1.65087092f, -1.89238167f, 1.54172635f,

    0.03966608f,  -0.24936394f, -0.77526885f,
    2.06740379f,  -1.51439476f, 1.43768692f,

    0.11771342f,  -0.23761693f, -0.65898693f,
    0.31088525f,  -1.55601168f, -0.87661445f,

    -0.89477462f, 1.67204106f,  -0.53235275f,
    -0.6230064f,  0.29819036f,  1.06939757f,
};

static float svdf_golden_output_rank_1[] = {
    0.014899f,    -0.0517661f,  -0.143725f,   -0.00271883f,
    -0.03004015f, 0.09565311f,  0.1587342f,   0.00784263f,

    0.068281f,    -0.162217f,   -0.152268f,   0.00323521f,
    0.01582633f,  0.03858774f,  -0.03001583f, -0.02671271f,

    -0.0317821f,  -0.0333089f,  0.0609602f,   0.0333759f,
    -0.01432795f, 0.05524484f,  0.1101355f,   -0.02382665f,

    -0.00623099f, -0.077701f,   -0.391193f,   -0.0136691f,
    -0.02333033f, 0.02293761f,  0.12338032f,  0.04326871f,

    0.201551f,    -0.164607f,   -0.179462f,   -0.0592739f,
    0.01064911f,  -0.17503069f, 0.07821996f,  -0.00224009f,

    0.0886511f,   -0.0875401f,  -0.269283f,   0.0281379f,
    -0.02282338f, 0.09741908f,  0.32973239f,  0.12281385f,

    -0.201174f,   -0.586145f,   -0.628624f,   -0.0330412f,
    0.24780814f,  -0.39304617f, -0.22473189f, 0.02589256f,

    -0.0839096f,  -0.299329f,   0.108746f,    0.109808f,
    0.10084175f,  -0.06416984f, 0.28936723f,  0.0026358f,

    0.419114f,    -0.237824f,   -0.422627f,   0.175115f,
    -0.2314795f,  -0.18584411f, -0.4228974f,  -0.12928449f,

    0.36726f,     -0.522303f,   -0.456502f,   -0.175475f,
    0.17012937f,  -0.34447709f, 0.38505614f,  -0.28158101f,
};

static float svdf_golden_output_rank_2[] = {
    -0.09623547f, -0.10193135f, 0.11083051f,  -0.0347917f,
    0.1141196f,   0.12965347f,  -0.12652366f, 0.01007236f,

    -0.16396809f, -0.21247184f, 0.11259045f,  -0.04156673f,
    0.10132131f,  -0.06143532f, -0.00924693f, 0.10084561f,

    0.01257364f,  0.0506071f,   -0.19287863f, -0.07162561f,
    -0.02033747f, 0.22673416f,  0.15487903f,  0.02525555f,

    -0.1411963f,  -0.37054959f, 0.01774767f,  0.05867489f,
    0.09607603f,  -0.0141301f,  -0.08995658f, 0.12867066f,

    -0.27142537f, -0.16955489f, 0.18521598f,  -0.12528358f,
    0.00331409f,  0.11167502f,  0.02218599f,  -0.07309391f,

    0.09593632f,  -0.28361851f, -0.0773851f,  0.17199151f,
    -0.00075242f, 0.33691186f,  -0.1536046f,  0.16572715f,

    -0.27916506f, -0.27626723f, 0.42615682f,  0.3225764f,
    -0.37472126f, -0.55655634f, -0.05013514f, 0.289112f,

    -0.24418658f, 0.07540751f,  -0.1940318f,  -0.08911639f,
    0.00732617f,  0.46737891f,  0.26449674f,  0.24888524f,

    -0.17225097f, -0.54660404f, -0.38795233f, 0.08389944f,
    0.07736043f,  -0.28260678f, 0.15666828f,  1.14949894f,

    -0.57454878f, -0.64704704f, 0.73235172f,  -0.34616736f,
    0.21120001f,  -0.22927976f, 0.02455296f,  -0.35906726f,
};

// Derived class of SingleOpModel, which is used to test SVDF TFLite op.
class SVDFOpModel : public SingleOpModel {
 public:
  SVDFOpModel(int batches, int units, int input_size, int memory_size, int rank)
      : batches_(batches),
        units_(units),
        input_size_(input_size),
        memory_size_(memory_size),
        rank_(rank) {
    input_ = AddInput(TensorType_FLOAT32);
    weights_feature_ = AddInput(TensorType_FLOAT32);
    weights_time_ = AddInput(TensorType_FLOAT32);
    bias_ = AddNullInput();
    state_ = AddOutput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(
        BuiltinOperator_SVDF, BuiltinOptions_SVDFOptions,
        CreateSVDFOptions(builder_, rank, ActivationFunctionType_NONE).Union());
    BuildInterpreter({
        {batches_, input_size_},        // Input tensor
        {units_ * rank, input_size_},   // weights_feature tensor
        {units_ * rank, memory_size_},  // weights_time tensor
        {units_}                        // bias tensor
    });
  }

  // Populates the weights_feature tensor.
  void SetWeightsFeature(std::initializer_list<float> f) {
    PopulateTensor(weights_feature_, f);
  }

  // Populates the weights_time tensor.
  void SetWeightsTime(std::initializer_list<float> f) {
    PopulateTensor(weights_time_, f);
  }

  // Populates the input tensor.
  void SetInput(int offset, float* begin, float* end) {
    PopulateTensor(input_, offset, begin, end);
  }

  // Resets the state of SVDF op by filling it with 0's.
  void ResetState() {
    const int zero_buffer_size = rank_ * units_ * batches_ * memory_size_;
    std::unique_ptr<float[]> zero_buffer(new float[zero_buffer_size]);
    memset(zero_buffer.get(), 0, zero_buffer_size * sizeof(float));
    PopulateTensor(state_, 0, zero_buffer.get(),
                   zero_buffer.get() + zero_buffer_size);
  }

  // Extracts the output tensor from the SVDF op.
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

  int input_size() { return input_size_; }
  int num_units() { return units_; }
  int num_batches() { return batches_; }

 private:
  int input_;
  int weights_feature_;
  int weights_time_;
  int bias_;
  int state_;
  int output_;

  int batches_;
  int units_;
  int input_size_;
  int memory_size_;
  int rank_;
};

TEST(SVDFOpTest, BlackBoxTestRank1) {
  SVDFOpModel svdf(/*batches=*/2, /*units=*/4, /*input_size=*/3,
                   /*memory_size=*/10, /*rank=*/1);
  svdf.SetWeightsFeature({-0.31930989f, -0.36118156f, 0.0079667f, 0.37613347f,
                          0.22197971f, 0.12416199f, 0.27901134f, 0.27557442f,
                          0.3905206f, -0.36137494f, -0.06634006f, -0.10640851f});

  svdf.SetWeightsTime(
      {-0.31930989f, 0.37613347f,  0.27901134f,  -0.36137494f, -0.36118156f,
       0.22197971f,  0.27557442f,  -0.06634006f, 0.0079667f,   0.12416199f,

       0.3905206f,   -0.10640851f, -0.0976817f,  0.15294972f,  0.39635518f,
       -0.02702999f, 0.39296314f,  0.15785322f,  0.21931258f,  0.31053296f,

       -0.36916667f, 0.38031587f,  -0.21580373f, 0.27072677f,  0.23622236f,
       0.34936687f,  0.18174365f,  0.35907319f,  -0.17493086f, 0.324846f,

       -0.10781813f, 0.27201805f,  0.14324132f,  -0.23681851f, -0.27115166f,
       -0.01580888f, -0.14943552f, 0.15465137f,  0.09784451f,  -0.0337657f});

  svdf.ResetState();
  const int svdf_num_batches = svdf.num_batches();
  const int svdf_input_size = svdf.input_size();
  const int svdf_num_units = svdf.num_units();
  const int input_sequence_size =
      sizeof(svdf_input) / sizeof(float) / (svdf_input_size * svdf_num_batches);
  // Going over each input batch, setting the input tensor, invoking the SVDF op
  // and checking the output with the expected golden values.
  for (int i = 0; i < input_sequence_size; i++) {
    float* batch_start = svdf_input + i * svdf_input_size * svdf_num_batches;
    float* batch_end = batch_start + svdf_input_size * svdf_num_batches;
    svdf.SetInput(0, batch_start, batch_end);

    svdf.Invoke();

    float* golden_start =
        svdf_golden_output_rank_1 + i * svdf_num_units * svdf_num_batches;
    float* golden_end = golden_start + svdf_num_units * svdf_num_batches;
    std::vector<float> expected;
    expected.insert(expected.end(), golden_start, golden_end);

    EXPECT_THAT(svdf.GetOutput(), ElementsAreArray(ArrayFloatNear(expected)));
  }
}

TEST(SVDFOpTest, BlackBoxTestRank2) {
  SVDFOpModel svdf(/*batches=*/2, /*units=*/4, /*input_size=*/3,
                   /*memory_size=*/10, /*rank=*/2);
  svdf.SetWeightsFeature({-0.31930989f, 0.0079667f,   0.39296314f,  0.37613347f,
                          0.12416199f,  0.15785322f,  0.27901134f,  0.3905206f,
                          0.21931258f,  -0.36137494f, -0.10640851f, 0.31053296f,
                          -0.36118156f, -0.0976817f,  -0.36916667f, 0.22197971f,
                          0.15294972f,  0.38031587f,  0.27557442f,  0.39635518f,
                          -0.21580373f, -0.06634006f, -0.02702999f, 0.27072677f});

  svdf.SetWeightsTime(
      {-0.31930989f, 0.37613347f,  0.27901134f,  -0.36137494f, -0.36118156f,
       0.22197971f,  0.27557442f,  -0.06634006f, 0.0079667f,   0.12416199f,

       0.3905206f,   -0.10640851f, -0.0976817f,  0.15294972f,  0.39635518f,
       -0.02702999f, 0.39296314f,  0.15785322f,  0.21931258f,  0.31053296f,

       -0.36916667f, 0.38031587f,  -0.21580373f, 0.27072677f,  0.23622236f,
       0.34936687f,  0.18174365f,  0.35907319f,  -0.17493086f, 0.324846f,

       -0.10781813f, 0.27201805f,  0.14324132f,  -0.23681851f, -0.27115166f,
       -0.01580888f, -0.14943552f, 0.15465137f,  0.09784451f,  -0.0337657f,

       -0.14884081f, 0.19931212f,  -0.36002168f, 0.34663299f,  -0.11405486f,
       0.12672701f,  0.39463779f,  -0.07886535f, -0.06384811f, 0.08249187f,

       -0.26816407f, -0.19905911f, 0.29211238f,  0.31264046f,  -0.28664589f,
       0.05698794f,  0.11613581f,  0.14078894f,  0.02187902f,  -0.21781836f,

       -0.15567942f, 0.08693647f,  -0.38256618f, 0.36580828f,  -0.22922277f,
       -0.0226903f,  0.12878349f,  -0.28122205f, -0.10850525f, -0.11955214f,

       0.27179423f,  -0.04710215f, 0.31069002f,  0.22672787f,  0.09580326f,
       0.08682203f,  0.1258215f,   0.1851041f,   0.29228821f,  0.12366763f});

  svdf.ResetState();
  const int svdf_num_batches = svdf.num_batches();
  const int svdf_input_size = svdf.input_size();
  const int svdf_num_units = svdf.num_units();
  const int input_sequence_size =
      sizeof(svdf_input) / sizeof(float) / (svdf_input_size * svdf_num_batches);
  // Going over each input batch, setting the input tensor, invoking the SVDF op
  // and checking the output with the expected golden values.
  for (int i = 0; i < input_sequence_size; i++) {
    float* batch_start = svdf_input + i * svdf_input_size * svdf_num_batches;
    float* batch_end = batch_start + svdf_input_size * svdf_num_batches;
    svdf.SetInput(0, batch_start, batch_end);

    svdf.Invoke();

    float* golden_start =
        svdf_golden_output_rank_2 + i * svdf_num_units * svdf_num_batches;
    float* golden_end = golden_start + svdf_num_units * svdf_num_batches;
    std::vector<float> expected;
    expected.insert(expected.end(), golden_start, golden_end);

    EXPECT_THAT(svdf.GetOutput(), ElementsAreArray(ArrayFloatNear(expected)));
  }
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
