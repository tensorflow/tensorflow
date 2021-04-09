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

#include <stdint.h>

#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

static float svdf_input[] = {
    0.12609188,  -0.46347019, -0.89598465,
    0.35867718,  0.36897406,  0.73463392,

    0.14278367,  -1.64410412, -0.75222826,
    -0.57290924, 0.12729003,  0.7567004,

    0.49837467,  0.19278903,  0.26584083,
    0.17660543,  0.52949083,  -0.77931279,

    -0.11186574, 0.13164264,  -0.05349274,
    -0.72674477, -0.5683046,  0.55900657,

    -0.68892461, 0.37783599,  0.18263303,
    -0.63690937, 0.44483393,  -0.71817774,

    -0.81299269, -0.86831826, 1.43940818,
    -0.95760226, 1.82078898,  0.71135032,

    -1.45006323, -0.82251364, -1.69082689,
    -1.65087092, -1.89238167, 1.54172635,

    0.03966608,  -0.24936394, -0.77526885,
    2.06740379,  -1.51439476, 1.43768692,

    0.11771342,  -0.23761693, -0.65898693,
    0.31088525,  -1.55601168, -0.87661445,

    -0.89477462, 1.67204106,  -0.53235275,
    -0.6230064,  0.29819036,  1.06939757,
};

static float svdf_golden_output_rank_1[] = {
    0.014899,    -0.0517661,  -0.143725,   -0.00271883,
    -0.03004015, 0.09565311,  0.1587342,   0.00784263,

    0.068281,    -0.162217,   -0.152268,   0.00323521,
    0.01582633,  0.03858774,  -0.03001583, -0.02671271,

    -0.0317821,  -0.0333089,  0.0609602,   0.0333759,
    -0.01432795, 0.05524484,  0.1101355,   -0.02382665,

    -0.00623099, -0.077701,   -0.391193,   -0.0136691,
    -0.02333033, 0.02293761,  0.12338032,  0.04326871,

    0.201551,    -0.164607,   -0.179462,   -0.0592739,
    0.01064911,  -0.17503069, 0.07821996,  -0.00224009,

    0.0886511,   -0.0875401,  -0.269283,   0.0281379,
    -0.02282338, 0.09741908,  0.32973239,  0.12281385,

    -0.201174,   -0.586145,   -0.628624,   -0.0330412,
    0.24780814,  -0.39304617, -0.22473189, 0.02589256,

    -0.0839096,  -0.299329,   0.108746,    0.109808,
    0.10084175,  -0.06416984, 0.28936723,  0.0026358,

    0.419114,    -0.237824,   -0.422627,   0.175115,
    -0.2314795,  -0.18584411, -0.4228974,  -0.12928449,

    0.36726,     -0.522303,   -0.456502,   -0.175475,
    0.17012937,  -0.34447709, 0.38505614,  -0.28158101,
};

static float svdf_golden_output_rank_2[] = {
    -0.09623547, -0.10193135, 0.11083051,  -0.0347917,
    0.1141196,   0.12965347,  -0.12652366, 0.01007236,

    -0.16396809, -0.21247184, 0.11259045,  -0.04156673,
    0.10132131,  -0.06143532, -0.00924693, 0.10084561,

    0.01257364,  0.0506071,   -0.19287863, -0.07162561,
    -0.02033747, 0.22673416,  0.15487903,  0.02525555,

    -0.1411963,  -0.37054959, 0.01774767,  0.05867489,
    0.09607603,  -0.0141301,  -0.08995658, 0.12867066,

    -0.27142537, -0.16955489, 0.18521598,  -0.12528358,
    0.00331409,  0.11167502,  0.02218599,  -0.07309391,

    0.09593632,  -0.28361851, -0.0773851,  0.17199151,
    -0.00075242, 0.33691186,  -0.1536046,  0.16572715,

    -0.27916506, -0.27626723, 0.42615682,  0.3225764,
    -0.37472126, -0.55655634, -0.05013514, 0.289112,

    -0.24418658, 0.07540751,  -0.1940318,  -0.08911639,
    0.00732617,  0.46737891,  0.26449674,  0.24888524,

    -0.17225097, -0.54660404, -0.38795233, 0.08389944,
    0.07736043,  -0.28260678, 0.15666828,  1.14949894,

    -0.57454878, -0.64704704, 0.73235172,  -0.34616736,
    0.21120001,  -0.22927976, 0.02455296,  -0.35906726,
};

// Derived class of SingleOpModel, which is used to test SVDF TFLite op.
class BaseSVDFOpModel : public SingleOpModel {
 public:
  BaseSVDFOpModel(int batches, int units, int input_size, int memory_size,
                  int rank,
                  TensorType weights_feature_type = TensorType_FLOAT32,
                  TensorType weights_time_type = TensorType_FLOAT32,
                  bool asymmetric_quantize_inputs = false)
      : batches_(batches),
        units_(units),
        input_size_(input_size),
        memory_size_(memory_size),
        rank_(rank) {
    input_ = AddInput(TensorType_FLOAT32);
    weights_feature_ = AddInput(weights_feature_type);
    weights_time_ = AddInput(weights_time_type);
    bias_ = AddNullInput();
    const int num_filters = units * rank;
    activation_state_ = AddVariableInput(
        TensorData{TensorType_FLOAT32, {batches, memory_size * num_filters}});
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_SVDF, BuiltinOptions_SVDFOptions,
                 CreateSVDFOptions(builder_, rank, ActivationFunctionType_NONE,
                                   asymmetric_quantize_inputs)
                     .Union());
    BuildInterpreter({
        {batches_, input_size_},              // input tensor
        {units_ * rank, input_size_},         // weights_feature tensor
        {units_ * rank, memory_size_},        // weights_time tensor
        {units_},                             // bias tensor
        {batches, memory_size * num_filters}  // activation_state tensor
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

  // Extracts the output tensor from the SVDF op.
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

  int input_size() { return input_size_; }
  int num_units() { return units_; }
  int num_batches() { return batches_; }

 protected:
  int input_;
  int weights_feature_;
  int weights_time_;
  int bias_;
  int activation_state_;
  int output_;

  int batches_;
  int units_;
  int input_size_;
  int memory_size_;
  int rank_;
};

class SVDFOpModel : public BaseSVDFOpModel {
 public:
  using BaseSVDFOpModel::BaseSVDFOpModel;
};

class HybridSVDFOpModel : public BaseSVDFOpModel {
 public:
  HybridSVDFOpModel(int batches, int units, int input_size, int memory_size,
                    int rank, TensorType tensor_type,
                    bool asymmetric_quantize_inputs)
      : BaseSVDFOpModel(batches, units, input_size, memory_size, rank,
                        tensor_type, tensor_type, asymmetric_quantize_inputs) {
    tensor_type_ = tensor_type;
  }

  void SetWeights(int weights_idx, const std::vector<float>& f) {
    if (tensor_type_ == TensorType_UINT8) {
      SymmetricQuantizeAndPopulate(weights_idx, f);
    } else {
      SignedSymmetricQuantizeAndPopulate(weights_idx, f);
    }
  }

  void SetWeightsFeature(std::initializer_list<float> f) {
    SetWeights(weights_feature_, f);
  }

  void SetWeightsTime(std::initializer_list<float> f) {
    SetWeights(weights_time_, f);
  }

 protected:
  TensorType tensor_type_;
};

class SVDFOpTest : public ::testing::TestWithParam<bool> {
 protected:
  void VerifyGoldens(float golden_input[], float golden_output[],
                     int golden_size, BaseSVDFOpModel* svdf,
                     float tolerance = 1e-5) {
    const int svdf_num_batches = svdf->num_batches();
    const int svdf_input_size = svdf->input_size();
    const int svdf_num_units = svdf->num_units();
    const int input_sequence_size =
        golden_size / sizeof(float) / (svdf_input_size * svdf_num_batches);
    // Going over each input batch, setting the input tensor, invoking the SVDF
    // op and checking the output with the expected golden values.
    for (int i = 0; i < input_sequence_size; i++) {
      float* batch_start =
          golden_input + i * svdf_input_size * svdf_num_batches;
      float* batch_end = batch_start + svdf_input_size * svdf_num_batches;
      svdf->SetInput(0, batch_start, batch_end);

      svdf->Invoke();

      const float* golden_start =
          golden_output + i * svdf_num_units * svdf_num_batches;
      const float* golden_end =
          golden_start + svdf_num_units * svdf_num_batches;
      std::vector<float> expected;
      expected.insert(expected.end(), golden_start, golden_end);

      EXPECT_THAT(svdf->GetOutput(),
                  ElementsAreArray(ArrayFloatNear(expected, tolerance)));
    }
  }
};

INSTANTIATE_TEST_SUITE_P(SVDFOpTest, SVDFOpTest,
                         ::testing::ValuesIn({false, true}));

TEST_F(SVDFOpTest, BlackBoxTestRank1) {
  SVDFOpModel svdf(/*batches=*/2, /*units=*/4, /*input_size=*/3,
                   /*memory_size=*/10, /*rank=*/1);
  svdf.SetWeightsFeature({-0.31930989, -0.36118156, 0.0079667, 0.37613347,
                          0.22197971, 0.12416199, 0.27901134, 0.27557442,
                          0.3905206, -0.36137494, -0.06634006, -0.10640851});

  svdf.SetWeightsTime(
      {-0.31930989, 0.37613347,  0.27901134,  -0.36137494, -0.36118156,
       0.22197971,  0.27557442,  -0.06634006, 0.0079667,   0.12416199,

       0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
       -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

       -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
       0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

       -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
       -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657});

  VerifyGoldens(svdf_input, svdf_golden_output_rank_1, sizeof(svdf_input),
                &svdf);
}

TEST_F(SVDFOpTest, BlackBoxTestRank2) {
  SVDFOpModel svdf(/*batches=*/2, /*units=*/4, /*input_size=*/3,
                   /*memory_size=*/10, /*rank=*/2);
  svdf.SetWeightsFeature({-0.31930989, 0.0079667,   0.39296314,  0.37613347,
                          0.12416199,  0.15785322,  0.27901134,  0.3905206,
                          0.21931258,  -0.36137494, -0.10640851, 0.31053296,
                          -0.36118156, -0.0976817,  -0.36916667, 0.22197971,
                          0.15294972,  0.38031587,  0.27557442,  0.39635518,
                          -0.21580373, -0.06634006, -0.02702999, 0.27072677});

  svdf.SetWeightsTime(
      {-0.31930989, 0.37613347,  0.27901134,  -0.36137494, -0.36118156,
       0.22197971,  0.27557442,  -0.06634006, 0.0079667,   0.12416199,

       0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
       -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

       -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
       0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

       -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
       -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657,

       -0.14884081, 0.19931212,  -0.36002168, 0.34663299,  -0.11405486,
       0.12672701,  0.39463779,  -0.07886535, -0.06384811, 0.08249187,

       -0.26816407, -0.19905911, 0.29211238,  0.31264046,  -0.28664589,
       0.05698794,  0.11613581,  0.14078894,  0.02187902,  -0.21781836,

       -0.15567942, 0.08693647,  -0.38256618, 0.36580828,  -0.22922277,
       -0.0226903,  0.12878349,  -0.28122205, -0.10850525, -0.11955214,

       0.27179423,  -0.04710215, 0.31069002,  0.22672787,  0.09580326,
       0.08682203,  0.1258215,   0.1851041,   0.29228821,  0.12366763});

  VerifyGoldens(svdf_input, svdf_golden_output_rank_2, sizeof(svdf_input),
                &svdf);
}

TEST_P(SVDFOpTest, BlackBoxTestHybridRank1Uint8) {
  HybridSVDFOpModel svdf(/*batches=*/2, /*units=*/4, /*input_size=*/3,
                         /*memory_size=*/10, /*rank=*/1, TensorType_UINT8,
                         GetParam());
  svdf.SetWeightsFeature({-0.31930989, -0.36118156, 0.0079667, 0.37613347,
                          0.22197971, 0.12416199, 0.27901134, 0.27557442,
                          0.3905206, -0.36137494, -0.06634006, -0.10640851});

  svdf.SetWeightsTime(
      {-0.31930989, 0.37613347,  0.27901134,  -0.36137494, -0.36118156,
       0.22197971,  0.27557442,  -0.06634006, 0.0079667,   0.12416199,

       0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
       -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

       -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
       0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

       -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
       -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657});

  VerifyGoldens(svdf_input, svdf_golden_output_rank_1, sizeof(svdf_input),
                &svdf,
                /*tolerance=*/0.004285);
}

TEST_P(SVDFOpTest, BlackBoxTestHybridRank2Uint8) {
  HybridSVDFOpModel svdf(/*batches=*/2, /*units=*/4, /*input_size=*/3,
                         /*memory_size=*/10, /*rank=*/2, TensorType_UINT8,
                         GetParam());
  svdf.SetWeightsFeature({-0.31930989, 0.0079667,   0.39296314,  0.37613347,
                          0.12416199,  0.15785322,  0.27901134,  0.3905206,
                          0.21931258,  -0.36137494, -0.10640851, 0.31053296,
                          -0.36118156, -0.0976817,  -0.36916667, 0.22197971,
                          0.15294972,  0.38031587,  0.27557442,  0.39635518,
                          -0.21580373, -0.06634006, -0.02702999, 0.27072677});

  svdf.SetWeightsTime(
      {-0.31930989, 0.37613347,  0.27901134,  -0.36137494, -0.36118156,
       0.22197971,  0.27557442,  -0.06634006, 0.0079667,   0.12416199,

       0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
       -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

       -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
       0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

       -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
       -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657,

       -0.14884081, 0.19931212,  -0.36002168, 0.34663299,  -0.11405486,
       0.12672701,  0.39463779,  -0.07886535, -0.06384811, 0.08249187,

       -0.26816407, -0.19905911, 0.29211238,  0.31264046,  -0.28664589,
       0.05698794,  0.11613581,  0.14078894,  0.02187902,  -0.21781836,

       -0.15567942, 0.08693647,  -0.38256618, 0.36580828,  -0.22922277,
       -0.0226903,  0.12878349,  -0.28122205, -0.10850525, -0.11955214,

       0.27179423,  -0.04710215, 0.31069002,  0.22672787,  0.09580326,
       0.08682203,  0.1258215,   0.1851041,   0.29228821,  0.12366763});

  VerifyGoldens(svdf_input, svdf_golden_output_rank_2, sizeof(svdf_input),
                &svdf,
                /*tolerance=*/0.007175);
}

TEST_P(SVDFOpTest, BlackBoxTestHybridRank1Int8) {
  HybridSVDFOpModel svdf(/*batches=*/2, /*units=*/4, /*input_size=*/3,
                         /*memory_size=*/10, /*rank=*/1, TensorType_INT8,
                         GetParam());
  svdf.SetWeightsFeature({-0.31930989, -0.36118156, 0.0079667, 0.37613347,
                          0.22197971, 0.12416199, 0.27901134, 0.27557442,
                          0.3905206, -0.36137494, -0.06634006, -0.10640851});

  svdf.SetWeightsTime(
      {-0.31930989, 0.37613347,  0.27901134,  -0.36137494, -0.36118156,
       0.22197971,  0.27557442,  -0.06634006, 0.0079667,   0.12416199,

       0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
       -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

       -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
       0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

       -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
       -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657});

  VerifyGoldens(svdf_input, svdf_golden_output_rank_1, sizeof(svdf_input),
                &svdf,
                /*tolerance=*/0.004285);
}

TEST_P(SVDFOpTest, BlackBoxTestHybridRank2Int8) {
  HybridSVDFOpModel svdf(/*batches=*/2, /*units=*/4, /*input_size=*/3,
                         /*memory_size=*/10, /*rank=*/2, TensorType_INT8,
                         GetParam());
  svdf.SetWeightsFeature({-0.31930989, 0.0079667,   0.39296314,  0.37613347,
                          0.12416199,  0.15785322,  0.27901134,  0.3905206,
                          0.21931258,  -0.36137494, -0.10640851, 0.31053296,
                          -0.36118156, -0.0976817,  -0.36916667, 0.22197971,
                          0.15294972,  0.38031587,  0.27557442,  0.39635518,
                          -0.21580373, -0.06634006, -0.02702999, 0.27072677});

  svdf.SetWeightsTime(
      {-0.31930989, 0.37613347,  0.27901134,  -0.36137494, -0.36118156,
       0.22197971,  0.27557442,  -0.06634006, 0.0079667,   0.12416199,

       0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
       -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

       -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
       0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

       -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
       -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657,

       -0.14884081, 0.19931212,  -0.36002168, 0.34663299,  -0.11405486,
       0.12672701,  0.39463779,  -0.07886535, -0.06384811, 0.08249187,

       -0.26816407, -0.19905911, 0.29211238,  0.31264046,  -0.28664589,
       0.05698794,  0.11613581,  0.14078894,  0.02187902,  -0.21781836,

       -0.15567942, 0.08693647,  -0.38256618, 0.36580828,  -0.22922277,
       -0.0226903,  0.12878349,  -0.28122205, -0.10850525, -0.11955214,

       0.27179423,  -0.04710215, 0.31069002,  0.22672787,  0.09580326,
       0.08682203,  0.1258215,   0.1851041,   0.29228821,  0.12366763});

  VerifyGoldens(svdf_input, svdf_golden_output_rank_2, sizeof(svdf_input),
                &svdf,
                /*tolerance=*/0.007175);
}

// Test case for full integer quantization of SVDF.
class IntegerSVDFOpModel : public SingleOpModel {
 public:
  IntegerSVDFOpModel(int batches, int units, int input_size, int memory_size,
                     int rank)
      : batches_(batches),
        units_(units),
        input_size_(input_size),
        memory_size_(memory_size),
        rank_(rank) {
    const int num_filters = units * rank;
    input_ = AddInput({TensorType_INT8, {batches, input_size}, -1, 1});
    weights_feature_ =
        AddInput({TensorType_INT8, {num_filters, input_size}, -0.5, 0.5});
    weights_time_ =
        AddInput({TensorType_INT16, {num_filters, memory_size}, -1, 1});
    bias_ = AddInput({TensorType_INT32, {units}, -512, 512});
    activation_state_ = AddVariableInput(
        {TensorType_INT16, {batches, memory_size * num_filters}, -16, 16});
    output_ = AddOutput({TensorType_INT8, {batches, units}, -0.5, 0.5});
    SetBuiltinOp(
        BuiltinOperator_SVDF, BuiltinOptions_SVDFOptions,
        CreateSVDFOptions(builder_, rank, ActivationFunctionType_RELU).Union());
    BuildInterpreter({
        {batches, input_size},                // input tensor
        {num_filters, input_size},            // weights_feature tensor
        {num_filters, memory_size},           // weights_time tensor
        {units},                              // bias tensor
        {batches, memory_size * num_filters}  // activation_state tensor
    });
  }

  // Populates the weights_feature tensor.
  void SetWeightsFeature(const std::vector<float>& f) {
    QuantizeAndPopulate<int8_t>(weights_feature_, f);
  }

  // Populates the weights_time tensor.
  void SetWeightsTime(const std::vector<float>& f) {
    QuantizeAndPopulate<int16_t>(weights_time_, f);
  }

  void SetBias(const std::vector<float>& f) {
    QuantizeAndPopulate<int32_t>(bias_, f);
  }

  // Populates the input tensor.
  void SetInput(const std::vector<float>& f) {
    QuantizeAndPopulate<int8_t>(input_, f);
  }

  // Extracts the output tensor from the SVDF op.
  std::vector<int8_t> GetOutput() { return ExtractVector<int8_t>(output_); }

 protected:
  int input_;
  int weights_feature_;
  int weights_time_;
  int bias_;
  int activation_state_;
  int output_;

  int batches_;
  int units_;
  int input_size_;
  int memory_size_;
  int rank_;
};

TEST_F(SVDFOpTest, BlackBoxTestInteger) {
  IntegerSVDFOpModel svdf(/*batches=*/2, /*units=*/4, /*input_size=*/3,
                          /*memory_size=*/10, /*rank=*/1);
  svdf.SetWeightsFeature({-0.31930989, -0.36118156, 0.0079667, 0.37613347,
                          0.22197971, 0.12416199, 0.27901134, 0.27557442,
                          0.3905206, -0.36137494, -0.06634006, -0.10640851});

  svdf.SetWeightsTime(
      {-0.31930989, 0.37613347,  0.27901134,  -0.36137494, -0.36118156,
       0.22197971,  0.27557442,  -0.06634006, 0.0079667,   0.12416199,

       0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
       -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

       -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
       0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

       -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
       -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657});

  svdf.SetBias({-0.0976817, 0.15294972, 0.39635518, -0.02702999});

  const std::vector<std::vector<float>> input_sequences = {
      {0.49837467, 0.19278903, 0.26584083, 0.17660543, 0.52949083, -0.77931279},
      {0.12609188, -0.46347019, -0.89598465, 0.35867718, 0.36897406,
       0.73463392},
      {0.14278367, -1.64410412, -0.75222826, -0.57290924, 0.12729003,
       0.7567004},
      {0.49837467, 0.19278903, 0.26584083, 0.17660543, 0.52949083, -0.77931279},
      {0.12609188, -0.46347019, -0.89598465, 0.35867718, 0.36897406,
       0.73463392},
      {0.14278367, -1.64410412, -0.75222826, -0.57290924, 0.12729003,
       0.7567004},
      {0.49837467, 0.19278903, 0.26584083, 0.17660543, 0.52949083, -0.77931279},
      {0.12609188, -0.46347019, -0.89598465, 0.35867718, 0.36897406,
       0.73463392},
      {0.14278367, -1.64410412, -0.75222826, -0.57290924, 0.12729003,
       0.7567004},
      {0.49837467, 0.19278903, 0.26584083, 0.17660543, 0.52949083, -0.77931279},
      {0.12609188, -0.46347019, -0.89598465, 0.35867718, 0.36897406,
       0.73463392},
      {0.14278367, -1.64410412, -0.75222826, -0.57290924, 0.12729003,
       0.7567004}};

  const std::vector<std::vector<int8_t>> expected_output = {
      {-9, 24, 31, 1, -10, 10, -3, 0},
      {2, 4, -44, -7, -10, 32, 52, 1},
      {12, -17, 9, -8, 7, 16, -11, -8},
      {-26, 29, 28, 16, -23, 26, 30, -6},
      {-8, -25, -86, -5, -44, 59, 81, 15},
      {62, -16, -37, 3, 27, 14, 34, -10},
      {1, 24, -25, 23, 31, 61, 67, 11},
      {-64, -65, -128, -25, -53, 59, 127, 20},
      {20, -29, -20, -15, -28, 0, 8, -27},
      {54, 61, -67, 38, 38, 64, 115, 0},
      {-44, -75, -128, -20, -19, 93, 101, 35},
      {-5, -56, 30, -18, -40, -9, -8, -31},
  };

  for (int sequence_index = 0; sequence_index < 12; ++sequence_index) {
    svdf.SetInput(input_sequences[sequence_index]);
    svdf.Invoke();
    const std::vector<int8_t> res = svdf.GetOutput();
    EXPECT_THAT(res, ElementsAreArray(expected_output[sequence_index]));
  }
}

}  // namespace
}  // namespace tflite
