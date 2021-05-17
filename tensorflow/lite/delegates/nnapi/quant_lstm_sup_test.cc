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
#include "tensorflow/lite/delegates/nnapi/quant_lstm_sup.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/testing/util.h"

namespace {

using ::testing::ElementsAreArray;
using ::testing::Test;

class DimsAllocatingTest : public Test {
 protected:
  DimsAllocatingTest() : allocated_dims_() {}

  ~DimsAllocatingTest() override {
    for (TfLiteIntArray* dim : allocated_dims_) {
      TfLiteIntArrayFree(dim);
    }
  }

  TfLiteIntArray* CreateDimArray(int size,
                                 std::initializer_list<int> dimensions) {
    TfLiteIntArray* dims = TfLiteIntArrayCreate(size);
    allocated_dims_.push_back(dims);

    int i = 0;
    for (const int dimension : dimensions) {
      dims->data[i++] = dimension;
    }

    return dims;
  }

 private:
  std::vector<TfLiteIntArray*> allocated_dims_;
};

using tflite::delegate::nnapi::ExtractQuantLstmWeightsSubmatrix;

class ExtractQuantLstmWeightsSubmatrixTest : public DimsAllocatingTest {};

TEST_F(ExtractQuantLstmWeightsSubmatrixTest, TopLeftSubmatrixIsExtracted) {
  std::vector<uint8_t> weights = {1,   2,   3,   4,   5,    //
                                  11,  12,  13,  14,  15,   //
                                  101, 102, 103, 104, 105,  //
                                  111, 112, 113, 114, 115,  //
                                  201, 202, 203, 204, 205,  //
                                  211, 212, 213, 214, 215,  //
                                  221, 222, 223, 224, 225,  //
                                  231, 232, 233, 234, 235};
  const TfLiteIntArray* weight_dims = CreateDimArray(2, {8, 5});

  std::vector<uint8_t> submatrix;
  const TfLiteIntArray* submatrix_dims = CreateDimArray(2, {2, 3});

  ExtractQuantLstmWeightsSubmatrix(submatrix_dims, 0 /* offset_row */,
                                   0 /* offset_column */, weight_dims,
                                   weights.data(), &submatrix);

  EXPECT_THAT(submatrix, ElementsAreArray({1, 2, 3, 11, 12, 13}));
}

TEST_F(ExtractQuantLstmWeightsSubmatrixTest, TopRightSubmatrixIsExtracted) {
  std::vector<uint8_t> weights = {1,   2,   3,   4,   5,    //
                                  11,  12,  13,  14,  15,   //
                                  101, 102, 103, 104, 105,  //
                                  111, 112, 113, 114, 115,  //
                                  201, 202, 203, 204, 205,  //
                                  211, 212, 213, 214, 215,  //
                                  221, 222, 223, 224, 225,  //
                                  231, 232, 233, 234, 235};
  const TfLiteIntArray* weight_dims = CreateDimArray(2, {8, 5});

  std::vector<uint8_t> submatrix;
  const TfLiteIntArray* submatrix_dims = CreateDimArray(2, {2, 2});

  ExtractQuantLstmWeightsSubmatrix(submatrix_dims, 0 /* offset_row */,
                                   3 /* offset_column */, weight_dims,
                                   weights.data(), &submatrix);

  EXPECT_THAT(submatrix, ElementsAreArray({4, 5, 14, 15}));
}

TEST_F(ExtractQuantLstmWeightsSubmatrixTest, RightCentralSubmatrixIsExtracted) {
  std::vector<uint8_t> weights = {1,   2,   3,   4,   5,    //
                                  11,  12,  13,  14,  15,   //
                                  101, 102, 103, 104, 105,  //
                                  111, 112, 113, 114, 115,  //
                                  201, 202, 203, 204, 205,  //
                                  211, 212, 213, 214, 215,  //
                                  221, 222, 223, 224, 225,  //
                                  231, 232, 233, 234, 235};
  const TfLiteIntArray* weight_dims = CreateDimArray(2, {8, 5});

  std::vector<uint8_t> submatrix;
  const TfLiteIntArray* submatrix_dims = CreateDimArray(2, {2, 2});

  ExtractQuantLstmWeightsSubmatrix(
      submatrix_dims, 1 * submatrix_dims->data[0] /* offset_row */,
      3 /* offset_column */, weight_dims, weights.data(), &submatrix);

  EXPECT_THAT(submatrix, ElementsAreArray({104, 105, 114, 115}));
}

using tflite::delegate::nnapi::DecomposeQuantLstmWeightsTensor;

class QuantLstmWeightDecompTest : public DimsAllocatingTest {
 protected:
  QuantLstmWeightDecompTest()
      : weights_({1,   2,   3,   4,   5,    //
                  11,  12,  13,  14,  15,   //
                  101, 102, 103, 104, 105,  //
                  111, 112, 113, 114, 115,  //
                  201, 202, 203, 204, 205,  //
                  211, 212, 213, 214, 215,  //
                  221, 222, 223, 224, 225,  //
                  231, 232, 233, 234, 235}),
        // Creating the arrays empty, the size is set by the decomposition
        // function
        recurrent_to_input_(),
        input_to_input_(),
        recurrent_to_cell_(),
        input_to_cell_(),
        recurrent_to_forget_(),
        input_to_forget_(),
        recurrent_to_output_(),
        input_to_output_() {
    weight_dims_ = CreateDimArray(2, {8, 5});
  }

  const std::vector<uint8_t> weights_;
  const TfLiteIntArray* weight_dims_;
  std::vector<uint8_t> recurrent_to_input_;
  std::vector<uint8_t> input_to_input_;
  std::vector<uint8_t> recurrent_to_cell_;
  std::vector<uint8_t> input_to_cell_;
  std::vector<uint8_t> recurrent_to_forget_;
  std::vector<uint8_t> input_to_forget_;
  std::vector<uint8_t> recurrent_to_output_;
  std::vector<uint8_t> input_to_output_;
};

TEST_F(QuantLstmWeightDecompTest, ExtractRecurrentToInput) {
  DecomposeQuantLstmWeightsTensor(
      weights_.data(), weight_dims_, &recurrent_to_input_, &input_to_input_,
      &recurrent_to_cell_, &input_to_cell_, &recurrent_to_forget_,
      &input_to_forget_, &recurrent_to_output_, &input_to_output_);

  EXPECT_THAT(recurrent_to_input_, ElementsAreArray({1, 2,  //
                                                     11, 12}));
}

TEST_F(QuantLstmWeightDecompTest, ExtractInputToInput) {
  DecomposeQuantLstmWeightsTensor(
      weights_.data(), weight_dims_, &recurrent_to_input_, &input_to_input_,
      &recurrent_to_cell_, &input_to_cell_, &recurrent_to_forget_,
      &input_to_forget_, &recurrent_to_output_, &input_to_output_);

  EXPECT_THAT(input_to_input_, ElementsAreArray({3, 4, 5,  //
                                                 13, 14, 15}));
}

TEST_F(QuantLstmWeightDecompTest, ExtractRecurrentToCell) {
  DecomposeQuantLstmWeightsTensor(
      weights_.data(), weight_dims_, &recurrent_to_input_, &input_to_input_,
      &recurrent_to_cell_, &input_to_cell_, &recurrent_to_forget_,
      &input_to_forget_, &recurrent_to_output_, &input_to_output_);

  EXPECT_THAT(recurrent_to_cell_, ElementsAreArray({101, 102,  //
                                                    111, 112}));
}

TEST_F(QuantLstmWeightDecompTest, ExtractInputToCell) {
  DecomposeQuantLstmWeightsTensor(
      weights_.data(), weight_dims_, &recurrent_to_input_, &input_to_input_,
      &recurrent_to_cell_, &input_to_cell_, &recurrent_to_forget_,
      &input_to_forget_, &recurrent_to_output_, &input_to_output_);

  EXPECT_THAT(input_to_cell_, ElementsAreArray({103, 104, 105,  //
                                                113, 114, 115}));
}

TEST_F(QuantLstmWeightDecompTest, ExtractRecurrentToForget) {
  DecomposeQuantLstmWeightsTensor(
      weights_.data(), weight_dims_, &recurrent_to_input_, &input_to_input_,
      &recurrent_to_cell_, &input_to_cell_, &recurrent_to_forget_,
      &input_to_forget_, &recurrent_to_output_, &input_to_output_);

  EXPECT_THAT(recurrent_to_forget_, ElementsAreArray({201, 202,  //
                                                      211, 212}));
}

TEST_F(QuantLstmWeightDecompTest, ExtractInputToForget) {
  DecomposeQuantLstmWeightsTensor(
      weights_.data(), weight_dims_, &recurrent_to_input_, &input_to_input_,
      &recurrent_to_cell_, &input_to_cell_, &recurrent_to_forget_,
      &input_to_forget_, &recurrent_to_output_, &input_to_output_);

  EXPECT_THAT(input_to_forget_, ElementsAreArray({203, 204, 205,  //
                                                  213, 214, 215}));
}

TEST_F(QuantLstmWeightDecompTest, ExtractRecurrentToOutput) {
  DecomposeQuantLstmWeightsTensor(
      weights_.data(), weight_dims_, &recurrent_to_input_, &input_to_input_,
      &recurrent_to_cell_, &input_to_cell_, &recurrent_to_forget_,
      &input_to_forget_, &recurrent_to_output_, &input_to_output_);

  EXPECT_THAT(recurrent_to_output_, ElementsAreArray({221, 222,  //
                                                      231, 232}));
}

TEST_F(QuantLstmWeightDecompTest, ExtractInputToOutput) {
  DecomposeQuantLstmWeightsTensor(
      weights_.data(), weight_dims_, &recurrent_to_input_, &input_to_input_,
      &recurrent_to_cell_, &input_to_cell_, &recurrent_to_forget_,
      &input_to_forget_, &recurrent_to_output_, &input_to_output_);

  EXPECT_THAT(input_to_output_, ElementsAreArray({223, 224, 225,  //
                                                  233, 234, 235}));
}

using tflite::delegate::nnapi::DecomposeBiasTensor;

TEST(DecomposeBiasTensor, ExtractInputBias) {
  // clang-format off
  std::vector<int32_t> biases
      // inputGateBias
      {-7876, 13488, -726, 32839,
      // cellGateBias
      39481, 48624, 48976, -21419,
      // forgetGateBias
      9206, -46884, -11693, -38724,
      // outputGateBias
      -58999, -17050, -41852, -40538};
  // clang-format on

  std::vector<int32_t> input_bias;
  std::vector<int32_t> cell_bias;
  std::vector<int32_t> forget_bias;
  std::vector<int32_t> output_bias;
  DecomposeBiasTensor(biases.data(), 4, &input_bias, &cell_bias, &forget_bias,
                      &output_bias);

  EXPECT_THAT(input_bias, ElementsAreArray({-7876, 13488, -726, 32839}));
}

TEST(DecomposeBiasTensor, ExtractCellBias) {
  // clang-format off
  std::vector<int32_t> biases
      // inputGateBias
      {-7876, 13488, -726, 32839,
      // cellGateBias
      39481, 48624, 48976, -21419,
      // forgetGateBias
      9206, -46884, -11693, -38724,
      // outputGateBias
      -58999, -17050, -41852, -40538};
  // clang-format on

  std::vector<int32_t> input_bias;
  std::vector<int32_t> cell_bias;
  std::vector<int32_t> forget_bias;
  std::vector<int32_t> output_bias;
  DecomposeBiasTensor(biases.data(), 4, &input_bias, &cell_bias, &forget_bias,
                      &output_bias);

  EXPECT_THAT(cell_bias, ElementsAreArray({39481, 48624, 48976, -21419}));
}

TEST(DecomposeBiasTensor, ExtractForgetBias) {
  // clang-format off
  std::vector<int32_t> biases
      // inputGateBias
      {-7876, 13488, -726, 32839,
      // cellGateBias
      39481, 48624, 48976, -21419,
      // forgetGateBias
      9206, -46884, -11693, -38724,
      // outputGateBias
      -58999, -17050, -41852, -40538};
  // clang-format on

  std::vector<int32_t> input_bias;
  std::vector<int32_t> cell_bias;
  std::vector<int32_t> forget_bias;
  std::vector<int32_t> output_bias;
  DecomposeBiasTensor(biases.data(), 4, &input_bias, &cell_bias, &forget_bias,
                      &output_bias);

  EXPECT_THAT(forget_bias, ElementsAreArray({9206, -46884, -11693, -38724}));
}

TEST(DecomposeBiasTensor, ExtractOutputBias) {
  // clang-format off
  std::vector<int32_t> biases
      // inputGateBias
      {-7876, 13488, -726, 32839,
      // cellGateBias
      39481, 48624, 48976, -21419,
      // forgetGateBias
      9206, -46884, -11693, -38724,
      // outputGateBias
      -58999, -17050, -41852, -40538};
  // clang-format on

  std::vector<int32_t> input_bias;
  std::vector<int32_t> cell_bias;
  std::vector<int32_t> forget_bias;
  std::vector<int32_t> output_bias;
  DecomposeBiasTensor(biases.data(), 4, &input_bias, &cell_bias, &forget_bias,
                      &output_bias);

  EXPECT_THAT(output_bias, ElementsAreArray({-58999, -17050, -41852, -40538}));
}

}  // namespace
