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
#include <stdint.h>

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

template <typename T>
class ReverseSequenceOpModel : public SingleOpModel {
 public:
  ReverseSequenceOpModel(const TensorData& input, const TensorData& seq_lengths,
                         int seq_dim, int batch_dim) {
    input_ = AddInput(input);
    seq_lengths_ = AddInput(seq_lengths);

    output_ = AddOutput({input.type, {}});

    SetBuiltinOp(
        BuiltinOperator_REVERSE_SEQUENCE, BuiltinOptions_ReverseSequenceOptions,
        CreateReverseSequenceOptions(builder_, seq_dim, batch_dim).Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }
  int seq_lengths() { return seq_lengths_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int seq_lengths_;
  int output_;
};

// float32 tests
TEST(ReverseSequenceOpTest, FloatSeqDimIsGreater) {
  ReverseSequenceOpModel<float> model({TensorType_FLOAT32, {4, 3, 2}},
                                      {TensorType_INT32, {4}}, 1, 0);
  model.PopulateTensor<float>(model.input(),
                              {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.PopulateTensor<int32_t>(model.seq_lengths(), {3, 2, 3, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({5,  6,  3,  4,  1,  2,  9,  10, 7,  8,  11, 12,
                        17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20}));
}

TEST(ReverseSequenceOpTest, FloatBatchDimIsGreater) {
  ReverseSequenceOpModel<float> model({TensorType_FLOAT32, {4, 3, 2}},
                                      {TensorType_INT32, {2}}, 0, 2);
  model.PopulateTensor<float>(model.input(),
                              {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.PopulateTensor<int32_t>(model.seq_lengths(), {3, 4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({13, 20, 15, 22, 17, 24, 7, 14, 9, 16, 11, 18, 1,
                                8,  3,  10, 5,  12, 19, 2, 21, 4, 23, 6}));
}

// int32 tests
TEST(ReverseSequenceOpTest, Int32SeqDimIsGreater) {
  ReverseSequenceOpModel<int32_t> model({TensorType_INT32, {4, 3, 2}},
                                        {TensorType_INT32, {4}}, 1, 0);
  model.PopulateTensor<int32_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.PopulateTensor<int32_t>(model.seq_lengths(), {3, 2, 3, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({5,  6,  3,  4,  1,  2,  9,  10, 7,  8,  11, 12,
                        17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20}));
}

TEST(ReverseSequenceOpTest, Int32BatchDimIsGreater) {
  ReverseSequenceOpModel<int32_t> model({TensorType_INT32, {4, 3, 2}},
                                        {TensorType_INT32, {2}}, 0, 2);
  model.PopulateTensor<int32_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.PopulateTensor<int32_t>(model.seq_lengths(), {3, 4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({13, 20, 15, 22, 17, 24, 7, 14, 9, 16, 11, 18, 1,
                                8,  3,  10, 5,  12, 19, 2, 21, 4, 23, 6}));
}

// int64 tests
TEST(ReverseSequenceOpTest, Int64SeqDimIsGreater) {
  ReverseSequenceOpModel<int64_t> model({TensorType_INT64, {4, 3, 2}},
                                        {TensorType_INT32, {4}}, 1, 0);
  model.PopulateTensor<int64_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.PopulateTensor<int32_t>(model.seq_lengths(), {3, 2, 3, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({5,  6,  3,  4,  1,  2,  9,  10, 7,  8,  11, 12,
                        17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20}));
}

TEST(ReverseSequenceOpTest, Int64BatchDimIsGreater) {
  ReverseSequenceOpModel<int64_t> model({TensorType_INT64, {4, 3, 2}},
                                        {TensorType_INT32, {2}}, 0, 2);
  model.PopulateTensor<int64_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.PopulateTensor<int32_t>(model.seq_lengths(), {3, 4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({13, 20, 15, 22, 17, 24, 7, 14, 9, 16, 11, 18, 1,
                                8,  3,  10, 5,  12, 19, 2, 21, 4, 23, 6}));
}

// uint8 tests
TEST(ReverseSequenceOpTest, Uint8SeqDimIsGreater) {
  ReverseSequenceOpModel<uint8_t> model({TensorType_UINT8, {4, 3, 2}},
                                        {TensorType_INT32, {4}}, 1, 0);
  model.PopulateTensor<uint8_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.PopulateTensor<int32_t>(model.seq_lengths(), {3, 2, 3, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({5,  6,  3,  4,  1,  2,  9,  10, 7,  8,  11, 12,
                        17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20}));
}

TEST(ReverseSequenceOpTest, Uint8BatchDimIsGreater) {
  ReverseSequenceOpModel<uint8_t> model({TensorType_UINT8, {4, 3, 2}},
                                        {TensorType_INT32, {2}}, 0, 2);
  model.PopulateTensor<uint8_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.PopulateTensor<int32_t>(model.seq_lengths(), {3, 4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({13, 20, 15, 22, 17, 24, 7, 14, 9, 16, 11, 18, 1,
                                8,  3,  10, 5,  12, 19, 2, 21, 4, 23, 6}));
}

// int16 tests
TEST(ReverseSequenceOpTest, Int16SeqDimIsGreater) {
  ReverseSequenceOpModel<int16_t> model({TensorType_INT16, {4, 3, 2}},
                                        {TensorType_INT32, {4}}, 1, 0);
  model.PopulateTensor<int16_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.PopulateTensor<int32_t>(model.seq_lengths(), {3, 2, 3, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({5,  6,  3,  4,  1,  2,  9,  10, 7,  8,  11, 12,
                        17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20}));
}

TEST(ReverseSequenceOpTest, Int16BatchDimIsGreater) {
  ReverseSequenceOpModel<int16_t> model({TensorType_INT16, {4, 3, 2}},
                                        {TensorType_INT32, {2}}, 0, 2);
  model.PopulateTensor<int16_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.PopulateTensor<int32_t>(model.seq_lengths(), {3, 4});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({13, 20, 15, 22, 17, 24, 7, 14, 9, 16, 11, 18, 1,
                                8,  3,  10, 5,  12, 19, 2, 21, 4, 23, 6}));
}

}  // namespace
}  // namespace tflite
