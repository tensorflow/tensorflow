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
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class QuantizeOpModel : public SingleOpModel {
 public:
  explicit QuantizeOpModel() {}

  QuantizeOpModel(const TensorData& input, const TensorData& output,
                  bool allocate_and_delegate = true) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_QUANTIZE, BuiltinOptions_QuantizeOptions,
                 CreateQuantizeOptions(builder_).Union());

    BuildInterpreter({GetShape(input_)}, /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true, allocate_and_delegate);
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  template <typename T>
  void SetInputAndQuantize(std::initializer_list<float> data) {
    QuantizeAndPopulate<T>(input_, data);
  }

  void SetInputInt8(const std::vector<int8_t>& data) {
    PopulateTensor<int8_t>(input_, data);
  }

  void SetInputInt4(const std::vector<int8_t>& data) {
    PopulateTensor4bit(input_, /*offset=*/0, data.data(),
                       data.data() + data.size());
  }

  void SetInputInt2(const std::vector<int8_t>& data) {
    PopulateTensor2bit(input_, /*offset=*/0, data.data(),
                       data.data() + data.size());
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  std::vector<int8_t> GetOutputUnpackedInt4() {
    TfLiteTensor* t = interpreter_->tensor(output_);
    int num_elements = NumElements(t);
    std::vector<int8_t> unpacked_output(num_elements);
    tensor_utils::UnpackPackedIntToInt8(t->data.int8, num_elements,
                                        /*bit_width=*/4,
                                        unpacked_output.data());
    return unpacked_output;
  }

  std::vector<int8_t> GetOutputUnpackedInt2() {
    TfLiteTensor* t = interpreter_->tensor(output_);
    int num_elements = NumElements(t);
    std::vector<int8_t> unpacked_output(num_elements);
    tensor_utils::UnpackPackedIntToInt8(t->data.int8, num_elements,
                                        /*bit_width=*/2,
                                        unpacked_output.data());
    return unpacked_output;
  }

  std::vector<int8_t> GetOutputUnpackedUInt4() {
    TfLiteTensor* t = interpreter_->tensor(output_);
    int num_elements = NumElements(t);
    std::vector<int8_t> unpacked_output(num_elements);
    const int8_t* src_buffer = t->data.int8;
    for (int i = 0; i < num_elements; i += 2) {
      int8_t src_val = src_buffer[i / 2];
      unpacked_output[i] = src_val & 0x0F;
      if (i + 1 < num_elements) {
        unpacked_output[i + 1] = (src_val >> 4) & 0x0F;
      }
    }
    return unpacked_output;
  }

 protected:
  int input_;
  int output_;
};

class QuantizePerChannelOpModel : public QuantizeOpModel {
 public:
  QuantizePerChannelOpModel(TensorType inputType, TensorType outputType,
                            std::initializer_list<int> shape,
                            std::initializer_list<float> scales,
                            std::initializer_list<int64_t> zero_points,
                            int channel_dim) {
    std::vector<float> per_channel_scales(scales);
    std::vector<int64_t> per_channel_quantization_offsets(zero_points);
    const TensorData output_tensor_data = {outputType,
                                           shape,
                                           0 /*=min*/,
                                           0 /*=max*/,
                                           0.0f /*=scale*/,
                                           0 /*=zero_point*/,
                                           true /*=per_channel_quantization*/,
                                           per_channel_scales,
                                           per_channel_quantization_offsets,
                                           channel_dim};
    input_ = AddInput({inputType, shape});
    output_ = AddOutput(output_tensor_data);
    SetBuiltinOp(BuiltinOperator_QUANTIZE, BuiltinOptions_QuantizeOptions,
                 CreateQuantizeOptions(builder_).Union());

    BuildInterpreter({GetShape(input_)});
  }
};

// Per-node quantization tests.

TEST(QuantizeOpTest, UINT8) {
  // [-63.5, 64] -> scale=0.5 zero_point=127 for UINT8
  QuantizeOpModel m({TensorType_FLOAT32, {2, 5}},
                    {TensorType_UINT8, {2, 5}, 0, 0, 0.5, 127});

  m.SetInput({-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 251, 252, 253, 254, 255}));
}

TEST(QuantizeOpTest, INT8) {
  // [-63.5, 64] -> scale=0.5, zero_point=1 for INT8
  QuantizeOpModel m({TensorType_FLOAT32, {2, 5}},
                    {TensorType_INT8, {2, 5}, 0, 0, 0.5, -1});

  m.SetInput({-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray(
                  {-128, -127, -126, -125, -124, 123, 124, 125, 126, 127}));
}

TEST(QuantizeOpTest, INT16) {
  QuantizeOpModel m({TensorType_FLOAT32, {2, 5}},
                    {TensorType_INT16, {2, 5}, 0, 0, 0.005, 0});

  m.SetInput({-63.5, -63, -3, -2, -1, 1, 2, 3, 63.5, 64});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray({-12700, -12600, -600, -400, -200, 200, 400, 600,
                                12700, 12800}));
}
TEST(QuantizeOpTest, INT4) {
  // [-3.5, 4] -> scale=0.5, zero_point=0 for INT4
  QuantizeOpModel m({TensorType_FLOAT32, {2, 5}},
                    {TensorType_INT4, {2, 5}, 0, 0, 0.5, 0});

  m.SetInput({-4.5, -3, -2.5, -2, -1.5, 2, 2.5, 3, 3.5, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  // Range of int4 is [-8, 7]. Values over the range are clamped to the range.
  EXPECT_THAT(m.GetOutputUnpackedInt4(),
              ElementsAreArray({-8, -6, -5, -4, -3, 4, 5, 6, 7, 7}));
}

TEST(QuantizeOpTest, UINT4) {
  // [-4, 3.5] -> scale=0.5, zero_point=8 for UINT4
  QuantizeOpModel m({TensorType_FLOAT32, {1, 8}},
                    {TensorType_UINT4, {1, 8}, 0, 0, 0.5, 8});

  m.SetInput({-4.5, -4, -3.5, 0, 0.5, 3, 3.5, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  // Range of uint4 is [0, 15]. Values over the range are clamped to the range.
  EXPECT_THAT(m.GetOutputUnpackedUInt4(),
              ElementsAreArray({0, 0, 1, 8, 9, 14, 15, 15}));
}

// Per-channel quantization tests.

TEST(QuantizePerChannelOpTest, UINT8) {
  // [-63.5, 64] -> scale=0.5 zero_point=127 for UINT8
  QuantizePerChannelOpModel m(TensorType_FLOAT32, TensorType_UINT8, {2, 5},
                              {0.5, 0.5}, {127, 127}, 0);

  m.SetInput({-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 251, 252, 253, 254, 255}));
}

TEST(QuantizePerChannelOpTest, INT8) {
  // [-63.5, 64] -> scale=0.5, zero_point=1 for INT8
  QuantizePerChannelOpModel m(TensorType_FLOAT32, TensorType_INT8, {2, 5},
                              {0.5, 0.5}, {-1, -1}, 0);

  m.SetInput({-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray(
                  {-128, -127, -126, -125, -124, 123, 124, 125, 126, 127}));
}

TEST(QuantizePerChannelOpTest, INT16) {
  // [-63.5, 64] -> scale=0.005, zero_point=0 for INT16
  QuantizePerChannelOpModel m(TensorType_FLOAT32, TensorType_INT16, {2, 5},
                              {0.005, 0.005}, {0, 0}, 0);

  m.SetInput({-63.5, -63, -3, -2, -1, 1, 2, 3, 63.5, 64});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray({-12700, -12600, -600, -400, -200, 200, 400, 600,
                                12700, 12800}));
}

// Requantization tests.
// Input scale 1.000000, output scale 0.500000, input zeropoint 0, output
// zeropoint 0
TEST(QuantizeOpTest, Int32Int16) {
  QuantizeOpModel m({TensorType_INT32, {1, 1, 2, 5}, 0, 0, 1.0, 0},
                    {TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, 0});

  m.SetInputAndQuantize<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray({2, 4, 6, 8, 10, 12, 14, 16, 18, 20}));
}

// Input scale 0.500000, output scale 0.500000, input zeropoint 0, output
// zeropoint 0
TEST(QuantizeOpTest, Int32Int16SameScale) {
  QuantizeOpModel m({TensorType_INT32, {1, 1, 2, 5}, 0, 0, 0.5, 0},
                    {TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, 0});
  m.SetInputAndQuantize<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 37767});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray({0, 2, 4, 6, 8, 10, 12, 14, 16, 32767}));
}

// Input scale 0.500000, output scale 0.500000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Int32Int8SameScale) {
  QuantizeOpModel m({TensorType_INT32, {1, 1, 2, 5}, 0, 0, 0.5, 0},
                    {TensorType_INT8, {1, 1, 2, 5}, 0, 0, 0.5, -1});

  // Input will quantized to {1,3,5,7,9,11,13,15,17,19}.
  m.SetInputAndQuantize<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Input scale 0.500000, output scale 1.000000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Int32Int8LargerScale) {
  QuantizeOpModel m({TensorType_INT32, {1, 1, 2, 5}, 0, 0, 0.5, 0},
                    {TensorType_INT8, {1, 1, 2, 5}, 0, 0, 1.0, -1});

  // Input will quantized to {1,3,5,7,9,11,13,15,17,19}.
  m.SetInputAndQuantize<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// Input scale 1.000000, output scale 0.500000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Int32Int8SmallerScale) {
  QuantizeOpModel m({TensorType_INT32, {1, 1, 2, 5}, 0, 0, 1.0, 0},
                    {TensorType_INT8, {1, 1, 2, 5}, 0, 0, 0.5, -1});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9}.
  m.SetInputAndQuantize<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Input scale 1.000000, output scale 0.500000, input zeropoint 0, output
// zeropoint 0
TEST(QuantizeOpTest, Int16Int16) {
  QuantizeOpModel m({TensorType_INT16, {1, 1, 2, 5}, 0, 0, 1.0, 0},
                    {TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, 0});

  m.SetInputAndQuantize<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray({2, 4, 6, 8, 10, 12, 14, 16, 18, 20}));
}

// Input scale 0.500000, output scale 0.500000, input zeropoint 0, output
// zeropoint 0
TEST(QuantizeOpTest, Int16Int16SameScale) {
  QuantizeOpModel m({TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, 0},
                    {TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, 0});
  m.SetInputAndQuantize<int16_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 37767});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray({0, 2, 4, 6, 8, 10, 12, 14, 16, 32767}));
}

// Input scale 0.500000, output scale 0.500000, input zeropoint -1, output
// zeropoint -1
TEST(QuantizeOpTest, Int8Int8SameScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -63.5, 64},
                    {TensorType_INT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {1,3,5,7,9,11,13,15,17,19}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Input scale 0.500000, output scale 1.000000, input zeropoint -1, output
// zeropoint -1
TEST(QuantizeOpTest, Int8Int8LargerScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -63.5, 64},
                    {TensorType_INT8, {1, 1, 2, 5}, -127, 128});

  // Input will quantized to {1,3,5,7,9,11,13,15,17,19}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// Input scale 1.000000, output scale 0.500000, input zeropoint -1, output
// zeropoint -1
TEST(QuantizeOpTest, Int8Int8SmallerScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -127, 128},
                    {TensorType_INT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Int8Int8SmallerScaleNeonPath) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 4, 5}, -127, 128},
                    {TensorType_INT8, {1, 1, 4, 5}, -63.5, 64});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1,0}.
  m.SetInputAndQuantize<int8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1,  3,  5,  7,  9,  11, 13, 15, 17, 19,
                                19, 17, 15, 13, 11, 9,  7,  5,  3,  1}));
}

// Input scale 0.500000, output scale 0.500000, input zeropoint 127, output
// zeropoint 127
TEST(QuantizeOpTest, UInt8UInt8SameScale) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, -63.5, 64},
                    {TensorType_UINT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {129,131,133,135,137,139,141,143,145,147}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({129, 131, 133, 135, 137, 139, 141, 143, 145, 147}));
}

// Input scale 0.500000, output scale 1.000000, input zeropoint 127, output
// zeropoint 127
TEST(QuantizeOpTest, Uint8Uint8LargerScale) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, -63.5, 64},
                    {TensorType_UINT8, {1, 1, 2, 5}, -127, 128});

  // Input will quantized to {129,131,133,135,137,139,141,143,145,147}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({128, 129, 130, 131, 132, 133, 134, 135, 136, 137}));
}

// Input scale 1.000000, output scale 0.500000, input zeropoint 127, output
// zeropoint 127
TEST(QuantizeOpTest, Uint8Uint8SmallerScale) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {128, 129, 130, 131, 132, 133, 134, 135, 136, 137}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({129, 131, 133, 135, 137, 139, 141, 143, 145, 147}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Uint8Uint8SmallerScaleNeonPath) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 4, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 4, 5}, -63.5, 64});

  // Input will quantized to {128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
  // 137, 136, 135, 134, 133, 132, 131, 130, 129, 128}.
  m.SetInputAndQuantize<uint8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({129, 131, 133, 135, 137, 139, 141, 143, 145, 147,
                        147, 145, 143, 141, 139, 137, 135, 133, 131, 129}));
}

// Input scale 1.000000, output scale 1.000000, input zeropoint -1, output
// zeropoint 127
TEST(QuantizeOpTest, Int8Uint8SameScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 2, 5}, -127, 128});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({128, 129, 130, 131, 132, 133, 134, 135, 136, 137}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Int8UInt8SameScaleNeonPath) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 4, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 4, 5}, -127, 128});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1,0}.
  m.SetInputAndQuantize<int8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
                        137, 136, 135, 134, 133, 132, 131, 130, 129, 128}));
}

//  Input scale 1.000000, output scale 0.500000, input zeropoint -1, output
//  zeropoint 127
TEST(QuantizeOpTest, Int8Uint8SmallerScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({129, 131, 133, 135, 137, 139, 141, 143, 145, 147}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Int8Uint8SmallerScaleNeonPath) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 4, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 4, 5}, -63.5, 64});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1,0}.
  m.SetInputAndQuantize<int8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({129, 131, 133, 135, 137, 139, 141, 143, 145, 147,
                        147, 145, 143, 141, 139, 137, 135, 133, 131, 129}));
}

//  Input scale 0.500000, output scale 1.000000, input zeropoint -1, output
//  zeropoint 127.
TEST(QuantizeOpTest, Int8Uint8LargerScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -63.5, 63},
                    {TensorType_UINT8, {1, 1, 2, 5}, -127, 128});

  // Input will quantized to {2,4,6,8,10,12,14,16,18,20}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({128, 129, 130, 131, 132, 133, 134, 135, 136, 137}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Int8Uint8LargerScaleNeonPath) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 4, 5}, -63.5, 63},
                    {TensorType_UINT8, {1, 1, 4, 5}, -127, 128});

  // Input will quantized to
  // {2,4,6,8,10,12,14,16,18,20,20,18,16,14,12,10,8,6,4,2}.
  m.SetInputAndQuantize<int8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
                        137, 136, 135, 134, 133, 132, 131, 130, 129, 128}));
}

// input scale 0.500000, output scale 0.500000, input zeropoint 127, output
// zeropoint -1
TEST(QuantizeOpTest, UInt8Int8SameScale128Diff) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, -127, 128},
                    {TensorType_INT8, {1, 1, 2, 5}, -127, 128});

  // Input will quantized to {128, 129, 130, 131, 132, 133, 134, 135, 136, 137}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, UInt8Int8SameScale128DiffNeonPath) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 4, 5}, -127, 128},
                    {TensorType_INT8, {1, 1, 4, 5}, -127, 128});

  // Input will quantized to {128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
  // 137, 136, 135, 134, 133, 132, 131, 130, 129, 128}.
  m.SetInputAndQuantize<uint8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                9, 8, 7, 6, 5, 4, 3, 2, 1, 0}));
}

// input scale 0.500000, output scale 0.500000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Uint8Int8SameScaleArbitraryDiff) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, 0, 127.5},
                    {TensorType_INT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {2,4,6,8,10,12,14,16,18,20}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Uint8Int8SameScaleArbitraryDiffNeonPath) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 4, 5}, 0, 127.5},
                    {TensorType_INT8, {1, 1, 4, 5}, -63.5, 64});

  // Input will quantized to
  // {2,4,6,8,10,12,14,16,18,20,20,18,16,14,12,10,8,6,4,2}.
  m.SetInputAndQuantize<uint8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1,  3,  5,  7,  9,  11, 13, 15, 17, 19,
                                19, 17, 15, 13, 11, 9,  7,  5,  3,  1}));
}

// input scale 0.500000, output scale 1.000000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Uint8Int8LargerScale) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, 0, 127.5},
                    {TensorType_INT8, {1, 1, 2, 5}, -127, 128});

  // Input will quantized to {2,4,6,8,10,12,14,16,18,20}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Uint8Int8LargerScaleNeonPath) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 4, 5}, 0, 127.5},
                    {TensorType_INT8, {1, 1, 4, 5}, -127, 128});

  // Input will quantized to
  // {2,4,6,8,10,12,14,16,18,20,20,18,16,14,12,10,8,6,4,2}.
  m.SetInputAndQuantize<uint8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                9, 8, 7, 6, 5, 4, 3, 2, 1, 0}));
}

// input scale 1.000000, output scale 0.500000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Uint8Int8SmallerScale) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, 0, 255},
                    {TensorType_INT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {1,2,3,4,5,6,7,8,9,10}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Input scale 0.500000, output scale 0.500000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Int16Int8SameScale) {
  QuantizeOpModel m({TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, 0},
                    {TensorType_INT8, {1, 1, 2, 5}, 0, 0, 0.5, -1});

  // Input will quantized to {2,4,6,8,10,12,14,16,18,20}.
  m.SetInputAndQuantize<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Input scale 0.500000, output scale 0.500000, input zeropoint -1, output
// zeropoint -1.
TEST(QuantizeOpTest, Int16ZeroPointInt8) {
  QuantizeOpModel m({TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, -1},
                    {TensorType_INT8, {1, 1, 2, 5}, 0, 0, 0.5, -1});

  // Input will quantized to {2,4,6,8,10,12,14,16,18,20}.
  m.SetInputAndQuantize<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Input scale 0.500000, output scale 1.000000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Int16Int8LargerScale) {
  QuantizeOpModel m({TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, 0},
                    {TensorType_INT8, {1, 1, 2, 5}, 0, 0, 1.0, -1});

  m.SetInputAndQuantize<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// Input scale 1.000000, output scale 0.500000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Int16Int8SmallerScale) {
  QuantizeOpModel m({TensorType_INT16, {1, 1, 2, 5}, 0, 0, 1.0, 0},
                    {TensorType_INT8, {1, 1, 2, 5}, 0, 0, 0.5, -1});

  m.SetInputAndQuantize<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Int16Int8SmallerScaleNeonPath) {
  QuantizeOpModel m({TensorType_INT16, {1, 1, 4, 5}, 0, 0, 1.0, 0},
                    {TensorType_INT8, {1, 1, 4, 5}, 0, 0, 0.5, -1});

  m.SetInputAndQuantize<int16_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1,  3,  5,  7,  9,  11, 13, 15, 17, 19,
                                19, 17, 15, 13, 11, 9,  7,  5,  3,  1}));
}

// Input scale 1.0, output scale 1.0, input zeropoint 0, output zeropoint 0
TEST(QuantizeOpTest, Int16Int32SameScale) {
  QuantizeOpModel m({TensorType_INT16,
                     {1, 1, 2, 5},
                     std::numeric_limits<int16_t>::min(),
                     std::numeric_limits<int16_t>::max()},
                    {TensorType_INT32,
                     {1, 1, 2, 5},
                     std::numeric_limits<int32_t>::min(),
                     static_cast<float>(std::numeric_limits<int32_t>::max())});

  // Input will quantized to {1,3,5,7,9,11,13,15,17,19}.
  m.SetInputAndQuantize<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int32_t>(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
}

// Input scale 0.500000, output scale 1.000000, input zeropoint -1, output
// zeropoint 0
TEST(QuantizeOpTest, Int16Int32LargerScale) {
  QuantizeOpModel m({TensorType_INT16,
                     {1, 1, 2, 5},
                     std::numeric_limits<int16_t>::min() / 2.0,
                     std::numeric_limits<int16_t>::max() / 2.0},
                    {TensorType_INT32,
                     {1, 1, 2, 5},
                     std::numeric_limits<int32_t>::min(),
                     static_cast<float>(std::numeric_limits<int32_t>::max())});

  m.SetInputAndQuantize<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int32_t>(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
}

// Input scale 1.000000, output scale 0.500000, input zeropoint -1, output
// zeropoint 0
TEST(QuantizeOpTest, Int16Int32SmallerScale) {
  QuantizeOpModel m({TensorType_INT16,
                     {1, 1, 2, 5},
                     std::numeric_limits<int16_t>::min(),
                     std::numeric_limits<int16_t>::max()},
                    {TensorType_INT32,
                     {1, 1, 2, 5},
                     std::numeric_limits<int32_t>::min() / 2.0,
                     std::numeric_limits<int32_t>::max() / 2.0});

  m.SetInputAndQuantize<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int32_t>(),
              ElementsAreArray({2, 4, 6, 8, 10, 12, 14, 16, 18, 20}));
}

// Input scale 0.5, output scale 0.5, input zeropoint -1, output zeropoint 0
TEST(QuantizeOpTest, Int8Int16SameScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, 0, 0, 0.5, -1},
                    {TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, 0});

  // Input will quantized to {1,3,5,7,9,11,13,15,17,19}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray({2, 4, 6, 8, 10, 12, 14, 16, 18, 20}));
}

// Input scale 0.5, output scale 0.5, input zeropoint 127, output zeropoint 0
TEST(QuantizeOpTest, UInt8Int16SameScale) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, 0, 0, 0.5, 127},
                    {TensorType_INT16, {1, 1, 2, 5}, 0, 0, 0.5, 0});

  // Input will quantized to {129,131,133,135,137,139,141,143,145,147}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray({2, 4, 6, 8, 10, 12, 14, 16, 18, 20}));
}

TEST(QuantizeOpTest, Int4ToInt8RequantizeOddCountAndZeroPoint) {
  // Odd element count (7) with non-zero zero_point:
  // dequant = (q_in - 1) * 1.0, requant = dequant / 0.5 + (-2) = 2*(q_in - 1) -
  // 2
  QuantizeOpModel m({TensorType_INT4, {1, 7}, 0, 0, 1.0f, 1},
                    {TensorType_INT8, {1, 7}, 0, 0, 0.5f, -2});

  m.SetInputInt4({-8, -4, 0, 1, 3, 5, 7});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({-20, -12, -4, -2, 2, 6, 10}));
}

TEST(QuantizeOpTest, Int4ToUInt8AndInt16Requantize) {
  QuantizeOpModel m_uint8({TensorType_INT4, {1, 5}, 0, 0, 0.5f, 0},
                          {TensorType_UINT8, {1, 5}, 0, 0, 0.5f, 128});
  m_uint8.SetInputInt4({-8, -1, 0, 3, 7});
  ASSERT_EQ(m_uint8.Invoke(), kTfLiteOk);
  EXPECT_THAT(m_uint8.GetOutput<uint8_t>(),
              ElementsAreArray({120, 127, 128, 131, 135}));

  QuantizeOpModel m_int16({TensorType_INT4, {1, 5}, 0, 0, 1.0f, 0},
                          {TensorType_INT16, {1, 5}, 0, 0, 0.5f, 0});
  m_int16.SetInputInt4({-8, -1, 0, 3, 7});
  ASSERT_EQ(m_int16.Invoke(), kTfLiteOk);
  EXPECT_THAT(m_int16.GetOutput<int16_t>(),
              ElementsAreArray({-16, -2, 0, 6, 14}));
}

TEST(QuantizeOpTest, Int2ToInt8RequantizeOddCountAndZeroPoint) {
  // Non-multiple-of-4 element count (7) with non-zero zero_point:
  // dequant = (q_in - (-1)) * 1.0 = q_in + 1, requant = (q_in + 1) / 0.5 + 3
  QuantizeOpModel m({TensorType_INT2, {1, 7}, 0, 0, 1.0f, -1},
                    {TensorType_INT8, {1, 7}, 0, 0, 0.5f, 3});

  m.SetInputInt2({-2, -1, 0, 1, -2, 0, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({1, 3, 5, 7, 1, 5, 7}));
}

TEST(QuantizeOpTest, Int2ToUInt8AndInt16Requantize) {
  QuantizeOpModel m_uint8({TensorType_INT2, {1, 5}, 0, 0, 0.5f, 0},
                          {TensorType_UINT8, {1, 5}, 0, 0, 0.5f, 128});
  m_uint8.SetInputInt2({-2, -1, 0, 1, -2});
  ASSERT_EQ(m_uint8.Invoke(), kTfLiteOk);
  EXPECT_THAT(m_uint8.GetOutput<uint8_t>(),
              ElementsAreArray({126, 127, 128, 129, 126}));

  QuantizeOpModel m_int16({TensorType_INT2, {1, 5}, 0, 0, 1.0f, 0},
                          {TensorType_INT16, {1, 5}, 0, 0, 0.5f, 0});
  m_int16.SetInputInt2({-2, -1, 0, 1, -2});
  ASSERT_EQ(m_int16.Invoke(), kTfLiteOk);
  EXPECT_THAT(m_int16.GetOutput<int16_t>(),
              ElementsAreArray({-4, -2, 0, 2, -4}));
}

TEST(QuantizeOpTest, Int8ToInt4RequantizeSaturationAndOddCount) {
  // Odd element count (7), scale 1.0 -> 1.0, zp 1 -> -1:
  // q_out = clamp((q_in - 1) + (-1), -8, 7) = clamp(q_in - 2, -8, 7)
  QuantizeOpModel m({TensorType_INT8, {1, 7}, 0, 0, 1.0f, 1},
                    {TensorType_INT4, {1, 7}, 0, 0, 1.0f, -1});

  m.SetInputInt8({-19, -6, -5, 1, 8, 9, 21});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputUnpackedInt4(),
              ElementsAreArray({-8, -8, -7, -1, 6, 7, 7}));
}

TEST(QuantizeOpTest, Int8ToInt2RequantizeSaturationAndOddCount) {
  // Non-multiple-of-4 element count (7), scale 1.0 -> 1.0, zp 0 -> -1:
  // q_out = clamp((q_in - 0) + (-1), -2, 1) = clamp(q_in - 1, -2, 1)
  QuantizeOpModel m({TensorType_INT8, {1, 7}, 0, 0, 1.0f, 0},
                    {TensorType_INT2, {1, 7}, 0, 0, 1.0f, -1});

  m.SetInputInt8({-10, -1, 0, 1, 2, 3, 10});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputUnpackedInt2(),
              ElementsAreArray({-2, -2, -1, 0, 1, 1, 1}));
}

TEST(QuantizeOpTest, Int4AndInt8MultiChunkRequantize) {
  // 1031 elements spans multiple 1024-element stack chunks, 16-wide SIMD loops,
  // and an odd tail.
  constexpr int kCount = 1031;
  std::vector<int8_t> int4_input(kCount);
  std::vector<int8_t> expected_int8(kCount);
  for (int i = 0; i < kCount; ++i) {
    int8_t v = static_cast<int8_t>((i % 16) - 8);
    int4_input[i] = v;
    expected_int8[i] = static_cast<int8_t>(2 * v);
  }

  QuantizeOpModel to_int8({TensorType_INT4, {1, kCount}, 0, 0, 1.0f, 0},
                          {TensorType_INT8, {1, kCount}, 0, 0, 0.5f, 0});
  to_int8.SetInputInt4(int4_input);
  ASSERT_EQ(to_int8.Invoke(), kTfLiteOk);
  EXPECT_THAT(to_int8.GetOutput<int8_t>(), ElementsAreArray(expected_int8));

  QuantizeOpModel to_int4({TensorType_INT8, {1, kCount}, 0, 0, 0.5f, 0},
                          {TensorType_INT4, {1, kCount}, 0, 0, 1.0f, 0});
  to_int4.SetInputInt8(expected_int8);
  ASSERT_EQ(to_int4.Invoke(), kTfLiteOk);
  EXPECT_THAT(to_int4.GetOutputUnpackedInt4(), ElementsAreArray(int4_input));
}

TEST(QuantizeOpTest, SubByteUnsupportedTypesAndPerChannelFailPrepare) {
  QuantizeOpModel uint8_to_int4({TensorType_UINT8, {1, 4}, 0, 0, 1.0f, 0},
                                {TensorType_INT4, {1, 4}, 0, 0, 1.0f, 0},
                                /*allocate_and_delegate=*/false);
  EXPECT_EQ(uint8_to_int4.AllocateTensors(), kTfLiteError);

  QuantizeOpModel int4_to_int4({TensorType_INT4, {1, 4}, 0, 0, 1.0f, 0},
                               {TensorType_INT4, {1, 4}, 0, 0, 1.0f, 0},
                               /*allocate_and_delegate=*/false);
  EXPECT_EQ(int4_to_int4.AllocateTensors(), kTfLiteError);

  TensorData per_channel_int4 = {
      TensorType_INT4,    {2, 2}, 0, 0, 0.0f, 0, true, {1.0f, 0.5f}, {0, 0},
      /*channel_index=*/0};
  TensorData per_channel_int8 = {
      TensorType_INT8,    {2, 2}, 0, 0, 0.0f, 0, true, {1.0f, 0.5f}, {0, 0},
      /*channel_index=*/0};
  QuantizeOpModel pc_int4_in(per_channel_int4,
                             {TensorType_INT8, {2, 2}, 0, 0, 1.0f, 0},
                             /*allocate_and_delegate=*/false);
  EXPECT_EQ(pc_int4_in.AllocateTensors(), kTfLiteError);

  QuantizeOpModel pc_int4_out({TensorType_INT4, {2, 2}, 0, 0, 1.0f, 0},
                              per_channel_int8,
                              /*allocate_and_delegate=*/false);
  EXPECT_EQ(pc_int4_out.AllocateTensors(), kTfLiteError);

  QuantizeOpModel pc_int8_to_int4({TensorType_INT8, {2, 2}, 0, 0, 1.0f, 0},
                                  per_channel_int4,
                                  /*allocate_and_delegate=*/false);
  EXPECT_EQ(pc_int8_to_int4.AllocateTensors(), kTfLiteError);
}

}  // namespace
}  // namespace tflite
