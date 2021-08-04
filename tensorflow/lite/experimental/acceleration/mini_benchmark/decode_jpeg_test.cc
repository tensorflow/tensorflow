/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_register.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_chessboard_jpeg.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_test_card_jpeg.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder_test_helper.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

namespace {

using testing::ElementsAre;

const int kHeight = 300, kWidth = 250, kChannels = 3;
const int kDecodedSize = kHeight * kWidth * kChannels;

class DecodeJPEGOpModel : public SingleOpModel {
 public:
  DecodeJPEGOpModel(const TensorData& input, const TensorData& output,
                    int num_images, int height, int width) {
    input_id_ = AddInput(input);
    output_id_ = AddOutput(output);
    flexbuffers::Builder fbb;
    fbb.Map([&] {
      fbb.Int("num_images", num_images);
      fbb.Int("height", height);
      fbb.Int("width", width);
    });
    fbb.Finish();
    SetCustomOp("DECODE_JPEG", fbb.GetBuffer(),
                tflite::acceleration::decode_jpeg_kernel::Register_DECODE_JPEG);
    BuildInterpreter({GetShape(input_id_)});
  }

  int input_buffer_id() { return input_id_; }
  int output_id() { return output_id_; }
  std::vector<uint8_t> GetOutput() {
    return ExtractVector<uint8_t>(output_id_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_id_); }

 protected:
  int input_id_;
  int shapes_id_;
  int output_id_;
};

// TODO(b/172544567): Add more tests to verify that invalid shapes, types and
// params are handled gracefully by the op.

TEST(DecodeJpegTest, TestMultipleJPEGImages) {
  // Set up model and populate the input.
  std::string chessboard_image(
      reinterpret_cast<const char*>(g_tflite_acceleration_chessboard_jpeg),
      g_tflite_acceleration_chessboard_jpeg_len);
  std::string test_card_image(
      reinterpret_cast<const char*>(g_tflite_acceleration_test_card_jpeg),
      g_tflite_acceleration_test_card_jpeg_len);
  const int kNumImages = 2;
  DecodeJPEGOpModel model({TensorType_STRING, {kNumImages}},
                          {TensorType_UINT8, {}}, kNumImages, kHeight, kWidth);
  model.PopulateStringTensor(model.input_buffer_id(),
                             {chessboard_image, test_card_image});

  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  // Check output values and shape.
  ASSERT_THAT(model.GetOutputShape(),
              ElementsAre(kNumImages, kHeight, kWidth, kChannels));
  std::vector<uint8_t> output_flattened = model.GetOutput();
  std::vector<uint8_t> img1(output_flattened.begin(),
                            output_flattened.begin() + kDecodedSize);
  EXPECT_THAT(img1, HasChessboardPatternWithTolerance(12));
  std::vector<uint8_t> img2(output_flattened.begin() + kDecodedSize,
                            output_flattened.end());
  EXPECT_THAT(img2, HasRainbowPatternWithTolerance(5));
}

}  // namespace
}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite
