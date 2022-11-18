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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_chessboard_jpeg.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

void PrintTo(const Status& status, std::ostream* os) {
  *os << "{ code: " + std::to_string(status.code) + ", error_message: '" +
             status.error_message + "'}";
}

}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite

namespace {

using ::testing::AllOf;
using ::testing::Eq;
using ::testing::Field;
using ::testing::Matcher;

using tflite::acceleration::decode_jpeg_kernel::JpegHeader;
using tflite::acceleration::decode_jpeg_kernel::ReadJpegHeader;

Matcher<JpegHeader> JpegHeaderEq(const JpegHeader& expected) {
  return AllOf(
      Field(&JpegHeader::channels, Eq(expected.channels)),
      Field(&JpegHeader::height, Eq(expected.height)),
      Field(&JpegHeader::width, Eq(expected.width)),
      Field(&JpegHeader::bits_per_sample, Eq(expected.bits_per_sample)));
}

using tflite::acceleration::decode_jpeg_kernel::Status;

Matcher<Status> StatusEq(const Status& expected) {
  return AllOf(Field(&Status::code, Eq(expected.code)),
               Field(&Status::error_message, Eq(expected.error_message)));
}

const int kChessboardImgHeight = 300;
const int kChessboardImgWidth = 250;
const int kChessboardImgChannels = 3;

TEST(ReadJpegHeader, ShouldParseValidJpgImage) {
  const tflite::StringRef chessboard_image{
      reinterpret_cast<const char*>(g_tflite_acceleration_chessboard_jpeg),
      g_tflite_acceleration_chessboard_jpeg_len};
  ASSERT_GT(chessboard_image.len, 4);

  JpegHeader header;

  ASSERT_THAT(ReadJpegHeader(chessboard_image, &header),
              StatusEq({kTfLiteOk, ""}));
  EXPECT_THAT(header, JpegHeaderEq({kChessboardImgHeight, kChessboardImgWidth,
                                    kChessboardImgChannels}));
}

TEST(ReadJpegHeader, ShouldFailForInvalidJpegImage) {
  const std::string invalid_image = "invalid image content";
  const tflite::StringRef invalid_image_ref{
      invalid_image.c_str(), static_cast<int>(invalid_image.size())};

  JpegHeader header;

  EXPECT_THAT(ReadJpegHeader(invalid_image_ref, &header),
              StatusEq({kTfLiteError, "Not a valid JPEG image."}));
}

TEST(ReadJpegHeader, ShouldFailForEmptyJpegImage) {
  const tflite::StringRef invalid_image_ref{"", 0};

  JpegHeader header;

  EXPECT_THAT(ReadJpegHeader(invalid_image_ref, &header),
              StatusEq({kTfLiteError, "Not a valid JPEG image."}));
}

TEST(ApplyHeaderToImage, ReturnsNewImageWithDifferentHeader) {
  const tflite::StringRef chessboard_image{
      reinterpret_cast<const char*>(g_tflite_acceleration_chessboard_jpeg),
      g_tflite_acceleration_chessboard_jpeg_len};

  JpegHeader new_header{
      .height = 20, .width = 30, .channels = 1, .bits_per_sample = 3};

  std::string new_image_data;

  ASSERT_THAT(
      BuildImageWithNewHeader(chessboard_image, new_header, new_image_data),
      StatusEq({kTfLiteOk, ""}));

  const tflite::StringRef altered_image{
      new_image_data.c_str(), static_cast<int>(new_image_data.size())};
  JpegHeader header;
  ASSERT_THAT(ReadJpegHeader(altered_image, &header),
              StatusEq({kTfLiteOk, ""}));
  EXPECT_THAT(header, JpegHeaderEq(new_header));
}

}  // namespace
