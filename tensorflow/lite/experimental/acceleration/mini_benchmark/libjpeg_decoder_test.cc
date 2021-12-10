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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.h"

#include <stddef.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_status.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_chessboard_jpeg.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_snow_jpeg.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_test_card_jpeg.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder_test_helper.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {
namespace {

using testing::IsEmpty;
using testing::NotNull;

constexpr JpegHeader kExpectedImageDimensions{
    .height = 300, .width = 250, .channels = 3};

constexpr int kDecodedSize = kExpectedImageDimensions.height *
                             kExpectedImageDimensions.width *
                             kExpectedImageDimensions.channels;

TEST(LibjpegDecoderTest, InitShouldSucceedOnAndroid) {
  Status status;
  std::unique_ptr<LibjpegDecoder> decoder = LibjpegDecoder::Create(status);
  EXPECT_EQ(status.code, kTfLiteOk);
  EXPECT_THAT(status.error_message, IsEmpty());
}

TEST(LibjpegDecoderTest, DecodingChessboardShouldSucceedOnAndroid) {
  Status status;
  std::unique_ptr<LibjpegDecoder> decoder = LibjpegDecoder::Create(status);
  ASSERT_THAT(decoder, NotNull());
  tflite::StringRef string_ref = {
      reinterpret_cast<const char*>(g_tflite_acceleration_chessboard_jpeg),
      g_tflite_acceleration_chessboard_jpeg_len};
  unsigned char decoded[kDecodedSize];

  status = decoder->DecodeImage(string_ref, kExpectedImageDimensions, decoded,
                                kDecodedSize);

  ASSERT_EQ(status.error_message, "");
  ASSERT_EQ(status.code, kTfLiteOk);
  std::vector<uint8_t> decoded_vec(decoded, decoded + kDecodedSize);
  EXPECT_THAT(decoded_vec, HasChessboardPatternWithTolerance(12));
}

TEST(LibjpegDecoderTest, DecodingRainbowTestCardShouldSucceedOnAndroid) {
  Status status;
  std::unique_ptr<LibjpegDecoder> decoder = LibjpegDecoder::Create(status);
  EXPECT_THAT(status.error_message, IsEmpty());
  EXPECT_EQ(status.code, kTfLiteOk);
  ASSERT_THAT(decoder, NotNull());
  std::string encoded(
      reinterpret_cast<const char*>(g_tflite_acceleration_test_card_jpeg),
      g_tflite_acceleration_test_card_jpeg_len);
  tflite::StringRef string_ref = {encoded.c_str(),
                                  static_cast<int>(encoded.length())};
  unsigned char decoded[kDecodedSize];

  status = decoder->DecodeImage(string_ref, kExpectedImageDimensions, decoded,
                                kDecodedSize);

  ASSERT_EQ(status.error_message, "");
  ASSERT_EQ(status.code, kTfLiteOk);
  std::vector<uint8_t> decoded_vec(decoded, decoded + kDecodedSize);
  EXPECT_THAT(decoded_vec, HasRainbowPatternWithTolerance(5));
}

TEST(LibjpegDecoderTest, ErrorsFromJpegLayerAreReturnedToCaller) {
  Status status;
  std::unique_ptr<LibjpegDecoder> decoder = LibjpegDecoder::Create(status);
  EXPECT_THAT(status.error_message, IsEmpty());
  EXPECT_EQ(status.code, kTfLiteOk);
  ASSERT_THAT(decoder, NotNull());
  std::string str = "this is not a jpeg image";
  tflite::StringRef encoded = {str.c_str(), static_cast<int>(str.length())};
  unsigned char decoded_image[12];

  status = decoder->DecodeImage(encoded, kExpectedImageDimensions,
                                decoded_image, 12);

  EXPECT_EQ(status.code, kTfLiteError);
  EXPECT_EQ(status.error_message, "Not a valid JPEG image.");
}

TEST(LibjpegDecoderTest, DecodingFailsWhenDecodeBufferIsSmall) {
  Status status;
  std::unique_ptr<LibjpegDecoder> decoder = LibjpegDecoder::Create(status);
  EXPECT_THAT(status.error_message, IsEmpty());
  EXPECT_EQ(status.code, kTfLiteOk);
  ASSERT_THAT(decoder, NotNull());
  std::string encoded(
      reinterpret_cast<const char*>(g_tflite_acceleration_test_card_jpeg),
      g_tflite_acceleration_test_card_jpeg_len);
  tflite::StringRef string_ref = {encoded.c_str(),
                                  static_cast<int>(encoded.length())};
  const int decoded_size = 100;
  unsigned char decoded[decoded_size];
  status = decoder->DecodeImage(string_ref, kExpectedImageDimensions, decoded,
                                decoded_size);
  EXPECT_EQ(status.code, kTfLiteError);
  EXPECT_EQ(status.error_message,
            "Size of buffer(100) for storing decoded image must be equal to "
            "the size of decoded image(225000).");
}

TEST(LibjpegDecoderTest, DecodingFailsWhenImageDimensionsDifferFromExpected) {
  Status status;
  std::unique_ptr<LibjpegDecoder> decoder = LibjpegDecoder::Create(status);
  EXPECT_THAT(status.error_message, IsEmpty());
  EXPECT_EQ(status.code, kTfLiteOk);
  ASSERT_THAT(decoder, NotNull());
  std::string encoded(
      reinterpret_cast<const char*>(g_tflite_acceleration_test_card_jpeg),
      g_tflite_acceleration_test_card_jpeg_len);
  tflite::StringRef string_ref = {encoded.c_str(),
                                  static_cast<int>(encoded.length())};
  unsigned char decoded[kDecodedSize];

  status = decoder->DecodeImage(string_ref,
                                {.height = 300, .width = 250, .channels = 1},
                                decoded, kDecodedSize);
  EXPECT_EQ(status.code, kTfLiteError);
  EXPECT_EQ(
      status.error_message,
      "Decoded image size (300, 250, 3, 8) is different from provided image "
      "size (300, 250, 1, 8)");
}

TEST(LibjpegDecoderTest, DecodingFailsWhenImageDimensionsAreOverThreshold) {
  Status status;
  std::unique_ptr<LibjpegDecoder> decoder = LibjpegDecoder::Create(status);
  EXPECT_THAT(status.error_message, IsEmpty());
  EXPECT_EQ(status.code, kTfLiteOk);
  ASSERT_THAT(decoder, NotNull());
  std::string encoded(
      reinterpret_cast<const char*>(g_tflite_acceleration_test_card_jpeg),
      g_tflite_acceleration_test_card_jpeg_len);
  tflite::StringRef origin_string_ref = {encoded.c_str(),
                                         static_cast<int>(encoded.length())};

  const JpegHeader kHeader{
      .height = static_cast<int>(LibjpegDecoder::kMaxImageHeight + 1),
      .width = static_cast<int>(LibjpegDecoder::kMaxImageWidth + 1),
      .channels = 3};

  const size_t decoded_size = static_cast<size_t>(kHeader.height) *
                              static_cast<size_t>(kHeader.width) *
                              static_cast<size_t>(kHeader.channels);

  std::string altered_image;
  Status alter_header_status =
      BuildImageWithNewHeader(origin_string_ref, kHeader, altered_image);
  ASSERT_EQ(alter_header_status.code, kTfLiteOk);

  tflite::StringRef altered_string_ref = {
      altered_image.c_str(), static_cast<int>(altered_image.length())};

  std::vector<unsigned char> decoded(decoded_size);
  status = decoder->DecodeImage(altered_string_ref, kHeader, decoded.data(),
                                decoded_size);
  EXPECT_EQ(status.code, kTfLiteError);
  EXPECT_EQ(status.error_message,
            "Image is too big, dimensions (" + std::to_string(kHeader.width) +
                "," + std::to_string(kHeader.width) +
                ") larger than the maximum allowed (" +
                std::to_string(LibjpegDecoder::kMaxImageHeight) + ", " +
                std::to_string(LibjpegDecoder::kMaxImageWidth) + ")");
}

TEST(LibjpegDecoderTest, DecodingFailsWhenImageHasUnsupportedNumberOfChannels) {
  Status status;
  std::unique_ptr<LibjpegDecoder> decoder = LibjpegDecoder::Create(status);
  EXPECT_THAT(status.error_message, IsEmpty());
  EXPECT_EQ(status.code, kTfLiteOk);
  ASSERT_THAT(decoder, NotNull());
  std::string encoded(
      reinterpret_cast<const char*>(g_tflite_acceleration_test_card_jpeg),
      g_tflite_acceleration_test_card_jpeg_len);
  tflite::StringRef string_ref = {encoded.c_str(),
                                  static_cast<int>(encoded.length())};
  unsigned char decoded[300 * 250 * 4];

  const JpegHeader kHeader{.height = 300, .width = 250, .channels = 4};
  status = decoder->DecodeImage(string_ref, kHeader, decoded, kDecodedSize);
  EXPECT_EQ(status.code, kTfLiteError);
  EXPECT_EQ(status.error_message,
            "Supporting only images with 1 or 3 channels");
}

TEST(LibjpegDecoderTest, DecodingFailsWhenExpectedBitPerSampleIsNot8) {
  Status status;
  std::unique_ptr<LibjpegDecoder> decoder = LibjpegDecoder::Create(status);
  EXPECT_THAT(status.error_message, IsEmpty());
  EXPECT_EQ(status.code, kTfLiteOk);
  ASSERT_THAT(decoder, NotNull());
  std::string encoded(
      reinterpret_cast<const char*>(g_tflite_acceleration_test_card_jpeg),
      g_tflite_acceleration_test_card_jpeg_len);
  tflite::StringRef string_ref = {encoded.c_str(),
                                  static_cast<int>(encoded.length())};
  unsigned char decoded[kDecodedSize];

  status = decoder->DecodeImage(
      string_ref,
      {.height = 300, .width = 250, .channels = 3, .bits_per_sample = 4},
      decoded, kDecodedSize);
  EXPECT_EQ(status.code, kTfLiteError);
  EXPECT_EQ(status.error_message,
            "Supporting only images with 8 bits per sample");
}

TEST(LibjpegDecoderTest, DoesNotDecodeBeyondWhatIsSpecifiedInHeader) {
  Status status;
  std::unique_ptr<LibjpegDecoder> decoder = LibjpegDecoder::Create(status);
  EXPECT_THAT(status.error_message, IsEmpty());
  EXPECT_EQ(status.code, kTfLiteOk);
  ASSERT_THAT(decoder, NotNull());
  std::string origin_encoded_img(
      reinterpret_cast<const char*>(g_tflite_acceleration_test_card_jpeg),
      g_tflite_acceleration_test_card_jpeg_len);
  tflite::StringRef origin_string_ref = {
      origin_encoded_img.c_str(),
      static_cast<int>(origin_encoded_img.length())};

  JpegHeader undersized_image_header = {
      .height = kExpectedImageDimensions.height / 2,
      .width = kExpectedImageDimensions.width / 2,
      .channels = kExpectedImageDimensions.channels};
  std::string altered_image;
  Status alter_header_status = BuildImageWithNewHeader(
      origin_string_ref, undersized_image_header, altered_image);
  ASSERT_EQ(alter_header_status.code, kTfLiteOk);

  tflite::StringRef altered_string_ref = {
      altered_image.c_str(), static_cast<int>(altered_image.length())};

  unsigned char decoded[kDecodedSize / 4];

  status = decoder->DecodeImage(altered_string_ref, undersized_image_header,
                                decoded, kDecodedSize / 4);
  EXPECT_EQ(status.code, kTfLiteOk);
}

TEST(LibjpegDecoderTest, CanReadImagesWithVeryLargeRows) {
  Status status;
  std::unique_ptr<LibjpegDecoder> decoder = LibjpegDecoder::Create(status);
  EXPECT_THAT(status.error_message, IsEmpty());
  EXPECT_EQ(status.code, kTfLiteOk);
  ASSERT_THAT(decoder, NotNull());
  std::string origin_encoded_img(
      reinterpret_cast<const char*>(g_tflite_acceleration_snow_jpeg),
      g_tflite_acceleration_snow_jpeg_len);
  tflite::StringRef origin_string_ref = {
      origin_encoded_img.c_str(),
      static_cast<int>(origin_encoded_img.length())};

  JpegHeader one_long_row_image_header = {
      .height = 1,
      .width = static_cast<int>(LibjpegDecoder::kMaxImageWidth),
      .channels = kExpectedImageDimensions.channels};
  std::string altered_image;
  Status alter_header_status = BuildImageWithNewHeader(
      origin_string_ref, one_long_row_image_header, altered_image);
  ASSERT_EQ(alter_header_status.code, kTfLiteOk);

  tflite::StringRef altered_string_ref = {
      altered_image.c_str(), static_cast<int>(altered_image.length())};

  const size_t kImageSize = LibjpegDecoder::kMaxImageWidth * 3;
  std::vector<unsigned char> decoded(kImageSize);

  status = decoder->DecodeImage(altered_string_ref, one_long_row_image_header,
                                decoded.data(), kImageSize);
  EXPECT_EQ(status.code, kTfLiteOk);
}

TEST(LibjpegDecoderTest, FailDecodingAnImageWithUnexpectedEofInDataStream) {
  Status status;
  std::unique_ptr<LibjpegDecoder> decoder = LibjpegDecoder::Create(status);
  EXPECT_THAT(status.error_message, IsEmpty());
  EXPECT_EQ(status.code, kTfLiteOk);
  ASSERT_THAT(decoder, NotNull());
  std::string img(
      reinterpret_cast<const char*>(g_tflite_acceleration_test_card_jpeg),
      g_tflite_acceleration_test_card_jpeg_len);
  tflite::StringRef truncated_image_ref = {
      img.c_str(), static_cast<int>(img.length() - 100)};

  unsigned char decoded[kDecodedSize];

  status = decoder->DecodeImage(truncated_image_ref, kExpectedImageDimensions,
                                decoded, kDecodedSize);
  EXPECT_EQ(status.code, kTfLiteError);
  // The image is not even read because is recognised as invalid JPEG image
  EXPECT_EQ(status.error_message, "Not a valid JPEG image.");
}

TEST(LibjpegDecoderTest, JpegErrorMsgParsingForValidMsg) {
  size_t extracted_size;
  Status status = ExtractSizeFromErrorMessage(
      "JPEG parameter struct mismatch: library thinks size is 480, caller "
      "expects 464.",
      extracted_size);
  ASSERT_EQ(status.code, kTfLiteOk);
  EXPECT_EQ(extracted_size, 480);
}

TEST(LibjpegDecoderTest, JpegErrorMsgParsingForMaformedMsg) {
  size_t extracted_size;
  std::string err_msg =
      "JPEG parameter struct mismatch: library thinks size is abcde, caller "
      "expects 464.";
  Status status = ExtractSizeFromErrorMessage(err_msg, extracted_size);
  EXPECT_EQ(status.code, kTfLiteError);
  EXPECT_EQ(status.error_message,
            "Couldn't parse the size from message: '" + err_msg + "'");
}

}  // namespace
}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite
