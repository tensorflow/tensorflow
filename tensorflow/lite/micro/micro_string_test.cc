/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/micro_string.h"

#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FormatPositiveIntShouldMatchExpected) {
  const int kBufferLen = 32;
  char buffer[kBufferLen];
  const char golden[] = "Int: 55";
  int bytes_written = MicroSnprintf(buffer, kBufferLen, "Int: %d", 55);
  TF_LITE_MICRO_EXPECT_EQ(sizeof(golden), bytes_written);
  TF_LITE_MICRO_EXPECT_STRING_EQ(golden, buffer);
}

TF_LITE_MICRO_TEST(FormatNegativeIntShouldMatchExpected) {
  const int kBufferLen = 32;
  char buffer[kBufferLen];
  const char golden[] = "Int: -55";
  int bytes_written = MicroSnprintf(buffer, kBufferLen, "Int: %d", -55);
  TF_LITE_MICRO_EXPECT_EQ(sizeof(golden), bytes_written);
  TF_LITE_MICRO_EXPECT_STRING_EQ(golden, buffer);
}

TF_LITE_MICRO_TEST(FormatUnsignedIntShouldMatchExpected) {
  const int kBufferLen = 32;
  char buffer[kBufferLen];
  const char golden[] = "UInt: 12345";
  int bytes_written = MicroSnprintf(buffer, kBufferLen, "UInt: %u", 12345);
  TF_LITE_MICRO_EXPECT_EQ(sizeof(golden), bytes_written);
  TF_LITE_MICRO_EXPECT_STRING_EQ(golden, buffer);
}

TF_LITE_MICRO_TEST(FormatHexShouldMatchExpected) {
  const int kBufferLen = 32;
  char buffer[kBufferLen];
  const char golden[] = "Hex: 0x12345";
  int bytes_written = MicroSnprintf(buffer, kBufferLen, "Hex: %x", 0x12345);
  TF_LITE_MICRO_EXPECT_EQ(sizeof(golden), bytes_written);
  TF_LITE_MICRO_EXPECT_STRING_EQ(golden, buffer);
}

TF_LITE_MICRO_TEST(FormatFloatShouldMatchExpected) {
  const int kBufferLen = 32;
  char buffer[kBufferLen];
  const char golden[] = "Float: 1.0*2^4";
  int bytes_written = MicroSnprintf(buffer, kBufferLen, "Float: %f", 16.f);
  TF_LITE_MICRO_EXPECT_EQ(sizeof(golden), bytes_written);
  TF_LITE_MICRO_EXPECT_STRING_EQ(golden, buffer);
}

TF_LITE_MICRO_TEST(BadlyFormattedStringShouldProduceReasonableString) {
  const int kBufferLen = 32;
  char buffer[kBufferLen];
  const char golden[] = "Test Badly % formated % string";
  int bytes_written =
      MicroSnprintf(buffer, kBufferLen, "Test Badly %% formated %% string%");
  TF_LITE_MICRO_EXPECT_EQ(sizeof(golden), bytes_written);
  TF_LITE_MICRO_EXPECT_STRING_EQ(golden, buffer);
}

TF_LITE_MICRO_TEST(IntFormatOverrunShouldTruncate) {
  const int kBufferLen = 8;
  char buffer[kBufferLen];
  const char golden[] = "Int: ";
  int bytes_written = MicroSnprintf(buffer, kBufferLen, "Int: %d", 12345);
  TF_LITE_MICRO_EXPECT_EQ(sizeof(golden), bytes_written);
  TF_LITE_MICRO_EXPECT_STRING_EQ(golden, buffer);
}

TF_LITE_MICRO_TEST(UnsignedIntFormatOverrunShouldTruncate) {
  const int kBufferLen = 8;
  char buffer[kBufferLen];
  const char golden[] = "UInt: ";
  int bytes_written = MicroSnprintf(buffer, kBufferLen, "UInt: %u", 12345);
  TF_LITE_MICRO_EXPECT_EQ(sizeof(golden), bytes_written);
  TF_LITE_MICRO_EXPECT_STRING_EQ(golden, buffer);
}

TF_LITE_MICRO_TEST(HexFormatOverrunShouldTruncate) {
  const int kBufferLen = 8;
  char buffer[kBufferLen];
  const char golden[] = "Hex: ";
  int bytes_written = MicroSnprintf(buffer, kBufferLen, "Hex: %x", 0x12345);
  TF_LITE_MICRO_EXPECT_EQ(sizeof(golden), bytes_written);
  TF_LITE_MICRO_EXPECT_STRING_EQ(golden, buffer);
}

TF_LITE_MICRO_TEST(FloatFormatOverrunShouldTruncate) {
  const int kBufferLen = 12;
  char buffer[kBufferLen];
  const char golden[] = "Float: ";
  int bytes_written = MicroSnprintf(buffer, kBufferLen, "Float: %x", 12345.f);
  TF_LITE_MICRO_EXPECT_EQ(sizeof(golden), bytes_written);
  TF_LITE_MICRO_EXPECT_STRING_EQ(golden, buffer);
}

TF_LITE_MICRO_TEST(StringFormatOverrunShouldTruncate) {
  const int kBufferLen = 10;
  char buffer[kBufferLen];
  const char golden[] = "String: h";
  int bytes_written =
      MicroSnprintf(buffer, kBufferLen, "String: %s", "hello world");
  TF_LITE_MICRO_EXPECT_EQ(sizeof(golden), bytes_written);
  TF_LITE_MICRO_EXPECT_STRING_EQ(golden, buffer);
}

TF_LITE_MICRO_TEST(StringFormatWithExactOutputSizeOverrunShouldTruncate) {
  const int kBufferLen = 10;
  char buffer[kBufferLen];
  const char golden[] = "format st";
  int bytes_written = MicroSnprintf(buffer, kBufferLen, "format str");
  TF_LITE_MICRO_EXPECT_EQ(sizeof(golden), bytes_written);
  TF_LITE_MICRO_EXPECT_STRING_EQ(golden, buffer);
}

TF_LITE_MICRO_TESTS_END
