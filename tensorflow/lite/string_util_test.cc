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
#include "tensorflow/lite/string_util.h"

#include <stdint.h>

#ifndef _WIN32
#include <sys/mman.h>
#endif

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {

TEST(StringUtil, TestStringUtil) {
  Interpreter interpreter;
  interpreter.AddTensors(3);

  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  TfLiteTensor* t1 = interpreter.tensor(1);
  t1->type = kTfLiteString;
  t1->allocation_type = kTfLiteDynamic;

  // String tensor with one string of length 3
  union {
    char raw_bytes[15];
    struct {
      int32_t num_strs;
      int32_t offsets[2];
      char str_data[3];
    } tensor_data;
  } data;
  data.tensor_data = {1, {12, 15}, {'X', 'Y', 'Z'}};

  TfLiteQuantization quant;
  quant.type = kTfLiteNoQuantization;
  quant.params = nullptr;
  interpreter.SetTensorParametersReadOnly(
      2, kTfLiteString, "", {1}, quant, data.raw_bytes, sizeof(data.raw_bytes));
  TfLiteTensor* t2 = interpreter.tensor(2);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  char s0[] = "ABC";
  string s1 = "DEFG";
  char s2[] = "";

  // Write strings to tensors
  DynamicBuffer buf0;
  ASSERT_EQ(buf0.AddString(s0, 3), kTfLiteOk);
  DynamicBuffer buf1;
  ASSERT_EQ(buf1.AddString(s1.data(), s1.length()), kTfLiteOk);
  ASSERT_EQ(buf0.AddString(s2, 0), kTfLiteOk);

  auto new_shape = TfLiteIntArrayCreate(2);
  new_shape->data[0] = 2;
  new_shape->data[1] = 1;
  buf0.WriteToTensor(t0, new_shape);
  buf1.WriteToTensorAsVector(t1);

  // Check tensor shapes.
  EXPECT_EQ(t0->dims->size, 2);
  EXPECT_EQ(t0->dims->data[0], 2);
  EXPECT_EQ(t0->dims->data[1], 1);

  EXPECT_EQ(t1->dims->size, 1);
  EXPECT_EQ(t1->dims->data[0], 1);

  // Read strings from tensors.
  ASSERT_EQ(GetStringCount(t0), 2);
  StringRef str_ref;
  str_ref = GetString(t0, 0);
  ASSERT_EQ(string(str_ref.str, str_ref.len), "ABC");
  str_ref = GetString(t0, 1);
  ASSERT_EQ(string(str_ref.str, str_ref.len), "");
  ASSERT_EQ(t0->bytes, 19);

  ASSERT_EQ(GetStringCount(t1), 1);
  str_ref = GetString(t1, 0);
  ASSERT_EQ(string(str_ref.str, str_ref.len), "DEFG");
  ASSERT_EQ(t1->bytes, 16);

  ASSERT_EQ(GetStringCount(t2), 1);
  str_ref = GetString(t2, 0);
  ASSERT_EQ(string(str_ref.str, str_ref.len), "XYZ");
  ASSERT_EQ(t2->bytes, 15);
}

TEST(StringUtil, AddStringOverflow32Length) {
  // Set max size to a small number so we can efficiently test for
  // overflows.
  const size_t max_size = 100;
  DynamicBuffer buf{max_size};
  std::string big_string(max_size + 1, 'A');
  ASSERT_EQ(buf.AddString({big_string.data(), big_string.length()}),
            kTfLiteError);
}

TEST(StringUtil, AddStringToFullBufferOverflow32Length) {
  const size_t max_size = 100;
  DynamicBuffer buf{max_size};
  std::string big_string((max_size / 2) + 1, 'A');
  ASSERT_EQ(buf.AddString({big_string.data(), big_string.length()}), kTfLiteOk);
  EXPECT_EQ(buf.AddString({big_string.data(), big_string.length()}),
            kTfLiteError);
}

TEST(StringUtil, TruncatesCharDataToLen) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  DynamicBuffer buf;
  char fake_big[] = "ABCADASDA";
  ASSERT_EQ(buf.AddString({fake_big, 3}), kTfLiteOk);
  buf.WriteToTensorAsVector(t0);

  StringRef added_string = GetString(t0, 0);
  EXPECT_EQ(added_string.len, 3);
  EXPECT_EQ(string(added_string.str, 3), "ABC");
}

TEST(StringUtil, TestAddJoinedStringCharSeparator) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  char s0[] = "";
  char s1[] = "ABC";
  char s2[] = "DEFG";
  char s3[] = "";
  char s4[] = "XYZ";

  DynamicBuffer buf;
  buf.AddJoinedString({{s0, 0}, {s1, 3}, {s2, 4}, {s3, 0}, {s4, 3}}, ' ');
  buf.WriteToTensorAsVector(t0);

  ASSERT_EQ(GetStringCount(t0), 1);
  StringRef str_ref;
  str_ref = GetString(t0, 0);
  ASSERT_EQ(string(str_ref.str, str_ref.len), " ABC DEFG  XYZ");
  ASSERT_EQ(t0->bytes, 26);
}

TEST(StringUtil, TestAddJoinedStringStringRefSeparator) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  char s[] = " - ";
  char s0[] = "";
  char s1[] = "ABC";
  char s2[] = "DEFG";
  char s3[] = "";
  char s4[] = "XYZ";

  DynamicBuffer buf;
  buf.AddJoinedString({{s0, 0}, {s1, 3}, {s2, 4}, {s3, 0}, {s4, 3}}, {s, 3});
  buf.WriteToTensorAsVector(t0);

  ASSERT_EQ(GetStringCount(t0), 1);
  StringRef str_ref;
  str_ref = GetString(t0, 0);
  ASSERT_EQ(string(str_ref.str, str_ref.len), " - ABC - DEFG -  - XYZ");
  ASSERT_EQ(t0->bytes, 34);
}

TEST(StringUtil, TestEmptyList) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;
  DynamicBuffer buf;
  buf.WriteToTensorAsVector(t0);

  ASSERT_EQ(GetStringCount(t0), 0);
  ASSERT_EQ(t0->bytes, 8);
}

TEST(StringUtil, TestShapes) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;
  t0->dims = TfLiteIntArrayCreate(2);
  t0->dims->data[0] = 2;
  t0->dims->data[1] = 1;

  // Not setting a new shape: number of strings must match
  DynamicBuffer buf;
  buf.AddString("ABC", 3);
  buf.AddString("X", 1);
  buf.WriteToTensor(t0, nullptr);

  ASSERT_EQ(t0->dims->size, 2);
  EXPECT_EQ(t0->dims->data[0], 2);
  EXPECT_EQ(t0->dims->data[1], 1);

  auto new_shape = TfLiteIntArrayCreate(2);
  new_shape->data[0] = 1;
  new_shape->data[1] = 2;

  buf.WriteToTensor(t0, new_shape);

  ASSERT_EQ(t0->dims->size, 2);
  EXPECT_EQ(t0->dims->data[0], 1);
  EXPECT_EQ(t0->dims->data[1], 2);
}

TEST(StringUtil, EmptyStringWithEmptyBuffer) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  DynamicBuffer buf;
  std::string empty_string;
  ASSERT_EQ(buf.AddString(empty_string.data(), empty_string.length()),
            kTfLiteOk);
  buf.WriteToTensorAsVector(t0);

  StringRef added_string = GetString(t0, 0);
  EXPECT_EQ(added_string.len, 0);
}

// Positive control: bounded AddJoinedString still appends joined text.
TEST(StringUtil, AddJoinedStringWithinMaxLengthSucceeds) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  DynamicBuffer buf{16};
  const char a[] = "AB";
  const char b[] = "CD";
  buf.AddJoinedString({{a, 2}, {b, 2}}, '-');
  buf.WriteToTensorAsVector(t0);

  ASSERT_EQ(GetStringCount(t0), 1);
  StringRef joined = GetString(t0, 0);
  EXPECT_EQ(std::string(joined.str, joined.len), "AB-CD");
}

TEST(StringUtil, AddJoinedStringAtMaxLengthSucceeds) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* tensor = interpreter.tensor(0);
  tensor->type = kTfLiteString;
  tensor->allocation_type = kTfLiteDynamic;

  // The limit counts payload bytes, including separators, but not the header.
  DynamicBuffer buf{8};
  ASSERT_EQ(buf.AddString("PRE", 3), kTfLiteOk);
  buf.AddJoinedString({{"AB", 2}, {"CD", 2}}, '-');
  buf.WriteToTensorAsVector(tensor);

  ASSERT_EQ(GetStringCount(tensor), 2);
  const StringRef first = GetString(tensor, 0);
  const StringRef joined = GetString(tensor, 1);
  EXPECT_EQ(std::string(first.str, first.len), "PRE");
  EXPECT_EQ(std::string(joined.str, joined.len), "AB-CD");
  EXPECT_EQ(tensor->bytes, 8 + 4 * sizeof(int32_t));

  // A further non-empty join must leave the full buffer unchanged.
  buf.AddJoinedString({{"X", 1}}, '-');
  buf.WriteToTensorAsVector(tensor);
  EXPECT_EQ(GetStringCount(tensor), 2);
  EXPECT_EQ(tensor->bytes, 8 + 4 * sizeof(int32_t));
}

TEST(StringUtil, AddJoinedStringEmptyVectorAppendsEmptyEntry) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  DynamicBuffer buf;
  ASSERT_EQ(buf.AddString("KEEP", 4), kTfLiteOk);
  const std::vector<StringRef> empty_parts;
  const char sep[] = ", ";
  buf.AddJoinedString(empty_parts, {sep, 2});
  buf.WriteToTensorAsVector(t0);

  ASSERT_EQ(GetStringCount(t0), 2);
  StringRef first = GetString(t0, 0);
  StringRef second = GetString(t0, 1);
  EXPECT_EQ(std::string(first.str, first.len), "KEEP");
  EXPECT_EQ(second.len, 0);
}

TEST(StringUtil, AddJoinedStringEmptyPartsInEmptyBufferSafe) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  DynamicBuffer buf;
  buf.AddJoinedString({{nullptr, 0}, {nullptr, 0}}, {nullptr, 0});
  buf.WriteToTensorAsVector(t0);

  ASSERT_EQ(GetStringCount(t0), 1);
  EXPECT_EQ(GetString(t0, 0).len, 0);
}

// Negative boundary: overflow and underflow lengths are safely ignored.
TEST(StringUtil, AddJoinedStringOverflowLengthIgnored) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  DynamicBuffer small_buf{8};
  ASSERT_EQ(small_buf.AddString("OK", 2), kTfLiteOk);
  const char part[] = "12345";
  small_buf.AddJoinedString({{part, 5}, {part, 5}}, '-');
  small_buf.WriteToTensorAsVector(t0);
  ASSERT_EQ(GetStringCount(t0), 1);
  StringRef kept = GetString(t0, 0);
  EXPECT_EQ(std::string(kept.str, kept.len), "OK");
}

#ifndef _WIN32
TEST(StringUtil, AddJoinedStringInt32WrapOverflowIgnored) {
  Interpreter interpreter;
  interpreter.AddTensors(1);
  TfLiteTensor* t0 = interpreter.tensor(0);
  t0->type = kTfLiteString;
  t0->allocation_type = kTfLiteDynamic;

  constexpr size_t kChunk = size_t{1} << 30;
  void* mapping =
      mmap(nullptr, kChunk, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapping != MAP_FAILED) {
    const char* ptr = static_cast<const char*>(mapping);
    DynamicBuffer wrap_buf;
    ASSERT_EQ(wrap_buf.AddString("SAFE", 4), kTfLiteOk);
    wrap_buf.AddJoinedString(
        {{ptr, kChunk}, {ptr, kChunk}, {ptr, kChunk}, {ptr, kChunk}}, ' ');
    munmap(mapping, kChunk);

    wrap_buf.WriteToTensorAsVector(t0);
    ASSERT_EQ(GetStringCount(t0), 1);
    StringRef still_safe = GetString(t0, 0);
    EXPECT_EQ(std::string(still_safe.str, still_safe.len), "SAFE");
  }
}
#endif

}  // namespace tflite
