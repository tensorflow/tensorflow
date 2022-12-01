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

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/testing/util.h"

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
  interpreter.AllocateTensors();

  char s0[] = "ABC";
  string s1 = "DEFG";
  char s2[] = "";

  // Write strings to tensors
  DynamicBuffer buf0;
  buf0.AddString(s0, 3);
  DynamicBuffer buf1;
  buf1.AddString(s1.data(), s1.length());
  buf0.AddString(s2, 0);

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

}  // namespace tflite
