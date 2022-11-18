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

#include "tensorflow/lite/type_to_tflitetype.h"

#include <string>
#include <type_traits>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"

namespace tflite {
namespace {

TEST(TypeToTfLiteType, TypeMapsAreInverseOfEachOther) {
  EXPECT_EQ(kTfLiteInt16,
            typeToTfLiteType<TfLiteTypeToType<kTfLiteInt16>::Type>());
  EXPECT_EQ(kTfLiteUInt16,
            typeToTfLiteType<TfLiteTypeToType<kTfLiteUInt16>::Type>());
  EXPECT_EQ(kTfLiteInt32,
            typeToTfLiteType<TfLiteTypeToType<kTfLiteInt32>::Type>());
  EXPECT_EQ(kTfLiteUInt32,
            typeToTfLiteType<TfLiteTypeToType<kTfLiteUInt32>::Type>());
  EXPECT_EQ(kTfLiteFloat32,
            typeToTfLiteType<TfLiteTypeToType<kTfLiteFloat32>::Type>());
  EXPECT_EQ(kTfLiteUInt8,
            typeToTfLiteType<TfLiteTypeToType<kTfLiteUInt8>::Type>());
  EXPECT_EQ(kTfLiteInt8,
            typeToTfLiteType<TfLiteTypeToType<kTfLiteInt8>::Type>());
  EXPECT_EQ(kTfLiteBool,
            typeToTfLiteType<TfLiteTypeToType<kTfLiteBool>::Type>());
  EXPECT_EQ(kTfLiteComplex64,
            typeToTfLiteType<TfLiteTypeToType<kTfLiteComplex64>::Type>());
  EXPECT_EQ(kTfLiteComplex128,
            typeToTfLiteType<TfLiteTypeToType<kTfLiteComplex128>::Type>());
  EXPECT_EQ(kTfLiteString,
            typeToTfLiteType<TfLiteTypeToType<kTfLiteString>::Type>());
  EXPECT_EQ(kTfLiteFloat16,
            typeToTfLiteType<TfLiteTypeToType<kTfLiteFloat16>::Type>());
  EXPECT_EQ(kTfLiteFloat64,
            typeToTfLiteType<TfLiteTypeToType<kTfLiteFloat64>::Type>());
}

TEST(TypeToTfLiteType, Sanity) {
  EXPECT_EQ(kTfLiteFloat32, typeToTfLiteType<float>());
  EXPECT_EQ(kTfLiteBool, typeToTfLiteType<bool>());
  EXPECT_EQ(kTfLiteString, typeToTfLiteType<std::string>());
  static_assert(
      std::is_same<float, TfLiteTypeToType<kTfLiteFloat32>::Type>::value,
      "TfLiteTypeToType test failure");
  static_assert(std::is_same<bool, TfLiteTypeToType<kTfLiteBool>::Type>::value,
                "TfLiteTypeToType test failure");
  static_assert(
      std::is_same<std::string, TfLiteTypeToType<kTfLiteString>::Type>::value,
      "TfLiteTypeToType test failure");
}

}  // namespace
}  // namespace tflite
