// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <any>
#include <cstdint>

#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_any.h"

TEST(Any, ConversionNone) {
  EXPECT_FALSE(
      litert::ToStdAny(LiteRtAny{/*.type=*/kLiteRtAnyTypeNone}).has_value());

  ASSERT_EQ(litert::ToLiteRtAny(std::any())->type, kLiteRtAnyTypeNone);
}

TEST(Any, ConversionBool) {
  ASSERT_EQ(std::any_cast<bool>(litert::ToStdAny(LiteRtAny{
                /*.type=*/kLiteRtAnyTypeBool, {/*.bool_value=*/true}})),
            true);
  ASSERT_EQ(std::any_cast<bool>(litert::ToStdAny(LiteRtAny{
                /*.type=*/kLiteRtAnyTypeBool, {/*.bool_value=*/false}})),
            false);

  ASSERT_EQ(litert::ToLiteRtAny(std::any(true))->type, kLiteRtAnyTypeBool);
  ASSERT_EQ(litert::ToLiteRtAny(std::any(true))->bool_value, true);
  ASSERT_EQ(litert::ToLiteRtAny(std::any(false))->type, kLiteRtAnyTypeBool);
  ASSERT_EQ(litert::ToLiteRtAny(std::any(false))->bool_value, false);
}

TEST(Any, ConversionInt) {
  LiteRtAny litert_any;
  litert_any.type = kLiteRtAnyTypeInt;
  litert_any.int_value = 1234;
  ASSERT_EQ(std::any_cast<int64_t>(litert::ToStdAny(litert_any)), 1234);

  ASSERT_EQ(litert::ToLiteRtAny(std::any(static_cast<int8_t>(12)))->type,
            kLiteRtAnyTypeInt);
  ASSERT_EQ(litert::ToLiteRtAny(std::any(static_cast<int8_t>(12)))->int_value,
            12);
  ASSERT_EQ(litert::ToLiteRtAny(std::any(static_cast<int16_t>(1234)))->type,
            kLiteRtAnyTypeInt);
  ASSERT_EQ(
      litert::ToLiteRtAny(std::any(static_cast<int16_t>(1234)))->int_value,
      1234);
  ASSERT_EQ(litert::ToLiteRtAny(std::any(static_cast<int32_t>(1234)))->type,
            kLiteRtAnyTypeInt);
  ASSERT_EQ(
      litert::ToLiteRtAny(std::any(static_cast<int32_t>(1234)))->int_value,
      1234);
  ASSERT_EQ(litert::ToLiteRtAny(std::any(static_cast<int64_t>(1234)))->type,
            kLiteRtAnyTypeInt);
  ASSERT_EQ(
      litert::ToLiteRtAny(std::any(static_cast<int64_t>(1234)))->int_value,
      1234);
}

TEST(Any, ConversionReal) {
  LiteRtAny litert_any;
  litert_any.type = kLiteRtAnyTypeReal;
  litert_any.real_value = 123.4;
  ASSERT_EQ(std::any_cast<double>(litert::ToStdAny(litert_any)), 123.4);

  ASSERT_EQ(litert::ToLiteRtAny(std::any(static_cast<float>(1.2)))->type,
            kLiteRtAnyTypeReal);
  EXPECT_NEAR(
      litert::ToLiteRtAny(std::any(static_cast<float>(1.2)))->real_value, 1.2,
      1e-7);
  ASSERT_EQ(litert::ToLiteRtAny(std::any(static_cast<double>(1.2)))->type,
            kLiteRtAnyTypeReal);
  EXPECT_NEAR(
      litert::ToLiteRtAny(std::any(static_cast<double>(1.2)))->real_value, 1.2,
      1e-7);
}

TEST(Any, ConversionString) {
  constexpr const char* kTestString = "test";
  LiteRtAny litert_any;
  litert_any.type = kLiteRtAnyTypeString;
  litert_any.str_value = kTestString;
  ASSERT_EQ(std::any_cast<const char*>(litert::ToStdAny(litert_any)),
            kTestString);

  ASSERT_EQ(litert::ToLiteRtAny(std::any("test"))->type, kLiteRtAnyTypeString);
  EXPECT_STREQ(litert::ToLiteRtAny(std::any("test"))->str_value, "test");
}

TEST(Any, ConversionPtr) {
  const void* kTestPtr = reinterpret_cast<const void*>(1234);
  LiteRtAny litert_any;
  litert_any.type = kLiteRtAnyTypeVoidPtr;
  litert_any.ptr_value = kTestPtr;
  ASSERT_EQ(std::any_cast<const void*>(litert::ToStdAny(litert_any)), kTestPtr);

  ASSERT_EQ(litert::ToLiteRtAny(std::any(kTestPtr))->type,
            kLiteRtAnyTypeVoidPtr);
  EXPECT_EQ(litert::ToLiteRtAny(std::any(kTestPtr))->ptr_value, kTestPtr);
}
