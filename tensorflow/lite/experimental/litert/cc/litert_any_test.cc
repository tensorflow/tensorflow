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
}

TEST(Any, ConversionBool) {
  ASSERT_EQ(std::any_cast<bool>(litert::ToStdAny(LiteRtAny{
                /*.type=*/kLiteRtAnyTypeBool, {/*.bool_value=*/true}})),
            true);
  ASSERT_EQ(std::any_cast<bool>(litert::ToStdAny(LiteRtAny{
                /*.type=*/kLiteRtAnyTypeBool, {/*.bool_value=*/false}})),
            false);
}

TEST(Any, ConversionInt) {
  LiteRtAny litert_any;
  litert_any.type = kLiteRtAnyTypeInt;
  litert_any.int_value = 1234;
  ASSERT_EQ(std::any_cast<int64_t>(litert::ToStdAny(litert_any)), 1234);
}

TEST(Any, ConversionReal) {
  LiteRtAny litert_any;
  litert_any.type = kLiteRtAnyTypeReal;
  litert_any.real_value = 123.4;
  ASSERT_EQ(std::any_cast<double>(litert::ToStdAny(litert_any)), 123.4);
}

TEST(Any, ConversionString) {
  constexpr const char* kTestString = "test";
  LiteRtAny litert_any;
  litert_any.type = kLiteRtAnyTypeString;
  litert_any.str_value = kTestString;
  ASSERT_EQ(std::any_cast<const char*>(litert::ToStdAny(litert_any)),
            kTestString);
}

TEST(Any, ConversionPtr) {
  const void* kTestPtr = reinterpret_cast<const void*>(1234);
  LiteRtAny litert_any;
  litert_any.type = kLiteRtAnyTypeVoidPtr;
  litert_any.ptr_value = kTestPtr;
  ASSERT_EQ(std::any_cast<const void*>(litert::ToStdAny(litert_any)), kTestPtr);
}
