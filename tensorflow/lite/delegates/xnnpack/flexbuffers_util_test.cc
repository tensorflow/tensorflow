/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/xnnpack/flexbuffers_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers

namespace tflite::xnnpack {
namespace {

using ::testing::Pointee;

TEST(FlexbuffersUtilTest, FloatPointer) {
  constexpr float kAValue = 3.14;
  constexpr float kBValue = 56;

  flexbuffers::Builder fbb;
  fbb.Map([&] {
    fbb.Float("a", kAValue);
    fbb.Float("b", kBValue);
  });
  fbb.Finish();

  const flexbuffers::Map map = flexbuffers::GetRoot(fbb.GetBuffer()).AsMap();

  const flexbuffers::Reference a = map["a"];
  EXPECT_TRUE(a.IsFloat());
  EXPECT_THAT(a.As<FloatPointer>().ptr, Pointee(kAValue));

  const flexbuffers::Reference b = map["b"];
  EXPECT_TRUE(b.IsFloat());
  EXPECT_THAT(b.As<FloatPointer>().ptr, Pointee(kBValue));

  const flexbuffers::Reference c = map["c"];
  ASSERT_TRUE(c.IsNull());
  EXPECT_EQ(c.As<FloatPointer>().ptr, nullptr);
}

}  // namespace
}  // namespace tflite::xnnpack
