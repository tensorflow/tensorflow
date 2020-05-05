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

#include "tensorflow/lite/tools/tool_params.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace tools {
namespace {

TEST(ToolParams, SetTest) {
  ToolParams params;
  params.AddParam("some-int1", ToolParam::Create<int>(13));
  params.AddParam("some-int2", ToolParam::Create<int>(17));

  ToolParams others;
  others.AddParam("some-int1", ToolParam::Create<int>(19));
  others.AddParam("some-bool", ToolParam::Create<bool>(true));

  params.Set(others);
  EXPECT_EQ(19, params.Get<int>("some-int1"));
  EXPECT_EQ(17, params.Get<int>("some-int2"));
  EXPECT_FALSE(params.HasParam("some-bool"));
}

TEST(ToolParams, MergeTestOverwriteTrue) {
  ToolParams params;
  params.AddParam("some-int1", ToolParam::Create<int>(13));
  params.AddParam("some-int2", ToolParam::Create<int>(17));

  ToolParams others;
  others.AddParam("some-int1", ToolParam::Create<int>(19));
  others.AddParam("some-bool", ToolParam::Create<bool>(true));

  params.Merge(others, true /* overwrite */);
  EXPECT_EQ(19, params.Get<int>("some-int1"));
  EXPECT_EQ(17, params.Get<int>("some-int2"));
  EXPECT_TRUE(params.Get<bool>("some-bool"));
}

TEST(ToolParams, MergeTestOverwriteFalse) {
  ToolParams params;
  params.AddParam("some-int1", ToolParam::Create<int>(13));
  params.AddParam("some-int2", ToolParam::Create<int>(17));

  ToolParams others;
  others.AddParam("some-int1", ToolParam::Create<int>(19));
  others.AddParam("some-bool", ToolParam::Create<bool>(true));

  params.Merge(others);  // default overwrite is false
  EXPECT_EQ(13, params.Get<int>("some-int1"));
  EXPECT_EQ(17, params.Get<int>("some-int2"));
  EXPECT_TRUE(params.Get<bool>("some-bool"));
}
}  // namespace
}  // namespace tools
}  // namespace tflite
