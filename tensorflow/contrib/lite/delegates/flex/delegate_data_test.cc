/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/contrib/lite/delegates/flex/delegate_data.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/c/c_api_internal.h"
#include "tensorflow/contrib/lite/testing/util.h"

namespace tflite {
namespace flex {
namespace {

TEST(DelegateDataTest, Basic) {
  std::unique_ptr<DelegateData> data;
  // We only check for success because it is hard to make initialization fail.
  // It only happens if we manage to not link the CPU device factory into the
  // binary.
  EXPECT_TRUE(DelegateData::Create(&data).ok());

  TfLiteContext dummy_context1 = {};
  TfLiteContext dummy_context2 = {};
  EXPECT_NE(data->GetEagerContext(), nullptr);
  EXPECT_NE(data->GetBufferMap(&dummy_context1), nullptr);
  EXPECT_NE(data->GetBufferMap(&dummy_context1),
            data->GetBufferMap(&dummy_context2));
}

}  // namespace
}  // namespace flex
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
