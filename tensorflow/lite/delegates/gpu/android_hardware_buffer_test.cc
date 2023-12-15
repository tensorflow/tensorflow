/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/android_hardware_buffer.h"

#include <gtest/gtest.h>

using tflite::gpu::OptionalAndroidHardwareBuffer;
auto Instance = OptionalAndroidHardwareBuffer::Instance;

namespace {

#ifndef __ANDROID__

TEST(OptionalAndroidHardwareBufferTest, NotSupportedOnNonAndroid) {
  EXPECT_EQ(Instance().Supported(), false);
}

#else  // defined(__ANDROID__)

TEST(OptionalAndroidHardwareBufferTest, SupportedOnAndroid) {
  EXPECT_EQ(Instance().Supported(), true);
}

TEST(OptionalAndroidHardwareBufferTest, CanAllocateAndReleaseOnAndroid) {
  EXPECT_EQ(Instance().Supported(), true);
  AHardwareBuffer* buffer;
  AHardwareBuffer_Desc description{};
  description.width = 1600;
  description.height = 1;
  description.layers = 1;
  description.rfu0 = 0;
  description.rfu1 = 0;
  description.stride = 1;
  description.format = AHARDWAREBUFFER_FORMAT_BLOB;
  description.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN;
  EXPECT_TRUE(Instance().IsSupported(&description));
  EXPECT_EQ(Instance().Allocate(&description, &buffer), 0);
  Instance().Release(buffer);
}

TEST(OptionalAndroidHardwareBufferTest, CanAcquireAndReleaseOnAndroid) {
  EXPECT_EQ(Instance().Supported(), true);
  AHardwareBuffer* buffer;
  AHardwareBuffer_Desc description{};
  description.width = 1600;
  description.height = 1;
  description.layers = 1;
  description.rfu0 = 0;
  description.rfu1 = 0;
  description.stride = 1;
  description.format = AHARDWAREBUFFER_FORMAT_BLOB;
  description.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN;
  EXPECT_TRUE(Instance().IsSupported(&description));
  EXPECT_EQ(Instance().Allocate(&description, &buffer), 0);
  Instance().Acquire(buffer);
  Instance().Release(buffer);  // To match Acquire
  Instance().Release(buffer);  // To match Allocate
}

#endif  // defined(__ANDROID__)

}  // namespace
