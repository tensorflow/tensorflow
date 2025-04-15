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

#include "tensorflow/lite/delegates/gpu/async_buffers.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/android_hardware_buffer.h"
#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"

namespace tflite {
namespace gpu {
namespace {

TEST(AsyncBufferTest, DuplicateTest) {
  if (__builtin_available(android 26, *)) {
    auto Instance = OptionalAndroidHardwareBuffer::Instance;
    // Create tie
    TensorObjectDef* tie = new TensorObjectDef();
    tie->object_def.data_type = DataType::FLOAT32;
    tie->object_def.data_layout = DataLayout::BHWC;
    tie->dimensions = Dimensions(2, 2, 2, 2);

    // Create AHWB
    AHardwareBuffer_Desc buffDesc = {};
    buffDesc.width = 1000;
    buffDesc.height = 1;
    buffDesc.layers = 1;
    buffDesc.format = AHARDWAREBUFFER_FORMAT_BLOB;
    buffDesc.usage = AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN |
                     AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                     AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;
    AHardwareBuffer* ahwb;
    EXPECT_TRUE(Instance().IsSupported(&buffDesc));
    EXPECT_EQ(Instance().Allocate(&buffDesc, &ahwb), 0);

    // Init GL Env to properly use gl fcns
    std::unique_ptr<gl::EglEnvironment> env;
    EXPECT_OK(gl::EglEnvironment::NewEglEnvironment(&env));
    AsyncBuffer async_buffer1 = AsyncBuffer(*tie, ahwb);
    GLuint buffer1, buffer2;
    EXPECT_OK(async_buffer1.GetOpenGlBuffer(buffer1));
    EXPECT_GE(buffer1, 0);
    EXPECT_OK(async_buffer1.GetOpenGlBuffer(buffer2));
    // Check that each instance of AsyncBuffer class has only one id
    EXPECT_EQ(buffer1, buffer2);
    AsyncBuffer async_buffer2 = AsyncBuffer(*tie, ahwb);
    EXPECT_OK(async_buffer2.GetOpenGlBuffer(buffer2));
    // Check that each different instance will produce unique id
    EXPECT_NE(buffer1, buffer2);
  } else {
    GTEST_SKIP();
  }
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
