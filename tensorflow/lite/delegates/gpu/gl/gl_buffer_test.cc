/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

TEST(Buffer, Read) {
  std::unique_ptr<EglEnvironment> env;
  ASSERT_TRUE(EglEnvironment::NewEglEnvironment(&env).ok());
  std::vector<float> test = {0, 1, 2, 3};
  GlBuffer buffer;
  ASSERT_TRUE(CreateReadOnlyShaderStorageBuffer<float>(test, &buffer).ok());
  std::vector<float> from_buffer;
  ASSERT_TRUE(AppendFromBuffer(buffer, &from_buffer).ok());
  EXPECT_EQ(test, from_buffer);
}

TEST(Buffer, Write) {
  std::unique_ptr<EglEnvironment> env;
  ASSERT_TRUE(EglEnvironment::NewEglEnvironment(&env).ok());
  GlBuffer buffer;
  ASSERT_TRUE(CreateReadWriteShaderStorageBuffer<float>(4, &buffer).ok());
  std::vector<float> test = {0, 1, 2, 3};
  ASSERT_TRUE(buffer.Write<float>(test).ok());
  std::vector<float> from_buffer;
  ASSERT_TRUE(AppendFromBuffer(buffer, &from_buffer).ok());
  EXPECT_EQ(test, from_buffer);
}

TEST(Buffer, View) {
  std::unique_ptr<EglEnvironment> env;
  ASSERT_TRUE(EglEnvironment::NewEglEnvironment(&env).ok());
  GlBuffer buffer;
  ASSERT_TRUE(CreateReadWriteShaderStorageBuffer<float>(6, &buffer).ok());
  EXPECT_TRUE(buffer.has_ownership());
  EXPECT_EQ(24, buffer.bytes_size());
  EXPECT_EQ(0, buffer.offset());

  // Create view and write data there.
  GlBuffer* buffer1_ptr = nullptr;
  ASSERT_TRUE(buffer.MakeView(4, 16, buffer1_ptr).ok());
  EXPECT_FALSE(buffer1_ptr->has_ownership());
  EXPECT_EQ(16, buffer1_ptr->bytes_size());
  EXPECT_EQ(4, buffer1_ptr->offset());
  std::vector<float> test = {1, 2, 3, 4};
  ASSERT_TRUE(buffer1_ptr->Write<float>(test).ok());

  // Check that data indeed landed in a buffer with proper offset.
  std::vector<float> from_buffer;
  ASSERT_TRUE(AppendFromBuffer(buffer, &from_buffer).ok());
  EXPECT_THAT(from_buffer, testing::ElementsAre(0, 1, 2, 3, 4, 0));

  std::vector<float> from_view;
  ASSERT_TRUE(AppendFromBuffer(*buffer1_ptr, &from_view).ok());
  EXPECT_THAT(from_view, testing::ElementsAre(1, 2, 3, 4));
}

TEST(Buffer, SubView) {
  std::unique_ptr<EglEnvironment> env;
  ASSERT_TRUE(EglEnvironment::NewEglEnvironment(&env).ok());
  GlBuffer buffer;
  ASSERT_TRUE(CreateReadWriteShaderStorageBuffer<float>(6, &buffer).ok());

  // Create view and another view over that view.

  GlBuffer* buffer1_ptr = nullptr;
  ASSERT_TRUE(buffer.MakeView(4, 16, buffer1_ptr).ok());
  GlBuffer* buffer2_ptr = nullptr;
  EXPECT_NE(buffer1_ptr->MakeView(1, 16, buffer2_ptr), OkStatus());
  ASSERT_TRUE(buffer1_ptr->MakeView(2, 2, buffer2_ptr).ok());

  EXPECT_FALSE(buffer2_ptr->has_ownership());
  EXPECT_EQ(2, buffer2_ptr->bytes_size());
  EXPECT_EQ(6, buffer2_ptr->offset());
}

TEST(Buffer, Copy) {
  std::unique_ptr<EglEnvironment> env;
  ASSERT_TRUE(EglEnvironment::NewEglEnvironment(&env).ok());
  GlBuffer buffer;
  ASSERT_TRUE(CreateReadWriteShaderStorageBuffer<float>(4, &buffer).ok());

  // Create view and write data there.
  GlBuffer* buffer1_ptr = nullptr;
  ASSERT_TRUE(buffer.MakeView(4, 4, buffer1_ptr).ok());

  GlBuffer* buffer2_ptr = nullptr;
  ASSERT_TRUE(buffer.MakeView(8, 4, buffer2_ptr).ok());

  // Copy data from one view to another
  ASSERT_TRUE(buffer1_ptr->Write<float>({1}).ok());
  ASSERT_TRUE(CopyBuffer(*buffer1_ptr, *buffer2_ptr).ok());

  // Check that data indeed landed correctly.
  std::vector<float> from_buffer;
  ASSERT_TRUE(AppendFromBuffer(buffer, &from_buffer).ok());
  EXPECT_THAT(from_buffer, testing::ElementsAre(0, 1, 1, 0));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
