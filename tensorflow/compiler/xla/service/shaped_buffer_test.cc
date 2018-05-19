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

#include "tensorflow/compiler/xla/service/shaped_buffer.h"

#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/util/ptr_util.h"

namespace {

TEST(ShapedBufferTest, ScopedShapeBufferAsShapedBufferB71629047) {
  TF_ASSERT_OK_AND_ASSIGN(auto platforms,
                          xla::PlatformUtil::GetSupportedPlatforms());
  ASSERT_FALSE(platforms.empty());
  auto* platform = platforms[0];
  TF_ASSERT_OK_AND_ASSIGN(auto executors,
                          xla::PlatformUtil::GetStreamExecutors(platform));
  xla::StreamExecutorMemoryAllocator allocator(platform, executors);
  const xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {});
  const int kDeviceOrdinal = 0;
  auto scoped_buffer = tensorflow::MakeUnique<xla::ScopedShapedBuffer>(
      shape, shape, &allocator, kDeviceOrdinal);
  std::unique_ptr<xla::ShapedBuffer> buffer = std::move(scoped_buffer);
  buffer = nullptr;
}

}  // namespace
