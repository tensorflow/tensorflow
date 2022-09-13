/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/tsl/platform/test.h"

namespace stream_executor {
namespace {

#if GOOGLE_CUDA

class StreamSearchTest : public ::testing::Test {
 public:
  Platform* GetPlatform() {
    return *MultiPlatformManager::PlatformWithName("CUDA");
  }
};

TEST_F(StreamSearchTest, NoMatchBadPtr) {
  void* bad_ptr = reinterpret_cast<void*>(0xdeadbeef);

  StreamExecutorConfig config;
  config.gpu_stream = bad_ptr;

  port::StatusOr<StreamExecutor*> found_executor =
      GetPlatform()->GetExecutor(config);

  // No executor found.
  EXPECT_FALSE(found_executor.ok());
}

TEST_F(StreamSearchTest, FoundPrevExecutor) {
  port::StatusOr<StreamExecutor*> executor =
      GetPlatform()->ExecutorForDevice(0);
  EXPECT_TRUE(executor.ok());

  Stream s(*executor);
  s.Init();

  Stream s2(*executor);
  s2.Init();

  void* gpu_ptr = s.implementation()->GpuStreamHack();
  void* gpu_ptr_2 = s2.implementation()->GpuStreamHack();

  StreamExecutorConfig c;
  c.gpu_stream = gpu_ptr;

  port::StatusOr<StreamExecutor*> found_executor =
      GetPlatform()->GetExecutor(c);
  EXPECT_TRUE(found_executor.ok());
  EXPECT_EQ(*found_executor, *executor);

  Stream* found1 = (*found_executor)->FindAllocatedStream(gpu_ptr);
  EXPECT_EQ(found1, &s);

  Stream* found2 = (*found_executor)->FindAllocatedStream(gpu_ptr_2);
  EXPECT_EQ(found2, &s2);
}

#endif  // GOOGLE_CUDA

}  // namespace
}  // namespace stream_executor
