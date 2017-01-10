/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/infeed_manager.h"

#include <memory>

#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class InfeedManagerTest : public ::testing::Test {};

class TestInfeedBuffer : public cpu::runtime::InfeedBuffer {
 public:
  explicit TestInfeedBuffer(int32 length)
      : done_called_(false), length_(length) {}
  ~TestInfeedBuffer() override { EXPECT_TRUE(done_called_); }

  int32 length() override { return length_; }
  void* data() override { return nullptr; }
  void Done() override {
    CHECK(!done_called_);
    done_called_ = true;
  }

 private:
  bool done_called_;
  int32 length_;
};

void ProcessNextBuffer(int32 length) {
  void* buffer = __xla_cpu_runtime_AcquireInfeedBufferForDequeue(length);
  __xla_cpu_runtime_ReleaseInfeedBufferAfterDequeue(length, buffer);
}

TEST_F(InfeedManagerTest, SingleThreadedSequential) {
  TestInfeedBuffer* a = new TestInfeedBuffer(64);
  TestInfeedBuffer* b = new TestInfeedBuffer(32);

  cpu::runtime::InfeedManager* infeed = cpu::runtime::GetInfeedManager();

  infeed->EnqueueBuffer(a);
  infeed->EnqueueBuffer(b);
  ProcessNextBuffer(a->length());
  ProcessNextBuffer(b->length());
}

TEST_F(InfeedManagerTest, SingleThreadedInterleaved) {
  TestInfeedBuffer* a = new TestInfeedBuffer(64);
  TestInfeedBuffer* b = new TestInfeedBuffer(32);

  cpu::runtime::InfeedManager* infeed = cpu::runtime::GetInfeedManager();

  infeed->EnqueueBuffer(a);
  ProcessNextBuffer(a->length());
  infeed->EnqueueBuffer(b);
  ProcessNextBuffer(b->length());
}

TEST_F(InfeedManagerTest, MultiThreaded) {
  tensorflow::thread::ThreadPool pool(tensorflow::Env::Default(), "test", 2);

  cpu::runtime::InfeedManager* infeed = cpu::runtime::GetInfeedManager();

  const int32 length = 64;

  pool.Schedule([infeed]() {
    // Spin for 100 milliseconds
    int64 start_micros = tensorflow::Env::Default()->NowMicros();
    while (true) {
      int64 end_micros = tensorflow::Env::Default()->NowMicros();
      if ((end_micros - start_micros) >= 100000) {  // 100 ms
        break;
      }
    }
    TestInfeedBuffer* a = new TestInfeedBuffer(length);
    infeed->EnqueueBuffer(a);
  });

  ProcessNextBuffer(length);
}

}  // namespace
}  // namespace xla
