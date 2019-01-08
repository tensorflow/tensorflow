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

#include "tensorflow/compiler/xla/service/cpu/xfeed_manager.h"

#include <memory>

#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class InfeedManagerTest : public ::testing::Test {};

class TestInfeedBuffer : public cpu::runtime::XfeedBuffer {
 public:
  explicit TestInfeedBuffer(int32 length, bool expect_shape_match = true)
      : shape_(ShapeUtil::MakeShape(U8, {length})),
        done_called_(false),
        length_(length),
        expect_shape_match_(expect_shape_match) {}
  ~TestInfeedBuffer() override { EXPECT_TRUE(done_called_); }

  int32 length() override { return length_; }
  void* data() override { return nullptr; }
  void Done(StatusOr<Shape> shape) override {
    CHECK(!done_called_);
    done_called_ = true;
    TF_ASSERT_OK(shape.status());
    EXPECT_EQ(expect_shape_match_, ShapeUtil::Equal(shape_, shape.ValueOrDie()))
        << "want " << ShapeUtil::HumanString(shape_) << " "
        << (expect_shape_match_ ? "==" : "!=") << " "
        << ShapeUtil::HumanString(shape.ValueOrDie());
  }

  const Shape& shape() const { return shape_; }

 private:
  Shape shape_;
  bool done_called_;
  int32 length_;
  bool expect_shape_match_;
};

// Performs the acquire/release sequence on the infeed, as the generated CPU
// code would in the process of executing the infeed operation.
void ProcessNextBuffer(int32 length) {
  auto shape = ShapeUtil::MakeShape(U8, {length});
  string bytes = shape.SerializeAsString();
  void* buffer = __xla_cpu_runtime_AcquireInfeedBufferForDequeue(
      /*run_options=*/nullptr, length, bytes.data(), bytes.size());
  __xla_cpu_runtime_ReleaseInfeedBufferAfterDequeue(
      /*run_options=*/nullptr, length, buffer, bytes.data(), bytes.size());
}

// Performs the acquire/release sequence on the outfeed, as the generated CPU
// code would in the process of executing the outfeed operation.
void ProcessNextOutfeedBuffer(int32 length, const Shape& shape) {
  string bytes = shape.SerializeAsString();
  void* buffer = __xla_cpu_runtime_AcquireOutfeedBufferForPopulation(
      /*run_options=*/nullptr, length, bytes.data(), bytes.size());
  __xla_cpu_runtime_ReleaseOutfeedBufferAfterPopulation(
      /*run_options=*/nullptr, length, buffer, bytes.data(), bytes.size());
}

TEST_F(InfeedManagerTest, SingleThreadedSequential) {
  TestInfeedBuffer* a = new TestInfeedBuffer(64);
  TestInfeedBuffer* b = new TestInfeedBuffer(32);

  cpu::runtime::XfeedManager* xfeed = cpu::runtime::GetXfeedManager(0);

  xfeed->infeed()->EnqueueBuffersAtomically({a});
  xfeed->infeed()->EnqueueBuffersAtomically({b});
  ProcessNextBuffer(a->length());
  ProcessNextBuffer(b->length());
}

TEST_F(InfeedManagerTest, SingleThreadedInterleaved) {
  TestInfeedBuffer* a = new TestInfeedBuffer(64);
  TestInfeedBuffer* b = new TestInfeedBuffer(32);

  cpu::runtime::XfeedManager* xfeed = cpu::runtime::GetXfeedManager(0);

  xfeed->infeed()->EnqueueBuffersAtomically({a});
  ProcessNextBuffer(a->length());
  xfeed->infeed()->EnqueueBuffersAtomically({b});
  ProcessNextBuffer(b->length());
}

TEST_F(InfeedManagerTest, MultiThreaded) {
  tensorflow::thread::ThreadPool pool(tensorflow::Env::Default(), "test", 2);

  cpu::runtime::XfeedManager* xfeed = cpu::runtime::GetXfeedManager(0);

  const int32 length = 64;

  pool.Schedule([xfeed]() {
    // Spin for 100 milliseconds
    int64 start_micros = tensorflow::Env::Default()->NowMicros();
    while (true) {
      int64 end_micros = tensorflow::Env::Default()->NowMicros();
      if ((end_micros - start_micros) >= 100000) {  // 100 ms
        break;
      }
    }
    TestInfeedBuffer* a = new TestInfeedBuffer(length);
    xfeed->infeed()->EnqueueBuffersAtomically({a});
  });

  ProcessNextBuffer(length);
}

TEST_F(InfeedManagerTest, OutfeedWrongShape) {
  TestInfeedBuffer* b = new TestInfeedBuffer(32, /*expect_shape_match=*/false);
  cpu::runtime::XfeedManager* xfeed = cpu::runtime::GetXfeedManager(0);
  xfeed->outfeed()->EnqueueBuffersAtomically({b});

  ProcessNextOutfeedBuffer(32, ShapeUtil::MakeShape(U8, {33}));
}

}  // namespace
}  // namespace xla
