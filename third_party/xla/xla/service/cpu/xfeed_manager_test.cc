/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/xfeed_manager.h"

#include <cstdint>
#include <string>

#include "xla/service/cpu/cpu_runtime.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace {

class InfeedManagerTest : public ::testing::Test {};

class TestInfeedBuffer : public cpu::runtime::XfeedBuffer {
 public:
  explicit TestInfeedBuffer(int32_t length, bool expect_shape_match = true)
      : shape_(ShapeUtil::MakeShape(U8, {length})),
        done_called_(false),
        length_(length),
        expect_shape_match_(expect_shape_match) {}
  ~TestInfeedBuffer() override { EXPECT_TRUE(done_called_); }

  int32_t length() override { return length_; }
  void* data() override { return nullptr; }
  void Done(absl::StatusOr<Shape> shape) override {
    CHECK(!done_called_);
    done_called_ = true;
    TF_ASSERT_OK(shape.status());
    EXPECT_EQ(expect_shape_match_, ShapeUtil::Equal(shape_, shape.value()))
        << "want " << ShapeUtil::HumanString(shape_) << " "
        << (expect_shape_match_ ? "==" : "!=") << " "
        << ShapeUtil::HumanString(shape.value());
    delete this;
  }

  const Shape& shape() const { return shape_; }

 private:
  Shape shape_;
  bool done_called_;
  int32_t length_;
  bool expect_shape_match_;
};

// Performs the acquire/release sequence on the infeed, as the generated CPU
// code would in the process of executing the infeed operation.
void ProcessNextBuffer(int32_t length) {
  auto shape = ShapeUtil::MakeShape(U8, {length});
  std::string bytes = shape.SerializeAsString();
  void* buffer = __xla_cpu_runtime_AcquireInfeedBufferForDequeue(
      /*run_options=*/nullptr, length, bytes.data(), bytes.size());
  __xla_cpu_runtime_ReleaseInfeedBufferAfterDequeue(
      /*run_options=*/nullptr, length, buffer, bytes.data(), bytes.size());
}

// Performs the acquire/release sequence on the outfeed, as the generated CPU
// code would in the process of executing the outfeed operation.
void ProcessNextOutfeedBuffer(int32_t length, const Shape& shape) {
  std::string bytes = shape.SerializeAsString();
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
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test", 2);

  cpu::runtime::XfeedManager* xfeed = cpu::runtime::GetXfeedManager(0);

  const int32_t length = 64;

  pool.Schedule([length, &xfeed]() {
    // Spin for 100 milliseconds
    int64_t start_micros = tsl::Env::Default()->NowMicros();
    while (true) {
      int64_t end_micros = tsl::Env::Default()->NowMicros();
      if ((end_micros - start_micros) >= 100000) {  // 100 ms
        break;
      }
    }
    TestInfeedBuffer* a = new TestInfeedBuffer(length);
    xfeed->infeed()->EnqueueBuffersAtomically({a});
  });

  ProcessNextBuffer(length);
}

TEST_F(InfeedManagerTest, OutfeedBasic) {
  TestInfeedBuffer* b = new TestInfeedBuffer(32, /*expect_shape_match=*/true);
  cpu::runtime::XfeedManager* xfeed = cpu::runtime::GetXfeedManager(0);
  xfeed->outfeed()->EnqueueBuffersAtomically({b});

  ProcessNextOutfeedBuffer(32, ShapeUtil::MakeShape(U8, {32}));
}

TEST_F(InfeedManagerTest, OutfeedEmpty) {
  TestInfeedBuffer* b = new TestInfeedBuffer(0, /*expect_shape_match=*/true);
  cpu::runtime::XfeedManager* xfeed = cpu::runtime::GetXfeedManager(0);
  xfeed->outfeed()->EnqueueBuffersAtomically({b});

  ProcessNextOutfeedBuffer(0, ShapeUtil::MakeShape(U8, {0}));
}

TEST_F(InfeedManagerTest, OutfeedWrongShape) {
  TestInfeedBuffer* b = new TestInfeedBuffer(32, /*expect_shape_match=*/false);
  cpu::runtime::XfeedManager* xfeed = cpu::runtime::GetXfeedManager(0);
  xfeed->outfeed()->EnqueueBuffersAtomically({b});

  ProcessNextOutfeedBuffer(32, ShapeUtil::MakeShape(U8, {33}));
}

}  // namespace
}  // namespace xla
