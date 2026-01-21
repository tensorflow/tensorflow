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

#include "xla/backends/cpu/runtime/xfeed_manager.h"

#include <cstdint>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

class InfeedManagerTest : public ::testing::Test {};

class TestInfeedBuffer : public XfeedBuffer {
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
    ASSERT_OK(shape.status());
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
void ProcessNextBuffer(XfeedManager* xfeed, int32_t length) {
  const auto shape = ShapeUtil::MakeShape(U8, {length});
  const std::string bytes = shape.ToProto().SerializeAsString();

  XfeedBuffer* buffer = xfeed->infeed()->BlockingDequeueBuffer();
  ASSERT_EQ(buffer->length(), length);

  xfeed->infeed()->ReleaseCurrentBuffer(buffer->length(), buffer->data(),
                                        std::move(shape));
}

// Performs the acquire/release sequence on the outfeed, as the generated CPU
// code would in the process of executing the outfeed operation.
void ProcessNextOutfeedBuffer(XfeedManager* xfeed, int32_t length,
                              const Shape& shape) {
  const std::string bytes = shape.ToProto().SerializeAsString();

  XfeedBuffer* buffer = xfeed->outfeed()->BlockingDequeueBuffer();
  ASSERT_EQ(buffer->length(), length);

  xfeed->outfeed()->ReleaseCurrentBuffer(buffer->length(), buffer->data(),
                                         std::move(shape));
}

TEST_F(InfeedManagerTest, SingleThreadedSequential) {
  TestInfeedBuffer* a = new TestInfeedBuffer(64);
  TestInfeedBuffer* b = new TestInfeedBuffer(32);

  XfeedManager* xfeed = GetXfeedManager(0);

  xfeed->infeed()->EnqueueBuffersAtomically({a});
  xfeed->infeed()->EnqueueBuffersAtomically({b});
  ProcessNextBuffer(xfeed, a->length());
  ProcessNextBuffer(xfeed, b->length());
}

TEST_F(InfeedManagerTest, SingleThreadedInterleaved) {
  TestInfeedBuffer* a = new TestInfeedBuffer(64);
  TestInfeedBuffer* b = new TestInfeedBuffer(32);

  XfeedManager* xfeed = GetXfeedManager(0);

  xfeed->infeed()->EnqueueBuffersAtomically({a});
  ProcessNextBuffer(xfeed, a->length());
  xfeed->infeed()->EnqueueBuffersAtomically({b});
  ProcessNextBuffer(xfeed, b->length());
}

TEST_F(InfeedManagerTest, MultiThreaded) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test", 2);

  XfeedManager* xfeed = GetXfeedManager(0);

  const int32_t length = 64;

  pool.Schedule([&xfeed]() {
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

  ProcessNextBuffer(xfeed, length);
}

TEST_F(InfeedManagerTest, OutfeedBasic) {
  TestInfeedBuffer* b = new TestInfeedBuffer(32, /*expect_shape_match=*/true);
  XfeedManager* xfeed = GetXfeedManager(0);
  xfeed->outfeed()->EnqueueBuffersAtomically({b});

  ProcessNextOutfeedBuffer(xfeed, 32, ShapeUtil::MakeShape(U8, {32}));
}

TEST_F(InfeedManagerTest, OutfeedEmpty) {
  TestInfeedBuffer* b = new TestInfeedBuffer(0, /*expect_shape_match=*/true);
  XfeedManager* xfeed = GetXfeedManager(0);
  xfeed->outfeed()->EnqueueBuffersAtomically({b});

  ProcessNextOutfeedBuffer(xfeed, 0, ShapeUtil::MakeShape(U8, {0}));
}

TEST_F(InfeedManagerTest, OutfeedWrongShape) {
  TestInfeedBuffer* b = new TestInfeedBuffer(32, /*expect_shape_match=*/false);
  XfeedManager* xfeed = GetXfeedManager(0);
  xfeed->outfeed()->EnqueueBuffersAtomically({b});

  ProcessNextOutfeedBuffer(xfeed, 32, ShapeUtil::MakeShape(U8, {33}));
}

}  // namespace
}  // namespace xla::cpu
