/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tsl/profiler/lib/continuous_profiler_orchestrator.h"

#include <any>
#include <atomic>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/test.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

using ::testing::Invoke;
using ::testing::Pointee;
using ::testing::Return;

class MockProfiler : public ProfilerInterface {
 public:
  MOCK_METHOD(absl::Status, Start, (), (override));
  MOCK_METHOD(absl::Status, Stop, (), (override));
  MOCK_METHOD(absl::Status, CollectData, (tensorflow::profiler::XSpace * space),
              (override));
  MOCK_METHOD(absl::StatusOr<ConsumeResult>, Consume, (), (override));

  absl::Status Serialize(std::any data,
                         tensorflow::profiler::XSpace* space) override {
    return SerializeMock(&data, space);
  }
  MOCK_METHOD(absl::Status, SerializeMock,
              (const std::any* data, tensorflow::profiler::XSpace* space));
};

// Custom Google Mock matcher to compare integers wrapped in std::any.
MATCHER_P(AnyEqInt, expected_value, "") {
  auto* val = std::any_cast<int>(&arg);
  return val != nullptr && *val == expected_value;
}

TEST(ContinuousProfilerOrchestratorTest,
     CircularBufferingAndCorrectSerialization) {
  auto mock_profiler = std::make_unique<MockProfiler>();
  MockProfiler* mock = mock_profiler.get();

  EXPECT_CALL(*mock, Start()).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock, Stop()).WillOnce(Return(absl::OkStatus()));

  // Setup Consume mock to return sequential chunks with high sizes to shrink
  // the interval
  std::atomic<int> consume_count(0);
  EXPECT_CALL(*mock, Consume())
      .WillRepeatedly(Invoke([&]() -> absl::StatusOr<ConsumeResult> {
        int count = ++consume_count;
        return ConsumeResult{
            .data = std::any(count),
            .estimated_size_bytes = 1000 * 1024 * 1024  // 1000MB (>512MB)
        };
      }));

  ContinuousProfilerOrchestrator<ProfilerInterface> orchestrator(
      std::move(mock_profiler));

  // Start orchestrator (spawns background loop)
  ASSERT_OK(orchestrator.Start());

  // Wait until we have consumed at least 4 chunks.
  // Due to high watermark scaling, interval shrinks from 1s -> 500ms -> 250ms
  // -> 125ms -> 100ms. This will happen in less than 1.0 second of real-time
  // sleep.
  while (consume_count < 4) {
    absl::SleepFor(absl::Milliseconds(50));
  }

  // Stop orchestrator (terminates background loop)
  ASSERT_OK(orchestrator.Stop());

  // Verify polling interval shrank from 1s due to high watermark
  EXPECT_LE(orchestrator.polling_interval(), absl::Milliseconds(500));

  // Verify that CollectData serializes the chunks in correct chronological
  // order!
  tensorflow::profiler::XSpace space;

  // We expect Serialize to be called for each chunk in order: 1, 2, 3...
  ::testing::InSequence seq;
  for (int i = 1; i <= consume_count; ++i) {
    EXPECT_CALL(*mock, SerializeMock(Pointee(AnyEqInt(i)), &space))
        .WillOnce(Return(absl::OkStatus()));
  }
  // And finally mock CollectData is called to collect remainder.
  EXPECT_CALL(*mock, CollectData(&space)).WillOnce(Return(absl::OkStatus()));

  // Call CollectData on the orchestrator.
  EXPECT_OK(orchestrator.CollectData(&space));
}

TEST(ContinuousProfilerOrchestratorTest, DynamicIntervalLowWatermarkScaling) {
  auto mock_profiler = std::make_unique<MockProfiler>();
  MockProfiler* mock = mock_profiler.get();

  EXPECT_CALL(*mock, Start()).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock, Stop()).WillOnce(Return(absl::OkStatus()));

  // Consume returns a very small chunk (1MB < 5MB low watermark)
  EXPECT_CALL(*mock, Consume())
      .WillRepeatedly([]() -> absl::StatusOr<ConsumeResult> {
        return ConsumeResult{
            .data = std::any(1),
            .estimated_size_bytes = 1 * 1024 * 1024  // 1MB (<5MB)
        };
      });

  ContinuousProfilerOrchestrator<ProfilerInterface> orchestrator(
      std::move(mock_profiler));
  EXPECT_EQ(orchestrator.polling_interval(), absl::Seconds(1));  // initial

  // Start
  ASSERT_OK(orchestrator.Start());

  // Wait a very short time for the first immediate consume to run and adjust
  // the interval
  absl::SleepFor(absl::Milliseconds(50));

  // Stop immediately
  ASSERT_OK(orchestrator.Stop());

  // Verify interval scaled up from 1s to 2s due to low watermark!
  EXPECT_EQ(orchestrator.polling_interval(), absl::Seconds(2));
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
