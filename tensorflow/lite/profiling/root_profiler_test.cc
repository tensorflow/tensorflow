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

#include "tensorflow/lite/profiling/root_profiler.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/api/profiler.h"

using ::testing::_;
using ::testing::StrictMock;

namespace tflite {
namespace profiling {

namespace {

constexpr char kTag[] = "tag";

class MockProfiler : public Profiler {
 public:
  MOCK_METHOD(uint32_t, BeginEvent,
              (const char* tag, EventType event_type, int64_t event_metadata1,
               int64_t event_metadata2),
              (override));
  MOCK_METHOD(void, EndEvent, (uint32_t event_handle), (override));
  MOCK_METHOD(void, EndEvent,
              (uint32_t event_handle, int64_t event_metadata1,
               int64_t event_metadata2),
              (override));
  MOCK_METHOD(void, AddEvent,
              (const char* tag, EventType event_type, uint64_t metric,
               int64_t event_metadata1, int64_t event_metadata2),
              (override));
  MOCK_METHOD(void, AddEventWithData,
              (const char* tag, EventType event_type, const void* data),
              (override));
};

using MockProfilerT = StrictMock<MockProfiler>;

TEST(RootProfilerTest, ChildProfilerTest) {
  auto mock_profiler = std::make_unique<MockProfilerT>();
  auto* mock = mock_profiler.get();
  RootProfiler root;
  root.AddProfiler(mock_profiler.get());

  ON_CALL(*mock, BeginEvent(_, _, _, _)).WillByDefault(testing::Return(42));

  EXPECT_CALL(*mock, BeginEvent(kTag, Profiler::EventType::DEFAULT, 1, 2));
  EXPECT_CALL(*mock, EndEvent(42, 3, 4));
  EXPECT_CALL(*mock, AddEvent(kTag, Profiler::EventType::OPERATOR_INVOKE_EVENT,
                              5, 6, 7));
  EXPECT_CALL(*mock, AddEventWithData(kTag, Profiler::EventType::DEFAULT, _));

  // Calls each method sequentially.
  auto begin = root.BeginEvent(kTag, Profiler::EventType::DEFAULT, 1, 2);
  root.EndEvent(begin, 3, 4);
  root.AddEvent(kTag, Profiler::EventType::OPERATOR_INVOKE_EVENT, 5, 6, 7);
  root.AddEventWithData(kTag, Profiler::EventType::DEFAULT, nullptr);
}

TEST(RootProfilerTest, OwnedProfilerTest) {
  auto mock_profiler = std::make_unique<MockProfilerT>();
  auto* mock = mock_profiler.get();
  RootProfiler root;
  root.AddProfiler(std::move(mock_profiler));

  ON_CALL(*mock, BeginEvent(_, _, _, _)).WillByDefault(testing::Return(42));

  EXPECT_CALL(*mock, BeginEvent(kTag, Profiler::EventType::DEFAULT, 1, 2));
  EXPECT_CALL(*mock, EndEvent(42));
  EXPECT_CALL(*mock, AddEvent(kTag, Profiler::EventType::OPERATOR_INVOKE_EVENT,
                              3, 4, 5));

  // Calls each method sequentially.
  auto begin = root.BeginEvent(kTag, Profiler::EventType::DEFAULT, 1, 2);
  root.EndEvent(begin);
  root.AddEvent(kTag, Profiler::EventType::OPERATOR_INVOKE_EVENT, 3, 4, 5);
}

TEST(RootProfilerTest, MultipleProfilerTest) {
  auto mock_profiler0 = std::make_unique<MockProfilerT>();
  auto* mock0 = mock_profiler0.get();
  auto mock_profiler1 = std::make_unique<MockProfilerT>();
  auto* mock1 = mock_profiler1.get();
  RootProfiler root;
  root.AddProfiler(std::move(mock_profiler0));
  root.AddProfiler(std::move(mock_profiler1));

  // Different child profilers might return different event id.
  ON_CALL(*mock0, BeginEvent(_, _, _, _)).WillByDefault(testing::Return(42));
  ON_CALL(*mock1, BeginEvent(_, _, _, _)).WillByDefault(testing::Return(24));

  EXPECT_CALL(*mock0, BeginEvent(kTag, Profiler::EventType::DEFAULT, 1, 2));
  EXPECT_CALL(*mock0, EndEvent(42));
  EXPECT_CALL(*mock1, BeginEvent(kTag, Profiler::EventType::DEFAULT, 1, 2));
  EXPECT_CALL(*mock1, EndEvent(24));

  // Calls each method sequentially.
  auto begin = root.BeginEvent(kTag, Profiler::EventType::DEFAULT, 1, 2);
  root.EndEvent(begin);
}

}  // namespace
}  // namespace profiling
}  // namespace tflite
