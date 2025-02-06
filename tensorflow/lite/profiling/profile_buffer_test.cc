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
#include "tensorflow/lite/profiling/profile_buffer.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace tflite {
namespace profiling {

namespace {

std::vector<const ProfileEvent*> GetProfileEvents(const ProfileBuffer& buffer) {
  std::vector<const ProfileEvent*> events;
  for (size_t i = 0; i < buffer.Size(); i++) {
    events.push_back(buffer.At(i));
  }
  return events;
}

TEST(ProfileBufferTest, Empty) {
  ProfileBuffer buffer(/*max_size*/ 0, /*enabled*/ true);
  EXPECT_EQ(0, buffer.Size());
}

TEST(ProfileBufferTest, AddEvent) {
  ProfileBuffer buffer(/*max_size*/ 10, /*enabled*/ true);
  EXPECT_EQ(0, buffer.Size());
  auto event_handle =
      buffer.BeginEvent("hello", ProfileEvent::EventType::DEFAULT,
                        /*event_metadata1*/ 42, /*event_metadata2*/ 0);

  EXPECT_GE(event_handle, 0);
  EXPECT_EQ(1, buffer.Size());

  auto event = GetProfileEvents(buffer)[0];
  EXPECT_EQ(event->tag, "hello");
  EXPECT_GT(event->begin_timestamp_us, 0);
  EXPECT_EQ(event->event_type, ProfileEvent::EventType::DEFAULT);
  EXPECT_EQ(event->event_metadata, 42);

  buffer.EndEvent(event_handle);
  EXPECT_EQ(1, buffer.Size());
  EXPECT_GE(event->elapsed_time, 0);
}

TEST(ProfileBufferTest, EndEventWithMetadata) {
  ProfileBuffer buffer(/*max_size*/ 10, /*enabled*/ true);
  EXPECT_EQ(0, buffer.Size());
  auto event_handle =
      buffer.BeginEvent("hello", ProfileEvent::EventType::DEFAULT,
                        /*event_metadata1*/ 42, /*event_metadata2*/ 0);
  const int64_t kEventMetadata1 = 18;
  const int64_t kEventMetadata2 = 36;
  buffer.EndEvent(event_handle, &kEventMetadata1, &kEventMetadata2);

  EXPECT_GE(event_handle, 0);
  EXPECT_EQ(1, buffer.Size());
  auto event = GetProfileEvents(buffer)[0];
  EXPECT_EQ(event->tag, "hello");
  EXPECT_GT(event->begin_timestamp_us, 0);
  EXPECT_EQ(event->event_type, ProfileEvent::EventType::DEFAULT);
  EXPECT_EQ(event->event_metadata, kEventMetadata1);
  EXPECT_EQ(event->extra_event_metadata, kEventMetadata2);
  EXPECT_EQ(1, buffer.Size());
  EXPECT_GE(event->elapsed_time, 0);
}

TEST(ProfileBufferTest, OverFlow) {
  const int max_size = 4;
  ProfileBuffer buffer{max_size, true};
  std::vector<std::string> eventNames = {"first", "second", "third", "fourth"};
  for (int i = 0; i < 2 * max_size; i++) {
    buffer.BeginEvent(eventNames[i % 4].c_str(),
                      ProfileEvent::EventType::DEFAULT, i, 0);
    size_t expected_size = std::min(i + 1, max_size);
    EXPECT_EQ(expected_size, buffer.Size());
  }
  EXPECT_EQ(max_size, buffer.Size());
  for (size_t j = 0; j < buffer.Size(); ++j) {
    auto event = buffer.At(j);
    EXPECT_EQ(eventNames[j % 4], event->tag);
    EXPECT_EQ(ProfileEvent::EventType::DEFAULT, event->event_type);
    EXPECT_EQ(j, event->event_metadata);
  }
}

TEST(ProfileBufferTest, DynamicIncrease) {
  const int max_initial_size = 4;
  ProfileBuffer buffer{max_initial_size, true,
                       true /*allow_dynamic_buffer_increase*/};
  std::vector<std::string> eventNames = {"first", "second", "third", "fourth"};
  for (int i = 0; i < 2 * max_initial_size; i++) {
    buffer.BeginEvent(eventNames[i % 4].c_str(),
                      ProfileEvent::EventType::DEFAULT, i, 0);
    const size_t expected_size = i + 1;
    EXPECT_EQ(expected_size, buffer.Size());
  }
  EXPECT_EQ(2 * max_initial_size, buffer.Size());
  for (size_t j = 0; j < buffer.Size(); ++j) {
    auto event = buffer.At(j);
    EXPECT_EQ(eventNames[j % 4], event->tag);
    EXPECT_EQ(ProfileEvent::EventType::DEFAULT, event->event_type);
    EXPECT_EQ(j, event->event_metadata);
  }
}

TEST(ProfileBufferTest, Enable) {
  ProfileBuffer buffer(/*max_size*/ 10, /*enabled*/ false);
  EXPECT_EQ(0, buffer.Size());
  auto event_handle =
      buffer.BeginEvent("hello", ProfileEvent::EventType::DEFAULT,
                        /*event_metadata1*/ 42, /*event_metadata2*/ 0);
  EXPECT_EQ(kInvalidEventHandle, event_handle);
  EXPECT_EQ(0, buffer.Size());
  buffer.SetEnabled(true);
  event_handle =
      buffer.BeginEvent("hello", ProfileEvent::EventType::DEFAULT,
                        /*event_metadata1*/ 42, /*event_metadata2*/ 0);
  EXPECT_GE(event_handle, 0);
  EXPECT_EQ(1, buffer.Size());
}

}  // namespace
}  // namespace profiling
}  // namespace tflite
