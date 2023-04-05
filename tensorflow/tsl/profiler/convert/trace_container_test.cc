/* Copyright 2023 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/tsl/profiler/convert/trace_container.h"

#include <string>

#include "tensorflow/tsl/platform/protobuf.h"
#include "tensorflow/tsl/platform/test.h"

namespace tsl {
namespace profiler {
namespace {

void PopulateDummyEvent(TraceEvent* const event) {
  event->set_device_id(1);
  event->set_resource_id(2);
  event->set_name("A");
  event->set_timestamp_ps(3);
  event->set_duration_ps(4);
}

TEST(TraceContainer, TraceEventAllocation) {
  TraceContainer container;
  PopulateDummyEvent(container.CreateEvent());
}

TEST(TraceContainer, FlushAndSerializeEvents) {
  TraceContainer container;

  PopulateDummyEvent(container.CreateEvent());

  EXPECT_EQ(container.UnsortedEvents().size(), 1);

  std::string serialized;
  container.FlushAndSerializeEvents(&serialized);

  EXPECT_EQ(container.UnsortedEvents().size(), 0);

  PopulateDummyEvent(container.CreateEvent());

  EXPECT_EQ(container.UnsortedEvents().size(), 1);

  std::string reserialized;
  container.FlushAndSerializeEvents(&reserialized);

  EXPECT_EQ(serialized, reserialized);
  EXPECT_EQ(container.UnsortedEvents().size(), 0);

  Trace trace;
  trace.ParseFromString(reserialized);

  EXPECT_EQ(trace.trace_events_size(), 1);
}

TEST(TraceContainer, CapEvents) {
  TraceContainer container;
  for (int i = 0; i < 100; i++) {
    container.CreateEvent()->set_timestamp_ps((100 - i) % 50);
  }
  // No dropping.
  container.CapEvents(101);
  EXPECT_EQ(container.UnsortedEvents().size(), 100);

  container.CapEvents(100);
  EXPECT_EQ(container.UnsortedEvents().size(), 100);

  container.CapEvents(99);
  EXPECT_EQ(container.UnsortedEvents().size(), 99);

  container.CapEvents(50);
  EXPECT_EQ(container.UnsortedEvents().size(), 50);
  for (const TraceEvent* const event : container.UnsortedEvents()) {
    EXPECT_LT(event->timestamp_ps(), 25);
  }
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
