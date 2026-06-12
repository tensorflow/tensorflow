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
#include "xla/backends/profiler/cpu/host_tracer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/synchronization/blocking_counter.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/backends/cpu/traceme_recorder.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

using ::tsl::Env;
using ::tsl::Thread;
using ::tsl::ThreadOptions;
using ::tsl::profiler::StatType;
using ::tsl::profiler::Timespan;
using ::tsl::profiler::TraceMe;
using ::tsl::profiler::XEventVisitor;
using ::tsl::profiler::XLineVisitor;
using ::tsl::profiler::XPlaneVisitor;
using ::tsl::profiler::XStatVisitor;

TEST(HostTracerTest, CollectsTraceMeEventsAsXSpace) {
  int64_t thread_id;
  std::string thread_name = "MyThreadName";
  tensorflow::profiler::XSpace space;

  // We start a thread with a known and controlled name. As of the time of
  // writing, not all platforms (example: Windows) allow reading through the
  // system to the current thread name/description. By starting a thread with a
  // name, we control this behavior entirely within the TensorFlow subsystems.
  std::unique_ptr<Thread> traced_thread(
      Env::Default()->StartThread(ThreadOptions(), thread_name, [&] {
        // Some implementations add additional information to the thread name.
        // Recapture this information.
        ASSERT_TRUE(Env::Default()->GetCurrentThreadName(&thread_name));
        thread_id = Env::Default()->GetCurrentThreadId();

        auto tracer = CreateHostTracer({});

        TF_ASSERT_OK(tracer->Start());
        { TraceMe traceme("hello"); }
        { TraceMe traceme("world"); }
        { TraceMe traceme("contains#inside"); }
        { TraceMe traceme("good#key1=value1#"); }
        { TraceMe traceme("morning#key1=value1,key2=value2#"); }
        { TraceMe traceme("incomplete#key1=value1,key2#"); }
        // Special cases for tf.data
        { TraceMe traceme("Iterator::XXX::YYY::ParallelMap"); }
        TF_ASSERT_OK(tracer->Stop());

        TF_ASSERT_OK(tracer->CollectData(&space));
      }));
  traced_thread.reset();      // Join thread, waiting for completion.
  ASSERT_NO_FATAL_FAILURE();  // Test for failure in child thread.

  ASSERT_EQ(space.planes_size(), 1);
  const auto& plane = space.planes(0);
  XPlaneVisitor xplane(&plane);
  ASSERT_EQ(plane.name(), ::tsl::profiler::kHostThreadsPlaneName);
  ASSERT_EQ(plane.lines_size(), 1);
  ASSERT_EQ(plane.event_metadata_size(), 7);
  ASSERT_EQ(plane.stat_metadata_size(), 4);
  const auto& line = plane.lines(0);
  EXPECT_EQ(line.id(), thread_id);
  EXPECT_EQ(line.name(), thread_name);
  ASSERT_EQ(line.events_size(), 7);
  const auto& events = line.events();

  XEventVisitor e0(&xplane, &line, &events[0]);
  EXPECT_EQ(e0.Name(), "hello");
  ASSERT_EQ(events[0].stats_size(), 0);

  XEventVisitor e1(&xplane, &line, &events[1]);
  EXPECT_EQ(e1.Name(), "world");
  ASSERT_EQ(events[1].stats_size(), 0);

  XEventVisitor e2(&xplane, &line, &events[2]);
  EXPECT_EQ(e2.Name(), "contains#inside");
  ASSERT_EQ(events[2].stats_size(), 0);

  XEventVisitor e3(&xplane, &line, &events[3]);
  EXPECT_EQ(e3.Name(), "good");
  ASSERT_EQ(events[3].stats_size(), 1);
  {
    std::optional<std::string> value;
    e3.ForEachStat([&](const XStatVisitor& stat) {
      if (stat.Name() == "key1") value = stat.ToString();
    });
    ASSERT_TRUE(value);           // The stat key is present.
    EXPECT_EQ(*value, "value1");  // The stat value is expected.
  }

  XEventVisitor e4(&xplane, &line, &events[4]);
  EXPECT_EQ(e4.Name(), "morning");
  ASSERT_EQ(events[4].stats_size(), 2);
  {
    std::optional<std::string> value1, value2;
    e4.ForEachStat([&](const XStatVisitor& stat) {
      if (stat.Name() == "key1") {
        value1 = stat.ToString();
      } else if (stat.Name() == "key2") {
        value2 = stat.ToString();
      }
    });
    ASSERT_TRUE(value1 && value2);  // The stat keys are presents.
    EXPECT_EQ(*value1, "value1");   // The stat value1 is expected.
    EXPECT_EQ(*value2, "value2");   // The stat value2 is expected.
  }

  XEventVisitor e5(&xplane, &line, &events[5]);
  EXPECT_EQ(e5.Name(), "incomplete");
  ASSERT_EQ(events[5].stats_size(), 1);
  {
    std::optional<std::string> value1, value2;
    e5.ForEachStat([&](const XStatVisitor& stat) {
      if (stat.Name() == "key1") {
        value1 = stat.ToString();
      } else if (stat.Name() == "key2") {
        value2 = stat.ToString();
      }
    });
    ASSERT_TRUE(value1 && !value2);  // One of the stat key is present.
    EXPECT_EQ(*value1, "value1");    // The stat value is expected.
  }

  // Dataset Ops will trim intermediate namespace.
  XEventVisitor e6(&xplane, &line, &events[6]);
  EXPECT_EQ(e6.Name(), "Iterator::XXX::YYY::ParallelMap");

  EXPECT_EQ(e6.DisplayName(), "Iterator::ParallelMap");
}

TEST(HostTracerTest, CollectEventsFromThreadPool) {
  auto thread_pool =
      std::make_unique<tsl::thread::ThreadPool>(/*env=*/Env::Default(),
                                                /*name=*/"HostTracerTest",
                                                /*num_threads=*/1);
  absl::BlockingCounter counter(1);
  auto tracer = CreateHostTracer({});
  TF_EXPECT_OK(tracer->Start());
  thread_pool->Schedule([&counter] {
    TraceMe traceme("hello");
    counter.DecrementCount();
  });
  counter.Wait();
  // Explicitly delete the thread_pool before trying to collect performance data
  // to ensure that there's no window between the ThreadPool ending the region
  // and collecting the trace data.  This was the cause of this test being racy
  // in the past.
  thread_pool.reset();
  TF_EXPECT_OK(tracer->Stop());
  tensorflow::profiler::XSpace space;
  TF_EXPECT_OK(tracer->CollectData(&space));

  EXPECT_THAT(space.planes(), testing::SizeIs(1));
  XPlaneVisitor xplane = tsl::profiler::CreateTfXPlaneVisitor(&space.planes(0));

  bool has_record_event = false;
  bool has_start_region_event = false;
  bool has_end_region_event = false;
  int64_t record_region_id = 0;
  int64_t start_region_id = 0;

  Timespan region_timespan;
  Timespan traceme_timespan;

  xplane.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      if (event.Name() == tsl::profiler::kThreadpoolListenerRecord) {
        has_record_event = true;
        const auto& stat = event.GetStat(StatType::kProducerId);
        EXPECT_TRUE(stat.has_value());
        record_region_id = stat->IntOrUintValue();
      } else if (event.Name() ==
                 tsl::profiler::kThreadpoolListenerStartRegion) {
        has_start_region_event = true;
        const auto& stat = event.GetStat(StatType::kConsumerId);
        EXPECT_TRUE(stat.has_value());
        start_region_id = stat->IntOrUintValue();
        region_timespan = event.GetTimespan();
      } else if (event.Name() == tsl::profiler::kThreadpoolListenerStopRegion) {
        has_end_region_event = true;
        region_timespan = Timespan::FromEndPoints(region_timespan.begin_ps(),
                                                  event.GetTimespan().end_ps());
      } else if (event.Name() == "hello") {
        traceme_timespan = event.GetTimespan();
      }
    });
  });

  EXPECT_TRUE(has_record_event);
  EXPECT_TRUE(has_start_region_event);
  EXPECT_TRUE(has_end_region_event);

  EXPECT_EQ(record_region_id, start_region_id);

  EXPECT_TRUE(region_timespan.Includes(traceme_timespan));
}

TEST(HostTracerTest, ConsumeSucceedsWhileRecording) {
  auto tracer = internal::CreateHostTracerImpl({});
  TF_ASSERT_OK(tracer->Start());

  {
    TraceMe traceme("streaming_event_1");
  }

  tsl::profiler::TraceMeRecorder::Events flush_events_1;
  absl::Status status =
      tracer->Consume(&flush_events_1, sizeof(flush_events_1));
  TF_EXPECT_OK(status);
  EXPECT_FALSE(flush_events_1.empty());

  {
    TraceMe traceme("streaming_event_2");
  }

  tsl::profiler::TraceMeRecorder::Events flush_events_2;
  status = tracer->Consume(&flush_events_2, sizeof(flush_events_2));
  TF_EXPECT_OK(status);
  EXPECT_FALSE(flush_events_2.empty());

  TF_ASSERT_OK(tracer->Stop());

  // Verify the trailing remainder
  tsl::profiler::TraceMeRecorder::Events flush_events_3;
  status = tracer->Consume(&flush_events_3, sizeof(flush_events_3));
  TF_EXPECT_OK(status);

  tensorflow::profiler::XSpace space1;
  TF_ASSERT_OK(
      tracer->Serialize(&flush_events_1, sizeof(flush_events_1), &space1, 0));
  tensorflow::profiler::XSpace space2;
  TF_ASSERT_OK(
      tracer->Serialize(&flush_events_2, sizeof(flush_events_2), &space2, 0));

  ASSERT_EQ(space1.planes_size(), 1);
  ASSERT_EQ(space2.planes_size(), 1);

  bool trace1_found = false;
  XPlaneVisitor xplane1(&space1.planes(0));
  xplane1.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      if (event.Name() == "streaming_event_1") {
        trace1_found = true;
      }
    });
  });
  EXPECT_TRUE(trace1_found);

  bool trace2_found = false;
  XPlaneVisitor xplane2(&space2.planes(0));
  xplane2.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      if (event.Name() == "streaming_event_2") {
        trace2_found = true;
      }
    });
  });
  EXPECT_TRUE(trace2_found);
}

TEST(HostTracerTest, ConsumeFailsWithNullPtr) {
  auto tracer = internal::CreateHostTracerImpl({});
  TF_ASSERT_OK(tracer->Start());
  TF_ASSERT_OK(tracer->Stop());

  absl::Status status = tracer->Consume(nullptr, 0);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(HostTracerTest, SerializeFailsWithNullPtrs) {
  auto tracer = internal::CreateHostTracerImpl({});
  tsl::profiler::TraceMeRecorder::Events temp_events;
  tensorflow::profiler::XSpace space;

  absl::Status status1 = tracer->Serialize(nullptr, 0, &space, 0);
  EXPECT_FALSE(status1.ok());
  EXPECT_EQ(status1.code(), absl::StatusCode::kInvalidArgument);

  absl::Status status2 =
      tracer->Serialize(&temp_events, sizeof(temp_events), nullptr, 0);
  EXPECT_FALSE(status2.ok());
  EXPECT_EQ(status2.code(), absl::StatusCode::kInvalidArgument);
}

TEST(HostTracerTest, SerializeOkWithEmptyEvents) {
  auto tracer = internal::CreateHostTracerImpl({});
  tsl::profiler::TraceMeRecorder::Events temp_events;
  tensorflow::profiler::XSpace space;

  absl::Status status =
      tracer->Serialize(&temp_events, sizeof(temp_events), &space, 0);
  TF_EXPECT_OK(status);
  EXPECT_EQ(space.planes_size(),
            0);  // Should not create planes for empty events
}

TEST(HostTracerTest, ConsumeAndSerializeHappyPath) {
  auto tracer = internal::CreateHostTracerImpl({});
  TF_ASSERT_OK(tracer->Start());
  {
    TraceMe traceme("consume_and_serialize_test");
  }
  TF_ASSERT_OK(tracer->Stop());

  tsl::profiler::TraceMeRecorder::Events temp_events;
  TF_ASSERT_OK(tracer->Consume(&temp_events, sizeof(temp_events)));

  // Data has been pulled out, should not be empty.
  EXPECT_FALSE(temp_events.empty());

  tensorflow::profiler::XSpace space;
  TF_ASSERT_OK(tracer->Serialize(&temp_events, sizeof(temp_events), &space, 0));

  // Verify the serialized output space is identical to the one-shot extraction
  ASSERT_EQ(space.planes_size(), 1);
  const auto& plane = space.planes(0);
  XPlaneVisitor xplane(&plane);
  ASSERT_EQ(plane.name(), ::tsl::profiler::kHostThreadsPlaneName);

  bool trace_found = false;
  xplane.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      if (event.Name() == "consume_and_serialize_test") {
        trace_found = true;
      }
    });
  });
  EXPECT_TRUE(trace_found);

  // input_events should be cleared after successful serialization
  EXPECT_TRUE(temp_events.empty());
}

}  // namespace
}  // namespace profiler
}  // namespace xla
