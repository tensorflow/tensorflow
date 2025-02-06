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
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/optional.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/platform/types.h"
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

}  // namespace
}  // namespace profiler
}  // namespace xla
