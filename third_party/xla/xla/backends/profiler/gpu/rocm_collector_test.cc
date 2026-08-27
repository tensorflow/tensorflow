/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/backends/profiler/gpu/rocm_collector.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace test {

using tsl::profiler::FindOrAddMutablePlaneWithName;
using tsl::profiler::XSpace;

TEST(RocmCollectorTest, TestAddKernelEventAndExport) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 100;
  options.max_activity_api_events = 100;
  options.max_annotation_strings = 100;
  options.num_gpus = 1;

  constexpr uint64_t kStartWallTimeNs = 1000;
  constexpr uint64_t kStartGpuTimeNs = 2000;

  RocmTraceCollectorImpl collector(options, kStartWallTimeNs, kStartGpuTimeNs);

  constexpr uint32_t kCorrelationId = 42;
  constexpr uint64_t kStartTimeNs = 3000;
  constexpr uint64_t kEndTimeNs = 4000;

  // === 1. Add API Callback Event ===
  RocmTracerEvent api_event;
  api_event.type = RocmTracerEventType::Kernel;
  api_event.source = RocmTracerEventSource::ApiCallback;
  api_event.domain = RocmTracerEventDomain::HIP_API;
  api_event.name = "test_rocm_kernel";
  api_event.correlation_id = kCorrelationId;
  api_event.thread_id = 999;
  api_event.kernel_info = KernelDetails{};
  api_event.kernel_info.private_segment_size = 32;
  api_event.kernel_info.group_segment_size = 1024;
  api_event.kernel_info.workgroup_x = 256;
  api_event.kernel_info.workgroup_y = 1;
  api_event.kernel_info.workgroup_z = 1;
  api_event.kernel_info.grid_x = 100;
  api_event.kernel_info.grid_y = 1;
  api_event.kernel_info.grid_z = 1;
  api_event.kernel_info.func_ptr = reinterpret_cast<void*>(0xdeadbeef);

  collector.AddEvent(std::move(api_event), /*is_auxiliary=*/false);

  // === 2. Add Activity Event ===
  RocmTracerEvent activity_event;
  activity_event.type = RocmTracerEventType::Kernel;
  activity_event.source = RocmTracerEventSource::Activity;
  activity_event.domain = RocmTracerEventDomain::HIP_OPS;
  activity_event.name = "test_rocm_kernel";
  activity_event.correlation_id = kCorrelationId;
  activity_event.start_time_ns = kStartTimeNs;
  activity_event.end_time_ns = kEndTimeNs;
  activity_event.device_id = 100;
  activity_event.stream_id = 123;

  collector.AddEvent(std::move(activity_event), /*is_auxiliary=*/false);

  // === 3. Finalize and Export ===
  collector.Flush();

  tensorflow::profiler::XSpace space;
  collector.Export(&space);

  // === 4. Check results ===
  ASSERT_GE(space.planes_size(), 1);
  const auto* gpu_plane =
      FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu_plane, nullptr);

  ASSERT_GT(gpu_plane->lines_size(), 0);
  const auto& line = gpu_plane->lines(0);
  ASSERT_GT(line.events_size(), 0);

  const auto& event = line.events(0);
  EXPECT_EQ(event.offset_ps(), (kStartTimeNs - kStartGpuTimeNs) * 1000);
  EXPECT_EQ(event.duration_ps(), (kEndTimeNs - kStartTimeNs) * 1000);
  EXPECT_EQ(gpu_plane->event_metadata().at(event.metadata_id()).name(),
            "test_rocm_kernel");
}

// Regression test for the .front()-only iteration bug in
// ApiActivityInfoExchange. When N activity events share one
// correlation_id (the rocprofiler-sdk pattern for hipGraphLaunch-replayed
// kernels), all N must reach the exported XPlane, not just the first.
TEST(RocmCollectorTest, MultipleActivitiesPerCorrelationIdAllExported) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 100;
  options.max_activity_api_events = 100;
  options.max_annotation_strings = 100;
  options.num_gpus = 1;

  constexpr uint64_t kStartWallTimeNs = 1000;
  constexpr uint64_t kStartGpuTimeNs = 2000;
  RocmTraceCollectorImpl collector(options, kStartWallTimeNs, kStartGpuTimeNs);

  // Single correlation_id shared by all events -- mirrors a hipGraphLaunch
  // that replays a captured graph: one API call, many kernel-dispatch
  // records emitted by rocprofiler-sdk under the same correlation_id.
  constexpr uint32_t kCorrelationId = 7;
  constexpr uint32_t kDeviceId = 100;
  constexpr uint64_t kStreamId = 123;

  RocmTracerEvent api_event;
  api_event.type = RocmTracerEventType::Kernel;
  api_event.source = RocmTracerEventSource::ApiCallback;
  api_event.domain = RocmTracerEventDomain::HIP_API;
  api_event.name = "hipGraphLaunch";
  api_event.correlation_id = kCorrelationId;
  api_event.thread_id = 999;
  api_event.kernel_info = KernelDetails{};
  api_event.kernel_info.func_ptr = reinterpret_cast<void*>(0xdeadbeef);
  collector.AddEvent(std::move(api_event), /*is_auxiliary=*/false);

  // Three GPU activity records, same correlation_id, same stream (so
  // they land on the same XLine), distinct names and timestamps.
  struct ActivityShape {
    const char* name;
    uint64_t start_ns;
    uint64_t end_ns;
  };
  constexpr ActivityShape kActivities[] = {
      {"kernel_a", 3000, 3500},
      {"kernel_b", 3500, 4000},
      {"kernel_c", 4000, 4500},
  };
  for (const auto& shape : kActivities) {
    RocmTracerEvent activity;
    activity.type = RocmTracerEventType::Kernel;
    activity.source = RocmTracerEventSource::Activity;
    activity.domain = RocmTracerEventDomain::HIP_OPS;
    activity.name = shape.name;
    activity.correlation_id = kCorrelationId;
    activity.start_time_ns = shape.start_ns;
    activity.end_time_ns = shape.end_ns;
    activity.device_id = kDeviceId;
    activity.stream_id = kStreamId;
    collector.AddEvent(std::move(activity), /*is_auxiliary=*/false);
  }

  collector.Flush();
  tensorflow::profiler::XSpace space;
  collector.Export(&space);

  const auto* gpu_plane =
      FindOrAddMutablePlaneWithName(&space, "/device:GPU:0");
  ASSERT_NE(gpu_plane, nullptr);

  // Pre-fix (.front()-only) would emit just one event here. The fix
  // iterates the entire vector, so all three activity records must
  // appear on the stream line. Dense stream remapping converts the raw
  // stream_id (123) to a sequential index (0), so we look for events on
  // any device line rather than matching a specific line ID.
  size_t total_kernel_events = 0;
  absl::flat_hash_set<std::string> seen_names;
  for (const auto& line : gpu_plane->lines()) {
    total_kernel_events += line.events_size();
    for (const auto& ev : line.events()) {
      seen_names.insert(
          gpu_plane->event_metadata().at(ev.metadata_id()).name());
    }
  }

  EXPECT_EQ(total_kernel_events, 3u)
      << "Expected all 3 activity records to be emitted under the same "
         "correlation_id; got "
      << total_kernel_events
      << " (this is the "
         "regression the .front()-only iteration introduced).";
  EXPECT_TRUE(seen_names.contains("kernel_a"));
  EXPECT_TRUE(seen_names.contains("kernel_b"));
  EXPECT_TRUE(seen_names.contains("kernel_c"));
}

// ---------------------------------------------------------------------------
// ROCTX marker (Generic event) handling.
//
// These live here rather than in rocm_tracer_test.cc deliberately: they drive
// RocmTraceCollectorImpl directly, with no rocprofiler context and no tracer
// singleton, so they exercise the collector contract on any host.
// ---------------------------------------------------------------------------

namespace {

RocmTracerEvent MakeMarkerEvent(absl::string_view label, uint64_t tid,
                                uint64_t start_ns, uint64_t end_ns) {
  RocmTracerEvent e;
  e.type = RocmTracerEventType::Generic;
  // ApiCallback is what routes this to a host line keyed on thread_id.
  e.source = RocmTracerEventSource::ApiCallback;
  e.domain = RocmTracerEventDomain::InvalidDomain;
  e.name = std::string(label);
  e.start_time_ns = start_ns;
  e.end_time_ns = end_ns;
  e.thread_id = tid;
  e.device_id = RocmTracerEvent::kInvalidDeviceId;
  e.correlation_id = RocmTracerEvent::kInvalidCorrelationId;
  e.stream_id = RocmTracerEvent::kInvalidStreamId;
  return e;
}

// Counts drops so the cap can be asserted on rather than inferred.
class DropCountingCollector : public RocmTraceCollectorImpl {
 public:
  using RocmTraceCollectorImpl::RocmTraceCollectorImpl;
  void OnEventsDropped(const std::string& reason, uint64_t id) override {
    ++drops_;
  }
  int drops() const { return drops_; }

 private:
  int drops_ = 0;
};

}  // namespace

// Marker events are routed to standalone_events_ by an early return that sits
// above the source-based branching, so they do not inherit the cap enforced
// there. Without an explicit guard an application emitting markers in a hot
// loop grows standalone_events_ without bound for the whole session -- the
// buffer is only drained at Flush(). This is the regression guard for that.
TEST(RocmCollectorTest, MarkerEventsRespectMaxCallbackApiEvents) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 8;
  options.max_activity_api_events = 100;
  options.max_annotation_strings = 100;
  options.num_gpus = 1;

  DropCountingCollector collector(options, /*start_walltime_ns=*/1000,
                                  /*start_gputime_ns=*/2000);

  constexpr int kEmitted = 25;
  for (int i = 0; i < kEmitted; ++i) {
    collector.AddEvent(MakeMarkerEvent(absl::StrCat("marker_", i), /*tid=*/7,
                                       3000 + i, 3100 + i),
                       /*is_auxiliary=*/false);
  }

  EXPECT_EQ(collector.drops(), kEmitted - options.max_callback_api_events)
      << "every marker past the cap must be reported through OnEventsDropped, "
         "not silently retained";

  collector.Flush();
  XSpace space;
  collector.Export(&space);

  int marker_events = 0;
  for (const auto& plane : space.planes()) {
    for (const auto& line : plane.lines()) {
      if (absl::EndsWith(line.name(), "/ROCTX")) {
        marker_events += line.events_size();
      }
    }
  }
  EXPECT_EQ(marker_events, static_cast<int>(options.max_callback_api_events))
      << "the cap must bound what is retained, not just what is reported";
}

// Flush() buckets standalone events into per_device_collector_[0], but
// Export() only iterates [0, num_gpus_). With no GPUs that slot is created and
// never exported, so the events would be silently lost; drop them explicitly
// instead, and do not crash.
TEST(RocmCollectorTest, MarkerEventsDroppedWhenNoGpusReported) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 100;
  options.max_activity_api_events = 100;
  options.max_annotation_strings = 100;
  options.num_gpus = 0;

  RocmTraceCollectorImpl collector(options, /*start_walltime_ns=*/1000,
                                   /*start_gputime_ns=*/2000);
  collector.AddEvent(MakeMarkerEvent("orphan", /*tid=*/11, 3000, 3100),
                     /*is_auxiliary=*/false);
  collector.Flush();

  XSpace space;
  collector.Export(&space);  // must not crash

  for (const auto& plane : space.planes()) {
    for (const auto& line : plane.lines()) {
      EXPECT_FALSE(absl::EndsWith(line.name(), "/ROCTX"))
          << "no device plane exists to carry marker events";
    }
  }
}

// The routing contract, stated once without a tracer singleton: a marker and a
// kernel-launch API event on the SAME thread must land on different lines --
// "Host Threads/<tid>/ROCTX" and "Host Threads/<tid>" -- so marker bands sort
// directly beneath their owning thread after the merge into /host:CPU.
TEST(RocmCollectorTest, MarkerAndApiEventsOnSameThreadGetSeparateLines) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 100;
  options.max_activity_api_events = 100;
  options.max_annotation_strings = 100;
  options.num_gpus = 1;

  RocmTraceCollectorImpl collector(options, /*start_walltime_ns=*/1000,
                                   /*start_gputime_ns=*/2000);

  constexpr uint64_t kTid = 4242;
  collector.AddEvent(MakeMarkerEvent("my_range", kTid, 3000, 4000),
                     /*is_auxiliary=*/false);

  // The API event needs its Activity counterpart: ApiActivityInfoExchange
  // drops any ApiCallback event whose correlation_id has no activity record
  // (rocm_collector.cc, "could not find activity counterpart"). Markers are
  // exempt because they never enter that exchange -- which is the asymmetry
  // this test is here to pin down.
  constexpr uint32_t kCorrelationId = 55;

  RocmTracerEvent api_event;
  api_event.type = RocmTracerEventType::Kernel;
  api_event.source = RocmTracerEventSource::ApiCallback;
  api_event.domain = RocmTracerEventDomain::HIP_API;
  api_event.name = "some_kernel_launch";
  api_event.correlation_id = kCorrelationId;
  api_event.thread_id = kTid;
  api_event.device_id = 0;
  api_event.start_time_ns = 3100;
  api_event.end_time_ns = 3200;
  api_event.kernel_info = KernelDetails{};
  collector.AddEvent(std::move(api_event), /*is_auxiliary=*/false);

  RocmTracerEvent activity_event;
  activity_event.type = RocmTracerEventType::Kernel;
  activity_event.source = RocmTracerEventSource::Activity;
  activity_event.domain = RocmTracerEventDomain::HIP_OPS;
  activity_event.name = "some_kernel_launch";
  activity_event.correlation_id = kCorrelationId;
  activity_event.thread_id = kTid;
  activity_event.device_id = 0;
  activity_event.stream_id = 1;
  activity_event.start_time_ns = 3150;
  activity_event.end_time_ns = 3250;
  activity_event.kernel_info = KernelDetails{};
  collector.AddEvent(std::move(activity_event), /*is_auxiliary=*/false);

  collector.Flush();
  XSpace space;
  collector.Export(&space);

  bool found_marker_line = false;
  bool found_plain_host_line = false;
  for (const auto& plane : space.planes()) {
    for (const auto& line : plane.lines()) {
      if (line.name() == absl::StrCat("Host Threads/", kTid, "/ROCTX")) {
        found_marker_line = true;
        EXPECT_EQ(line.events_size(), 1);
      } else if (line.name() == absl::StrCat("Host Threads/", kTid)) {
        found_plain_host_line = true;
      }
    }
  }
  EXPECT_TRUE(found_marker_line) << "marker must get its own /ROCTX line";
  EXPECT_TRUE(found_plain_host_line)
      << "the API event must stay on the plain host-thread line";
}

}  // namespace test
}  // namespace profiler
}  // namespace xla
