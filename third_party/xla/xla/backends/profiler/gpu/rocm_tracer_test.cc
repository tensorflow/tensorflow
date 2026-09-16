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

#include "xla/backends/profiler/gpu/rocm_tracer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/rocprofiler-sdk-roctx/roctx.h"
#include "rocm/include/rocprofiler-sdk/callback_tracing.h"
#include "rocm/include/rocprofiler-sdk/context.h"
#include "rocm/include/rocprofiler-sdk/fwd.h"
#include "rocm/include/rocprofiler-sdk/marker.h"
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env_time.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

using tsl::profiler::XSpace;

// Minimal mock collector implementation based on RocmTraceCollectorImpl.
class TestRocmTraceCollector : public RocmTraceCollectorImpl {
 public:
  TestRocmTraceCollector(const RocmTraceCollectorOptions& options,
                         uint64_t start_walltime_ns, uint64_t start_gputime_ns)
      : RocmTraceCollectorImpl(options, start_walltime_ns, start_gputime_ns) {}

  void Export(XSpace* space) override {
    exported_ = true;
    exported_space_ = space;
  }

  void OnEventsDropped(const std::string& reason,
                       uint64_t correlation_id) override {
    dropped_reason_ = reason;
    dropped_id_ = correlation_id;
  }

  bool exported() const { return exported_; }
  const std::string& dropped_reason() const { return dropped_reason_; }
  uint64_t dropped_id() const { return dropped_id_; }
  XSpace* exported_space() const { return exported_space_; }

 private:
  bool exported_ = false;
  std::string dropped_reason_;
  uint64_t dropped_id_ = 0;
  XSpace* exported_space_ = nullptr;
};

// Utility to create valid options for the test collector.
std::unique_ptr<TestRocmTraceCollector> CreateTestCollector() {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 2 * 1024 * 1024;
  options.max_activity_api_events = 2 * 1024 * 1024;
  options.max_annotation_strings = 1024 * 1024;
  options.num_gpus = 1;

  uint64_t walltime_ns = RocmTracer::GetTimestamp();
  uint64_t gputime_ns = RocmTracer::GetTimestamp();

  return std::make_unique<TestRocmTraceCollector>(options, walltime_ns,
                                                  gputime_ns);
}

TEST(RocmTracerTest, SingletonInstance) {
  LOG(INFO) << "Before RocmTracer::GetRocmTracerSingleton()";
  RocmTracer& tracer1 = RocmTracer::GetRocmTracerSingleton();
  RocmTracer& tracer2 = RocmTracer::GetRocmTracerSingleton();
  LOG(INFO) << "Before RocmTracer::GetRocmTracerSingleton()";
  EXPECT_EQ(&tracer1, &tracer2) << "RocmTracer must be a singleton";
}

TEST(RocmTracerTest, GpuAgentDataMatchesHipDeviceProperties) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  const auto& agents = tracer.GpuAgents();
  ASSERT_GT(agents.size(), 0u);

  hipDeviceProp_t props;
  ASSERT_EQ(hipGetDeviceProperties(&props, 0), hipSuccess);
  const auto& agent = agents[0];

  EXPECT_EQ(agent.cu_count, static_cast<uint32_t>(props.multiProcessorCount));
  // Agent clocks are in MHz, hipDeviceProp_t clocks are in KHz.
  EXPECT_EQ(static_cast<uint64_t>(agent.max_engine_clk_fcompute) * 1000,
            static_cast<uint64_t>(props.clockRate));

  auto gfx_major = (agent.gfx_target_version / 10000) % 100;
  auto gfx_minor = (agent.gfx_target_version / 100) % 100;
  EXPECT_EQ(gfx_major, static_cast<uint32_t>(props.major));
  EXPECT_EQ(gfx_minor, static_cast<uint32_t>(props.minor));

  uint64_t vram_total = 0;
  uint32_t vram_clock_mhz = 0;
  uint32_t vram_bus_width = 0;
  for (uint32_t i = 0; i < agent.mem_banks_count; ++i) {
    if (agent.mem_banks[i].heap_type == HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC ||
        agent.mem_banks[i].heap_type == HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE) {
      vram_total += agent.mem_banks[i].size_in_bytes;
      if (vram_clock_mhz == 0) {
        vram_clock_mhz = agent.mem_banks[i].mem_clk_max;
        vram_bus_width = agent.mem_banks[i].width;
      }
    }
  }
  EXPECT_EQ(vram_total, props.totalGlobalMem);
  EXPECT_EQ(static_cast<uint64_t>(vram_clock_mhz) * 1000,
            static_cast<uint64_t>(props.memoryClockRate));
  EXPECT_EQ(vram_bus_width, static_cast<uint32_t>(props.memoryBusWidth));
}

TEST(RocmTracerTest, InitialStateIsAvailable) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  EXPECT_TRUE(tracer.IsAvailable())
      << "Tracer should be available before Enable()";
}

TEST(RocmTracerTest, EnableAndDisableLifecycle) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  auto collector = CreateTestCollector();

  RocmTracerOptions tracer_options{/*max_annotation_strings=*/128};
  TF_ASSERT_OK(tracer.Enable(tracer_options, collector.get()));

  EXPECT_FALSE(tracer.IsAvailable())
      << "Tracer should not be available after Enable()";
  EXPECT_EQ(tracer.collector(), collector.get())
      << "Collector should be set after Enable()";
  ASSERT_NE(tracer.annotation_map(), nullptr)
      << "Annotation map should be initialized";

  tracer.Disable();

  EXPECT_TRUE(tracer.IsAvailable())
      << "Tracer should be available after Disable()";
}

TEST(RocmTracerTest, AnnotationMapWorks) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  AnnotationMap* map = tracer.annotation_map();
  ASSERT_NE(map, nullptr);

  uint64_t id = 42;
  std::string annotation = "matmul_fused_op";
  map->Add(id, annotation);

  absl::string_view result = map->LookUp(id);
  EXPECT_EQ(result, annotation);
}

TEST(RocmTracerTest, AnnotationMapClear) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  AnnotationMap* map = tracer.annotation_map();
  ASSERT_NE(map, nullptr);

  map->Add(100, "op_a");
  map->Add(101, "op_b");
  EXPECT_EQ(map->LookUp(100), "op_a");
  EXPECT_EQ(map->LookUp(101), "op_b");

  map->Clear();

  EXPECT_TRUE(map->LookUp(100).empty());
  EXPECT_TRUE(map->LookUp(101).empty());
}

// Simple collector that tracks received events for verification.
class EventCapturingCollector : public RocmTraceCollector {
 public:
  EventCapturingCollector() : RocmTraceCollector(MakeCollectorOptions()) {}

  void AddEvent(RocmTracerEvent&& event, bool is_auxiliary) override {
    event_count_++;
  }

  void OnEventsDropped(const std::string& reason,
                       uint64_t num_events) override {}
  void Flush() override {}
  void Export(tsl::profiler::XSpace* space) override {}

  int event_count() const { return event_count_; }

 private:
  static RocmTraceCollectorOptions MakeCollectorOptions() {
    RocmTraceCollectorOptions options;
    options.max_callback_api_events = 2 * 1024 * 1024;
    options.max_activity_api_events = 2 * 1024 * 1024;
    options.max_annotation_strings = 1024 * 1024;
    options.num_gpus = RocmTracer::GetRocmTracerSingleton().NumGpus();
    return options;
  }
  int event_count_ = 0;
};

std::unique_ptr<EventCapturingCollector> CreateEventCapturingCollector() {
  return std::make_unique<EventCapturingCollector>();
}

TEST(RocmTracerTest, CapturesHipEvents) {
#define HIP_ASSERT_OK(expr) ASSERT_EQ((expr), hipSuccess) << #expr " failed"

  int device_count = 0;
  HIP_ASSERT_OK(hipGetDeviceCount(&device_count));
  ASSERT_GT(device_count, 0) << "No HIP devices available";

  auto collector = CreateEventCapturingCollector();
  EventCapturingCollector* collector_ptr = collector.get();

  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  RocmTracerOptions tracer_options{/*max_annotation_strings=*/1024 * 1024};
  TF_ASSERT_OK(tracer.Enable(tracer_options, collector.get()));

  constexpr size_t kNumFloats = 1024;
  constexpr size_t kSize = kNumFloats * sizeof(float);
  std::vector<float> host_data(kNumFloats, 1.0f);
  void* device_data = nullptr;

  HIP_ASSERT_OK(hipMalloc(&device_data, kSize));
  HIP_ASSERT_OK(
      hipMemcpy(device_data, host_data.data(), kSize, hipMemcpyHostToDevice));
  HIP_ASSERT_OK(
      hipMemcpy(host_data.data(), device_data, kSize, hipMemcpyDeviceToHost));
  HIP_ASSERT_OK(hipDeviceSynchronize());

  tracer.Disable();
  HIP_ASSERT_OK(hipFree(device_data));

#undef HIP_ASSERT_OK

  EXPECT_GT(collector_ptr->event_count(), 0)
      << "Expected to capture at least one trace event";
}

// Regression guards: Disable() must stop the rocprofiler context it started
// in Enable(). Otherwise the buffer keeps collecting events between sessions
// and the next Enable()'s collector receives stale events.

TEST(RocmTracerTest, DisableStopsRocprofilerContext) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  ASSERT_TRUE(tracer.IsAvailable());

  auto collector = CreateTestCollector();
  RocmTracerOptions tracer_options{/*max_annotation_strings=*/128};
  TF_ASSERT_OK(tracer.Enable(tracer_options, collector.get()));

  int active = -1;
  ASSERT_EQ(rocprofiler_context_is_active(tracer.context_, &active),
            ROCPROFILER_STATUS_SUCCESS);
  EXPECT_NE(active, 0) << "Context should be active after Enable()";

  tracer.Disable();

  active = -1;
  ASSERT_EQ(rocprofiler_context_is_active(tracer.context_, &active),
            ROCPROFILER_STATUS_SUCCESS);
  EXPECT_EQ(active, 0)
      << "Disable() should call rocprofiler_stop_context(context_)";
}

TEST(RocmTracerTest, DisableIsolatesNextSession) {
  int device_count = 0;
  ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
  ASSERT_GT(device_count, 0) << "No HIP devices available";

  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  ASSERT_TRUE(tracer.IsAvailable());

  RocmTracerOptions tracer_options{/*max_annotation_strings=*/1024 * 1024};
  constexpr size_t kNumFloats = 1024;
  constexpr size_t kSize = kNumFloats * sizeof(float);
  std::vector<float> host_data(kNumFloats, 1.0f);
  void* device_data = nullptr;
  ASSERT_EQ(hipMalloc(&device_data, kSize), hipSuccess);

  // Session 1: minimal Enable -> Disable to put the rocprofiler context
  // into the post-Disable state. The 100 ms sleep before Disable lets the
  // async HIP_OPS activity record land in the buffer in time for the flush.
  auto collector1 = CreateEventCapturingCollector();
  TF_ASSERT_OK(tracer.Enable(tracer_options, collector1.get()));
  ASSERT_EQ(
      hipMemcpy(device_data, host_data.data(), kSize, hipMemcpyHostToDevice),
      hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  absl::SleepFor(absl::Milliseconds(100));
  tracer.Disable();
  ASSERT_GT(collector1->event_count(), 0)
      << "Sanity: profiler should capture events during a normal session";

  // No profiler. If Disable() stopped the context correctly, these HIP calls
  // must not be recorded into the rocprofiler-owned buffer.
  constexpr int kLeakedPairs = 50;
  for (int i = 0; i < kLeakedPairs; ++i) {
    ASSERT_EQ(
        hipMemcpy(device_data, host_data.data(), kSize, hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpy(host_data.data(), device_data, kSize, hipMemcpyDeviceToHost),
        hipSuccess);
  }
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  // Session 2: Enable -> Disable with no user HIP activity. With the fix
  // the buffer is empty here so collector2 receives zero events; with the
  // bug, leaked-window events drain into collector2.
  auto collector2 = CreateEventCapturingCollector();
  TF_ASSERT_OK(tracer.Enable(tracer_options, collector2.get()));
  tracer.Disable();

  ASSERT_EQ(hipFree(device_data), hipSuccess);

  EXPECT_EQ(collector2->event_count(), 0)
      << "Session 2 captured " << collector2->event_count()
      << " events despite no HIP activity between its Enable() and Disable();"
      << " these must have leaked from the preceding no-profiler window of "
      << kLeakedPairs << " hipMemcpy pairs";
}

// MarkerCallback unit tests — exercise MarkerCallback() directly without
// requiring real ROCTX API calls, using a capturing collector.
// ============================================================================

// Collector variant that captures the full RocmTracerEvent for inspection.
class MarkerCapturingCollector : public RocmTraceCollector {
 public:
  MarkerCapturingCollector() : RocmTraceCollector(MakeCollectorOptions()) {}

  void AddEvent(RocmTracerEvent&& event, bool) override {
    absl::MutexLock lock(&mu_);
    events_.push_back(std::move(event));
  }
  void OnEventsDropped(const std::string&, uint64_t) override {}
  void Flush() override {}
  void Export(tsl::profiler::XSpace*) override {}

  std::vector<RocmTracerEvent> TakeEvents() {
    absl::MutexLock lock(&mu_);
    return std::exchange(events_, {});
  }

 private:
  static RocmTraceCollectorOptions MakeCollectorOptions() {
    RocmTraceCollectorOptions o;
    o.max_callback_api_events = 1024;
    o.max_activity_api_events = 1024;
    o.max_annotation_strings = 1024;
    o.num_gpus = 1;
    return o;
  }
  absl::Mutex mu_;
  std::vector<RocmTracerEvent> events_ ABSL_GUARDED_BY(mu_);
};

// Build a minimal rocprofiler_callback_tracing_record_t for MARKER_CORE_API.
// `payload` must point to a live rocprofiler_callback_tracing_marker_api_data_t
// for the duration of the MarkerCallback call.
static rocprofiler_callback_tracing_record_t MakeMarkerRecord(
    rocprofiler_marker_core_api_id_t op, rocprofiler_callback_phase_t phase,
    uint64_t thread_id, void* payload) {
  rocprofiler_callback_tracing_record_t rec{};
  rec.kind = ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API;
  rec.operation = static_cast<rocprofiler_tracing_operation_t>(op);
  rec.phase = phase;
  rec.thread_id = thread_id;
  rec.correlation_id.internal = 99;
  rec.payload = payload;
  return rec;
}

TEST(RocmTracerTest, MarkerCallbackPushPopEmitsRoctxRange) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  ASSERT_TRUE(tracer.IsAvailable());

  auto collector = std::make_unique<MarkerCapturingCollector>();
  MarkerCapturingCollector* cptr = collector.get();

  RocmTracerOptions opts{/*max_annotation_strings=*/1024};
  TF_ASSERT_OK(tracer.Enable(opts, cptr));

  const uint64_t tid = 12345;
  const char* label = "my_roctx_range";

  // Simulate roctxRangePushA ENTER
  rocprofiler_callback_tracing_marker_api_data_t push_data{};
  push_data.args.roctxRangePushA.message = label;
  auto push_rec =
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA,
                       ROCPROFILER_CALLBACK_PHASE_ENTER, tid, &push_data);
  tracer.MarkerCallback(push_rec);

  // No event yet — PUSH doesn't emit
  EXPECT_TRUE(cptr->TakeEvents().empty())
      << "roctxRangePushA must not emit an event until the matching Pop";

  // Simulate roctxRangePop EXIT
  rocprofiler_callback_tracing_marker_api_data_t pop_data{};
  auto pop_rec =
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop,
                       ROCPROFILER_CALLBACK_PHASE_EXIT, tid, &pop_data);
  tracer.MarkerCallback(pop_rec);

  tracer.Disable();

  auto events = cptr->TakeEvents();
  ASSERT_EQ(events.size(), 1u)
      << "Expected exactly one range event from Push+Pop";

  const RocmTracerEvent& e = events[0];
  EXPECT_EQ(e.type, RocmTracerEventType::Generic);
  // ApiCallback is what makes PerDeviceCollector place this on a per-thread
  // line rather than a device stream line.
  EXPECT_EQ(e.source, RocmTracerEventSource::ApiCallback);
  // Markers own their label in `name`; `roctx_range` is reserved for
  // kernel/HIP-API events, where it views AnnotationMap's pool.
  EXPECT_EQ(e.name, label);
  EXPECT_TRUE(e.roctx_range.empty())
      << "marker events must not carry a roctx_range view";
  EXPECT_EQ(e.thread_id, tid);
  // GE, not GT: both timestamps come from rocprofiler_get_timestamp and a
  // zero-length range is legal, so strict inequality is a flake tail.
  EXPECT_GE(e.end_time_ns, e.start_time_ns);
}

TEST(RocmTracerTest, MarkerCallbackMarkEmitsInstantaneousEvent) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  ASSERT_TRUE(tracer.IsAvailable());

  auto collector = std::make_unique<MarkerCapturingCollector>();
  MarkerCapturingCollector* cptr = collector.get();

  RocmTracerOptions opts{/*max_annotation_strings=*/1024};
  TF_ASSERT_OK(tracer.Enable(opts, cptr));

  const uint64_t tid = 77777;
  const char* label = "checkpoint";

  rocprofiler_callback_tracing_marker_api_data_t mark_data{};
  mark_data.args.roctxMarkA.message = label;
  auto mark_rec =
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA,
                       ROCPROFILER_CALLBACK_PHASE_ENTER, tid, &mark_data);
  tracer.MarkerCallback(mark_rec);

  tracer.Disable();

  auto events = cptr->TakeEvents();
  ASSERT_EQ(events.size(), 1u) << "roctxMarkA must emit exactly one event";

  const RocmTracerEvent& e = events[0];
  EXPECT_EQ(e.type, RocmTracerEventType::Generic);
  EXPECT_EQ(e.name, label);
  EXPECT_TRUE(e.roctx_range.empty());
  EXPECT_EQ(e.thread_id, tid);
  EXPECT_EQ(e.start_time_ns, e.end_time_ns)
      << "roctxMarkA produces an instantaneous event (start == end)";
}

TEST(RocmTracerTest, MarkerCallbackUnmatchedPopIsIgnored) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  ASSERT_TRUE(tracer.IsAvailable());

  auto collector = std::make_unique<MarkerCapturingCollector>();
  MarkerCapturingCollector* cptr = collector.get();

  RocmTracerOptions opts{/*max_annotation_strings=*/1024};
  TF_ASSERT_OK(tracer.Enable(opts, cptr));

  // Pop without any preceding Push — must not crash, must not emit any event.
  rocprofiler_callback_tracing_marker_api_data_t pop_data{};
  auto pop_rec =
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop,
                       ROCPROFILER_CALLBACK_PHASE_EXIT, 1111, &pop_data);
  tracer.MarkerCallback(pop_rec);

  tracer.Disable();

  EXPECT_TRUE(cptr->TakeEvents().empty())
      << "An unmatched roctxRangePop must not emit an event";
}

TEST(RocmTracerTest, MarkerCallbackNullLabelRangeIsDroppedNotEmitted) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  ASSERT_TRUE(tracer.IsAvailable());

  auto collector = std::make_unique<MarkerCapturingCollector>();
  MarkerCapturingCollector* cptr = collector.get();

  RocmTracerOptions opts{/*max_annotation_strings=*/1024};
  TF_ASSERT_OK(tracer.Enable(opts, cptr));

  const uint64_t tid = 2222;

  // Push with a null message — must not crash, and must still push so the
  // matching pop consumes THIS frame rather than an enclosing one.
  rocprofiler_callback_tracing_marker_api_data_t push_data{};
  push_data.args.roctxRangePushA.message = nullptr;
  auto push_rec =
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA,
                       ROCPROFILER_CALLBACK_PHASE_ENTER, tid, &push_data);
  tracer.MarkerCallback(push_rec);

  rocprofiler_callback_tracing_marker_api_data_t pop_data{};
  auto pop_rec =
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop,
                       ROCPROFILER_CALLBACK_PHASE_EXIT, tid, &pop_data);
  tracer.MarkerCallback(pop_rec);

  tracer.Disable();

  // An unlabelled range carries no information and renders in the trace
  // viewer as an anonymous "Generic" band (CreateXEvent falls back to the
  // event-type name when `name` is empty). Drop it, matching how roctxMarkA
  // already treats a null/empty message.
  EXPECT_THAT(cptr->TakeEvents(), ::testing::IsEmpty())
      << "A range with no label must be dropped, not emitted as \"Generic\"";
}

// The counterpart to the above: dropping the unlabelled range must NOT
// desynchronise the thread's stack. If the null push were skipped entirely,
// this pop would consume "outer" and report it with the inner end time.
TEST(RocmTracerTest, MarkerCallbackNullLabelRangeKeepsStackBalanced) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  ASSERT_TRUE(tracer.IsAvailable());

  auto collector = std::make_unique<MarkerCapturingCollector>();
  MarkerCapturingCollector* cptr = collector.get();

  RocmTracerOptions opts{/*max_annotation_strings=*/1024};
  TF_ASSERT_OK(tracer.Enable(opts, cptr));

  const uint64_t tid = 2223;

  rocprofiler_callback_tracing_marker_api_data_t outer{};
  outer.args.roctxRangePushA.message = "outer";
  tracer.MarkerCallback(
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA,
                       ROCPROFILER_CALLBACK_PHASE_ENTER, tid, &outer));

  rocprofiler_callback_tracing_marker_api_data_t inner{};
  inner.args.roctxRangePushA.message = nullptr;  // unlabelled inner range
  tracer.MarkerCallback(
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA,
                       ROCPROFILER_CALLBACK_PHASE_ENTER, tid, &inner));

  rocprofiler_callback_tracing_marker_api_data_t pop{};
  tracer.MarkerCallback(
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop,
                       ROCPROFILER_CALLBACK_PHASE_EXIT, tid, &pop));  // inner
  tracer.MarkerCallback(
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop,
                       ROCPROFILER_CALLBACK_PHASE_EXIT, tid, &pop));  // outer

  tracer.Disable();

  auto events = cptr->TakeEvents();
  ASSERT_EQ(events.size(), 1u) << "only the labelled range should be emitted";
  EXPECT_EQ(events[0].name, "outer")
      << "the surviving event must be the outer range, not a shifted frame";
}

// Integration test: verifies the full pipeline — MarkerCallback → AddEvent →
// PerDeviceCollector::Export — produces a Generic event in the XSpace host
// plane. Uses the unit-test collector path (real rocprofiler context is live
// because Enable() starts it; we inject the event via MarkerCallback directly
// rather than going through the real ROCTX library).
TEST(RocmTracerTest, MarkerEventAppearsInExportedXSpace) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  ASSERT_TRUE(tracer.IsAvailable());

  RocmTraceCollectorOptions col_opts;
  col_opts.max_callback_api_events = 1024;
  col_opts.max_activity_api_events = 1024;
  col_opts.max_annotation_strings = 1024;
  col_opts.num_gpus = tracer.NumGpus() > 0 ? tracer.NumGpus() : 1;

  uint64_t start_gpu = RocmTracer::GetTimestamp();
  uint64_t start_wall = tsl::EnvTime::NowNanos();
  auto collector =
      std::make_unique<RocmTraceCollectorImpl>(col_opts, start_wall, start_gpu);
  collector->SetGpuAgents(tracer.GpuAgents());

  RocmTracerOptions opts{/*max_annotation_strings=*/1024};
  TF_ASSERT_OK(tracer.Enable(opts, collector.get()));

  const uint64_t tid = 4242;
  const char* label = "integration_label";

  rocprofiler_callback_tracing_marker_api_data_t push_data{};
  push_data.args.roctxRangePushA.message = label;
  auto push_rec =
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA,
                       ROCPROFILER_CALLBACK_PHASE_ENTER, tid, &push_data);
  tracer.MarkerCallback(push_rec);

  rocprofiler_callback_tracing_marker_api_data_t pop_data{};
  auto pop_rec =
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop,
                       ROCPROFILER_CALLBACK_PHASE_EXIT, tid, &pop_data);
  tracer.MarkerCallback(pop_rec);

  tracer.Disable();

  tsl::profiler::XSpace space;
  collector->Export(&space);

  // Assert on the LINE name, not the plane name. The marker plane is a
  // transient routing token that PostProcessSingleHostXSpace merges into
  // /host:CPU and deletes, so a plane-name assertion pins an implementation
  // detail that never reaches a user. The line name does survive, and the
  // "/ROCTX" suffix is applied only to lines on the marker plane -- so this
  // single check covers both halves of the routing contract: markers land on
  // the marker plane, and they do NOT land on the HIP-API host plane (whose
  // lines are named "Host Threads/<tid>" with no suffix).
  bool found_nvtx_stat = false;
  bool found_correct_label = false;
  bool found_on_non_marker_line = false;
  for (const auto& plane : space.planes()) {
    // Build a map from stat metadata ID -> stat name for this plane.
    absl::flat_hash_map<int64_t, std::string> stat_id_to_name;
    for (const auto& [id, stat_md] : plane.stat_metadata()) {
      stat_id_to_name[id] = stat_md.name();
    }

    // CreateXEvent writes the label via
    //   AddStatValue(md(kNVTXRange), *plane->GetOrCreateStatMetadata(label))
    // whose second argument is an XStatMetadata&, so the label arrives as a
    // ref_value naming another stat_metadata entry -- NOT as a str_value.
    // Reading str_value alone silently yields "" and the assertion can never
    // pass. Handle both encodings, as RealRoctxCallsProduceNvtxRangeInXSpace
    // below already does.
    auto label_of =
        [&](const tensorflow::profiler::XStat& stat) -> std::string {
      if (stat.value_case() == tensorflow::profiler::XStat::kRefValue) {
        auto it = plane.stat_metadata().find(stat.ref_value());
        return it != plane.stat_metadata().end() ? it->second.name() : "";
      }
      return stat.str_value();
    };

    for (const auto& line : plane.lines()) {
      const bool is_marker_line = absl::EndsWith(line.name(), "/ROCTX");
      for (const auto& event : line.events()) {
        for (const auto& stat : event.stats()) {
          auto name_it = stat_id_to_name.find(stat.metadata_id());
          if (name_it == stat_id_to_name.end() ||
              name_it->second != "nvtx_range") {
            continue;
          }
          found_nvtx_stat = true;
          if (!is_marker_line) {
            found_on_non_marker_line = true;
            continue;
          }
          if (label_of(stat) == label) found_correct_label = true;
        }
      }
    }
  }
  EXPECT_TRUE(found_nvtx_stat)
      << "XSpace should contain an nvtx_range stat after a ROCTX range";
  EXPECT_TRUE(found_correct_label)
      << "nvtx_range stat on a \"Host Threads/<tid>/ROCTX\" line must equal "
         "the pushed label: "
      << label;
  EXPECT_FALSE(found_on_non_marker_line)
      << "ROCTX markers must not be routed onto the HIP-API host plane";
}

// ============================================================================
// Integration test: real librocprofiler-sdk-roctx.so → MarkerCallback →
// kNVTXRange stat in exported XSpace.
//
// Linked directly against @local_config_rocm//rocm:rocprofiler_sdk_roctx,
// which stages the library and its librocprofiler-register.so dependency so
// rocprofiler-sdk intercepts the calls. libroctx64.so (old roctracer-era
// roctx) would NOT be intercepted, which is why this test requires the
// rocprofiler-sdk-integrated variant.
// ============================================================================

// Test: real roctxRangePushA/roctxRangePop → kNVTXRange stat in XSpace.
TEST(RocmTracerTest, RealRoctxCallsProduceNvtxRangeInXSpace) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  ASSERT_TRUE(tracer.IsAvailable());

  RocmTraceCollectorOptions col_opts;
  col_opts.max_callback_api_events = 1024;
  col_opts.max_activity_api_events = 1024;
  col_opts.max_annotation_strings = 1024;
  col_opts.num_gpus = tracer.NumGpus() > 0 ? tracer.NumGpus() : 1;

  uint64_t start_gpu = RocmTracer::GetTimestamp();
  uint64_t start_wall = tsl::EnvTime::NowNanos();
  auto collector =
      std::make_unique<RocmTraceCollectorImpl>(col_opts, start_wall, start_gpu);
  collector->SetGpuAgents(tracer.GpuAgents());

  RocmTracerOptions opts{/*max_annotation_strings=*/1024};
  TF_ASSERT_OK(tracer.Enable(opts, collector.get()));

  // Emit real ROCTX ranges — rocprofiler-sdk intercepts these and fires
  // MarkerCallback, which emits Generic RocmTracerEvents.
  EXPECT_GE(roctxRangePushA("unit_test_outer"), 0);
  EXPECT_GE(roctxRangePushA("unit_test_inner"), 0);
  roctxMarkA("unit_test_mark");
  roctxRangePop();  // end unit_test_inner
  roctxRangePop();  // end unit_test_outer

  tracer.Disable();

  // Export and verify kNVTXRange stat appears in XSpace.
  tsl::profiler::XSpace space;
  collector->Export(&space);

  bool found_nvtx_stat = false;
  bool found_on_marker_line = false;
  std::vector<std::string> found_labels;

  for (const auto& plane : space.planes()) {
    // Find the nvtx_range stat metadata id in this plane.
    int64_t nvtx_stat_id = -1;
    for (const auto& [sid, smd] : plane.stat_metadata()) {
      if (smd.name() == "nvtx_range") {
        nvtx_stat_id = sid;
        found_nvtx_stat = true;
        break;
      }
    }
    if (nvtx_stat_id < 0) continue;

    // Collect the label strings from event stats. Routing is asserted via the
    // line name ("Host Threads/<tid>/ROCTX"), which is what survives the merge
    // into /host:CPU -- the marker plane itself is deleted in post-processing.
    for (const auto& line : plane.lines()) {
      const bool is_marker_line = absl::EndsWith(line.name(), "/ROCTX");
      for (const auto& event : line.events()) {
        for (const auto& stat : event.stats()) {
          if (stat.metadata_id() != nvtx_stat_id) continue;
          if (is_marker_line) found_on_marker_line = true;
          if (stat.value_case() == tensorflow::profiler::XStat::kRefValue) {
            int64_t ref = stat.ref_value();
            auto it = plane.stat_metadata().find(ref);
            if (it != plane.stat_metadata().end()) {
              found_labels.push_back(it->second.name());
            }
          } else if (stat.value_case() ==
                     tensorflow::profiler::XStat::kStrValue) {
            found_labels.push_back(stat.str_value());
          }
        }
      }
    }
  }

  EXPECT_TRUE(found_nvtx_stat)
      << "XSpace should contain 'nvtx_range' stat metadata after real ROCTX "
         "calls via librocprofiler-sdk-roctx.so";
  EXPECT_TRUE(found_on_marker_line)
      << "ROCTX events must land on a \"Host Threads/<tid>/ROCTX\" line";

  if (found_nvtx_stat) {
    std::set<std::string> label_set(found_labels.begin(), found_labels.end());
    EXPECT_TRUE(label_set.count("unit_test_outer"))
        << "Expected 'unit_test_outer' in nvtx_range labels. Found: "
        << absl::StrJoin(found_labels, ", ");
    EXPECT_TRUE(label_set.count("unit_test_inner"))
        << "Expected 'unit_test_inner' in nvtx_range labels. Found: "
        << absl::StrJoin(found_labels, ", ");
    // roctxMarkA emits an instantaneous event — label is present
    EXPECT_TRUE(label_set.count("unit_test_mark"))
        << "Expected 'unit_test_mark' in nvtx_range labels. Found: "
        << absl::StrJoin(found_labels, ", ");
  }
}

// ============================================================================
// GetCurrentRoctxLabel and AnnotationMap roctx_range tests (Commit B)
// ============================================================================

TEST(RocmTracerTest, GetCurrentRoctxLabelReturnsTopOfStack) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  ASSERT_TRUE(tracer.IsAvailable());

  auto collector = std::make_unique<MarkerCapturingCollector>();
  RocmTracerOptions opts{/*max_annotation_strings=*/1024};
  TF_ASSERT_OK(tracer.Enable(opts, collector.get()));

  const uint64_t tid = 55555;

  EXPECT_EQ(tracer.GetCurrentRoctxLabel(), "");

  rocprofiler_callback_tracing_marker_api_data_t push_data{};
  push_data.args.roctxRangePushA.message = "outer";
  tracer.MarkerCallback(
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA,
                       ROCPROFILER_CALLBACK_PHASE_ENTER, tid, &push_data));
  EXPECT_EQ(tracer.GetCurrentRoctxLabel(), "outer");

  push_data.args.roctxRangePushA.message = "inner";
  tracer.MarkerCallback(
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA,
                       ROCPROFILER_CALLBACK_PHASE_ENTER, tid, &push_data));
  EXPECT_EQ(tracer.GetCurrentRoctxLabel(), "inner");

  tracer.Disable();
}

TEST(RocmTracerTest, GetCurrentRoctxLabelEmptyAfterPop) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  ASSERT_TRUE(tracer.IsAvailable());

  auto collector = std::make_unique<MarkerCapturingCollector>();
  RocmTracerOptions opts{/*max_annotation_strings=*/1024};
  TF_ASSERT_OK(tracer.Enable(opts, collector.get()));

  const uint64_t tid = 55556;

  rocprofiler_callback_tracing_marker_api_data_t push_data{};
  push_data.args.roctxRangePushA.message = "only_range";
  tracer.MarkerCallback(
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA,
                       ROCPROFILER_CALLBACK_PHASE_ENTER, tid, &push_data));
  EXPECT_EQ(tracer.GetCurrentRoctxLabel(), "only_range");

  rocprofiler_callback_tracing_marker_api_data_t pop_data{};
  tracer.MarkerCallback(
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop,
                       ROCPROFILER_CALLBACK_PHASE_EXIT, tid, &pop_data));
  EXPECT_EQ(tracer.GetCurrentRoctxLabel(), "");

  tracer.Disable();
}

TEST(RocmTracerTest, AnnotationMapStoresRoctxRange) {
  AnnotationMap map(1024);
  map.Add(99, "my_annotation", "my_roctx_label", {});
  EXPECT_EQ(map.LookUp(99), "my_annotation");
  EXPECT_EQ(map.LookUpRoctxRange(99), "my_roctx_label");

  EXPECT_EQ(map.LookUpRoctxRange(100), "");

  map.Clear();
  EXPECT_EQ(map.LookUpRoctxRange(99), "");
}

TEST(RocmTracerTest, AnnotationMapRoctxRangeEmptyWhenNotProvided) {
  AnnotationMap map(1024);
  map.Add(42, "some_op", {}, {});
  EXPECT_EQ(map.LookUp(42), "some_op");
  EXPECT_EQ(map.LookUpRoctxRange(42), "");
}

// Verify that Add() stores the roctx_range even when annotation is empty,
// so that standalone ROCTX annotations (no XLA AnnotationStack text) still
// produce kNVTXRange on kernel events.
TEST(RocmTracerTest, AnnotationMapStoresRoctxRangeWhenAnnotationEmpty) {
  AnnotationMap map(1024);
  // annotation is empty, roctx_range is not.
  map.Add(77, /*annotation=*/"", "roctx_only_label", {});
  EXPECT_EQ(map.LookUp(77), "")
      << "correlation_map should not have an entry when annotation is empty";
  EXPECT_EQ(map.LookUpRoctxRange(77), "roctx_only_label")
      << "roctx_range_map must store the label even with no annotation";
}

// GetCurrentRoctxLabel returns a view aliasing the thread_local frame. It is
// valid until this thread makes its next roctx call, which is the whole window
// the HIP API callback needs: that callback runs synchronously on the pushing
// thread, so no pop can interleave, and AnnotationMap::Add interns a copy
// before returning. This models that sequence -- read the view, materialise it
// the way Add does, then pop -- and confirms the materialised copy outlives the
// frame and that the stack reads empty afterwards.
TEST(RocmTracerTest, GetCurrentRoctxLabelViewIsValidUntilNextRoctxCall) {
  RocmTracer& tracer = RocmTracer::GetRocmTracerSingleton();
  ASSERT_TRUE(tracer.IsAvailable());

  auto collector = std::make_unique<MarkerCapturingCollector>();
  RocmTracerOptions opts{/*max_annotation_strings=*/1024};
  TF_ASSERT_OK(tracer.Enable(opts, collector.get()));

  const uint64_t tid = 66666;
  const char* label = "lifetime_check";

  // Push a range.
  rocprofiler_callback_tracing_marker_api_data_t push_data{};
  push_data.args.roctxRangePushA.message = label;
  tracer.MarkerCallback(
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA,
                       ROCPROFILER_CALLBACK_PHASE_ENTER, tid, &push_data));

  // Read the view while the frame is live, then intern it -- this is the
  // HIP API callback's sequence, all on the pushing thread.
  absl::string_view view = tracer.GetCurrentRoctxLabel();
  EXPECT_EQ(view, label);
  std::string interned(view);

  // Pop the range: this destroys the RoctxFrame::message inside roctx_stack_.
  // `view` dangles from here on and must not be read again.
  rocprofiler_callback_tracing_marker_api_data_t pop_data{};
  tracer.MarkerCallback(
      MakeMarkerRecord(ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop,
                       ROCPROFILER_CALLBACK_PHASE_EXIT, tid, &pop_data));

  EXPECT_EQ(interned, label)
      << "Copy taken before the pop must outlive the source frame";
  EXPECT_EQ(tracer.GetCurrentRoctxLabel(), "")
      << "Stack must be empty after pop";

  tracer.Disable();
}

}  // namespace
}  // namespace profiler
}  // namespace xla
