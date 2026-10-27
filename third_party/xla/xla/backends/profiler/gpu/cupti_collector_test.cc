/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/profiler/gpu/cupti_collector.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/backends/profiler/gpu/cupti_buffer_events.h"
#include "xla/stream_executor/cuda/cuda_graph_annotation_registry.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

using ::stream_executor::gpu::CudaGraphAnnotationRegistry;
using ::tensorflow::profiler::XSpace;
using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::Partially;

TEST(CuptiCollectorTest, TestPmSamplingDataToCounterLine) {
  PmSamples pm_samples({"metric1", "metric2"},
                       {{/*range_index=*/0,
                         /*start_timestamp_ns=*/100,
                         /*end_timestamp_ns=*/200,
                         /*metric_values=*/{1.0, 2.0}},
                        {/*range_index=*/1,
                         /*start_timestamp_ns=*/200,
                         /*end_timestamp_ns=*/300,
                         /*metric_values=*/{3.0, 4.0}}},
                       0);
  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  pm_samples.PopulateCounterLine(&plane_builder, 0);

  EXPECT_EQ(plane.lines_size(), 1);
  EXPECT_EQ(plane.lines(0).events_size(), 4);
  EXPECT_EQ(plane.event_metadata_size(), 2);
  EXPECT_EQ(plane.stat_metadata_size(), 2);
  absl::flat_hash_map<std::string, absl::flat_hash_map<uint64_t, double>>
      counter_events_values;
  for (const auto& event : plane.lines(0).events()) {
    counter_events_values[plane.event_metadata().at(event.metadata_id()).name()]
                         [event.offset_ps()] = event.stats(0).double_value();
  }
  EXPECT_THAT(counter_events_values,
              UnorderedElementsAre(
                  Pair("metric1", UnorderedElementsAre(Pair(100000, 1.0),
                                                       Pair(200000, 3.0))),
                  Pair("metric2", UnorderedElementsAre(Pair(100000, 2.0),
                                                       Pair(200000, 4.0)))));
}

TEST(CuptiCollectorTest, TestPmSamplingDefaultMetricsToCounterLineRenaming) {
  PmSamples pm_samples({"gpc__cycles_elapsed.avg.per_second", "metric2"},
                       {{/*range_index=*/0,
                         /*start_timestamp_ns=*/100,
                         /*end_timestamp_ns=*/200,
                         /*metric_values=*/{1.0, 2.0}},
                        {/*range_index=*/1,
                         /*start_timestamp_ns=*/200,
                         /*end_timestamp_ns=*/300,
                         /*metric_values=*/{3.0, 4.0}}},
                       0);
  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  pm_samples.PopulateCounterLine(&plane_builder, 0);

  EXPECT_EQ(plane.lines_size(), 1);
  EXPECT_EQ(plane.lines(0).events_size(), 4);
  EXPECT_EQ(plane.event_metadata_size(), 2);
  EXPECT_EQ(plane.stat_metadata_size(), 2);
  absl::flat_hash_map<std::string, absl::flat_hash_map<uint64_t, double>>
      counter_events_values;
  for (const auto& event : plane.lines(0).events()) {
    counter_events_values[plane.event_metadata().at(event.metadata_id()).name()]
                         [event.offset_ps()] = event.stats(0).double_value();
  }
  EXPECT_THAT(
      counter_events_values,
      UnorderedElementsAre(
          Pair("GPC Clock Frequency (Hz)",
               UnorderedElementsAre(Pair(100000, 1.0), Pair(200000, 3.0))),
          Pair("metric2",
               UnorderedElementsAre(Pair(100000, 2.0), Pair(200000, 4.0)))));
}

TEST(CuptiCollectorTest, ExportCallbackActivityAndNvtxEvents) {
  CuptiTracerCollectorOptions options;
  options.max_activity_api_events = 100;
  options.max_callback_api_events = 100;
  options.num_gpus = 1;
  std::unique_ptr<CuptiTraceCollector> collector =
      CreateCuptiCollector(options, 0, 0);

  collector->AddEvent(CuptiTracerEvent{
      /*type=*/CuptiTracerEventType::CudaGraph,
      /*source=*/CuptiTracerEventSource::Activity,
      /*name=*/"CudaGraphExec:2",
      /*annotation=*/"annotation",
      /*nvtx_range=*/"",
      /*start_time_ns=*/100,
      /*end_time_ns=*/200,
      /*device_id=*/0,
      /*correlation_id=*/8,
      /*thread_id=*/100,
      /*context_id=*/1,
      /*stream_id=*/2,
      /*graph_id=*/5,
  });

  collector->AddEvent(CuptiTracerEvent{
      /*type=*/CuptiTracerEventType::Generic,
      /*source=*/CuptiTracerEventSource::DriverCallback,
      /*name=*/"cudaGraphLaunch",
      /*annotation=*/"annotation",
      /*nvtx_range=*/"",
      /*start_time_ns=*/90,
      /*end_time_ns=*/120,
      /*device_id=*/0,
      /*correlation_id=*/8,
      /*thread_id=*/100,
      /*context_id=*/1,
      /*stream_id=*/2,
      /*graph_id=*/5,
  });

  collector->AddEvent(CuptiTracerEvent{
      /*type=*/CuptiTracerEventType::ThreadMarkerRange,
      /*source=*/CuptiTracerEventSource::Activity,
      /*name=*/"NVTX::MarkCudaGraphLaunch",
      /*annotation=*/"annotation",
      /*nvtx_range=*/"",
      /*start_time_ns=*/85,
      /*end_time_ns=*/125,
      /*device_id=*/0,
      /*correlation_id=*/0,
      /*thread_id=*/100,
      /*context_id=*/1,
      /*stream_id=*/2,
      /*graph_id=*/5,
  });

  XSpace space;
  collector->Export(&space, /*end_gpu_ns=*/210);

  // All the three planes must exist in the space:
  // Cupti-Driver-API, Cupti-NVTX, GpuDevice.
  const std::string gpu_device_plane_name = ::tsl::profiler::GpuPlaneName(0);
  const absl::flat_hash_set<absl::string_view> plane_names = {
      ::tsl::profiler::kCuptiDriverApiPlaneName,
      ::tsl::profiler::kCuptiActivityNvtxPlaneName, gpu_device_plane_name};
  int num_planes_to_check = 0;
  for (const auto& plane : space.planes()) {
    if (plane_names.contains(plane.name())) {
      ++num_planes_to_check;
    }
  }
  EXPECT_EQ(num_planes_to_check, static_cast<int>(plane_names.size()));

  // In each above plane, only one line is created, and it has one event.
  for (const auto& plane : space.planes()) {
    if (plane_names.contains(plane.name())) {
      ASSERT_EQ(plane.lines_size(), 1);
      ASSERT_EQ(plane.lines(0).events_size(), 1);
    }
  }
}

TEST(CuptiCollectorTest, ExportEnvironmentEvents) {
  CuptiTracerCollectorOptions options;
  options.max_activity_api_events = 100;
  options.max_callback_api_events = 100;
  options.num_gpus = 1;
  std::unique_ptr<CuptiTraceCollector> collector =
      CreateCuptiCollector(options, 0, 0);

  collector->AddEvent(CuptiTracerEvent{
      /*type=*/CuptiTracerEventType::Environment,
      /*source=*/CuptiTracerEventSource::Activity,
      /*name=*/"power_mw",
      /*annotation=*/"",
      /*nvtx_range=*/"",
      /*start_time_ns=*/100,
      /*end_time_ns=*/100,
      /*device_id=*/0,
      /*correlation_id=*/0,
      /*thread_id=*/0,
      /*context_id=*/0,
      /*stream_id=*/0,
      /*graph_id=*/0,
      /*scope_range_id=*/0,
      /*graph_node_id=*/0,
      /*environment_info=*/{{/*metric_value=*/1000}},
  });

  collector->AddEvent(CuptiTracerEvent{
      /*type=*/CuptiTracerEventType::Environment,
      /*source=*/CuptiTracerEventSource::Activity,
      /*name=*/"gpu_temp_c",
      /*annotation=*/"",
      /*nvtx_range=*/"",
      /*start_time_ns=*/110,
      /*end_time_ns=*/110,
      /*device_id=*/0,
      /*correlation_id=*/0,
      /*thread_id=*/0,
      /*context_id=*/0,
      /*stream_id=*/0,
      /*graph_id=*/0,
      /*scope_range_id=*/0,
      /*graph_node_id=*/0,
      /*environment_info=*/{{/*metric_value=*/60}},
  });

  XSpace space;
  collector->Export(&space, 210);

  const std::string gpu_device_plane_name = ::tsl::profiler::GpuPlaneName(0);
  int num_planes_to_check = 0;
  for (const auto& plane : space.planes()) {
    if (plane.name() != gpu_device_plane_name) {
      continue;
    }
    ++num_planes_to_check;
    EXPECT_EQ(plane.lines_size(), 1);
    EXPECT_EQ(plane.lines(0).events_size(), 2);
    absl::flat_hash_map<std::string, absl::flat_hash_map<uint64_t, double>>
        counter_events_values;
    for (const auto& event : plane.lines(0).events()) {
      counter_events_values[plane.event_metadata()
                                .at(event.metadata_id())
                                .name()][event.offset_ps()] =
          event.stats(0).double_value();
    }
    EXPECT_THAT(
        counter_events_values,
        UnorderedElementsAre(
            Pair("power_mw", UnorderedElementsAre(Pair(100000, 1000.0))),
            Pair("gpu_temp_c", UnorderedElementsAre(Pair(110000, 60.0)))));
  }
  EXPECT_EQ(num_planes_to_check, 1);
}

TEST(PmSamplesTest, PopulateCounterLineSkipsNan) {
  XSpace space;
  tsl::profiler::XPlaneBuilder plane_builder(space.add_planes());
  PmSamples pm_samples({"metric1", "metric2"},
                       {{/*range_index=*/0,
                         /*start_timestamp_ns=*/100,
                         /*end_timestamp_ns=*/200,
                         {123.0, std::numeric_limits<double>::quiet_NaN()}}},
                       /*device_id=*/0);

  uint64_t start_gpu_time_ns = 50;
  pm_samples.PopulateCounterLine(&plane_builder, start_gpu_time_ns);

  const auto& plane = space.planes(0);
  ASSERT_EQ(plane.lines_size(), 1);
  const auto& line = plane.lines(0);

  ASSERT_EQ(line.events_size(), 1);
  const auto& event = line.events(0);
  // metric2 is skipped because it's NaN.
  ASSERT_EQ(event.stats_size(), 1);
  const auto& stat = event.stats(0);
  EXPECT_EQ(plane.stat_metadata().at(stat.metadata_id()).name(), "metric1");
  EXPECT_EQ(stat.double_value(), 123.0);
}

TEST(CuptiCollectorTest, AggregatedTracingStats) {
  CuptiTracerCollectorOptions options;
  options.num_gpus = 1;
  options.aggregated_tracing = true;
  std::unique_ptr<CuptiTraceCollector> collector =
      CreateCuptiCollector(options, 0, 0);

  collector->AddEvent(CuptiTracerEvent{
      /*type=*/CuptiTracerEventType::Kernel,
      /*source=*/CuptiTracerEventSource::Activity,
      /*name=*/"kernel1",
      /*annotation=*/"Thunk:#hlo_op=foo#",
      /*nvtx_range=*/"",
      /*start_time_ns=*/100,
      /*end_time_ns=*/200,
      /*device_id=*/0,
      /*correlation_id=*/1,
      /*thread_id=*/100,
      /*context_id=*/1,
      /*stream_id=*/2,
      /*graph_id=*/0,
      /*scope_range_id=*/0,
      /*graph_node_id=*/0,
  });

  collector->AddEvent(CuptiTracerEvent{
      /*type=*/CuptiTracerEventType::Kernel,
      /*source=*/CuptiTracerEventSource::Activity,
      /*name=*/"kernel1",
      /*annotation=*/"Thunk:#hlo_op=foo#",
      /*nvtx_range=*/"",
      /*start_time_ns=*/300,
      /*end_time_ns=*/450,
      /*device_id=*/0,
      /*correlation_id=*/2,
      /*thread_id=*/100,
      /*context_id=*/1,
      /*stream_id=*/2,
      /*graph_id=*/0,
      /*scope_range_id=*/0,
      /*graph_node_id=*/0,
  });

  collector->AddEvent(CuptiTracerEvent{
      /*type=*/CuptiTracerEventType::Kernel,
      /*source=*/CuptiTracerEventSource::Activity,
      /*name=*/"kernel2",
      /*annotation=*/"Thunk:#hlo_op=bar#",
      /*nvtx_range=*/"",
      /*start_time_ns=*/500,
      /*end_time_ns=*/600,
      /*device_id=*/0,
      /*correlation_id=*/3,
      /*thread_id=*/100,
      /*context_id=*/1,
      /*stream_id=*/2,
      /*graph_id=*/0,
      /*scope_range_id=*/0,
      /*graph_node_id=*/0,
  });

  XSpace space;
  collector->Export(&space, /*end_gpu_ns=*/700);

  EXPECT_THAT(
      space.planes(), Contains(Partially(EqualsProto(R"pb(
        name: "/device:GPU:0"
        lines {
          events { metadata_id: 1 duration_ps: 250000 num_occurrences: 2 }
          events { metadata_id: 2 duration_ps: 100000 num_occurrences: 1 }
        }
        event_metadata {
          key: 1
          value { name: "kernel1" }
        }
        event_metadata {
          key: 2
          value { name: "kernel2" }
        }
      )pb"))));
}

TEST(CuptiCollectorTest, AggregatedTracingOutOfRangeEvents) {
  CuptiTracerCollectorOptions options;
  options.num_gpus = 1;
  options.aggregated_tracing = true;
  std::unique_ptr<CuptiTraceCollector> collector =
      CreateCuptiCollector(options, 0, 400);

  collector->AddEvent(CuptiTracerEvent{
      /*type=*/CuptiTracerEventType::Kernel,
      /*source=*/CuptiTracerEventSource::Activity,
      /*name=*/"kernel1",
      /*annotation=*/"Thunk:#hlo_op=foo#",
      /*nvtx_range=*/"",
      /*start_time_ns=*/100,
      /*end_time_ns=*/200,
      /*device_id=*/0,
      /*correlation_id=*/1,
      /*thread_id=*/100,
      /*context_id=*/1,
      /*stream_id=*/2,
      /*graph_id=*/0,
      /*scope_range_id=*/0,
      /*graph_node_id=*/0,
  });

  collector->AddEvent(CuptiTracerEvent{
      /*type=*/CuptiTracerEventType::Kernel,
      /*source=*/CuptiTracerEventSource::Activity,
      /*name=*/"kernel1",
      /*annotation=*/"Thunk:#hlo_op=foo#",
      /*nvtx_range=*/"",
      /*start_time_ns=*/300,
      /*end_time_ns=*/450,
      /*device_id=*/0,
      /*correlation_id=*/2,
      /*thread_id=*/100,
      /*context_id=*/1,
      /*stream_id=*/2,
      /*graph_id=*/0,
      /*scope_range_id=*/0,
      /*graph_node_id=*/0,
  });

  collector->AddEvent(CuptiTracerEvent{
      /*type=*/CuptiTracerEventType::Kernel,
      /*source=*/CuptiTracerEventSource::Activity,
      /*name=*/"kernel2",
      /*annotation=*/"Thunk:#hlo_op=bar#",
      /*nvtx_range=*/"",
      /*start_time_ns=*/500,
      /*end_time_ns=*/600,
      /*device_id=*/0,
      /*correlation_id=*/3,
      /*thread_id=*/100,
      /*context_id=*/1,
      /*stream_id=*/2,
      /*graph_id=*/0,
      /*scope_range_id=*/0,
      /*graph_node_id=*/0,
  });

  XSpace space;
  collector->Export(&space, /*end_gpu_ns=*/700);

  EXPECT_THAT(
      space.planes(), Contains(Partially(EqualsProto(R"pb(
        name: "/device:GPU:0"
        lines {
          events { metadata_id: 1 duration_ps: 100000 num_occurrences: 1 }
        }
        event_metadata {
          key: 1
          value { name: "kernel2" }
        }
      )pb"))));
}

TEST(CuptiCollectorTest, ExportScopeRangeIdTreeSkipsIfEmpty) {
  CuptiTracerCollectorOptions options;
  options.num_gpus = 1;
  std::unique_ptr<CuptiTraceCollector> collector =
      CreateCuptiCollector(options, 0, 0);

  // No events with scope_range_id_tree are added.

  XSpace space;
  collector->Export(&space, /*end_gpu_ns=*/210);

  tensorflow::profiler::XPlane* tree_plane =
      tsl::profiler::FindMutablePlaneWithName(
          &space, tsl::profiler::kScopeRangeIdTreePlaneName);
  EXPECT_EQ(tree_plane, nullptr);
}

TEST(CuptiCollectorTest, ExportScopeRangeIdTreePreventsIdOverlap) {
  CuptiTracerCollectorOptions options;
  options.num_gpus = 1;
  std::unique_ptr<CuptiTraceCollector> collector =
      CreateCuptiCollector(options, 0, 0);

  std::vector<CallbackAnnotationsAndEvents> callback_events;
  CallbackAnnotationsAndEvents events;
  events.scope_range_id_tree().insert({1, 2});
  events.scope_range_id_tree().insert({3, 10});
  callback_events.push_back(std::move(events));

  collector->OnTracerCollectedCallbackData(std::move(callback_events),
                                           /*need_callback_events=*/true);

  XSpace space;
  collector->Export(&space, /*end_gpu_ns=*/210);

  tensorflow::profiler::XPlane* tree_plane =
      tsl::profiler::FindMutablePlaneWithName(
          &space, tsl::profiler::kScopeRangeIdTreePlaneName);
  ASSERT_NE(tree_plane, nullptr);

  // Create a new metadata using XPlaneBuilder and verify its ID is greater
  // than 10.
  tsl::profiler::XPlaneBuilder plane_builder(tree_plane);
  auto* new_stat = plane_builder.GetOrCreateStatMetadata("new_stat");
  EXPECT_GT(new_stat->id(), 10);
}

constexpr absl::string_view kCommandBufferLaunchAnnotation =
    "XlaModule:#hlo_module=test_mod,program_id=1#::command_buffer:#tf_op="
    "cmd_buf,hlo_op=command_buffer#";
constexpr int64_t kLaunchScopeRangeId = 7;
// ID of the executable graph and of its node that ran the kernel.
constexpr uint32_t kExecGraphId = 99;
constexpr uint64_t kExecGraphNodeId = 501;

// Returns the kernel record of a graph node as CuptiTracer hands it to the
// collector: it carries the annotation and scope range of the command buffer
// launch.
CuptiTracerEvent MakeGraphNodeKernelEvent(uint32_t source_graph_id,
                                          uint64_t source_graph_node_id,
                                          uint64_t start_time_ns = 100,
                                          uint64_t end_time_ns = 200) {
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::Kernel;
  event.source = CuptiTracerEventSource::Activity;
  event.name = "graph_node_kernel";
  event.annotation = kCommandBufferLaunchAnnotation;
  event.start_time_ns = start_time_ns;
  event.end_time_ns = end_time_ns;
  event.device_id = 0;
  event.correlation_id = 42;
  event.context_id = 1;
  event.stream_id = 2;
  event.scope_range_id = kLaunchScopeRangeId;
  event.graph_id = kExecGraphId;
  event.graph_node_id = kExecGraphNodeId;
  event.source_graph_id = source_graph_id;
  event.source_graph_node_id = source_graph_node_id;
  return event;
}

std::unique_ptr<CuptiTraceCollector> CreateSingleGpuCollector(
    bool aggregated_tracing = false) {
  CuptiTracerCollectorOptions options;
  options.num_gpus = 1;
  options.aggregated_tracing = aggregated_tracing;
  return CreateCuptiCollector(options, /*start_walltime_ns=*/0,
                              /*start_gputime_ns=*/0);
}

// Returns the stat named `stat_name` of `event` on `plane`, or null.
const tensorflow::profiler::XStat* FindStat(
    const tensorflow::profiler::XPlane& plane,
    const tensorflow::profiler::XEvent& event, absl::string_view stat_name) {
  for (const tensorflow::profiler::XStat& stat : event.stats()) {
    if (plane.stat_metadata().at(stat.metadata_id()).name() == stat_name) {
      return &stat;
    }
  }
  return nullptr;
}

// Returns the string value of the stat named `stat_name` of `event` on
// `plane`, or "" if the event has no such stat.
std::string GetStringStat(const tensorflow::profiler::XPlane& plane,
                          const tensorflow::profiler::XEvent& event,
                          absl::string_view stat_name) {
  const tensorflow::profiler::XStat* stat = FindStat(plane, event, stat_name);
  if (stat == nullptr) {
    return "";
  }
  if (stat->value_case() == tensorflow::profiler::XStat::kRefValue) {
    return plane.stat_metadata().at(stat->ref_value()).name();
  }
  return stat->str_value();
}

TEST(CuptiCollectorTest,
     ResolvesCommandBufferNodeAnnotationsFromSourceGraphNodeId) {
  CudaGraphAnnotationRegistry& registry =
      CudaGraphAnnotationRegistry::Default();
  registry.ClearForTesting();

  constexpr uint32_t kRootGraphId = 10;
  constexpr uint32_t kChildGraphId = 20;
  constexpr uint64_t kKernelToolsId = 1001;
  constexpr uint64_t kMemcpyToolsId = 1002;
  constexpr uint64_t kMemsetToolsId = 2001;

  constexpr auto kNewGraph = CudaGraphAnnotationRegistry::TrackMode::kNewGraph;
  const uint64_t root_token =
      registry.TrackGraph(kRootGraphId, kNewGraph).token;
  const uint64_t child_token =
      registry.TrackGraph(kChildGraphId, kNewGraph).token;
  ASSERT_TRUE(registry.RegisterNodeAnnotation(
      kRootGraphId, root_token, kKernelToolsId,
      "XlaModule:#hlo_module=test_mod,program_id=1#::fusion_1:#tf_op=Add,"
      "hlo_op=fusion.1#"));
  ASSERT_TRUE(registry.RegisterNodeAnnotation(
      kRootGraphId, root_token, kMemcpyToolsId,
      "XlaModule:#hlo_module=test_mod,program_id=1#::copy_1:#tf_op=Identity,"
      "hlo_op=copy.1#"));
  ASSERT_TRUE(registry.RegisterNodeAnnotation(
      kChildGraphId, child_token, kMemsetToolsId,
      "XlaModule:#hlo_module=test_mod,program_id=1#::memset_1:#tf_op=ZerosLike,"
      "hlo_op=broadcast.1#"));

  std::unique_ptr<CuptiTraceCollector> collector = CreateSingleGpuCollector();

  collector->AddEvent(MakeGraphNodeKernelEvent(kRootGraphId, kKernelToolsId));

  CuptiTracerEvent memcpy_event =
      MakeGraphNodeKernelEvent(kRootGraphId, kMemcpyToolsId, 210, 250);
  memcpy_event.type = CuptiTracerEventType::MemcpyD2D;
  memcpy_event.name = "MemcpyD2D";
  memcpy_event.graph_node_id = 502;
  collector->AddEvent(std::move(memcpy_event));

  CuptiTracerEvent memset_event =
      MakeGraphNodeKernelEvent(kChildGraphId, kMemsetToolsId, 260, 300);
  memset_event.type = CuptiTracerEventType::Memset;
  memset_event.name = "Memset 1";
  memset_event.graph_node_id = 503;
  collector->AddEvent(std::move(memset_event));

  XSpace space;
  collector->Export(&space, /*end_gpu_ns=*/400);

  const tensorflow::profiler::XPlane* gpu_plane =
      tsl::profiler::FindPlaneWithName(space, tsl::profiler::GpuPlaneName(0));
  ASSERT_NE(gpu_plane, nullptr);
  ASSERT_EQ(gpu_plane->lines_size(), 1);
  ASSERT_EQ(gpu_plane->lines(0).events_size(), 3);

  EXPECT_EQ(GetStringStat(*gpu_plane, gpu_plane->lines(0).events(0), "hlo_op"),
            "fusion.1");
  EXPECT_EQ(
      GetStringStat(*gpu_plane, gpu_plane->lines(0).events(0), "hlo_module"),
      "test_mod");
  EXPECT_EQ(GetStringStat(*gpu_plane, gpu_plane->lines(0).events(1), "hlo_op"),
            "copy.1");
  EXPECT_EQ(
      GetStringStat(*gpu_plane, gpu_plane->lines(0).events(1), "hlo_module"),
      "test_mod");
  EXPECT_EQ(GetStringStat(*gpu_plane, gpu_plane->lines(0).events(2), "hlo_op"),
            "broadcast.1");
  EXPECT_EQ(
      GetStringStat(*gpu_plane, gpu_plane->lines(0).events(2), "hlo_module"),
      "test_mod");

  registry.ClearForTesting();
}

TEST(CuptiCollectorTest,
     ResolvesCommandBufferNodeAnnotationsInAggregatedTracing) {
  CudaGraphAnnotationRegistry& registry =
      CudaGraphAnnotationRegistry::Default();
  registry.ClearForTesting();

  constexpr uint32_t kGraphId = 10;
  constexpr uint64_t kNode1ToolsId = 1001;
  constexpr uint64_t kNode2ToolsId = 1002;

  constexpr auto kNewGraph = CudaGraphAnnotationRegistry::TrackMode::kNewGraph;
  const uint64_t token = registry.TrackGraph(kGraphId, kNewGraph).token;
  ASSERT_TRUE(registry.RegisterNodeAnnotation(
      kGraphId, token, kNode1ToolsId,
      "XlaModule:#hlo_module=test_mod#::fusion.1:#hlo_op=fusion.1#"));
  ASSERT_TRUE(registry.RegisterNodeAnnotation(
      kGraphId, token, kNode2ToolsId,
      "XlaModule:#hlo_module=test_mod#::fusion.2:#hlo_op=fusion.2#"));

  std::unique_ptr<CuptiTraceCollector> collector =
      CreateSingleGpuCollector(/*aggregated_tracing=*/true);
  collector->AddEvent(
      MakeGraphNodeKernelEvent(kGraphId, kNode1ToolsId, 100, 200));
  collector->AddEvent(
      MakeGraphNodeKernelEvent(kGraphId, kNode2ToolsId, 210, 360));
  collector->AddEvent(
      MakeGraphNodeKernelEvent(kGraphId, kNode1ToolsId, 400, 500));

  XSpace space;
  collector->Export(&space, /*end_gpu_ns=*/600);

  const tensorflow::profiler::XPlane* gpu_plane =
      tsl::profiler::FindPlaneWithName(space, tsl::profiler::GpuPlaneName(0));
  ASSERT_NE(gpu_plane, nullptr);
  ASSERT_EQ(gpu_plane->lines_size(), 1);
  // Two distinct aggregated events because node 1 and node 2 resolve to
  // distinct per-node annotations despite sharing the same kernel name and
  // graph launch correlation annotation.
  ASSERT_EQ(gpu_plane->lines(0).events_size(), 2);
  EXPECT_EQ(gpu_plane->lines(0).events(0).num_occurrences(), 2);
  EXPECT_EQ(gpu_plane->lines(0).events(0).duration_ps(), 200000);
  EXPECT_EQ(gpu_plane->lines(0).events(1).num_occurrences(), 1);
  EXPECT_EQ(gpu_plane->lines(0).events(1).duration_ps(), 150000);

  registry.ClearForTesting();
}

// While a harvest hold is active, the registry parks the entry of a graph
// destroyed during the session, so the graph's records still resolve when
// they are harvested. The collector copies the annotation, so the event keeps
// it after the last hold's release purges the entry.
TEST(CuptiCollectorTest, ResolvesParkedEntry) {
  CudaGraphAnnotationRegistry& registry =
      CudaGraphAnnotationRegistry::Default();
  registry.ClearForTesting();
  // Parked entries are purged only when the last hold is released.
  ASSERT_EQ(registry.GetStats().active_holds, 0);

  constexpr uint32_t kGraphId = 10;
  constexpr uint64_t kNodeId = 1001;
  std::unique_ptr<CuptiTraceCollector> collector = CreateSingleGpuCollector();
  {
    CudaGraphAnnotationRegistry::HarvestHold hold =
        registry.AcquireHarvestHold();
    const CudaGraphAnnotationRegistry::Token token =
        registry
            .TrackGraph(kGraphId,
                        CudaGraphAnnotationRegistry::TrackMode::kNewGraph)
            .token;
    ASSERT_TRUE(registry.RegisterNodeAnnotation(
        kGraphId, token, kNodeId,
        "XlaModule:#hlo_module=test_mod#::fusion.1:#hlo_op=fusion.1#"));
    // CUDA destroys the graph during the session.
    registry.UnregisterGraph(kGraphId, token);
    ASSERT_EQ(registry.GetStats().parked_graphs, 1);

    collector->AddEvent(MakeGraphNodeKernelEvent(kGraphId, kNodeId));
  }
  EXPECT_TRUE(registry.empty());

  XSpace space;
  collector->Export(&space, /*end_gpu_ns=*/400);
  const tensorflow::profiler::XPlane* gpu_plane =
      tsl::profiler::FindPlaneWithName(space, tsl::profiler::GpuPlaneName(0));
  ASSERT_NE(gpu_plane, nullptr);
  ASSERT_EQ(gpu_plane->lines_size(), 1);
  ASSERT_EQ(gpu_plane->lines(0).events_size(), 1);
  const tensorflow::profiler::XEvent& event = gpu_plane->lines(0).events(0);
  EXPECT_EQ(GetStringStat(*gpu_plane, event, "hlo_op"), "fusion.1");
  // The resolved annotation replaces the launch annotation, but the event
  // keeps the launch's scope range.
  const tensorflow::profiler::XStat* scope_range_id = FindStat(
      *gpu_plane, event,
      tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kScopeRangeId));
  ASSERT_NE(scope_range_id, nullptr);
  EXPECT_EQ(scope_range_id->int64_value(), kLaunchScopeRangeId);

  registry.ClearForTesting();
}

// Without command buffer telemetry the registry is empty, and the collector
// leaves the launch annotation in place.
TEST(CuptiCollectorTest, EmptyRegistryKeepsLaunchAnnotation) {
  CudaGraphAnnotationRegistry& registry =
      CudaGraphAnnotationRegistry::Default();
  registry.ClearForTesting();
  ASSERT_TRUE(registry.empty());

  constexpr uint32_t kGraphId = 10;
  constexpr uint64_t kNodeId = 1001;
  std::unique_ptr<CuptiTraceCollector> collector = CreateSingleGpuCollector();
  collector->AddEvent(MakeGraphNodeKernelEvent(kGraphId, kNodeId));

  XSpace space;
  collector->Export(&space, /*end_gpu_ns=*/400);
  const tensorflow::profiler::XPlane* gpu_plane =
      tsl::profiler::FindPlaneWithName(space, tsl::profiler::GpuPlaneName(0));
  ASSERT_NE(gpu_plane, nullptr);
  ASSERT_EQ(gpu_plane->lines_size(), 1);
  ASSERT_EQ(gpu_plane->lines(0).events_size(), 1);
  EXPECT_EQ(GetStringStat(*gpu_plane, gpu_plane->lines(0).events(0), "hlo_op"),
            "command_buffer");
  EXPECT_THAT(space.warnings(),
              Not(Contains(HasSubstr("CUDA graph node records"))));

  registry.ClearForTesting();
}

// A record whose source node isn't registered falls back to the annotations
// that the CUDA graph callbacks collected during the session.
TEST(CuptiCollectorTest, FallsBackToCallbackMap) {
  CudaGraphAnnotationRegistry& registry =
      CudaGraphAnnotationRegistry::Default();
  registry.ClearForTesting();
  // Register another node of the graph, so that the collector looks the record
  // up and misses instead of skipping an empty registry.
  constexpr uint32_t kGraphId = 10;
  constexpr uint64_t kRegisteredNodeId = 1001;
  constexpr uint64_t kUnregisteredNodeId = 1002;
  const CudaGraphAnnotationRegistry::Token token =
      registry
          .TrackGraph(kGraphId,
                      CudaGraphAnnotationRegistry::TrackMode::kNewGraph)
          .token;
  ASSERT_TRUE(registry.RegisterNodeAnnotation(
      kGraphId, token, kRegisteredNodeId,
      "XlaModule:#hlo_module=test_mod#::fusion.1:#hlo_op=fusion.1#"));

  std::unique_ptr<CuptiTraceCollector> collector = CreateSingleGpuCollector();

  // The node-creation callback recorded the annotation of node 70 of graph 7.
  constexpr uint32_t kOrigGraphId = 7;
  constexpr uint64_t kOrigGraphNodeId = 70;
  std::vector<CallbackAnnotationsAndEvents> callback_data(1);
  CuptiTracerEvent node_created{};
  node_created.type = CuptiTracerEventType::CudaGraph;
  node_created.source = CuptiTracerEventSource::DriverCallback;
  node_created.name = "cuGraphAddKernelNode";
  node_created.annotation = callback_data[0].DedupAnnotation(
      "XlaModule:#hlo_module=test_mod#::fusion.2:#hlo_op=fusion.2#");
  node_created.start_time_ns = 10;
  node_created.end_time_ns = 20;
  node_created.device_id = 0;
  node_created.correlation_id = 5;
  node_created.graph_id = kOrigGraphId;
  node_created.graph_node_id = kOrigGraphNodeId;
  callback_data[0].event_queue().Push(std::move(node_created));
  collector->OnTracerCollectedCallbackData(std::move(callback_data),
                                           /*need_callback_events=*/false);

  // The instantiation callback mapped the executable graph's node to it.
  CuptiTracerEvent node_map{};
  node_map.type = CuptiTracerEventType::CudaGraphNodeMap;
  node_map.source = CuptiTracerEventSource::DriverCallback;
  node_map.device_id = 0;
  node_map.graph_id = kExecGraphId;
  node_map.graph_node_id = kExecGraphNodeId;
  node_map.cuda_graph_info.orig_graph_id = kOrigGraphId;
  node_map.cuda_graph_info.orig_graph_node_id = kOrigGraphNodeId;
  collector->AddEvent(std::move(node_map));

  collector->AddEvent(MakeGraphNodeKernelEvent(kGraphId, kUnregisteredNodeId));

  XSpace space;
  collector->Export(&space, /*end_gpu_ns=*/400);
  const tensorflow::profiler::XPlane* gpu_plane =
      tsl::profiler::FindPlaneWithName(space, tsl::profiler::GpuPlaneName(0));
  ASSERT_NE(gpu_plane, nullptr);
  ASSERT_EQ(gpu_plane->lines_size(), 1);
  ASSERT_EQ(gpu_plane->lines(0).events_size(), 1);
  EXPECT_EQ(GetStringStat(*gpu_plane, gpu_plane->lines(0).events(0), "hlo_op"),
            "fusion.2");

  registry.ClearForTesting();
}

// Returns the `hlo_op` of the only event that `collector` exports for `event`,
// or "" if it exports a different number of events.
std::string ResolveHloOp(CuptiTracerEvent event) {
  std::unique_ptr<CuptiTraceCollector> collector = CreateSingleGpuCollector();
  collector->AddEvent(std::move(event));
  XSpace space;
  collector->Export(&space, /*end_gpu_ns=*/400);
  const tensorflow::profiler::XPlane* gpu_plane =
      tsl::profiler::FindPlaneWithName(space, tsl::profiler::GpuPlaneName(0));
  if (gpu_plane == nullptr || gpu_plane->lines_size() != 1 ||
      gpu_plane->lines(0).events_size() != 1) {
    return "";
  }
  return GetStringStat(*gpu_plane, gpu_plane->lines(0).events(0), "hlo_op");
}

// The collector looks up by the IDs each record carries: the source graph IDs
// in `kTemplateGraph` if the record has them (CUDA 13.4+), otherwise the graph
// IDs in `kPre134ExecGraph` (CUDA 13.2-13.3), otherwise nothing.
TEST(CuptiCollectorTest, LooksUpByRecordIds) {
  CudaGraphAnnotationRegistry& registry =
      CudaGraphAnnotationRegistry::Default();
  registry.ClearForTesting();

  constexpr uint32_t kSourceGraphId = 10;
  constexpr uint64_t kSourceNodeId = 1001;
  const CudaGraphAnnotationRegistry::Token exec_token =
      registry
          .TrackGraph(kExecGraphId,
                      CudaGraphAnnotationRegistry::TrackMode::kNewGraph,
                      CudaGraphAnnotationRegistry::KeySpace::kPre134ExecGraph)
          .token;
  const CudaGraphAnnotationRegistry::Token source_token =
      registry
          .TrackGraph(kSourceGraphId,
                      CudaGraphAnnotationRegistry::TrackMode::kNewGraph)
          .token;
  ASSERT_TRUE(registry.RegisterNodeAnnotation(
      kExecGraphId, exec_token, kExecGraphNodeId,
      "XlaModule:#hlo_module=test_mod#::pre134_exec:#hlo_op=pre134_exec#",
      CudaGraphAnnotationRegistry::KeySpace::kPre134ExecGraph));
  ASSERT_TRUE(registry.RegisterNodeAnnotation(
      kSourceGraphId, source_token, kSourceNodeId,
      "XlaModule:#hlo_module=test_mod#::cuda134_src:#hlo_op=cuda134_src#"));

  // Source graph IDs only.
  CuptiTracerEvent source_only =
      MakeGraphNodeKernelEvent(kSourceGraphId, kSourceNodeId);
  source_only.graph_id = 0;
  source_only.graph_node_id = 0;
  EXPECT_EQ(ResolveHloOp(std::move(source_only)), "cuda134_src");

  // Executable graph IDs only, as in a CUDA 13.2-13.3 record.
  EXPECT_EQ(ResolveHloOp(MakeGraphNodeKernelEvent(/*source_graph_id=*/0,
                                                  /*source_graph_node_id=*/0)),
            "pre134_exec");

  // Both: the source graph IDs take precedence.
  EXPECT_EQ(
      ResolveHloOp(MakeGraphNodeKernelEvent(kSourceGraphId, kSourceNodeId)),
      "cuda134_src");

  // Neither: the event keeps the launch annotation.
  CuptiTracerEvent neither = MakeGraphNodeKernelEvent(
      /*source_graph_id=*/0, /*source_graph_node_id=*/0);
  neither.graph_id = 0;
  neither.graph_node_id = 0;
  EXPECT_EQ(ResolveHloOp(std::move(neither)), "command_buffer");

  registry.ClearForTesting();
}

// If the registry has annotations but no graph node record of the session
// resolves, the collector warns, also in the exported XSpace.
TEST(CuptiCollectorTest, WarnsIfNoGraphNodeRecordResolves) {
  CudaGraphAnnotationRegistry& registry =
      CudaGraphAnnotationRegistry::Default();
  registry.ClearForTesting();
  constexpr uint32_t kGraphId = 10;
  constexpr uint64_t kNodeId = 1001;
  const CudaGraphAnnotationRegistry::Token token =
      registry
          .TrackGraph(kGraphId,
                      CudaGraphAnnotationRegistry::TrackMode::kNewGraph)
          .token;
  ASSERT_TRUE(registry.RegisterNodeAnnotation(
      kGraphId, token, kNodeId,
      "XlaModule:#hlo_module=test_mod#::fusion.1:#hlo_op=fusion.1#"));

  std::unique_ptr<CuptiTraceCollector> collector = CreateSingleGpuCollector();
  collector->AddEvent(MakeGraphNodeKernelEvent(kGraphId, kNodeId + 1));
  XSpace space;
  collector->Export(&space, /*end_gpu_ns=*/400);
  EXPECT_THAT(space.warnings(),
              Contains("None of the 1 CUDA graph node records with graph IDs "
                       "resolved to a command buffer node annotation recorded "
                       "by --xla_gpu_enable_command_buffer_telemetry, so they "
                       "keep the command buffer launch annotation."));

  registry.ClearForTesting();
}

// No warning once at least one graph node record resolves.
TEST(CuptiCollectorTest, NoResolutionWarningIfARecordResolves) {
  CudaGraphAnnotationRegistry& registry =
      CudaGraphAnnotationRegistry::Default();
  registry.ClearForTesting();
  constexpr uint32_t kGraphId = 10;
  constexpr uint64_t kNodeId = 1001;
  const CudaGraphAnnotationRegistry::Token token =
      registry
          .TrackGraph(kGraphId,
                      CudaGraphAnnotationRegistry::TrackMode::kNewGraph)
          .token;
  ASSERT_TRUE(registry.RegisterNodeAnnotation(
      kGraphId, token, kNodeId,
      "XlaModule:#hlo_module=test_mod#::fusion.1:#hlo_op=fusion.1#"));

  std::unique_ptr<CuptiTraceCollector> collector = CreateSingleGpuCollector();
  collector->AddEvent(MakeGraphNodeKernelEvent(kGraphId, kNodeId));
  collector->AddEvent(MakeGraphNodeKernelEvent(kGraphId, kNodeId + 1,
                                               /*start_time_ns=*/210,
                                               /*end_time_ns=*/250));
  XSpace space;
  collector->Export(&space, /*end_gpu_ns=*/400);
  EXPECT_THAT(space.warnings(),
              Not(Contains(HasSubstr("CUDA graph node records"))));

  registry.ClearForTesting();
}

}  // namespace
}  // namespace profiler
}  // namespace xla
