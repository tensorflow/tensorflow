/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/xplane_to_step_events.h"

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/utils/group_events.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

// Tests with a sample profile with two steps captured on the host but only one
// step on the device. On the host, each step consists of TraceContext ->
// FunctionRun -> ExecutorState::Process -> matmul. On the host, each step
// consists of matmul. The host's step db should be created only for the step
// observed on the host.
TEST(ConvertXPlaneToOpStats, CpuOnlyStepDbTest) {
  XSpace space;
  XPlane* host_plane = space.add_planes();
  XPlaneBuilder host_plane_builder(host_plane);
  host_plane_builder.SetName(kHostThreads);
  host_plane_builder.ReserveLines(2);

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kTraceContext,
               0, 100, {{StatType::kStepNum, 123}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               10, 90, {{StatType::kStepId, 0}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kTraceContext,
               300, 100, {{StatType::kStepNum, 456}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               310, 90, {{StatType::kStepId, 1}});

  auto tf_executor_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 20, 20,
               {{StatType::kStepId, 0}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "matmul", 30, 10,
               {{StatType::kCorrelationId, 100}, {StatType::kDeviceId, 1}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 320, 20,
               {{StatType::kStepId, 1}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "matmul", 330, 10,
               {{StatType::kCorrelationId, 200}, {StatType::kDeviceId, 1}});

  XPlane* device_plane = space.add_planes();
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.ReserveLines(1);

  auto stream = device_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&device_plane_builder, &stream, "matmul", 50, 40,
               {{StatType::kCorrelationId, 100}});

  GroupTfEvents(&space, nullptr);
  StepEvents device_step_events =
      ConvertDeviceTraceXPlaneToStepEvents(*device_plane);
  EXPECT_EQ(device_step_events.size(), 1);
  EXPECT_EQ(device_step_events[0].Events().size(), 1);
  StepEvents host_step_events = ConvertHostThreadsXPlaneToStepEvents(
      *host_plane, true, device_step_events);
  // Should contain only the step which is also present on the device.
  EXPECT_EQ(host_step_events.size(), 1);
  // TraceContext should be added as a step marker.
  EXPECT_EQ(host_step_events[0].Markers().size(), 1);
  // FunctionRun shouldn't be added.
  EXPECT_EQ(host_step_events[0].Events().size(), 2);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
