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

#include "tensorflow/tsl/profiler/convert/xplane_to_profile_instructions.h"

#include <string>

#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/profiler/protobuf/profiled_instructions.pb.h"
#include "tensorflow/tsl/profiler/protobuf/xplane.pb.h"
#include "tensorflow/tsl/profiler/rpc/client/save_profile.h"
#include "tensorflow/tsl/profiler/utils/file_system_utils.h"
#include "tensorflow/tsl/profiler/utils/xplane_builder.h"
#include "tensorflow/tsl/profiler/utils/xplane_schema.h"

namespace tsl {
namespace profiler {
namespace {

using tensorflow::profiler::XSpace;

void CreateXSpace(XSpace* space, uint32 first_device_latency,
                  uint32 second_device_latency) {
  XPlaneBuilder host_plane(space->add_planes());
  host_plane.SetName(kHostThreadsPlaneName);
  XLineBuilder thread1 = host_plane.GetOrCreateLine(10);
  thread1.SetName("thread1");
  XEventBuilder event1 =
      thread1.AddEvent(*host_plane.GetOrCreateEventMetadata("event1"));
  event1.SetTimestampNs(150000);
  event1.SetDurationNs(10000);
  event1.AddStatValue(*host_plane.GetOrCreateStatMetadata("tf_op"),
                      *host_plane.GetOrCreateStatMetadata("Relu"));
  XLineBuilder thread2 = host_plane.GetOrCreateLine(20);
  thread2.SetName("thread2");
  XEventBuilder event2 =
      thread2.AddEvent(*host_plane.GetOrCreateEventMetadata("event2"));
  event2.SetTimestampNs(160000);
  event2.SetDurationNs(10000);
  event2.AddStatValue(*host_plane.GetOrCreateStatMetadata("tf_op"),
                      *host_plane.GetOrCreateStatMetadata("Conv2D"));

  XPlaneBuilder device_plane(space->add_planes());
  device_plane.SetName(GpuPlaneName(0));
  device_plane.SetId(0);
  XLineBuilder stream1 = device_plane.GetOrCreateLine(30);
  stream1.SetName("gpu stream 1");
  XEventBuilder event3 =
      stream1.AddEvent(*device_plane.GetOrCreateEventMetadata("kernel1"));
  event3.SetTimestampNs(180000);
  event3.SetDurationNs(first_device_latency);
  event3.AddStatValue(
      *device_plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kHloOp)),
      *device_plane.GetOrCreateStatMetadata("fusion"));

  XPlaneBuilder device_plane_2(space->add_planes());
  device_plane_2.SetName(GpuPlaneName(1));
  device_plane_2.SetId(0);
  XLineBuilder stream2 = device_plane.GetOrCreateLine(30);
  stream2.SetName("gpu stream 1");
  XEventBuilder event5 =
      stream1.AddEvent(*device_plane.GetOrCreateEventMetadata("kernel1"));
  event5.SetTimestampNs(180000);
  event5.SetDurationNs(second_device_latency);
  event5.AddStatValue(
      *device_plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kHloOp)),
      *device_plane.GetOrCreateStatMetadata("fusion"));
}

TEST(XplaneToProfiledInstructionsProtoTest,
     ConvertXplaneToProfiledInstructionsProto) {
  tensorflow::profiler::ProfiledInstructionsProto profile_proto;
  std::string logdir = testing::TmpDir() + "/logdir";
  std::string run = tsl::profiler::GetCurrentTimeStampAsString();
  const std::string path = ProfilerJoinPath(logdir, run);

  XSpace xspace_first_host;
  CreateXSpace(&xspace_first_host, 10000, 10000);
  Status status =
      tsl::profiler::SaveXSpace(logdir, run, "host_0", xspace_first_host);
  EXPECT_TRUE(status.ok());

  XSpace xspace_2nd_host;
  CreateXSpace(&xspace_2nd_host, 15000, 5000);
  status = tsl::profiler::SaveXSpace(logdir, run, "host_1", xspace_2nd_host);
  EXPECT_TRUE(status.ok());

  EXPECT_TRUE(
      ConvertXplaneToProfiledInstructionsProto(path, &profile_proto).ok());
  EXPECT_EQ(profile_proto.costs_size(), 1);
  EXPECT_EQ(profile_proto.costs(0).cost_us(), 10);
  EXPECT_EQ(profile_proto.costs(0).name(), "fusion");
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
