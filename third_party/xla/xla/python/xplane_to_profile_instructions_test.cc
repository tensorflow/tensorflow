/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/python/xplane_to_profile_instructions.h"

#include <cstdint>
#include <memory>
#include <string>

#include "xla/service/hlo.pb.h"
#include "xla/tests/verified_hlo_module.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/rpc/client/save_profile.h"
#include "xla/tsl/profiler/utils/file_system_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/platform/test.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace {

using tensorflow::profiler::XSpace;
using tsl::profiler::GetStatTypeStr;
using tsl::profiler::GpuPlaneName;
using tsl::profiler::kHostThreadsPlaneName;
using tsl::profiler::kMetadataPlaneName;
using tsl::profiler::StatType;
using tsl::profiler::XEventBuilder;
using tsl::profiler::XLineBuilder;
using tsl::profiler::XPlaneBuilder;

void CreateXSpace(XSpace* space, int first_device_latency,
                  int second_device_latency) {
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

  int64_t program_id = 1;
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
      *device_plane.GetOrCreateStatMetadata("custom-call"));
  event3.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                          GetStatTypeStr(StatType::kHloModule)),
                      *device_plane.GetOrCreateStatMetadata("test_module"));
  event3.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                          GetStatTypeStr(StatType::kProgramId)),
                      program_id);

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
      *device_plane.GetOrCreateStatMetadata("custom-call"));
  event5.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                          GetStatTypeStr(StatType::kHloModule)),
                      *device_plane.GetOrCreateStatMetadata("test_module"));
  event5.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                          GetStatTypeStr(StatType::kProgramId)),
                      program_id);
}

void CreateXSpaceWithFingerprint(XSpace* space, int first_device_latency,
                                 int second_device_latency) {
  XPlaneBuilder metadata_plane(space->add_planes());
  metadata_plane.SetName(kMetadataPlaneName);
  const char* hlo_text = R"(
  HloModule test_module
  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY ar {
    p0 = f32[32] parameter(0)
    p1 = f32[32, 32] parameter(1)
    p2 = f32[32, 32] parameter(2)
    p3 = f32[32] parameter(3)

    dot0 = f32[32,32]{1,0} custom-call(p1, p2), custom_call_target="__cublas$gemm"
    dot1 = f32[32,32]{1,0} custom-call(dot0, p2), custom_call_target="__cublas$gemm"
    dot2 = f32[32,32]{1,0} custom-call(dot1, p2), custom_call_target="__cublas$gemm"
    dot3 = f32[32,32]{1,0} custom-call(dot2, p2), custom_call_target="__cublas$gemm"
    dot4 = f32[32,32]{1,0} custom-call(dot3, p2), custom_call_target="__cublas$gemm"
    dot5 = f32[32,32]{1,0} custom-call(dot4, p2), custom_call_target="__cublas$gemm"
    dot6 = f32[32,32]{1,0} custom-call(dot5, p2), custom_call_target="__cublas$gemm"

    ar-start = f32[32] all-reduce-start(p0), to_apply=apply_op
    ar-done = f32[32] all-reduce-done(ar-start)

    %ag-start = (f32[32], f32[64]) all-gather-start(p3), dimensions={0}
    %ag-done = f32[64] all-gather-done(%ag-start)

    add0 = f32[32,32] add(dot0, dot1)
    add1 = f32[32,32] add(add0, dot2)
    add2 = f32[32,32] add(add1, dot3)
    add3 = f32[32,32] add(add2, dot4)
    add4 = f32[32,32] add(add3, dot5)
    add5 = f32[32,32] add(add4, dot6)

    ROOT t = (f32[32], f32[64], f32[32,32]) tuple(ar-done, %ag-done, add5)
  })";

  xla::HloModuleConfig config;
  auto module = std::make_unique<VerifiedHloModule>(
      "test_module", config, /*verifier_layout_sensitive=*/false,
      /*allow_mixed_precision_in_hlo_verifier=*/true,
      ShapeUtil::ByteSizeOfElements);
  if (module->ParseHloStringAndVerifyModule(hlo_text).ok()) {
    HloInstruction* root = module->entry_computation()->root_instruction();
    FrontendAttributes attributes;
    (*attributes.mutable_map())["fingerprint_before_lhs"] = "08a5";
    root->add_frontend_attributes(attributes);
    xla::HloModuleProto hlo_module_proto = module->ToProto();
    hlo_module_proto.set_id(1);
    xla::HloProto hlo_proto;
    *hlo_proto.mutable_hlo_module() = hlo_module_proto;
    int64_t program_id = 1;
    tsl::profiler::XEventMetadata* event_metadata =
        metadata_plane.GetOrCreateEventMetadata(program_id);
    event_metadata->set_name(tsl::profiler::HloModuleNameWithProgramId(
        hlo_proto.hlo_module().name(), program_id));
    tsl::profiler::XStatsBuilder<tsl::profiler::XEventMetadata> event_stats(
        event_metadata, &metadata_plane);
    auto* hlo_proto_stat = metadata_plane.GetOrCreateStatMetadata(
        GetStatTypeStr(tsl::profiler::StatType::kHloProto));
    event_stats.AddStatValue(*hlo_proto_stat, hlo_proto);
  }

  return CreateXSpace(space, first_device_latency, second_device_latency);
}

TEST(XplaneToProfiledInstructionsProtoTest,
     ConvertXplaneUnderLogdirToProfiledInstructionsProto) {
  tensorflow::profiler::ProfiledInstructionsProto profile_proto;
  std::string logdir = testing::TempDir() + "/logdir";
  std::string run = tsl::profiler::GetCurrentTimeStampAsString();
  const std::string path = tsl::profiler::ProfilerJoinPath(logdir, run);

  XSpace xspace_first_host;
  CreateXSpace(&xspace_first_host, 10000, 10000);
  auto status =
      tsl::profiler::SaveXSpace(logdir, run, "host_0", xspace_first_host);
  EXPECT_TRUE(status.ok());

  XSpace xspace_2nd_host;
  CreateXSpace(&xspace_2nd_host, 15000, 5000);
  status = tsl::profiler::SaveXSpace(logdir, run, "host_1", xspace_2nd_host);
  EXPECT_TRUE(status.ok());

  EXPECT_TRUE(
      ConvertXplaneUnderLogdirToProfiledInstructionsProto(path, &profile_proto)
          .ok());
  EXPECT_EQ(profile_proto.costs_size(), 1);
  EXPECT_EQ(profile_proto.costs(0).cost_us(), 10);
  EXPECT_EQ(profile_proto.costs(0).name(), "custom-call");
}

TEST(XplaneToProfiledInstructionsProtoTest,
     ConvertXplaneUnderLogdirToProfiledInstructionsProtoWithFingerprint) {
  tensorflow::profiler::ProfiledInstructionsProto profile_proto;
  std::string logdir = testing::TempDir() + "/logdir";
  std::string run = tsl::profiler::GetCurrentTimeStampAsString();
  const std::string path = tsl::profiler::ProfilerJoinPath(logdir, run);

  XSpace xspace_first_host;
  CreateXSpaceWithFingerprint(&xspace_first_host, 10000, 10000);
  auto status =
      tsl::profiler::SaveXSpace(logdir, run, "host_0", xspace_first_host);
  EXPECT_TRUE(status.ok());

  XSpace xspace_2nd_host;
  CreateXSpaceWithFingerprint(&xspace_2nd_host, 15000, 5000);
  status = tsl::profiler::SaveXSpace(logdir, run, "host_1", xspace_2nd_host);
  EXPECT_TRUE(status.ok());

  EXPECT_TRUE(
      ConvertXplaneUnderLogdirToProfiledInstructionsProto(path, &profile_proto)
          .ok());
  EXPECT_EQ(profile_proto.costs_size(), 1);
  EXPECT_EQ(profile_proto.costs(0).cost_us(), 10);
  EXPECT_EQ(profile_proto.costs(0).name(), "08a5::custom-call");
}

TEST(XplaneToProfiledInstructionsProtoTest,
     ConvertXplaneToProfiledInstructionsProto) {
  tensorflow::profiler::ProfiledInstructionsProto profile_proto;

  XSpace xspace_a;
  CreateXSpace(&xspace_a, 10000, 10000);

  XSpace xspace_b;
  CreateXSpace(&xspace_b, 15000, 5000);

  EXPECT_TRUE(ConvertXplaneToProfiledInstructionsProto({xspace_a, xspace_b},
                                                       &profile_proto)
                  .ok());
  EXPECT_EQ(profile_proto.costs_size(), 1);
  EXPECT_EQ(profile_proto.costs(0).cost_us(), 10);
  EXPECT_EQ(profile_proto.costs(0).name(), "custom-call");
}

TEST(XplaneToProfiledInstructionsProtoTest,
     ConvertXplaneToProfiledInstructionsProtoWithFingerprint) {
  tensorflow::profiler::ProfiledInstructionsProto profile_proto;

  XSpace xspace_a;
  CreateXSpaceWithFingerprint(&xspace_a, 10000, 10000);

  XSpace xspace_b;
  CreateXSpaceWithFingerprint(&xspace_b, 15000, 5000);

  EXPECT_TRUE(ConvertXplaneToProfiledInstructionsProto({xspace_a, xspace_b},
                                                       &profile_proto)
                  .ok());
  EXPECT_EQ(profile_proto.costs_size(), 1);
  EXPECT_EQ(profile_proto.costs(0).cost_us(), 10);
  EXPECT_EQ(profile_proto.costs(0).name(), "08a5::custom-call");
}

}  // namespace
}  // namespace xla
