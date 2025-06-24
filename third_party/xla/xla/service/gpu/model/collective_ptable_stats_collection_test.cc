/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/model/collective_ptable_stats_collection.h"

#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"

namespace xla::gpu {
namespace {

constexpr const char* kFile = "profiles.pbtxt";

DeviceHloInstructionProfiles TestProfiles(
    const se::DeviceDescription& device_info) {
  DeviceHloInstructionProfiles profiles;
  HloInstructionProfileList profile_list;
  HloInstructionProfile profile_entry;

  // Create a simple AllReduce instruction w/ shape f32[1024] running on a
  // single host (8 devices).
  HloInstructionProto instr;
  *instr.mutable_opcode() = HloOpcodeString(HloOpcode::kAllReduce);
  Shape shape = ShapeUtil::MakeShape(F32, {1024});
  *instr.mutable_shape() = shape.ToProto();
  instr.set_constrain_layout(false);
  instr.set_use_global_device_ids(true);
  instr.set_channel_id(1);
  IotaReplicaGroupList iota(/*num_replica_groups=*/1,
                            /*num_devices_per_group=*/8);
  CollectiveDeviceList collective_device_list(iota);
  *instr.mutable_collective_device_list() = collective_device_list.ToProto();

  *profile_entry.mutable_instruction() = std::move(instr);
  profile_entry.set_network_throughput_bytes_per_sec(4 * 1024);

  *profile_list.add_entries() = std::move(profile_entry);
  profiles.mutable_entries()->insert(
      {HloOpProfiles::GetProfileName(device_info), std::move(profile_list)});

  return profiles;
}

class CollectivePerfTableStatsCollectionTest
    : public HloHardwareIndependentTestBase {
 public:
  explicit CollectivePerfTableStatsCollectionTest()
      : device_info_(TestGpuDeviceInfo::RTXA6000DeviceInfo(
            se::CudaComputeCapability(9, 0))),
        profiles_path_(tsl::io::JoinPath(tsl::testing::TmpDir(), kFile)) {}

  void SetUp() override {
    TF_ASSERT_OK(tsl::WriteTextProto(tsl::Env::Default(), profiles_path_,
                                     TestProfiles(device_info_)));
  }

 protected:
  const se::DeviceDescription device_info_;
  const std::string profiles_path_;
};

TEST_F(CollectivePerfTableStatsCollectionTest,
       CollectsCollectivePerfTableData) {
  constexpr absl::string_view kHloText = R"(
  HloModule m

  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT _ = f32[] add(x, y)
  }

  ENTRY ar {
    p0 = f32[1024] parameter(0)

    ar-start = f32[1024] all-reduce-start(p0), to_apply=add,
      replica_groups=[1,8]<=[8]
    ROOT ar-done = f32[1024] all-reduce-done(ar-start)
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, CollectivePerfTableStatsCollection(
                                            profiles_path_, device_info_)
                                            .Run(module.get()));

  VLOG(1) << module->ToString();

  EXPECT_FALSE(changed);
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
  CHECK: ar-start
  CHECK-SAME: "exec_time_us":1000000
  )"));
}

}  // namespace
}  // namespace xla::gpu
