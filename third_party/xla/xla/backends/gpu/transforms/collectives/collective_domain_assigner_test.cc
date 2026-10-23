/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/collectives/collective_domain_assigner.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu_topology.h"

namespace xla::gpu {
namespace {

void ExpectFileCheck(absl::string_view input, absl::string_view pattern) {
  ASSERT_OK_AND_ASSIGN(bool matches, RunFileCheck(std::string(input), pattern));
  EXPECT_TRUE(matches);
}

class CollectiveDomainAssignerTest : public HloHardwareIndependentTestBase {
 protected:
  void EnableAssignment(HloModule& module) {
    module.mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_collective_domain_assignment("scale_up_fabric");
  }

  absl::StatusOr<bool> RunAssigner(HloModule& module) {
    return RunHloPass(CollectiveDomainAssigner(gpu_topology_), &module);
  }

  GpuTopology gpu_topology_{"", /*num_partitions=*/2,
                            /*num_hosts_per_partition=*/1,
                            /*num_devices_per_host=*/8};
};

TEST_F(CollectiveDomainAssignerTest, AssignsDomainFromReplicaGroups) {
  constexpr absl::string_view kHlo = R"(
  HloModule m, replica_count=16

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT sum = f32[] add(lhs, rhs)
  }

  ENTRY main {
    p0 = f32[8] parameter(0)
    local = f32[8] all-reduce(p0), to_apply=add,
        replica_groups={{0,1,2,3,4,5,6,7},{8,9,10,11,12,13,14,15}}
    cross = f32[8] all-reduce(p0), to_apply=add,
        replica_groups={{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15}}
    ROOT result = (f32[8], f32[8]) tuple(local, cross)
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  module->mutable_config().reset_static_device_assignment();
  EnableAssignment(*module);

  ASSERT_OK_AND_ASSIGN(bool changed, RunAssigner(*module));
  EXPECT_TRUE(changed);
  const absl::string_view expected_hlo = R"(
  //       CHECK: %local = {{.*}} all-reduce({{.*}})
  //  CHECK-SAME:   backend_config={{.*}}communication_domain
  //  CHECK-SAME:   COLLECTIVE_COMMUNICATION_DOMAIN_SCALE_UP_FABRIC
  //       CHECK: %cross = {{.*}} all-reduce({{.*}})
  //   CHECK-NOT:   communication_domain
  //       CHECK: ROOT %result
  )";
  ExpectFileCheck(module->ToString(), expected_hlo);
}

TEST_F(CollectiveDomainAssignerTest, AssignsExplicitCollectivesGroupCall) {
  constexpr absl::string_view kHlo = R"(
  HloModule m, replica_count=2

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT sum = f32[] add(lhs, rhs)
  }

  collectives {
    p0 = f32[8] parameter(0)
    ar = f32[8] all-reduce(p0), to_apply=add, replica_groups={{0,1}}
    ag = f32[16] all-gather(p0), dimensions={0}, replica_groups={{0,1}}
    ROOT result = (f32[8], f32[16]) tuple(ar, ag)
  }

  ENTRY main {
    p0 = f32[8] parameter(0)
    ROOT group = (f32[8], f32[16]) call(p0), to_apply=collectives,
        frontend_attributes={_collectives_group=""}
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EnableAssignment(*module);

  ASSERT_OK_AND_ASSIGN(bool changed, RunAssigner(*module));
  EXPECT_TRUE(changed);
  const absl::string_view expected_hlo = R"(
  //       CHECK: ROOT %group = {{.*}} call({{.*}})
  //  CHECK-SAME:   frontend_attributes={_collectives_group=""}
  //  CHECK-SAME:   backend_config={{.*}}communication_domain
  //  CHECK-SAME:   COLLECTIVE_COMMUNICATION_DOMAIN_SCALE_UP_FABRIC
  )";
  ExpectFileCheck(module->ToString(), expected_hlo);
}

}  // namespace
}  // namespace xla::gpu
