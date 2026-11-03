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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/all_to_all.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu_topology.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::Field;
using ::testing::HasSubstr;

class BuildAllToAllInfoTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<AllToAllInfo> BuildInfo(PrimitiveType element_type,
                                         std::vector<int64_t> input_dims,
                                         std::vector<int32_t> replica_groups,
                                         int num_partitions = 1,
                                         int active_links = 18) {
    const int num_replicas =
        replica_groups.empty() ? 1 : static_cast<int>(replica_groups.size());
    const std::string element_type_str =
        primitive_util::LowercasePrimitiveTypeName(element_type);
    const std::string shape_str = absl::StrJoin(input_dims, ",");
    const std::string replica_groups_str =
        replica_groups.empty()
            ? ""
            : absl::StrFormat("{%s}", absl::StrJoin(replica_groups, ","));

    constexpr absl::string_view kModuleStr = R"(
    HloModule test
    ENTRY test_computation {
      param_0 = %1$s[%2$s] parameter(0)
      ROOT all-to-all = %1$s[%2$s] all-to-all(param_0),
          dimensions={0}, replica_groups={%3$s}
    }
    )";
    const std::string module_str = absl::StrFormat(
        kModuleStr, element_type_str, shape_str, replica_groups_str);

    SCOPED_TRACE(testing::Message() << "module_str: " << module_str);

    se::DeviceDescription device_info = TestGpuDeviceInfo::H100SXMDeviceInfo();
    stream_executor::GpuTargetConfigProto target_config_proto;
    *target_config_proto.mutable_gpu_device_info() = device_info.ToProto();
    target_config_proto.mutable_gpu_device_info()
        ->mutable_device_interconnect_info()
        ->set_active_links(active_links);
    target_config_proto.set_platform_name("CUDA");
    ABSL_ASSIGN_OR_RETURN(gpu::GpuTargetConfig target_config,
                     gpu::GpuTargetConfig::FromProto(target_config_proto));

    const int num_devices_per_host =
        num_replicas > 0 ? num_replicas / num_partitions : 1;
    GpuTopology gpu_topology("platform_version", num_partitions,
                             /*num_hosts_per_partition=*/1,
                             /*num_devices_per_host=*/num_devices_per_host,
                             target_config);

    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                     ParseAndReturnVerifiedModule(module_str, num_replicas));

    const HloInstruction* hlo_instr =
        HloHardwareIndependentTestBase::FindInstruction(module.get(),
                                                        HloOpcode::kAllToAll);
    return BuildAllToAllInfo(gpu_topology,
                             Cast<HloAllToAllInstruction>(hlo_instr),
                             /*device_assignment=*/nullptr);
  }

  absl::StatusOr<AllToAllInfo> BuildInfo(PrimitiveType element_type,
                                         int64_t num_elements,
                                         std::vector<int32_t> replica_groups,
                                         int num_partitions = 1,
                                         int active_links = 18) {
    return BuildInfo(element_type, std::vector<int64_t>{num_elements},
                     std::move(replica_groups), num_partitions, active_links);
  }
};

TEST_F(BuildAllToAllInfoTest, SucceedsForSupportedF32) {
  EXPECT_THAT(BuildInfo(F32, /*num_elements=*/512,
                        /*replica_groups=*/{0, 1}),
              IsOkAndHolds(AllOf(Field(&AllToAllInfo::num_devices, 2),
                                 Field(&AllToAllInfo::num_elements, 512),
                                 Field(&AllToAllInfo::element_type, F32))));
}

TEST_F(BuildAllToAllInfoTest, SucceedsForSupportedBF16) {
  EXPECT_THAT(BuildInfo(BF16,
                        /*num_elements=*/512, /*replica_groups=*/{0, 1}),
              IsOkAndHolds(AllOf(Field(&AllToAllInfo::num_devices, 2),
                                 Field(&AllToAllInfo::num_elements, 512),
                                 Field(&AllToAllInfo::element_type, BF16))));
}

TEST_F(BuildAllToAllInfoTest, SucceedsForFourDevices) {
  EXPECT_THAT(BuildInfo(F32, /*num_elements=*/512,
                        /*replica_groups=*/{0, 1, 2, 3}),
              IsOkAndHolds(AllOf(Field(&AllToAllInfo::num_devices, 4),
                                 Field(&AllToAllInfo::num_elements, 512),
                                 Field(&AllToAllInfo::element_type, F32))));
}

TEST_F(BuildAllToAllInfoTest, FailsForSingleDevicePerReplicaGroup) {
  EXPECT_THAT(
      BuildInfo(F32, /*num_elements=*/512, /*replica_groups=*/{0}),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("Degenerate collectives with a single device")));
}

TEST_F(BuildAllToAllInfoTest, FailsForNonPowerOfTwoDevices) {
  EXPECT_THAT(BuildInfo(F32, /*num_elements=*/768,
                        /*replica_groups=*/{0, 1, 2}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("only supported for power of 2")));
}

TEST_F(BuildAllToAllInfoTest, FailsForUnsupportedType) {
  EXPECT_THAT(
      BuildInfo(S4,
                /*num_elements=*/512, /*replica_groups=*/{0, 1}),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("is not supported for the all-to-all kernel")));
}

TEST_F(BuildAllToAllInfoTest, FailsForNonPowerOfTwoSplitDimSize) {
  EXPECT_THAT(BuildInfo(F32, /*num_elements=*/6,
                        /*replica_groups=*/{0, 1}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("per-rank size along the split dimension to "
                                 "be a power of 2")));
}

TEST_F(BuildAllToAllInfoTest, FailsForLargeOutputs) {
  // For 2 devices, max per-device size is 256 MB / 2 = 128 MB.
  // 64 * 1024 * 1024 F32 elements = 256 MB > 128 MB limit -> unimplemented.
  EXPECT_THAT(BuildInfo(F32,
                        /*num_elements=*/64LL * 1024 * 1024,
                        /*replica_groups=*/{0, 1}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("only supported for small outputs")));
}

TEST_F(BuildAllToAllInfoTest, SucceedsForMaxOutput) {
  // For 2 devices, max per-device size is 256 MB / 2 = 128 MB.
  // 32 * 1024 * 1024 F32 elements = 128 MB == 128 MB limit -> ok.
  EXPECT_THAT(
      BuildInfo(F32,
                /*num_elements=*/32LL * 1024 * 1024,
                /*replica_groups=*/{0, 1}),
      IsOkAndHolds(AllOf(Field(&AllToAllInfo::num_devices, 2),
                         Field(&AllToAllInfo::num_elements, 32LL * 1024 * 1024),
                         Field(&AllToAllInfo::element_type, F32),
                         Field(&AllToAllInfo::all_to_all_strategy,
                               AllToAllStrategy::kTwoShot))));
}

TEST_F(BuildAllToAllInfoTest,
       CreateAllToAllKernelSpecUsesDoubleBufferedScratchForOneShot) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = f32[1024] parameter(0)
    ROOT all-to-all = f32[1024] all-to-all(param_0),
        dimensions={0}, replica_groups={{0,1}}
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleStr, 2));
  const HloInstruction* instr = HloHardwareIndependentTestBase::FindInstruction(
      module.get(), HloOpcode::kAllToAll);
  ASSERT_OK_AND_ASSIGN(
      CollectiveKernelSpec spec,
      CreateAllToAllKernelSpec(instr, LaunchDimensions(4, 128),
                               SymmetricMemoryType::kXlaRendezvous));
  EXPECT_FALSE(spec.codegen_config.copy_input_to_scratch);
  ASSERT_EQ(spec.codegen_config.output_buffer_specs.size(), 1);
  EXPECT_EQ(spec.codegen_config.output_buffer_specs[0].symmetric_memory_type,
            SymmetricMemoryType::kNone);
  EXPECT_EQ(spec.codegen_config.sync_count_increment, 1u);
  ASSERT_EQ(spec.scratch_buffers.size(), 2);
  ASSERT_EQ(spec.codegen_config.argument_descriptors.size(), 6);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[0].type,
            KernelArgType::kInputBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[1].type,
            KernelArgType::kOutputBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[2].type,
            KernelArgType::kRuntimeRank);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[3].type,
            KernelArgType::kInvocationCount);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[4].type,
            KernelArgType::kScratchBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[5].type,
            KernelArgType::kScratchBuffer);
}

TEST_F(BuildAllToAllInfoTest,
       CreateAllToAllKernelSpecUsesSingleBufferedScratchForTwoShot) {
  // For 2 devices, one-shot threshold is 32 MB / 2 = 16 MB.
  // 8388608 F32 elements = 32 MB > 16 MB -> two-shot.
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = f32[8388608] parameter(0)
    ROOT all-to-all = f32[8388608] all-to-all(param_0),
        dimensions={0}, replica_groups={{0,1}}
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleStr, 2));
  const HloInstruction* instr = HloHardwareIndependentTestBase::FindInstruction(
      module.get(), HloOpcode::kAllToAll);
  ASSERT_OK_AND_ASSIGN(
      CollectiveKernelSpec spec,
      CreateAllToAllKernelSpec(instr, LaunchDimensions(4, 128),
                               SymmetricMemoryType::kXlaRendezvous));
  EXPECT_FALSE(spec.codegen_config.copy_input_to_scratch);
  ASSERT_EQ(spec.codegen_config.output_buffer_specs.size(), 1);
  EXPECT_EQ(spec.codegen_config.output_buffer_specs[0].symmetric_memory_type,
            SymmetricMemoryType::kNone);
  EXPECT_EQ(spec.codegen_config.sync_count_increment, 2u);
  ASSERT_EQ(spec.scratch_buffers.size(), 2);
  EXPECT_FALSE(spec.scratch_buffers[1].should_double_buffer);
  ASSERT_EQ(spec.codegen_config.argument_descriptors.size(), 6);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[5].type,
            KernelArgType::kScratchBuffer);
}

}  // namespace
}  // namespace xla::gpu
