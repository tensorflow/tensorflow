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
#include "xla/backends/gpu/runtime/all_gather.h"
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

class BuildAllGatherInfoTest : public HloHardwareIndependentTestBase {
 protected:
  // Builds an all-gather HLO module and invokes BuildAllGatherInfo.
  //
  // `num_elements` is the number of elements in the input operand (per
  // replica). The all-gather is 1-D along dimension 0.
  //
  // `replica_groups` is a flat list of device IDs forming a single replica
  // group. An empty list means replica_groups={} (empty) in the HLO, which
  // will trigger the "replica groups must be provided" error path.
  absl::StatusOr<AllGatherInfo> BuildInfo(PrimitiveType element_type,
                                          std::vector<int64_t> input_dims,
                                          std::vector<int32_t> replica_groups,
                                          int num_partitions = 1,
                                          int active_links = 18) {
    const int num_replicas =
        replica_groups.empty() ? 1 : static_cast<int>(replica_groups.size());
    const std::string element_type_str =
        primitive_util::LowercasePrimitiveTypeName(element_type);
    std::vector<int64_t> output_dims = input_dims;
    output_dims[0] *= num_replicas;
    const std::string input_shape_str = absl::StrJoin(input_dims, ",");
    const std::string output_shape_str = absl::StrJoin(output_dims, ",");
    const std::string replica_groups_str =
        replica_groups.empty()
            ? ""
            : absl::StrFormat("{%s}", absl::StrJoin(replica_groups, ","));

    // The all-gather gathers along dimension 0: output_dim0 = num_replicas *
    // input_dim0.
    constexpr absl::string_view kModuleStr = R"(
    HloModule test
    ENTRY test_computation {
      param_0 = %1$s[%2$s] parameter(0)
      ROOT all-gather = %1$s[%3$s] all-gather(param_0),
          dimensions={0}, replica_groups={%4$s}
    }
    )";
    const std::string module_str =
        absl::StrFormat(kModuleStr, element_type_str, input_shape_str,
                        output_shape_str, replica_groups_str);

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

    // Each partition is a single host (i.e. a separate NVLink domain) with
    // num_devices_per_host = num_replicas / num_partitions (for single group).
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
                                                        HloOpcode::kAllGather);
    return BuildAllGatherInfo(gpu_topology,
                              Cast<HloAllGatherInstruction>(hlo_instr),
                              /*device_assignment=*/nullptr);
  }

  absl::StatusOr<AllGatherInfo> BuildInfo(PrimitiveType element_type,
                                          int64_t num_elements,
                                          std::vector<int32_t> replica_groups,
                                          int num_partitions = 1,
                                          int active_links = 18) {
    return BuildInfo(element_type, std::vector<int64_t>{num_elements},
                     std::move(replica_groups), num_partitions, active_links);
  }
};

TEST_F(BuildAllGatherInfoTest, SucceedsForSupportedF32) {
  EXPECT_THAT(
      BuildInfo(F32, /*num_elements=*/512, /*replica_groups=*/{0, 1}),
      IsOkAndHolds(AllOf(Field(&AllGatherInfo::num_devices, 2),
                         Field(&AllGatherInfo::max_output_elements, 1024))));
}

TEST_F(BuildAllGatherInfoTest, SucceedsForSupportedBF16) {
  EXPECT_THAT(
      BuildInfo(BF16, /*num_elements=*/512, /*replica_groups=*/{0, 1}),
      IsOkAndHolds(AllOf(Field(&AllGatherInfo::num_devices, 2),
                         Field(&AllGatherInfo::max_output_elements, 1024))));
}

TEST_F(BuildAllGatherInfoTest, SucceedsForFourDevices) {
  EXPECT_THAT(
      BuildInfo(F32, /*num_elements=*/512, /*replica_groups=*/{0, 1, 2, 3}),
      IsOkAndHolds(AllOf(Field(&AllGatherInfo::num_devices, 4),
                         Field(&AllGatherInfo::max_output_elements, 2048))));
}

TEST_F(BuildAllGatherInfoTest, FailsForSingleDevicePerReplicaGroup) {
  EXPECT_THAT(
      BuildInfo(F32, /*num_elements=*/512, /*replica_groups=*/{0}),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("Degenerate collectives with a single device")));
}

TEST_F(BuildAllGatherInfoTest, FailsForNonPowerOfTwoDevices) {
  EXPECT_THAT(
      BuildInfo(F32, /*num_elements=*/512, /*replica_groups=*/{0, 1, 2}),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("only supported for power of 2")));
}

TEST_F(BuildAllGatherInfoTest, SucceedsForSupportedF8E4M3FN) {
  EXPECT_THAT(
      BuildInfo(F8E4M3FN, /*num_elements=*/512, /*replica_groups=*/{0, 1}),
      IsOkAndHolds(AllOf(Field(&AllGatherInfo::num_devices, 2),
                         Field(&AllGatherInfo::max_output_elements, 1024))));
}

TEST_F(BuildAllGatherInfoTest, FailsForUnsupportedType) {
  // Sub-byte types such as S4 are not supported by the Triton all-gather
  // kernel.
  EXPECT_THAT(
      BuildInfo(S4, /*num_elements=*/512, /*replica_groups=*/{0, 1}),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("is not supported for the all-gather kernel")));
}

TEST_F(BuildAllGatherInfoTest, FailsForNonPowerOfTwoGatherDimSize) {
  EXPECT_THAT(BuildInfo(F32, /*num_elements=*/7, /*replica_groups=*/{0, 1}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("per-rank size along the gather dimension to "
                                 "be a power of 2")));
  EXPECT_THAT(BuildInfo(S32, /*input_dims=*/{3, 4}, /*replica_groups=*/{0, 1}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("per-rank size along the gather dimension to "
                                 "be a power of 2")));
}

TEST_F(BuildAllGatherInfoTest,
       SucceedsForSubTransactionAndUnalignedNonGatherDim) {
  // s32[2] = 8 bytes (< 16-byte transaction).
  EXPECT_THAT(
      BuildInfo(S32, /*num_elements=*/2, /*replica_groups=*/{0, 1}),
      IsOkAndHolds(AllOf(Field(&AllGatherInfo::num_devices, 2),
                         Field(&AllGatherInfo::max_output_elements, 4))));
  // s32[2, 7] = 56 bytes (gather dim = 2, non-gather dim = 7).
  EXPECT_THAT(
      BuildInfo(S32, /*input_dims=*/{2, 7}, /*replica_groups=*/{0, 1}),
      IsOkAndHolds(AllOf(Field(&AllGatherInfo::num_devices, 2),
                         Field(&AllGatherInfo::max_output_elements, 28))));
  // s32[2, 529] = 4,232 bytes (gather dim = 2, non-gather dim = 529).
  EXPECT_THAT(
      BuildInfo(S32, /*input_dims=*/{2, 529}, /*replica_groups=*/{0, 1}),
      IsOkAndHolds(AllOf(Field(&AllGatherInfo::num_devices, 2),
                         Field(&AllGatherInfo::max_output_elements, 2116))));
}

TEST_F(BuildAllGatherInfoTest, FailsIfReplicaGroupsEmpty) {
  EXPECT_THAT(
      BuildInfo(F32, /*num_elements=*/512, /*replica_groups=*/{}),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("Replica groups must be explicitly provided")));
}

TEST_F(BuildAllGatherInfoTest, FailsForCrossHostCollective) {
  // 2 replicas split across 2 partitions → not local → should be rejected.
  EXPECT_THAT(BuildInfo(F32, /*num_elements=*/512, /*replica_groups=*/{0, 1},
                        /*num_partitions=*/2),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Cross-host symmetric memory collectives")));
}

TEST_F(BuildAllGatherInfoTest, FailsWithoutNvlink) {
  // Devices with no active NVLink/UALink connections cannot use symmetric
  // memory collectives.
  EXPECT_THAT(
      BuildInfo(F32, /*num_elements=*/512, /*replica_groups=*/{0, 1},
                /*num_partitions=*/1, /*active_links=*/0),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("NVLink/UALink")));
}

TEST_F(BuildAllGatherInfoTest, FailsForLargeOutputs) {
  // 4 * 1024 * 1024 F32 elements * 2 devices = 32 MB > 16 MB limit ->
  // unimplemented.
  EXPECT_THAT(BuildInfo(F32, /*num_elements=*/4 * 1024 * 1024,
                        /*replica_groups=*/{0, 1}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("only supported for small outputs")));
}

TEST_F(BuildAllGatherInfoTest, SucceedsFor16MBOutput) {
  // 2 * 1024 * 1024 F32 elements * 2 devices = 16 MB == 16 MB limit -> ok.
  EXPECT_THAT(BuildInfo(F32, /*num_elements=*/2 * 1024 * 1024,
                        /*replica_groups=*/{0, 1}),
              IsOkAndHolds(AllOf(Field(&AllGatherInfo::num_devices, 2),
                                 Field(&AllGatherInfo::max_output_elements,
                                       4 * 1024 * 1024))));
}

TEST_F(BuildAllGatherInfoTest, VariadicChecksSizePerOperandAndUsesMaxElements) {
  // Each operand's output is <= 16 MB (16 MB and 8 MB), while the combined
  // output is 24 MB. Since each operand tile is processed sequentially, the
  // byte limit applies per operand and `max_output_elements` is the maximum
  // across operands.
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = f32[2097152] parameter(0)
    param_1 = f32[1048576] parameter(1)
    ROOT all-gather = (f32[4194304], f32[2097152]) all-gather(param_0, param_1),
        dimensions={0}, replica_groups={{0,1}}
  }
  )";
  se::DeviceDescription device_info = TestGpuDeviceInfo::H100SXMDeviceInfo();
  stream_executor::GpuTargetConfigProto target_config_proto;
  *target_config_proto.mutable_gpu_device_info() = device_info.ToProto();
  target_config_proto.mutable_gpu_device_info()
      ->mutable_device_interconnect_info()
      ->set_active_links(18);
  target_config_proto.set_platform_name("CUDA");
  ASSERT_OK_AND_ASSIGN(gpu::GpuTargetConfig target_config,
                       gpu::GpuTargetConfig::FromProto(target_config_proto));
  GpuTopology gpu_topology("platform_version", /*num_partitions=*/1,
                           /*num_hosts_per_partition=*/1,
                           /*num_devices_per_host=*/2, target_config);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleStr, 2));
  const HloInstruction* hlo_instr =
      HloHardwareIndependentTestBase::FindInstruction(module.get(),
                                                      HloOpcode::kAllGather);
  EXPECT_THAT(
      BuildAllGatherInfo(gpu_topology, Cast<HloAllGatherInstruction>(hlo_instr),
                         /*device_assignment=*/nullptr),
      IsOkAndHolds(
          AllOf(Field(&AllGatherInfo::num_devices, 2),
                Field(&AllGatherInfo::max_output_elements, 4 * 1024 * 1024))));
}

TEST_F(BuildAllGatherInfoTest, FailsForMoreThanMaxOperands) {
  std::vector<std::string> params;
  std::vector<std::string> param_names;
  std::vector<std::string> output_shapes;
  for (int i = 0; i <= kAllGatherMaxNumOperands; ++i) {
    params.push_back(absl::StrFormat("param_%d = f32[8] parameter(%d)", i, i));
    param_names.push_back(absl::StrFormat("param_%d", i));
    output_shapes.push_back("f32[16]");
  }
  const std::string module_str = absl::StrFormat(
      R"(
  HloModule test
  ENTRY test_computation {
    %s
    ROOT all-gather = (%s) all-gather(%s),
        dimensions={0}, replica_groups={{0,1}}
  }
  )",
      absl::StrJoin(params, "\n"), absl::StrJoin(output_shapes, ", "),
      absl::StrJoin(param_names, ", "));
  se::DeviceDescription device_info = TestGpuDeviceInfo::H100SXMDeviceInfo();
  stream_executor::GpuTargetConfigProto target_config_proto;
  *target_config_proto.mutable_gpu_device_info() = device_info.ToProto();
  target_config_proto.mutable_gpu_device_info()
      ->mutable_device_interconnect_info()
      ->set_active_links(18);
  target_config_proto.set_platform_name("CUDA");
  ASSERT_OK_AND_ASSIGN(gpu::GpuTargetConfig target_config,
                       gpu::GpuTargetConfig::FromProto(target_config_proto));
  GpuTopology gpu_topology("platform_version", /*num_partitions=*/1,
                           /*num_hosts_per_partition=*/1,
                           /*num_devices_per_host=*/2, target_config);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str, 2));
  const HloInstruction* hlo_instr =
      HloHardwareIndependentTestBase::FindInstruction(module.get(),
                                                      HloOpcode::kAllGather);
  EXPECT_THAT(
      BuildAllGatherInfo(gpu_topology, Cast<HloAllGatherInstruction>(hlo_instr),
                         /*device_assignment=*/nullptr),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("supports at most 16 operands")));
}

TEST_F(BuildAllGatherInfoTest,
       CreateAllGatherKernelSpecMatchesAllReduceArgumentLayout) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = f32[512] parameter(0)
    ROOT all-gather = f32[1024] all-gather(param_0),
        dimensions={0}, replica_groups={{0,1}}
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleStr, 2));
  const HloInstruction* instr = HloHardwareIndependentTestBase::FindInstruction(
      module.get(), HloOpcode::kAllGather);
  ASSERT_OK_AND_ASSIGN(
      CollectiveKernelSpec spec,
      CreateAllGatherKernelSpec(instr, LaunchDimensions(4, 128),
                                SymmetricMemoryType::kXlaRendezvous));
  EXPECT_FALSE(spec.codegen_config.copy_input_to_scratch);
  ASSERT_EQ(spec.codegen_config.argument_descriptors.size(), 6);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[0].type,
            KernelArgType::kInputBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[0].index, 0);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[1].type,
            KernelArgType::kOutputBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[1].index, 0);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[2].type,
            KernelArgType::kRuntimeRank);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[3].type,
            KernelArgType::kInvocationCount);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[4].type,
            KernelArgType::kScratchBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[4].index, 0);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[5].type,
            KernelArgType::kScratchBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[5].index, 1);
}

}  // namespace
}  // namespace xla::gpu
