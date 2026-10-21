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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/reduce_scatter.h"
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

class BuildReduceScatterInfoTest : public HloHardwareIndependentTestBase {
 protected:
  // Builds a reduce-scatter HLO module and invokes BuildReduceScatterInfo.
  //
  // `output_dims` specifies the per-replica output shape after scattering along
  // `scatter_dimension`. The input shape has `scatter_dimension` scaled by the
  // number of replicas.
  //
  // `replica_groups` is a flat list of device IDs forming a single replica
  // group. An empty list means replica_groups={} (empty) in the HLO, which
  // will trigger the "replica groups must be provided" error path.
  absl::StatusOr<ReduceScatterInfo> BuildInfo(
      PrimitiveType element_type, std::vector<int64_t> output_dims,
      HloOpcode hlo_opcode, std::vector<int32_t> replica_groups,
      int64_t scatter_dimension = 0, int num_partitions = 1,
      int active_links = 18,
      se::DeviceDescription device_info =
          TestGpuDeviceInfo::H100SXMDeviceInfo()) {
    const int num_replicas =
        replica_groups.empty() ? 1 : static_cast<int>(replica_groups.size());
    const std::string element_type_str =
        primitive_util::LowercasePrimitiveTypeName(element_type);
    std::vector<int64_t> input_dims = output_dims;
    input_dims[scatter_dimension] *= num_replicas;
    const std::string input_shape_str = absl::StrJoin(input_dims, ",");
    const std::string output_shape_str = absl::StrJoin(output_dims, ",");
    const std::string replica_groups_str =
        replica_groups.empty()
            ? ""
            : absl::StrFormat("{%s}", absl::StrJoin(replica_groups, ","));

    constexpr absl::string_view kModuleStr = R"(
    HloModule test
    apply_op {
      x = %1$s[] parameter(0)
      y = %1$s[] parameter(1)
      ROOT apply_op = %1$s[] %5$s(x, y)
    }
    ENTRY test_computation {
      param_0 = %1$s[%2$s] parameter(0)
      ROOT reduce-scatter = %1$s[%3$s] reduce-scatter(param_0),
          dimensions={%6$d}, replica_groups={%4$s}, to_apply=apply_op
    }
    )";
    const std::string module_str = absl::StrFormat(
        kModuleStr, element_type_str, input_shape_str, output_shape_str,
        replica_groups_str, HloOpcodeString(hlo_opcode), scatter_dimension);

    SCOPED_TRACE(testing::Message() << "module_str: " << module_str);

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
        HloHardwareIndependentTestBase::FindInstruction(
            module.get(), HloOpcode::kReduceScatter);
    return BuildReduceScatterInfo(gpu_topology,
                                  Cast<HloReduceScatterInstruction>(hlo_instr),
                                  /*device_assignment=*/nullptr);
  }

  absl::StatusOr<ReduceScatterInfo> BuildInfo(
      PrimitiveType element_type, int64_t num_output_elements,
      HloOpcode hlo_opcode, std::vector<int32_t> replica_groups,
      int num_partitions = 1, int active_links = 18) {
    return BuildInfo(element_type, std::vector<int64_t>{num_output_elements},
                     hlo_opcode, std::move(replica_groups),
                     /*scatter_dimension=*/0, num_partitions, active_links);
  }
};

TEST_F(BuildReduceScatterInfoTest, SucceedsForSupportedF32) {
  EXPECT_THAT(
      BuildInfo(F32, /*num_output_elements=*/512, HloOpcode::kAdd,
                /*replica_groups=*/{0, 1}),
      IsOkAndHolds(AllOf(Field(&ReduceScatterInfo::num_devices, 2),
                         Field(&ReduceScatterInfo::num_output_elements, 512))));
}

TEST_F(BuildReduceScatterInfoTest, SucceedsForSupportedBF16) {
  EXPECT_THAT(
      BuildInfo(BF16, /*num_output_elements=*/512, HloOpcode::kAdd,
                /*replica_groups=*/{0, 1}),
      IsOkAndHolds(AllOf(Field(&ReduceScatterInfo::num_devices, 2),
                         Field(&ReduceScatterInfo::num_output_elements, 512))));
}

TEST_F(BuildReduceScatterInfoTest,
       SucceedsForSupportedIntegralAndPredicateReductions) {
  EXPECT_THAT(
      BuildInfo(S32, /*num_output_elements=*/512, HloOpcode::kMaximum,
                /*replica_groups=*/{0, 1}),
      IsOkAndHolds(AllOf(Field(&ReduceScatterInfo::num_devices, 2),
                         Field(&ReduceScatterInfo::num_output_elements, 512))));
  EXPECT_THAT(
      BuildInfo(PRED, /*num_output_elements=*/512, HloOpcode::kOr,
                /*replica_groups=*/{0, 1}),
      IsOkAndHolds(AllOf(Field(&ReduceScatterInfo::num_devices, 2),
                         Field(&ReduceScatterInfo::num_output_elements, 512))));
}

TEST_F(BuildReduceScatterInfoTest, SucceedsForFourDevices) {
  EXPECT_THAT(
      BuildInfo(F32, /*num_output_elements=*/512, HloOpcode::kAdd,
                /*replica_groups=*/{0, 1, 2, 3}),
      IsOkAndHolds(AllOf(Field(&ReduceScatterInfo::num_devices, 4),
                         Field(&ReduceScatterInfo::num_output_elements, 512))));
}

TEST_F(BuildReduceScatterInfoTest, FailsForSingleDevicePerReplicaGroup) {
  EXPECT_THAT(
      BuildInfo(F32, /*num_output_elements=*/512, HloOpcode::kAdd,
                /*replica_groups=*/{0}),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("Degenerate collectives with a single device")));
}

TEST_F(BuildReduceScatterInfoTest, FailsForNonPowerOfTwoDevices) {
  EXPECT_THAT(BuildInfo(F32, /*num_output_elements=*/512, HloOpcode::kAdd,
                        /*replica_groups=*/{0, 1, 2}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("only supported for power of 2")));
}

TEST_F(BuildReduceScatterInfoTest, FailsForUnsupportedTypeAndReduction) {
  EXPECT_THAT(BuildInfo(PRED, /*num_output_elements=*/512, HloOpcode::kAdd,
                        /*replica_groups=*/{0, 1}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("combination is not supported")));
  EXPECT_THAT(BuildInfo(F8E4M3FN, /*num_output_elements=*/512, HloOpcode::kAdd,
                        /*replica_groups=*/{0, 1}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("combination is not supported")));
}

TEST_F(BuildReduceScatterInfoTest, FailsForUnsupportedReductionComputation) {
  EXPECT_THAT(BuildInfo(F32, /*num_output_elements=*/512, HloOpcode::kSubtract,
                        /*replica_groups=*/{0, 1}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Unsupported reduction computation")));
}

TEST_F(BuildReduceScatterInfoTest, FailsIfReplicaGroupsEmpty) {
  EXPECT_THAT(
      BuildInfo(F32, /*num_output_elements=*/512, HloOpcode::kAdd,
                /*replica_groups=*/{}),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("Replica groups must be explicitly provided")));
}

TEST_F(BuildReduceScatterInfoTest, FailsForCrossHostCollective) {
  // 2 replicas split across 2 partitions -> not local -> should be rejected.
  EXPECT_THAT(BuildInfo(F32, /*num_output_elements=*/512, HloOpcode::kAdd,
                        /*replica_groups=*/{0, 1}, /*num_partitions=*/2),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Cross-host symmetric memory collectives")));
}

TEST_F(BuildReduceScatterInfoTest, FailsWithoutNvlink) {
  EXPECT_THAT(BuildInfo(F32, /*num_output_elements=*/512, HloOpcode::kAdd,
                        /*replica_groups=*/{0, 1}, /*num_partitions=*/1,
                        /*active_links=*/0),
              StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("NVLink")));
}

TEST_F(BuildReduceScatterInfoTest, FailsOnPreHopperDevice) {
  EXPECT_THAT(BuildInfo(F32, /*output_dims=*/{512}, HloOpcode::kAdd,
                        /*replica_groups=*/{0, 1}, /*scatter_dimension=*/0,
                        /*num_partitions=*/1, /*active_links=*/18,
                        TestGpuDeviceInfo::A100SXMDeviceInfo()),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("requires CUDA compute capability >= 9.0")));
}

TEST_F(BuildReduceScatterInfoTest,
       FailsWhenOutputElementsNotDivisibleByNumDevices) {
  EXPECT_THAT(BuildInfo(F32, /*num_output_elements=*/3, HloOpcode::kAdd,
                        /*replica_groups=*/{0, 1}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("divisible by the number of devices")));
}

TEST_F(BuildReduceScatterInfoTest,
       FailsWhenScatterDimensionIsNotMajorMostNonTrivial) {
  EXPECT_THAT(
      BuildInfo(F32, /*output_dims=*/{2, 4}, HloOpcode::kAdd,
                /*replica_groups=*/{0, 1}, /*scatter_dimension=*/1),
      StatusIs(
          absl::StatusCode::kUnimplemented,
          HasSubstr(
              "scatter dimension to be the physically major-most non-trivial "
              "dimension")));
}

TEST_F(BuildReduceScatterInfoTest,
       SucceedsWhenMajorDimensionsBeforeScatterAreTrivial) {
  EXPECT_THAT(
      BuildInfo(F32, /*output_dims=*/{1, 2, 4}, HloOpcode::kAdd,
                /*replica_groups=*/{0, 1}, /*scatter_dimension=*/1),
      IsOkAndHolds(AllOf(Field(&ReduceScatterInfo::num_devices, 2),
                         Field(&ReduceScatterInfo::num_output_elements, 8))));
}

TEST_F(BuildReduceScatterInfoTest, FailsWhenInputAndOutputLayoutsDoNotMatch) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }
  ENTRY test_computation {
    param_0 = f32[4,8]{1,0} parameter(0)
    ROOT reduce-scatter = f32[2,8]{0,1} reduce-scatter(param_0),
        dimensions={0}, replica_groups={{0,1}}, to_apply=apply_op
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleStr, 2));
  const auto* rs = Cast<HloReduceScatterInstruction>(
      HloHardwareIndependentTestBase::FindInstruction(
          module.get(), HloOpcode::kReduceScatter));
  EXPECT_FALSE(IsReduceScatterFlattenable(rs));
}

TEST_F(BuildReduceScatterInfoTest,
       VariadicChecksSizePerOperandAndUsesMaxElements) {
  // 8 MB + 16 MB = 24 MB total input across operands, but each operand is <=
  // the 16 MB per-operand limit, and `num_output_elements` is the maximum
  // across operands (2M elements from the second operand).
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }
  ENTRY test_computation {
    param_0 = f32[2097152] parameter(0)
    param_1 = f32[4194304] parameter(1)
    ROOT reduce-scatter = (f32[1048576], f32[2097152])
        reduce-scatter(param_0, param_1),
        dimensions={0}, replica_groups={{0,1}}, to_apply=apply_op
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
  const auto* rs = Cast<HloReduceScatterInstruction>(
      HloHardwareIndependentTestBase::FindInstruction(
          module.get(), HloOpcode::kReduceScatter));
  EXPECT_TRUE(IsReduceScatterFlattenable(rs));
  EXPECT_THAT(
      BuildReduceScatterInfo(gpu_topology, rs, /*device_assignment=*/nullptr),
      IsOkAndHolds(AllOf(
          Field(&ReduceScatterInfo::num_devices, 2),
          Field(&ReduceScatterInfo::num_output_elements, 2 * 1024 * 1024))));
}

TEST_F(BuildReduceScatterInfoTest, FailsForVariadicWhenSecondOperandInvalid) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }
  ENTRY test_computation {
    param_0 = f32[1,4,8] parameter(0)
    param_1 = f32[2,4,8] parameter(1)
    ROOT reduce-scatter = (f32[1,2,8], f32[2,2,8])
        reduce-scatter(param_0, param_1),
        dimensions={1}, replica_groups={{0,1}}, to_apply=apply_op
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleStr, 2));
  const auto* rs = Cast<HloReduceScatterInstruction>(
      HloHardwareIndependentTestBase::FindInstruction(
          module.get(), HloOpcode::kReduceScatter));
  EXPECT_FALSE(IsReduceScatterFlattenable(rs));
}

TEST_F(BuildReduceScatterInfoTest, FailsForVariadicWithMixedElementTypes) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  apply_op {
    x0 = f32[] parameter(0)
    x1 = s32[] parameter(1)
    y0 = f32[] parameter(2)
    y1 = s32[] parameter(3)
    add0 = f32[] add(x0, y0)
    add1 = s32[] add(x1, y1)
    ROOT t = (f32[], s32[]) tuple(add0, add1)
  }
  ENTRY test_computation {
    param_0 = f32[1024] parameter(0)
    param_1 = s32[1024] parameter(1)
    ROOT reduce-scatter = (f32[512], s32[512]) reduce-scatter(param_0, param_1),
        dimensions={0}, replica_groups={{0,1}}, to_apply=apply_op
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
  const auto* rs = Cast<HloReduceScatterInstruction>(
      HloHardwareIndependentTestBase::FindInstruction(
          module.get(), HloOpcode::kReduceScatter));
  EXPECT_THAT(BuildReduceScatterInfo(gpu_topology, rs,
                                     /*device_assignment=*/nullptr),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("same element type")));
}

TEST_F(BuildReduceScatterInfoTest, FailsForLargeInputs) {
  // 4 * 1024 * 1024 F32 output elements * 2 devices = 32 MB input > 16 MB
  // limit -> unimplemented.
  EXPECT_THAT(BuildInfo(F32, /*num_output_elements=*/4 * 1024 * 1024,
                        HloOpcode::kAdd, /*replica_groups=*/{0, 1}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("only supported for small inputs")));
}

TEST_F(BuildReduceScatterInfoTest, SucceedsFor16MBInput) {
  // 2 * 1024 * 1024 F32 output elements * 2 devices = 16 MB input == 16 MB
  // limit -> ok.
  EXPECT_THAT(
      BuildInfo(F32, /*num_output_elements=*/2 * 1024 * 1024, HloOpcode::kAdd,
                /*replica_groups=*/{0, 1}),
      IsOkAndHolds(AllOf(
          Field(&ReduceScatterInfo::num_devices, 2),
          Field(&ReduceScatterInfo::num_output_elements, 2 * 1024 * 1024))));
}

TEST_F(BuildReduceScatterInfoTest, CreateReduceScatterKernelSpec) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }
  ENTRY test_computation {
    param_0 = f32[1024] parameter(0)
    ROOT reduce-scatter = f32[512] reduce-scatter(param_0),
        dimensions={0}, replica_groups={{0,1}}, to_apply=apply_op
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleStr, 2));
  const HloInstruction* instr = HloHardwareIndependentTestBase::FindInstruction(
      module.get(), HloOpcode::kReduceScatter);
  ASSERT_OK_AND_ASSIGN(
      CollectiveKernelSpec spec,
      CreateReduceScatterKernelSpec(instr, LaunchDimensions(4, 128),
                                    SymmetricMemoryType::kXlaRendezvous));
  EXPECT_FALSE(spec.codegen_config.copy_input_to_scratch);
  EXPECT_EQ(spec.codegen_config.sync_count_increment, 1u);
  EXPECT_TRUE(spec.codegen_config.device_sync_count);
  ASSERT_EQ(spec.codegen_config.argument_descriptors.size(), 5);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[0].type,
            KernelArgType::kInputBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[0].index, 0);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[1].type,
            KernelArgType::kOutputBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[1].index, 0);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[2].type,
            KernelArgType::kRuntimeRank);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[3].type,
            KernelArgType::kScratchBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[3].index, 0);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[4].type,
            KernelArgType::kScratchBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[4].index, 1);
  ASSERT_EQ(spec.scratch_buffers.size(), 2);
  EXPECT_EQ(spec.scratch_buffers[0].symmetric_memory_type,
            SymmetricMemoryType::kXlaRendezvous);
  EXPECT_TRUE(spec.scratch_buffers[0].should_memzero);
  EXPECT_TRUE(spec.scratch_buffers[0].should_double_buffer);
  EXPECT_EQ(spec.scratch_buffers[1].symmetric_memory_type,
            SymmetricMemoryType::kXlaRendezvous);
  EXPECT_FALSE(spec.scratch_buffers[1].should_memzero);
  EXPECT_TRUE(spec.scratch_buffers[1].should_double_buffer);
}

TEST_F(BuildReduceScatterInfoTest, CreateReduceScatterKernelSpecVariadic) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }
  ENTRY test_computation {
    param_0 = f32[1024] parameter(0)
    param_1 = f32[8192] parameter(1)
    ROOT reduce-scatter = (f32[512], f32[4096])
        reduce-scatter(param_0, param_1),
        dimensions={0}, replica_groups={{0,1}}, to_apply=apply_op
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleStr, 2));
  const HloInstruction* instr = HloHardwareIndependentTestBase::FindInstruction(
      module.get(), HloOpcode::kReduceScatter);
  ASSERT_OK_AND_ASSIGN(
      CollectiveKernelSpec spec,
      CreateReduceScatterKernelSpec(instr, LaunchDimensions(4, 128),
                                    SymmetricMemoryType::kXlaRendezvous));
  ASSERT_EQ(spec.codegen_config.input_buffer_specs.size(), 2);
  ASSERT_EQ(spec.codegen_config.output_buffer_specs.size(), 2);
  ASSERT_EQ(spec.codegen_config.argument_descriptors.size(), 8);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[0].type,
            KernelArgType::kInputBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[0].index, 0);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[1].type,
            KernelArgType::kInputBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[1].index, 1);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[2].type,
            KernelArgType::kOutputBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[2].index, 0);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[3].type,
            KernelArgType::kOutputBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[3].index, 1);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[4].type,
            KernelArgType::kRuntimeRank);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[5].type,
            KernelArgType::kScratchBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[5].index, 0);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[6].type,
            KernelArgType::kScratchBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[6].index, 1);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[7].type,
            KernelArgType::kScratchBuffer);
  EXPECT_EQ(spec.codegen_config.argument_descriptors[7].index, 2);
  ASSERT_EQ(spec.scratch_buffers.size(), 3);
  EXPECT_EQ(spec.scratch_buffers[1].size_bytes, 1024 * sizeof(float));
  EXPECT_EQ(spec.scratch_buffers[2].size_bytes, 8192 * sizeof(float));
}

}  // namespace
}  // namespace xla::gpu
