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

#include "xla/service/gpu/model/collective_interpolator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

constexpr int kNumGpusPerHost = 8;

using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

struct SpaceSpec {
  // Discrete key.
  HloOpcode opcode;
  CollectiveInterpolator::CommunicationType comm;

  // Euclidean space.
  int tensor_size;
  int num_nodes;

  // Value.
  int network_througput_bytes;
};

struct ParametrizedTestCase {
  std::string test_name;
  SpaceSpec spec;
  absl::Duration expected_duration;
};

class CollectiveInterpolationTest : public TestWithParam<ParametrizedTestCase> {
  void SetUp() override {
    HloInstructionProfileList profiles;
    for (auto space_spec : test_space_) {
      HloInstructionProfile entry = CollectiveInstruction(space_spec);
      entry.set_network_throughput_bytes_per_sec(
          space_spec.network_througput_bytes);
      *profiles.add_entries() = entry;
    }
    interpolator_ = *CollectiveInterpolator::Create(
        profiles, TestGpuDeviceInfo::RTXA6000DeviceInfo());
  }

 protected:
  HloInstructionProfile CollectiveInstruction(const SpaceSpec& test_spec) {
    return CollectiveInstruction(test_spec.opcode, test_spec.comm,
                                 test_spec.tensor_size, test_spec.num_nodes);
  }

  HloInstructionProfile CollectiveInstruction(
      HloOpcode opcode, CollectiveInterpolator::CommunicationType comm,
      int64_t tensor_size, int num_hosts) {
    Shape shape;
    CollectiveDeviceList device_list;
    switch (opcode) {
      case HloOpcode::kAllReduce:
      case HloOpcode::kAllReduceStart:
        device_list = CollectiveDeviceList(CommToDeviceList(comm, num_hosts));
        shape = ShapeUtil::MakeShape(PrimitiveType::F32, {tensor_size / 4});
        break;
      case HloOpcode::kReduceScatter:
        device_list = CollectiveDeviceList(CommToDeviceList(comm, num_hosts));
        shape = ShapeUtil::MakeShape(
            PrimitiveType::F32,
            {tensor_size /
             (4 *
              device_list.iota_replica_group_list()->num_devices_per_group())});
        break;
      case HloOpcode::kAllGather:
      case HloOpcode::kAllGatherStart:
        device_list = CollectiveDeviceList(CommToDeviceList(comm, num_hosts));
        shape = ShapeUtil::MakeShape(PrimitiveType::F32, {tensor_size / 4});
        break;
      default:
        LOG(FATAL) << "Unsupported test spec.";
    };
    HloInstructionProfile profile;
    *profile.mutable_instruction()->mutable_opcode() = HloOpcodeString(opcode);
    *profile.mutable_instruction()->mutable_shape() = shape.ToProto();
    *profile.mutable_instruction()->mutable_collective_device_list() =
        device_list.ToProto();
    profile.mutable_instruction()->set_use_global_device_ids(true);
    profile.mutable_instruction()->set_channel_id(1);
    return profile;
  }

  std::optional<absl::Duration> EstimateRuntime(
      HloOpcode opcode, CollectiveInterpolator::CommunicationType comm,
      int64_t tensor_size, int num_hosts) {
    auto instr = CollectiveInstruction(opcode, comm, tensor_size, num_hosts);
    auto module = CollectiveInterpolator::ConstructModule(instr);
    auto* eval = Cast<HloCollectiveInstruction>(
        module->entry_computation()->root_instruction());
    return interpolator().EstimatedRuntime(*eval);
  }

  CollectiveInterpolator& interpolator() { return *interpolator_; }

 private:
  IotaReplicaGroupList CommToDeviceList(
      CollectiveInterpolator::CommunicationType comm, int num_hosts) {
    IotaReplicaGroupList iota(1, 1);
    switch (comm) {
      case CollectiveInterpolator::CommunicationType::SINGLE_HOST:
        iota = IotaReplicaGroupList(num_hosts, kNumGpusPerHost);
        break;
      case CollectiveInterpolator::CommunicationType::RAIL_ALIGNED:
        iota = IotaReplicaGroupList(1, num_hosts * kNumGpusPerHost);
        break;
      case CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED:
        iota = IotaReplicaGroupList(kNumGpusPerHost, num_hosts,
                                    {num_hosts, kNumGpusPerHost}, {1, 0});
        break;
      default:
        LOG(FATAL) << "Unsupported comm option.";
    }
    return iota;
  }

  std::unique_ptr<CollectiveInterpolator> interpolator_;
  std::vector<SpaceSpec> test_space_ = {
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/1024,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/512,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/CollectiveInterpolator::CommunicationType::SINGLE_HOST,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/CollectiveInterpolator::CommunicationType::SINGLE_HOST,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 2048,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/CollectiveInterpolator::CommunicationType::SINGLE_HOST,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/CollectiveInterpolator::CommunicationType::SINGLE_HOST,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2 * 2048,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/1024,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/512,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 512,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 512,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 512,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/CollectiveInterpolator::CommunicationType::SINGLE_HOST,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/CollectiveInterpolator::CommunicationType::SINGLE_HOST,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 2048,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/CollectiveInterpolator::CommunicationType::SINGLE_HOST,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/CollectiveInterpolator::CommunicationType::SINGLE_HOST,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2 * 2048,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/1024,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/512,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/CollectiveInterpolator::CommunicationType::SINGLE_HOST,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/CollectiveInterpolator::CommunicationType::SINGLE_HOST,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 2048,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/CollectiveInterpolator::CommunicationType::SINGLE_HOST,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/CollectiveInterpolator::CommunicationType::SINGLE_HOST,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2 * 2048,
      },
  };
};

TEST_P(CollectiveInterpolationTest, NextNeighbourInterpolation) {
  const auto& [_, spec, expected_duration] = GetParam();
  EXPECT_EQ(
      EstimateRuntime(spec.opcode, spec.comm, spec.tensor_size, spec.num_nodes),
      expected_duration);
}

INSTANTIATE_TEST_SUITE_P(
    CollectiveInterpolationTestInstantiation, CollectiveInterpolationTest,
    ValuesIn<ParametrizedTestCase>({
        {
            /*test_name=*/"AR_rail_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
                /*tensor_size=*/1024,
                /*num_nodes=*/8,
            },
            /*expected_duration=*/absl::Milliseconds(250),
        },
        {
            /*test_name=*/"AR_rail_aligned_extrapolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
                /*tensor_size=*/4 * 1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(2),
        },
        {
            /*test_name=*/"AR_rail_aligned_interpolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
                /*tensor_size=*/1024,
                /*num_nodes=*/3,
            },
            /*expected_duration=*/absl::Seconds(1),
        },
        {
            /*test_name=*/"AR_rail_aligned_interpolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(1250),
        },
        {
            /*test_name=*/"AR_nonrail_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
                /*tensor_size=*/1024,
                /*num_nodes=*/8,
            },
            /*expected_duration=*/absl::Milliseconds(500),
        },
        {
            /*test_name=*/"AR_nonrail_aligned_extrapolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
                /*tensor_size=*/4 * 1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(4),
        },
        {
            /*test_name=*/"AR_nonrail_aligned_interpolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
                /*tensor_size=*/1024,
                /*num_nodes=*/3,
            },
            /*expected_duration=*/absl::Seconds(2),
        },
        {
            /*test_name=*/"AR_nonrail_aligned_interpolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(2500),
        },
        {
            /*test_name=*/"AR_single_host_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::SINGLE_HOST,
                /*tensor_size=*/1024,
                /*num_nodes=*/8,
            },
            /*expected_duration=*/absl::Milliseconds(500),
        },
        {
            /*test_name=*/"AR_single_host_aligned_extrapolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::SINGLE_HOST,
                /*tensor_size=*/4 * 1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(1),
        },
        {
            /*test_name=*/"AR_single_host_aligned_interpolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::SINGLE_HOST,
                /*tensor_size=*/1024,
                /*num_nodes=*/3,
            },
            /*expected_duration=*/absl::Milliseconds(500),
        },
        {
            /*test_name=*/"AR_single_host_aligned_interpolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::SINGLE_HOST,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(625),
        },
        {
            /*test_name=*/"ARS_single_host_aligned_interpolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduceStart,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::SINGLE_HOST,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(625),
        },
        {
            /*test_name=*/"RS_rail_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
                /*tensor_size=*/1024,
                /*num_nodes=*/8,
            },
            /*expected_duration=*/absl::Milliseconds(250),
        },
        {
            /*test_name=*/"RS_rail_aligned_extrapolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
                /*tensor_size=*/4 * 1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(2),
        },
        {
            /*test_name=*/"RS_rail_aligned_interpolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
                /*tensor_size=*/1024,
                /*num_nodes=*/3,
            },
            /*expected_duration=*/absl::Milliseconds(937.5),
        },
        {
            /*test_name=*/"RS_rail_aligned_interpolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(1250),
        },
        {
            /*test_name=*/"RS_nonrail_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
                /*tensor_size=*/1024,
                /*num_nodes=*/8,
            },
            /*expected_duration=*/absl::Milliseconds(500),
        },
        {
            /*test_name=*/"RS_nonrail_aligned_extrapolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
                /*tensor_size=*/4 * 1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(4),
        },
        {
            /*test_name=*/"RS_nonrail_aligned_interpolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
                /*tensor_size=*/1032,
                /*num_nodes=*/3,
            },
            /*expected_duration=*/absl::Seconds(2.015625),
        },
        {
            /*test_name=*/"RS_nonrail_aligned_interpolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(2500),
        },
        {
            /*test_name=*/"RS_single_host_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::SINGLE_HOST,
                /*tensor_size=*/1024,
                /*num_nodes=*/8,
            },
            /*expected_duration=*/absl::Milliseconds(500),
        },
        {
            /*test_name=*/"RS_single_host_aligned_extrapolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::SINGLE_HOST,
                /*tensor_size=*/4 * 1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(1),
        },
        {
            /*test_name=*/"RS_single_host_aligned_interpolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::SINGLE_HOST,
                /*tensor_size=*/1024,
                /*num_nodes=*/3,
            },
            /*expected_duration=*/absl::Milliseconds(500),
        },
        {
            /*test_name=*/"RS_single_host_aligned_interpolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::SINGLE_HOST,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(625),
        },
        {
            /*test_name=*/"AG_rail_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
                /*tensor_size=*/1024,
                /*num_nodes=*/8,
            },
            /*expected_duration=*/absl::Milliseconds(250),
        },
        {
            /*test_name=*/"AG_rail_aligned_extrapolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
                /*tensor_size=*/4 * 1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(2),
        },
        {
            /*test_name=*/"AG_rail_aligned_interpolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
                /*tensor_size=*/1056,
                /*num_nodes=*/3,
            },
            /*expected_duration=*/absl::Milliseconds(1031.25),
        },
        {
            /*test_name=*/"AG_rail_aligned_interpolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::RAIL_ALIGNED,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(1250),
        },
        {
            /*test_name=*/"AG_nonrail_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
                /*tensor_size=*/1024,
                /*num_nodes=*/8,
            },
            /*expected_duration=*/absl::Milliseconds(500),
        },
        {
            /*test_name=*/"AG_nonrail_aligned_extrapolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
                /*tensor_size=*/4 * 1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(4),
        },
        {
            /*test_name=*/"AG_nonrail_aligned_interpolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
                /*tensor_size=*/1032,
                /*num_nodes=*/3,
            },
            /*expected_duration=*/absl::Seconds(2.015625),
        },
        {
            /*test_name=*/"AG_nonrail_aligned_interpolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::NON_RAIL_ALIGNED,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(2500),
        },
        {
            /*test_name=*/"AG_single_host_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::SINGLE_HOST,
                /*tensor_size=*/1024,
                /*num_nodes=*/8,
            },
            /*expected_duration=*/absl::Milliseconds(500),
        },
        {
            /*test_name=*/"AG_single_host_aligned_extrapolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::SINGLE_HOST,
                /*tensor_size=*/4 * 1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(1),
        },
        {
            /*test_name=*/"AG_single_host_aligned_interpolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::SINGLE_HOST,
                /*tensor_size=*/1024,
                /*num_nodes=*/3,
            },
            /*expected_duration=*/absl::Milliseconds(500),
        },
        {
            /*test_name=*/"AG_single_host_aligned_interpolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::SINGLE_HOST,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(625),
        },
        {
            /*test_name=*/"AGS_single_host_aligned_interpolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGatherStart,
                /*comm=*/
                CollectiveInterpolator::CommunicationType::SINGLE_HOST,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(625),
        },
    }),
    [](const TestParamInfo<CollectiveInterpolationTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace xla::gpu
