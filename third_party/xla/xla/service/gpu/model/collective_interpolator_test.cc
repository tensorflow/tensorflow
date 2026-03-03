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
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/replica_group.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
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
  GPUCommunicationType comm;

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
    device_info_ = TestGpuDeviceInfo::RTXA6000DeviceInfo();
    interpolator_ = *CollectiveInterpolator::Create(kNumGpusPerHost, profiles,
                                                    device_info_);
  }

 protected:
  HloInstructionProfile CollectiveInstruction(const SpaceSpec& test_spec) {
    return CollectiveInstruction(test_spec.opcode, test_spec.comm,
                                 test_spec.tensor_size, test_spec.num_nodes);
  }

  HloInstructionProfile CollectiveInstruction(HloOpcode opcode,
                                              GPUCommunicationType comm,
                                              int64_t tensor_size,
                                              int num_hosts) {
    Shape shape;

    switch (opcode) {
      case HloOpcode::kAllReduce:
      case HloOpcode::kAllReduceStart:
      case HloOpcode::kAllToAll:
        shape = ShapeUtil::MakeShape(PrimitiveType::F32, {tensor_size / 4});
        break;
      case HloOpcode::kReduceScatter: {
        auto iota_list = CommToDeviceList(comm, num_hosts);
        shape = ShapeUtil::MakeShape(
            PrimitiveType::F32,
            {tensor_size / (4 * iota_list.num_devices_per_group())});
        break;
      }
      case HloOpcode::kAllGather:
      case HloOpcode::kAllGatherStart:
        shape = ShapeUtil::MakeShape(PrimitiveType::F32, {tensor_size / 4});
        break;
      default:
        LOG(FATAL) << "Unsupported test spec.";
    };
    HloInstructionProfile profile;
    *profile.mutable_instruction()->mutable_opcode() = HloOpcodeString(opcode);
    *profile.mutable_instruction()->mutable_shape() = shape.ToProto();
    *profile.mutable_instruction()->mutable_iota_collective_device_list() =
        CommToDeviceList(comm, num_hosts).ToProto();
    profile.mutable_instruction()->set_use_global_device_ids(true);
    profile.mutable_instruction()->set_channel_id(1);
    return profile;
  }

  absl::StatusOr<absl::Duration> EstimateRuntime(HloOpcode opcode,
                                                 GPUCommunicationType comm,
                                                 int64_t tensor_size,
                                                 int num_hosts) {
    auto instr = CollectiveInstruction(opcode, comm, tensor_size, num_hosts);
    auto module = CollectiveInterpolator::ConstructModule(instr);
    auto* eval = Cast<HloCollectiveInstruction>(
        module->entry_computation()->root_instruction());
    return interpolator().EstimatedRuntime(*eval);
  }

  CollectiveInterpolator& interpolator() { return *interpolator_; }

 private:
  IotaReplicaGroupList CommToDeviceList(GPUCommunicationType comm,
                                        int num_hosts) {
    IotaReplicaGroupList iota(1, 1);
    switch (comm) {
      case GPUCommunicationType::SINGLE_PARTITION:
        iota = IotaReplicaGroupList(num_hosts, kNumGpusPerHost);
        break;
      case GPUCommunicationType::MULTI_HOST_WORLD_LEVEL:
        iota = IotaReplicaGroupList(1, num_hosts * kNumGpusPerHost);
        break;
      case GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL:
        iota = IotaReplicaGroupList(kNumGpusPerHost, num_hosts,
                                    {num_hosts, kNumGpusPerHost}, {1, 0});
        break;
      default:
        LOG(FATAL) << "Unsupported comm option.";
    }
    return iota;
  }

  se::DeviceDescription device_info_;
  std::unique_ptr<CollectiveInterpolator> interpolator_;
  std::vector<SpaceSpec> test_space_ = {
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/1024,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/512,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::SINGLE_PARTITION,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::SINGLE_PARTITION,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 2048,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::SINGLE_PARTITION,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::SINGLE_PARTITION,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2 * 2048,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/1024,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/512,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 512,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 512,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 512,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::SINGLE_PARTITION,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::SINGLE_PARTITION,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 2048,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::SINGLE_PARTITION,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::SINGLE_PARTITION,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2 * 2048,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/1024,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/512,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::SINGLE_PARTITION,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::SINGLE_PARTITION,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 2048,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::SINGLE_PARTITION,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::SINGLE_PARTITION,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2 * 2048,
      },
      {
          /*opcode=*/HloOpcode::kAllToAll,
          /*comm=*/GPUCommunicationType::SINGLE_PARTITION,
          /*tensor_size=*/1024,
          /*num_nodes=*/1,
          /*network_througput_bytes=*/1024,
      },
      {
          /*opcode=*/HloOpcode::kAllToAll,
          /*comm=*/GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kAllToAll,
          /*comm=*/GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/4096,
      },
  };
};

TEST_P(CollectiveInterpolationTest, NextNeighbourInterpolation) {
  const auto& [_, spec, expected_duration] = GetParam();
  EXPECT_EQ(*EstimateRuntime(spec.opcode, spec.comm, spec.tensor_size,
                             spec.num_nodes),
            expected_duration);
}

INSTANTIATE_TEST_SUITE_P(
    CollectiveInterpolationTestInstantiation, CollectiveInterpolationTest,
    ValuesIn<ParametrizedTestCase>({
        {
            /*test_name=*/"AR_rail_aligned_exact_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
                /*tensor_size=*/1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(1),
        },
        {
            /*test_name=*/"AR_rail_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(1250),
        },
        {
            /*test_name=*/"AR_nonrail_aligned_exact_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
                /*tensor_size=*/1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(2),
        },
        {
            /*test_name=*/"AR_nonrail_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(2500),
        },
        {
            /*test_name=*/"AR_SINGLE_PARTITION_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                GPUCommunicationType::SINGLE_PARTITION,
                /*tensor_size=*/1024,
                /*num_nodes=*/8,
            },
            /*expected_duration=*/absl::Milliseconds(500),
        },
        {
            /*test_name=*/"AR_SINGLE_PARTITION_aligned_extrapolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                GPUCommunicationType::SINGLE_PARTITION,
                /*tensor_size=*/4 * 1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(1),
        },
        {
            /*test_name=*/"AR_SINGLE_PARTITION_aligned_interpolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                GPUCommunicationType::SINGLE_PARTITION,
                /*tensor_size=*/1024,
                /*num_nodes=*/3,
            },
            /*expected_duration=*/absl::Milliseconds(500),
        },
        {
            /*test_name=*/"AR_SINGLE_PARTITION_aligned_interpolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                GPUCommunicationType::SINGLE_PARTITION,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(625),
        },
        {
            /*test_name=*/"ARS_SINGLE_PARTITION_aligned_interpolate_tensor_"
                          "size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduceStart,
                /*comm=*/
                GPUCommunicationType::SINGLE_PARTITION,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(625),
        },
        {
            /*test_name=*/"RS_rail_aligned_exact_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
                /*tensor_size=*/1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(1),
        },
        {
            /*test_name=*/"RS_rail_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(1250),
        },
        {
            /*test_name=*/"RS_nonrail_aligned_exact_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
                /*tensor_size=*/1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(2),
        },
        {
            /*test_name=*/"RS_nonrail_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(2500),
        },
        {
            /*test_name=*/"RS_SINGLE_PARTITION_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                GPUCommunicationType::SINGLE_PARTITION,
                /*tensor_size=*/1024,
                /*num_nodes=*/8,
            },
            /*expected_duration=*/absl::Milliseconds(500),
        },
        {
            /*test_name=*/"RS_SINGLE_PARTITION_aligned_extrapolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                GPUCommunicationType::SINGLE_PARTITION,
                /*tensor_size=*/4 * 1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(1),
        },
        {
            /*test_name=*/"RS_SINGLE_PARTITION_aligned_interpolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                GPUCommunicationType::SINGLE_PARTITION,
                /*tensor_size=*/1024,
                /*num_nodes=*/3,
            },
            /*expected_duration=*/absl::Milliseconds(500),
        },
        {
            /*test_name=*/"RS_SINGLE_PARTITION_aligned_interpolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kReduceScatter,
                /*comm=*/
                GPUCommunicationType::SINGLE_PARTITION,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(625),
        },
        {
            /*test_name=*/"AG_rail_aligned_exact_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
                /*tensor_size=*/1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(1),
        },
        {
            /*test_name=*/"AG_rail_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(1250),
        },
        {
            /*test_name=*/"AG_nonrail_aligned_exact_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
                /*tensor_size=*/1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(2),
        },
        {
            /*test_name=*/"AG_nonrail_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
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
                GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(2500),
        },
        {
            /*test_name=*/"AG_SINGLE_PARTITION_aligned_extrapolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                GPUCommunicationType::SINGLE_PARTITION,
                /*tensor_size=*/1024,
                /*num_nodes=*/8,
            },
            /*expected_duration=*/absl::Milliseconds(500),
        },
        {
            /*test_name=*/"AG_SINGLE_PARTITION_aligned_extrapolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                GPUCommunicationType::SINGLE_PARTITION,
                /*tensor_size=*/4 * 1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Seconds(1),
        },
        {
            /*test_name=*/"AG_SINGLE_PARTITION_aligned_interpolate_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                GPUCommunicationType::SINGLE_PARTITION,
                /*tensor_size=*/1024,
                /*num_nodes=*/3,
            },
            /*expected_duration=*/absl::Milliseconds(500),
        },
        {
            /*test_name=*/"AG_SINGLE_PARTITION_aligned_interpolate_tensor_size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGather,
                /*comm=*/
                GPUCommunicationType::SINGLE_PARTITION,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(625),
        },
        {
            /*test_name=*/"AGS_SINGLE_PARTITION_aligned_interpolate_tensor_"
                          "size",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllGatherStart,
                /*comm=*/
                GPUCommunicationType::SINGLE_PARTITION,
                /*tensor_size=*/1024 + 256,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(625),
        },
        {
            /*test_name=*/"A2A_rail_aligned_exact_match",
            {
                /*opcode=*/HloOpcode::kAllToAll,
                /*comm=*/
                GPUCommunicationType::MULTI_HOST_WORLD_LEVEL,
                /*tensor_size=*/1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(500),
        },
        {
            /*test_name=*/"A2A_nonrail_aligned_exact_match",
            {
                /*opcode=*/HloOpcode::kAllToAll,
                /*comm=*/
                GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL,
                /*tensor_size=*/1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(250),
        },
        {
            /*test_name=*/"A2A_SINGLE_PARTITION_exact_match",
            {
                /*opcode=*/HloOpcode::kAllToAll,
                /*comm=*/
                GPUCommunicationType::SINGLE_PARTITION,
                /*tensor_size=*/1024,
                /*num_nodes=*/1,
            },
            /*expected_duration=*/absl::Seconds(1),
        },
    }),
    [](const TestParamInfo<CollectiveInterpolationTest::ParamType>& info) {
      return info.param.test_name;
    });

struct CollectiveInterpolationWithDefaultProfileTestCase {
  std::string test_name;
  stream_executor::GpuComputeCapability cc;
  absl::Duration expected_duration;
};

class CollectiveInterpolationWithDefaultProfileTest
    : public TestWithParam<CollectiveInterpolationWithDefaultProfileTestCase> {
};

TEST_P(CollectiveInterpolationWithDefaultProfileTest, LoadsDefaultProfile) {
  const auto& [test_name, cc, expected_duration] = GetParam();
  auto device_info = test_name == "B200"
                         ? TestGpuDeviceInfo::B200SXMDeviceInfo(cc)
                         : TestGpuDeviceInfo::RTXA6000DeviceInfo(cc);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CollectiveInterpolator> interpolator,
      CollectiveInterpolator::Create(kNumGpusPerHost, device_info));
  absl::string_view kHlo = R"(
    HloModule m, num_partitions=8

    wrapped_add {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT _ = f32[] add(a,b)
    }

    ENTRY main {
      p = f32[256] parameter(0)
      ROOT _ = f32[256] all-reduce(p), to_apply=wrapped_add,
          replica_groups=[1,8]<=[8], use_global_device_ids=true, channel_id=1
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));
  HloCollectiveInstruction* instr = Cast<HloCollectiveInstruction>(
      module->entry_computation()->root_instruction());

  absl::StatusOr<absl::Duration> runtime =
      interpolator->EstimatedRuntime(*instr);
  EXPECT_TRUE(runtime.ok());
  EXPECT_EQ(runtime.value(), expected_duration);
}

INSTANTIATE_TEST_SUITE_P(
    CollectiveInterpolationWithDefaultProfileTestInstantiation,
    CollectiveInterpolationWithDefaultProfileTest,
    ValuesIn<CollectiveInterpolationWithDefaultProfileTestCase>(
        {{
             "H100",
             se::CudaComputeCapability(9, 0),
             absl::Microseconds(49.312),
         },
         {
             "B200",
             se::CudaComputeCapability(10, 0),
             absl::Microseconds(45.024),
         }}),
    [](const TestParamInfo<
        CollectiveInterpolationWithDefaultProfileTest::ParamType>& info) {
      return info.param.test_name;
    });

struct CollectivePermuteTestCase {
  std::string test_name;
  CollectivePermuteCostModelType permute_type;
  absl::Duration expected_duration;
};

class CollectivePermuteInterpolationTest
    : public TestWithParam<CollectivePermuteTestCase> {
  void SetUp() override {
    HloInstructionProfileList profiles;
    for (auto& [permute_type, points] : collective_permute_profiles_) {
      for (auto& [tensor_size, throughput] : points) {
        HloInstructionProfile entry =
            CollectivePermuteInstruction(permute_type, tensor_size);
        entry.set_network_throughput_bytes_per_sec(throughput);
        *profiles.add_entries() = entry;
      }
    }
    device_info_ = TestGpuDeviceInfo::RTXA6000DeviceInfo();
    interpolator_ = *CollectiveInterpolator::Create(kNumGpusPerHost, profiles,
                                                    device_info_);
  }

 protected:
  absl::StatusOr<absl::Duration> EstimateRuntime(
      CollectivePermuteCostModelType permute_type, int64_t tensor_size) {
    auto instr = CollectivePermuteInstruction(permute_type, tensor_size);
    auto module = CollectiveInterpolator::ConstructModule(instr);
    module->mutable_config().set_num_partitions(kNumGpusPerHost);
    auto* eval = module->entry_computation()->root_instruction();
    return interpolator_->EstimatedRuntime(*eval);
  }

 private:
  // Creates a collective permute instruction with a given `permute_type` and
  // `tensor_size`.
  // The perf table only supports the intra-partition collective permutes,
  // including one-way, two-way all-mutual and two-way has non-mutual.
  HloInstructionProfile CollectivePermuteInstruction(
      CollectivePermuteCostModelType permute_type, int64_t tensor_size) {
    Shape shape = ShapeUtil::MakeShape(PrimitiveType::F32, {tensor_size / 4});

    HloInstructionProfile profile;
    profile.mutable_instruction()->set_opcode(
        HloOpcodeString(HloOpcode::kCollectivePermute));
    *profile.mutable_instruction()->mutable_shape() = shape.ToProto();
    profile.mutable_instruction()->set_channel_id(1);

    std::vector<std::pair<int64_t, int64_t>> pairs;
    if (permute_type == CollectivePermuteCostModelType::kIntraPartitionOneWay) {
      pairs = {{0, 2}, {1, 3}};
    } else if (permute_type ==
               CollectivePermuteCostModelType::kIntraPartitionTwoWayAllMutual) {
      pairs = {{0, 1}, {1, 0}, {2, 3}, {3, 2}};
    } else if (permute_type == CollectivePermuteCostModelType::
                                   kIntraPartitionTwoWayHasNonMutual) {
      pairs = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
    }
    for (const auto& pair : pairs) {
      auto* p = profile.mutable_instruction()->add_source_target_pairs();
      p->set_source(pair.first);
      p->set_target(pair.second);
    }
    return profile;
  }

  se::DeviceDescription device_info_;
  std::unique_ptr<CollectiveInterpolator> interpolator_;
  // The collective permute testing profiles for the perf table.
  // format: {collectove_permute_type: {{tensor_size_1, throughput_1},
  //                                    {tensor_size_2, throughput_2},
  //                                    ...}}
  absl::flat_hash_map<CollectivePermuteCostModelType,
                      std::vector<std::pair<int64_t, int64_t>>>
      collective_permute_profiles_ = {
          {CollectivePermuteCostModelType::kIntraPartitionOneWay,
           {{512, 512}, {1024, 1024}, {2048, 1536}}},
          {CollectivePermuteCostModelType::kIntraPartitionTwoWayAllMutual,
           {{512, 1024}, {1024, 2048}, {2048, 3072}}},
          {CollectivePermuteCostModelType::kIntraPartitionTwoWayHasNonMutual,
           {{512, 1536}, {1024, 3072}, {2048, 4096}}}};
};

TEST_P(CollectivePermuteInterpolationTest, InterpolatesCorrectly) {
  const auto& [_, permute_type, expected_duration] = GetParam();
  auto runtime = EstimateRuntime(permute_type, 1024);
  ASSERT_TRUE(runtime.ok());
  EXPECT_NEAR(absl::ToDoubleSeconds(*runtime),
              absl::ToDoubleSeconds(expected_duration), 1e-5);
}

INSTANTIATE_TEST_SUITE_P(
    CollectivePermuteInterpolationTestInstantiation,
    CollectivePermuteInterpolationTest,
    ValuesIn<CollectivePermuteTestCase>({
        {"OneWay", CollectivePermuteCostModelType::kIntraPartitionOneWay,
         absl::Seconds(1)},
        {"TwoWayAllMutual",
         CollectivePermuteCostModelType::kIntraPartitionTwoWayAllMutual,
         absl::Milliseconds(500)},
        {"TwoWayHasNonMutual",
         CollectivePermuteCostModelType::kIntraPartitionTwoWayHasNonMutual,
         absl::Seconds(1.0 / 3.0)},
    }),
    [](const TestParamInfo<CollectivePermuteInterpolationTest::ParamType>&
           info) { return info.param.test_name; });

}  // namespace
}  // namespace xla::gpu
