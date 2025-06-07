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
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
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
    device_info_ = TestGpuDeviceInfo::RTXA6000DeviceInfo(
        stream_executor::CudaComputeCapability::Hopper());
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
    CollectiveDeviceList device_list;
    switch (opcode) {
      case HloOpcode::kAllReduce:
      case HloOpcode::kAllReduceStart:
      case HloOpcode::kAllToAll:
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

  std::optional<absl::Duration> EstimateRuntime(HloOpcode opcode,
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
      case GPUCommunicationType::SINGLE_HOST:
        iota = IotaReplicaGroupList(num_hosts, kNumGpusPerHost);
        break;
      case GPUCommunicationType::RAIL_ALIGNED:
        iota = IotaReplicaGroupList(1, num_hosts * kNumGpusPerHost);
        break;
      case GPUCommunicationType::NON_RAIL_ALIGNED:
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
          /*comm=*/GPUCommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/1024,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/512,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::SINGLE_HOST,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::SINGLE_HOST,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 2048,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::SINGLE_HOST,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kAllReduce,
          /*comm=*/GPUCommunicationType::SINGLE_HOST,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2 * 2048,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/1024,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/512,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 512,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 512,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 512,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::SINGLE_HOST,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::SINGLE_HOST,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 2048,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::SINGLE_HOST,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kReduceScatter,
          /*comm=*/GPUCommunicationType::SINGLE_HOST,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2 * 2048,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/1024,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 1024,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/512,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/4 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/5 * 512,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::SINGLE_HOST,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::SINGLE_HOST,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2 * 2048,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::SINGLE_HOST,
          /*tensor_size=*/1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kAllGather,
          /*comm=*/GPUCommunicationType::SINGLE_HOST,
          /*tensor_size=*/2 * 1024,
          /*num_nodes=*/4,
          /*network_througput_bytes=*/2 * 2048,
      },
      {
          /*opcode=*/HloOpcode::kAllToAll,
          /*comm=*/GPUCommunicationType::SINGLE_HOST,
          /*tensor_size=*/1024,
          /*num_nodes=*/1,
          /*network_througput_bytes=*/1024,
      },
      {
          /*opcode=*/HloOpcode::kAllToAll,
          /*comm=*/GPUCommunicationType::RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/2048,
      },
      {
          /*opcode=*/HloOpcode::kAllToAll,
          /*comm=*/GPUCommunicationType::NON_RAIL_ALIGNED,
          /*tensor_size=*/1024,
          /*num_nodes=*/2,
          /*network_througput_bytes=*/4096,
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
            /*test_name=*/"AR_rail_aligned_exact_nodes",
            /*spec=*/
            {
                /*opcode=*/HloOpcode::kAllReduce,
                /*comm=*/
                GPUCommunicationType::RAIL_ALIGNED,
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
                GPUCommunicationType::RAIL_ALIGNED,
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
                GPUCommunicationType::RAIL_ALIGNED,
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
                GPUCommunicationType::RAIL_ALIGNED,
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
                GPUCommunicationType::RAIL_ALIGNED,
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
                GPUCommunicationType::NON_RAIL_ALIGNED,
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
                GPUCommunicationType::NON_RAIL_ALIGNED,
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
                GPUCommunicationType::NON_RAIL_ALIGNED,
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
                GPUCommunicationType::NON_RAIL_ALIGNED,
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
                GPUCommunicationType::NON_RAIL_ALIGNED,
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
                GPUCommunicationType::SINGLE_HOST,
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
                GPUCommunicationType::SINGLE_HOST,
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
                GPUCommunicationType::SINGLE_HOST,
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
                GPUCommunicationType::SINGLE_HOST,
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
                GPUCommunicationType::SINGLE_HOST,
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
                GPUCommunicationType::RAIL_ALIGNED,
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
                GPUCommunicationType::RAIL_ALIGNED,
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
                GPUCommunicationType::RAIL_ALIGNED,
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
                GPUCommunicationType::RAIL_ALIGNED,
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
                GPUCommunicationType::RAIL_ALIGNED,
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
                GPUCommunicationType::NON_RAIL_ALIGNED,
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
                GPUCommunicationType::NON_RAIL_ALIGNED,
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
                GPUCommunicationType::NON_RAIL_ALIGNED,
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
                GPUCommunicationType::NON_RAIL_ALIGNED,
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
                GPUCommunicationType::NON_RAIL_ALIGNED,
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
                GPUCommunicationType::SINGLE_HOST,
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
                GPUCommunicationType::SINGLE_HOST,
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
                GPUCommunicationType::SINGLE_HOST,
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
                GPUCommunicationType::SINGLE_HOST,
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
                GPUCommunicationType::RAIL_ALIGNED,
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
                GPUCommunicationType::RAIL_ALIGNED,
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
                GPUCommunicationType::RAIL_ALIGNED,
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
                GPUCommunicationType::RAIL_ALIGNED,
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
                GPUCommunicationType::RAIL_ALIGNED,
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
                GPUCommunicationType::NON_RAIL_ALIGNED,
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
                GPUCommunicationType::NON_RAIL_ALIGNED,
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
                GPUCommunicationType::NON_RAIL_ALIGNED,
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
                GPUCommunicationType::NON_RAIL_ALIGNED,
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
                GPUCommunicationType::NON_RAIL_ALIGNED,
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
                GPUCommunicationType::SINGLE_HOST,
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
                GPUCommunicationType::SINGLE_HOST,
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
                GPUCommunicationType::SINGLE_HOST,
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
                GPUCommunicationType::SINGLE_HOST,
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
                GPUCommunicationType::SINGLE_HOST,
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
                GPUCommunicationType::RAIL_ALIGNED,
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
                GPUCommunicationType::NON_RAIL_ALIGNED,
                /*tensor_size=*/1024,
                /*num_nodes=*/2,
            },
            /*expected_duration=*/absl::Milliseconds(250),
        },
        {
            /*test_name=*/"A2A_single_host_exact_match",
            {
                /*opcode=*/HloOpcode::kAllToAll,
                /*comm=*/
                GPUCommunicationType::SINGLE_HOST,
                /*tensor_size=*/1024,
                /*num_nodes=*/1,
            },
            /*expected_duration=*/absl::Seconds(1),
        },
    }),
    [](const TestParamInfo<CollectiveInterpolationTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST(CollectiveInterpolatorTest, LoadsDefaultProfile) {
  auto device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo(
      stream_executor::CudaComputeCapability::Hopper());
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
        ROOT _ = f32[256] all-reduce(p),
        to_apply=wrapped_add,
        replica_groups=[1,8]<=[8],
        use_global_device_ids=true,
        channel_id=1
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(kHlo));
  HloCollectiveInstruction* instr = Cast<HloCollectiveInstruction>(
      module->entry_computation()->root_instruction());

  EXPECT_TRUE(interpolator->EstimatedRuntime(*instr).has_value());
}

}  // namespace
}  // namespace xla::gpu
