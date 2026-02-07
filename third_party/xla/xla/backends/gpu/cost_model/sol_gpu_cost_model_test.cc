/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/cost_model/sol_gpu_cost_model.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

constexpr int64_t kEightMB = 8 * 1024 * 1024;  // 8MB

using ::testing::TestWithParam;
using ::testing::ValuesIn;

struct LatencyTestCase {
  SolGPUCostModel::CollectiveType collective_type;
  int num_nodes;
  absl::Duration expected_latency;
};

class SolGPUCostModelTest : public TestWithParam<LatencyTestCase> {
 protected:
  SolGPUCostModelTest()
      : model_({
            /*nccl_op_launch_time=*/absl::Microseconds(100),
            /*nic_speed_gbps=*/100,
            /*chunk_prep_time=*/absl::Microseconds(100),
            /*rtt=*/absl::Microseconds(100),
            /*gpus_per_node=*/8,
            /*chunk_size_bytes=*/4 * 1024 * 1024,
        }) {}
  SolGPUCostModel model_;
};

TEST_P(SolGPUCostModelTest, TestLatency) {
  const LatencyTestCase& test_case = GetParam();
  absl::Duration actual_latency;
  if (test_case.collective_type == SolGPUCostModel::CollectiveType::kAllToAll) {
    actual_latency =
        absl::Trunc(*model_.AllToAllLatency(kEightMB, test_case.num_nodes,
                                            /*num_communicators=*/1),
                    absl::Microseconds(1));
  } else {
    actual_latency =
        absl::Trunc(*model_.RingLatency(kEightMB, test_case.num_nodes,
                                        test_case.collective_type,
                                        /*num_communicators=*/1),
                    absl::Microseconds(1));
  }
  EXPECT_EQ(actual_latency, test_case.expected_latency);
}

INSTANTIATE_TEST_SUITE_P(SolGPUCostModelTests, SolGPUCostModelTest,
                         ValuesIn<LatencyTestCase>({
                             {SolGPUCostModel::CollectiveType::kAllGather,
                              /*num_nodes=*/1, absl::Microseconds(284)},
                             {SolGPUCostModel::CollectiveType::kAllGather,
                              /*num_nodes=*/2, absl::Microseconds(485)},
                             {SolGPUCostModel::CollectiveType::kAllGather,
                              /*num_nodes=*/4, absl::Microseconds(885)},
                             {SolGPUCostModel::CollectiveType::kAllReduce,
                              /*num_nodes=*/1, absl::Microseconds(468)},
                             {SolGPUCostModel::CollectiveType::kAllReduce,
                              /*num_nodes=*/2, absl::Microseconds(870)},
                             {SolGPUCostModel::CollectiveType::kAllReduce,
                              /*num_nodes=*/4, absl::Microseconds(1670)},
                             {SolGPUCostModel::CollectiveType::kReduceScatter,
                              /*num_nodes=*/1, absl::Microseconds(284)},
                             {SolGPUCostModel::CollectiveType::kReduceScatter,
                              /*num_nodes=*/2, absl::Microseconds(485)},
                             {SolGPUCostModel::CollectiveType::kReduceScatter,
                              /*num_nodes=*/4, absl::Microseconds(885)},
                             {SolGPUCostModel::CollectiveType::kSendRecv,
                              /*num_nodes=*/1, absl::Microseconds(292)},
                             {SolGPUCostModel::CollectiveType::kSendRecv,
                              /*num_nodes=*/2, absl::Microseconds(485)},
                             {SolGPUCostModel::CollectiveType::kAllToAll,
                              /*num_nodes=*/1, absl::Microseconds(100)},
                             {SolGPUCostModel::CollectiveType::kAllToAll,
                              /*num_nodes=*/2, absl::Microseconds(1745)},
                             {SolGPUCostModel::CollectiveType::kAllToAll,
                              /*num_nodes=*/4, absl::Microseconds(4966)},
                         }));

TEST(SolGPUCostModelGetConfigTest, ConfigForHopper) {
  constexpr absl::string_view kDummyModule = R"(
    HloModule noop

    ENTRY main {
      ROOT constant = f32[] constant(0)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kDummyModule));

  se::DeviceDescription device_info;
  device_info.set_name("NVIDIA H100 80GB HBM3");
  SolGPUCostModel::Config config =
      SolGPUCostModel::GetConfig(module.get(), device_info);
  EXPECT_EQ(static_cast<int>(config.nic_speed_gbps), 25);
}

TEST(SolGPUCostModelGetConfigTest, ConfigForBlackwell) {
  constexpr absl::string_view kDummyModule = R"(
    HloModule noop

    ENTRY main {
      ROOT constant = f32[] constant(0)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kDummyModule));

  se::DeviceDescription device_info;
  device_info.set_name("NVIDIA B200");
  SolGPUCostModel::Config config =
      SolGPUCostModel::GetConfig(module.get(), device_info);
  EXPECT_EQ(static_cast<int>(config.nic_speed_gbps), 50);
  // Allow a tolerance of 10 nanoseconds for chunk_prep_time
  EXPECT_LT(absl::AbsDuration(config.chunk_prep_time - absl::Microseconds(2)),
            absl::Nanoseconds(10));
}

TEST(SolGPUCostModelGetConfigTest, ConfigForDefaultGPU) {
  constexpr absl::string_view kDummyModule = R"(
    HloModule noop

    ENTRY main {
      ROOT constant = f32[] constant(0)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kDummyModule));

  se::DeviceDescription device_info;
  device_info.set_name("NVIDIA H200");
  SolGPUCostModel::Config config =
      SolGPUCostModel::GetConfig(module.get(), device_info);
  EXPECT_EQ(static_cast<int>(config.nic_speed_gbps), 50);
}

}  // namespace
}  // namespace xla::gpu
