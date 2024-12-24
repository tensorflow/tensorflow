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

#include "xla/service/gpu/model/sol_gpu_cost_model.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "absl/time/time.h"
namespace xla {
namespace gpu {
namespace {
constexpr int64_t kTenMB = 10 * 1024 * 1024;  // 10MB

using ::testing::TestWithParam;
using ::testing::ValuesIn;

struct RingLatencyTestCase {
  SolGPUCostModel::CollectiveType collective_type;
  absl::Duration expected_latency;
};

class SolGPUCostModelTest : public TestWithParam<RingLatencyTestCase> {
 protected:
  SolGPUCostModelTest()
      : model_({
            /*nccl_op_launch_time=*/absl::Microseconds(100),
            /*nic_speed_gbps=*/100,
            /*chunk_prep_time=*/absl::Microseconds(100),
            /*rtt=*/absl::Microseconds(100),
            /*gpus_per_node=*/100,
            /*chunk_size_bytes=*/4 * 1024 * 1024,
        }) {}
  SolGPUCostModel model_;
};

TEST_P(SolGPUCostModelTest, TestRingLatency) {
  const RingLatencyTestCase& test_case = GetParam();
  absl::Duration actual_latency =
      absl::Trunc(model_.RingLatency(kTenMB, 1, test_case.collective_type),
                  absl::Microseconds(1));
  EXPECT_EQ(actual_latency, test_case.expected_latency);
}

INSTANTIATE_TEST_SUITE_P(
    SolGPUCostModelTests, SolGPUCostModelTest,
    ValuesIn<RingLatencyTestCase>({
        {SolGPUCostModel::CollectiveType::kAllGather, absl::Microseconds(298)},
        {SolGPUCostModel::CollectiveType::kAllReduce, absl::Microseconds(497)},
        {SolGPUCostModel::CollectiveType::kReduceScatter,
         absl::Microseconds(298)},
        {SolGPUCostModel::CollectiveType::kSendRecv, absl::Microseconds(350)},
    }));
}  // namespace
}  // namespace gpu
}  // namespace xla
