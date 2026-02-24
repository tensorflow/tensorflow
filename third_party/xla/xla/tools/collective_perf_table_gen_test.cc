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

#include "xla/tools/collective_perf_table_gen.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/tests/hlo_test_base.h"

namespace xla::gpu {
namespace {

using ::testing::Each;
using ::testing::Gt;
using ::testing::Property;

class CollectivePerfTableGenTest : public HloTestBase {
  void SetUp() override {
    if (!backend()
             .default_stream_executor()
             ->GetDeviceDescription()
             .gpu_compute_capability()
             .IsCuda()) {
      GTEST_SKIP() << "Not built with --config=cuda";
    }
    cfg_.dry_run = true;
  }

 protected:
  CollectivePerfTableGen::Config cfg_;
};

TEST_F(CollectivePerfTableGenTest, EmptyConfigReturnsEmptyProto) {
  std::unique_ptr<CollectivePerfTableGen> gen =
      CollectivePerfTableGen::Create(cfg_);
  EXPECT_EQ(gen->ComputeTable().entries_size(), 0);
}

TEST_F(CollectivePerfTableGenTest, ConstantStepGeneratesConfigs) {
  cfg_.collective_types = {
      CollectivePerfTableGen::CollectiveType::ALL_REDUCE,
      CollectivePerfTableGen::CollectiveType::ALL_GATHER,
      CollectivePerfTableGen::CollectiveType::REDUCE_SCATTER,
  };
  cfg_.replica_groups_list.emplace_back("[1,1]<=[1]");
  CollectivePerfTableGen::StepSpec spec{
      /*start=*/4,
      /*stop=*/20,
      /*step=*/4,
      /*factor=*/0,
  };
  cfg_.tensor_size_bytes_spec = spec;

  std::unique_ptr<CollectivePerfTableGen> gen =
      CollectivePerfTableGen::Create(cfg_);

  DeviceHloInstructionProfiles profiles = gen->ComputeTable();
  EXPECT_EQ(profiles.entries_size(), 1);
  EXPECT_EQ(profiles.entries().begin()->second.entries_size(), 15);
}

TEST_F(CollectivePerfTableGenTest, FactorStepGeneratesConfigsForCollective) {
  cfg_.collective_types = {
      CollectivePerfTableGen::CollectiveType::ALL_REDUCE,
      CollectivePerfTableGen::CollectiveType::ALL_GATHER,
      CollectivePerfTableGen::CollectiveType::REDUCE_SCATTER,
      CollectivePerfTableGen::CollectiveType::ALL_TO_ALL,
  };
  cfg_.replica_groups_list.emplace_back("[1,1]<=[1]");
  CollectivePerfTableGen::StepSpec spec{
      /*start=*/4,
      /*stop=*/32,
      /*step=*/0,
      /*factor=*/2,
  };
  cfg_.tensor_size_bytes_spec = spec;

  std::unique_ptr<CollectivePerfTableGen> gen =
      CollectivePerfTableGen::Create(cfg_);

  DeviceHloInstructionProfiles profiles = gen->ComputeTable();
  for (const auto& value : profiles.entries().begin()->second.entries()) {
    LOG(INFO) << "  value: " << value.DebugString();
  }
  EXPECT_EQ(profiles.entries_size(), 1);

  // 28 = 4 * (4 for collective + 1 for collective permute * 3)
  EXPECT_EQ(profiles.entries().begin()->second.entries_size(), 16);
}

TEST_F(CollectivePerfTableGenTest, FactorStepGeneratesConfigsForPermute) {
  cfg_.collective_types = {
      CollectivePerfTableGen::CollectiveType::COLLECTIVE_PERMUTE,
  };
  cfg_.replica_groups_list.emplace_back("[1,4]<=[4]");
  CollectivePerfTableGen::StepSpec spec{
      /*start=*/4,
      /*stop=*/32,
      /*step=*/0,
      /*factor=*/2,
  };
  cfg_.tensor_size_bytes_spec = spec;

  std::unique_ptr<CollectivePerfTableGen> gen =
      CollectivePerfTableGen::Create(cfg_);

  DeviceHloInstructionProfiles profiles = gen->ComputeTable();
  for (const auto& value : profiles.entries().begin()->second.entries()) {
    LOG(INFO) << "  value: " << value.DebugString();
  }
  EXPECT_EQ(profiles.entries_size(), 1);

  EXPECT_EQ(profiles.entries().begin()->second.entries_size(), 12);
}

TEST_F(CollectivePerfTableGenTest, HappyPathWorks) {
  cfg_.collective_types = {
      CollectivePerfTableGen::CollectiveType::ALL_REDUCE,
  };
  cfg_.replica_groups_list.emplace_back("[1,1]<=[1]");
  cfg_.dry_run = false;
  CollectivePerfTableGen::StepSpec spec{
      /*start=*/4,
      /*stop=*/32,
      /*step=*/0,
      /*factor=*/2,
  };
  cfg_.tensor_size_bytes_spec = spec;
  std::unique_ptr<CollectivePerfTableGen> gen =
      CollectivePerfTableGen::Create(cfg_);

  DeviceHloInstructionProfiles profiles = gen->ComputeTable();

  EXPECT_EQ(profiles.entries_size(), 1);
  EXPECT_THAT(
      profiles.entries().begin()->second.entries(),
      Each(Property(&HloInstructionProfile::network_throughput_bytes_per_sec,
                    Gt(0))));
}

TEST_F(CollectivePerfTableGenTest, BuildSourceTargetPairsOneWay) {
  EXPECT_EQ(BuildSourceTargetPairs(
                CollectivePermuteCostModelType::kIntraPartitionOneWay, 4),
            "{{0,2},{1,3}}");
  EXPECT_EQ(BuildSourceTargetPairs(
                CollectivePermuteCostModelType::kIntraPartitionOneWay, 8),
            "{{0,4},{1,5},{2,6},{3,7}}");
}

TEST_F(CollectivePerfTableGenTest, BuildSourceTargetPairsTwoWayAllMutual) {
  EXPECT_EQ(
      BuildSourceTargetPairs(
          CollectivePermuteCostModelType::kIntraPartitionTwoWayAllMutual, 4),
      "{{0,1},{1,0},{2,3},{3,2}}");
  EXPECT_EQ(
      BuildSourceTargetPairs(
          CollectivePermuteCostModelType::kIntraPartitionTwoWayAllMutual, 8),
      "{{0,1},{1,0},{2,3},{3,2},{4,5},{5,4},{6,7},{7,6}}");
}

TEST_F(CollectivePerfTableGenTest, BuildSourceTargetPairsTwoWayHasNonMutual) {
  EXPECT_EQ(
      BuildSourceTargetPairs(
          CollectivePermuteCostModelType::kIntraPartitionTwoWayHasNonMutual, 4),
      "{{0,1},{1,2},{2,3},{3,0}}");
  EXPECT_EQ(
      BuildSourceTargetPairs(
          CollectivePermuteCostModelType::kIntraPartitionTwoWayHasNonMutual, 8),
      "{{0,1},{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},{7,0}}");
}

}  // namespace
}  // namespace xla::gpu
