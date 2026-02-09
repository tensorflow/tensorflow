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
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

class CollectiveMetadataTest : public CollectiveOpsE2ETestBase {
 protected:
  CollectiveMetadataTest()
      : CollectiveOpsE2ETestBase(/*memory_size=*/32 * kMB,
                                 /*collectives_memory_size=*/1 * kMB) {}

  void SetUp() override {
    CollectiveOpsE2ETestBase::SetUp();
    if (!IsHopperAndHigher()) {
      GTEST_SKIP() << "Test requires Hopper or newer architecture since it's "
                      "using a multicast.";
    }
  }
};

TEST_F(CollectiveMetadataTest, ConstructCollectiveMetadata) {
  const absl::string_view kModuleStr = R"(
  HloModule test, replica_count=2

  ENTRY test_computation {
    param_0 = f32[4] parameter(0)
    param_1 = f32[4] parameter(1)
    copy_1 = f32[4]{0:S(1)} copy(param_1)

    const_0 = f32[1] constant({10})

    result_tuple = (f32[4], f32[4]{0:S(1)}, f32[1], u64[9]) custom-call(param_0, copy_1, const_0), custom_call_target="CollectiveMetadata", output_to_operand_aliasing={{0}: (0, {}), {1}: (1, {})}
    ROOT get_tuple_element = u64[9] get-tuple-element(result_tuple), index=3
  })";

  constexpr int kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto unoptimized_module,
      ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  Literal input_0 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  Literal input_1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(unoptimized_module),
                        /*arguments=*/std::vector<Literal*>{&input_0, &input_1},
                        /*run_hlo_passes=*/false));
  const std::vector<Literal>& result = execution_result.results;
  ASSERT_EQ(result.size(), kNumReplicas);

  absl::Span<const uint64_t> first_result_data = result[0].data<uint64_t>();
  absl::Span<const uint64_t> second_result_data = result[1].data<uint64_t>();
  constexpr int kNumElements = 9;
  ASSERT_EQ(first_result_data.size(), kNumElements);
  ASSERT_EQ(second_result_data.size(), kNumElements);

  EXPECT_EQ(first_result_data[0], 0) << "First result rank is not 0.";
  EXPECT_EQ(second_result_data[0], 1) << "Second result rank is not 1.";

  EXPECT_NE(first_result_data[1], 0)
      << "First result pointer to peers is NULL.";
  EXPECT_NE(second_result_data[1], 0)
      << "Second result pointer to peers is NULL.";

  EXPECT_NE(first_result_data[2], 0)
      << "First result pointer to multimem metadata is not set.";
  EXPECT_NE(second_result_data[2], 0)
      << "Second result pointer to multimem metadata is not set.";

  for (int i = 3; i < kNumElements; ++i) {
    EXPECT_NE(first_result_data[i], 0)
        << "First result param_to_peers is NULL.";
    EXPECT_EQ(second_result_data[i], first_result_data[i])
        << "Param_to_peers mismatch at index " << i
        << " in the first result: " << first_result_data[i]
        << " and in the second result: " << second_result_data[i];
  }
}

TEST_F(CollectiveMetadataTest, ConstructCollectiveMetadataForPartitions) {
  const absl::string_view kModuleStr = R"(
  HloModule test, allow_spmd_sharding_propagation_to_parameters={true}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=2

  ENTRY test_computation {
    param_0 = f32[4] parameter(0)
    param_1 = f32[4] parameter(1)

    const_0 = f32[1] constant({10})

    result_tuple = (f32[4], f32[4]{0}, f32[1], u64[9]) custom-call(param_0, param_1, const_0), custom_call_target="CollectiveMetadata", output_to_operand_aliasing={{0}: (0, {}), {1}: (1, {})}
    ROOT get_tuple_element = u64[9] get-tuple-element(result_tuple), index=3
  })";

  constexpr int kNumPartitions = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumPartitions)
      << "Test requires at least " << kNumPartitions << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto unoptimized_module,
      ParseAndReturnVerifiedModule(kModuleStr, /*replica_count=*/1,
                                   /*num_partitions=*/kNumPartitions));

  Literal input_0 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  Literal input_1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(unoptimized_module),
                        /*arguments=*/std::vector<Literal*>{&input_0, &input_1},
                        /*run_hlo_passes=*/false));
  const std::vector<Literal>& result = execution_result.results;
  ASSERT_EQ(result.size(), kNumPartitions);

  absl::Span<const uint64_t> first_result_data = result[0].data<uint64_t>();
  absl::Span<const uint64_t> second_result_data = result[1].data<uint64_t>();
  constexpr int kNumElements = 9;
  ASSERT_EQ(first_result_data.size(), kNumElements);
  ASSERT_EQ(second_result_data.size(), kNumElements);
}

TEST_F(CollectiveMetadataTest, BuildMultimemOnlyOncePerModuleExecution) {
  const absl::string_view kModuleStr = R"(
  HloModule test, replica_count=2

  ENTRY test_computation {
    param_0 = f32[1] parameter(0)
    copy_1 = f32[1]{0:S(1)} copy(param_0)

    first_result_tuple = (f32[1]{0:S(1)}, u64[5]) custom-call(copy_1), custom_call_target="CollectiveMetadata", output_to_operand_aliasing={{0}: (0, {})}
    first_result = u64[5] get-tuple-element(first_result_tuple), index=1
    second_result_tuple = (f32[1]{0:S(1)}, u64[5]) custom-call(copy_1), custom_call_target="CollectiveMetadata", output_to_operand_aliasing={{0}: (0, {})}
    second_result = u64[5] get-tuple-element(second_result_tuple), index=1
    ROOT result_tuple = (u64[5], u64[5]) tuple(first_result, second_result)
  })";

  constexpr int kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  Literal input_0 = LiteralUtil::CreateR1<float>({1.0f});
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/std::vector<Literal*>{&input_0},
                        /*run_hlo_passes=*/false));

  std::vector<Literal>& literals = execution_result.results;
  ASSERT_EQ(literals.size(), kNumReplicas);

  std::vector<Literal> first_result = literals[0].DecomposeTuple();
  std::vector<Literal> second_result = literals[1].DecomposeTuple();

  absl::Span<const uint64_t> first_device_first_result =
      first_result[0].data<uint64_t>();
  absl::Span<const uint64_t> first_device_second_result =
      first_result[1].data<uint64_t>();
  absl::Span<const uint64_t> second_device_first_result =
      second_result[0].data<uint64_t>();
  absl::Span<const uint64_t> second_device_second_result =
      second_result[1].data<uint64_t>();
  constexpr int kNumElements = 5;
  ASSERT_EQ(first_device_first_result.size(), kNumElements);
  ASSERT_EQ(first_device_second_result.size(), kNumElements);
  ASSERT_EQ(second_device_first_result.size(), kNumElements);
  ASSERT_EQ(second_device_second_result.size(), kNumElements);

  EXPECT_EQ(first_device_first_result[2], first_device_second_result[2])
      << "Multimem metadata should be the same for both results.";
  EXPECT_EQ(second_device_first_result[2], second_device_second_result[2])
      << "Multimem metadata should be the same for both results.";
}

TEST_F(CollectiveMetadataTest, ConstructCollectiveMetadataWithReplicaGroup) {
  const absl::string_view kModuleStr = R"(
  HloModule test, replica_count=4

  ENTRY test_computation {
    param_0 = f32[4] parameter(0)
    param_1 = f32[4] parameter(1)
    copy_1 = f32[4]{0:S(1)} copy(param_1)

    result_tuple = (f32[4], f32[4]{0:S(1)}, u64[7]) custom-call(param_0, copy_1), custom_call_target="CollectiveMetadata", output_to_operand_aliasing={{0}: (0, {}), {1}: (1, {})}, backend_config="{\"collective_metadata_backend_config\":{\"collective_devices\": { \"replica_groups\": [{\"replica_ids\": [0,1]}, {\"replica_ids\": [2,3]}]}}}"
    ROOT get_tuple_element = u64[7] get-tuple-element(result_tuple), index=2
  })";

  constexpr int kNumReplicas = 4;
  if (hlo_runner_->device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << hlo_runner_->device_count() << " available)";
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  Literal input_0 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  Literal input_1 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/std::vector<Literal*>{&input_0, &input_1},
                        /*run_hlo_passes=*/false));
  const std::vector<Literal>& result = execution_result.results;
  ASSERT_EQ(result.size(), kNumReplicas);
  absl::Span<const uint64_t> replica_0_result_0_data =
      result[0].data<uint64_t>();
  absl::Span<const uint64_t> replica_0_result_1_data =
      result[1].data<uint64_t>();
  absl::Span<const uint64_t> replica_1_result_0_data =
      result[2].data<uint64_t>();
  absl::Span<const uint64_t> replica_1_result_1_data =
      result[3].data<uint64_t>();

  // Check the rank in the first position.
  constexpr int kNumElements = 7;
  ASSERT_EQ(replica_0_result_0_data.size(), kNumElements);
  ASSERT_EQ(replica_0_result_1_data.size(), kNumElements);
  ASSERT_EQ(replica_1_result_0_data.size(), kNumElements);
  ASSERT_EQ(replica_1_result_1_data.size(), kNumElements);

  EXPECT_EQ(replica_0_result_0_data[0], 0);
  EXPECT_EQ(replica_0_result_1_data[0], 1);
  EXPECT_EQ(replica_1_result_0_data[0], 0);
  EXPECT_EQ(replica_1_result_1_data[0], 1);

  // Check pointer to peers in the second position.
  EXPECT_NE(replica_0_result_0_data[1], 0);
  EXPECT_NE(replica_0_result_1_data[1], 0);
  EXPECT_NE(replica_1_result_0_data[1], 0);
  EXPECT_NE(replica_1_result_1_data[1], 0);

  // Check pointer to multimem metadata in the third position.
  EXPECT_NE(replica_0_result_0_data[2], 0);
  EXPECT_NE(replica_0_result_1_data[2], 0);
  EXPECT_NE(replica_1_result_0_data[2], 0);
  EXPECT_NE(replica_1_result_1_data[2], 0);

  // Check param_to_peers structure.
  for (int i = 3; i < kNumElements; ++i) {
    EXPECT_NE(replica_0_result_0_data[i], 0);
    EXPECT_EQ(replica_0_result_1_data[i], replica_0_result_0_data[i]);
    EXPECT_NE(replica_1_result_0_data[i], 0);
    EXPECT_EQ(replica_1_result_1_data[i], replica_1_result_0_data[i]);
  }
}

}  // namespace
}  // namespace xla
