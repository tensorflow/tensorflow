/* Copyright 2023 The OpenXLA Authors.

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
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner.h"
#include "xla/tests/collective_ops_e2e_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::NotNull;

enum class RaggedAllToAllImplType {
  kNccl,
  kDecomposer,
  kOneShot,
};

class RaggedAllToAllTestBase : public CollectiveOpsWithFlagsBase {
 public:
  RaggedAllToAllTestBase(bool enable_async, RaggedAllToAllImplType impl_type)
      : CollectiveOpsWithFlagsBase(enable_async, /*enable_p2p_memcpy=*/false,
                                   /*memory_size=*/64 * kMB,
                                   /*collectives_memory_size=*/0),
        impl_type_(impl_type) {}

  // Creates random test data for a ragged-all-to-all.
  //
  // Ragged tensors which are ragged (have various size) along the second most
  // changing dimension only, i.e. shape such as [8, (4), 3]. In memory those
  // tensors are flattened out the outermost dimension.
  //
  // A ragged tensor is represented by three arrays: data, offsets, and sizes.
  //   * The data array holds the elements of the ragged tensor.
  //   * The offsets array holds the starting offset of each ragged row.
  //   * The sizes array holds the number of elements in each ragged row.
  //
  // A ragged-all-to-all of N replicas performance a collective transpose of the
  // ragged tensors. Each pair of replicas exchanges one ragged row. To generate
  // the test data we need to know the sizes of all ragged rows for each
  // replica.
  //
  // `input_sizes` is an array of shape [num_replicas, num_replicas,
  // num_updates_per_replica]. For concenivence, `input_sizes` can be a 2D
  // array, in that case `num_updates_per_replica` is assumed to be 1.
  absl::Status CreateRandomTestData(HloModule* module,
                                    Array<int64_t> input_sizes) {
    CHECK(inputs_.empty());
    if (input_sizes.num_dimensions() == 2) {
      input_sizes.Reshape({input_sizes.dim(0), input_sizes.dim(1), 1});
    }
    auto ragged_all_to_all =
        FindInstruction(module, HloOpcode::kRaggedAllToAll);
    EXPECT_THAT(ragged_all_to_all, NotNull());

    const std::vector<ReplicaGroup>& replica_groups =
        ragged_all_to_all->replica_groups();
    EXPECT_FALSE(replica_groups.empty());

    int64_t num_total_replicas = input_sizes.dim(0);
    int64_t num_replicas = replica_groups[0].replica_ids_size();

    EXPECT_TRUE(
        absl::c_all_of(replica_groups, [&](const ReplicaGroup& replica_group) {
          return replica_group.replica_ids_size() == num_replicas;
        }));

    inputs_.resize(num_total_replicas);
    expected_outputs_.resize(num_total_replicas);
    input_offsets_.resize(num_total_replicas);
    input_sizes_.resize(num_total_replicas);
    output_offsets_.resize(num_total_replicas);
    output_sizes_.resize(num_total_replicas);

    HloInstruction* output_param =
        module->entry_computation()->parameter_instruction(1);

    // The ragged-all-to-all accepts an output tensor as a parameter to allow
    // buffer reuse. We initialize the output tensor with -1 to make sure that
    // we don't accidentally overwrite data that is not part of the
    // ragged-all-to-all update.
    Array<float> output_init_data(output_param->shape().dimensions());
    output_init_data.Fill(-1);

    // Iterate over all replica groups and create random test data for each
    // group.
    for (const ReplicaGroup& replica_group : replica_groups) {
      Array<int64_t> input_sizes_per_replica_group(
          {num_replicas, input_sizes.dim(1), input_sizes.dim(2)});

      for (int64_t i = 0; i < num_replicas; ++i) {
        int64_t replica_id = replica_group.replica_ids(i);
        input_sizes_per_replica_group.UpdateSlice(
            input_sizes.Slice(
                {replica_id, 0, 0},
                {replica_id + 1, input_sizes.dim(1), input_sizes.dim(2)}),
            {i, 0, 0});
      }

      TF_RETURN_IF_ERROR(CreateRandomTestDataForReplicaGroup(
          module, input_sizes_per_replica_group, output_init_data,
          replica_group));
    }

    TF_ASSIGN_OR_RETURN(output_init_,
                        LiteralUtil::CreateFromArrayWithLayout(
                            output_init_data, output_param->shape().layout())
                            .Convert(output_param->shape().element_type()));
    return absl::OkStatus();
  }

  // Create random test data for a ragged-all-to-all for a single replica group.
  absl::Status CreateRandomTestDataForReplicaGroup(
      HloModule* module, Array<int64_t> input_sizes,
      const Array<float>& output_init_data, const ReplicaGroup& replica_group) {
    HloInstruction* input_param =
        module->entry_computation()->parameter_instruction(0);
    HloInstruction* output_param =
        module->entry_computation()->parameter_instruction(1);
    int64_t num_replicas = replica_group.replica_ids_size();

    Array<int64_t> output_sizes = input_sizes;
    output_sizes.TransposeDimensions({1, 0, 2});

    Array<int64_t> input_offsets = CalculateOffsetsFromSizes(input_sizes);
    Array<int64_t> output_offsets = CalculateOffsetsFromSizes(output_sizes);
    output_offsets.TransposeDimensions({1, 0, 2});

    std::vector<Array<float>> input_data(
        num_replicas, Array<float>(input_param->shape().dimensions()));
    std::vector<Array<float>> output_data(num_replicas, output_init_data);
    FillWithRandomData(input_data, output_data, input_offsets, output_offsets,
                       input_sizes);

    // Create literals from array data.
    for (int64_t i = 0; i < num_replicas; ++i) {
      int64_t replica_id = replica_group.replica_ids(i);
      TF_ASSIGN_OR_RETURN(inputs_[replica_id],
                          LiteralUtil::CreateFromArrayWithLayout(
                              input_data[i], input_param->shape().layout())
                              .Convert(input_param->shape().element_type()));

      TF_ASSIGN_OR_RETURN(expected_outputs_[replica_id],
                          LiteralUtil::CreateFromArrayWithLayout(
                              output_data[i], output_param->shape().layout())
                              .Convert(output_param->shape().element_type()));

      TF_ASSIGN_OR_RETURN(
          input_offsets_[replica_id],
          GetParameterLiteral(module, /*parameter_index=*/2, i, input_offsets));

      TF_ASSIGN_OR_RETURN(
          input_sizes_[replica_id],
          GetParameterLiteral(module, /*parameter_index=*/3, i, input_sizes));

      TF_ASSIGN_OR_RETURN(output_offsets_[replica_id],
                          GetParameterLiteral(module, /*parameter_index=*/4, i,
                                              output_offsets));
      TF_ASSIGN_OR_RETURN(
          output_sizes_[replica_id],
          GetParameterLiteral(module, /*parameter_index=*/5, i, output_sizes));
    }
    return absl::OkStatus();
  }

  // Returns a vector of pointers to the literals in the format needed for
  // ExecuteReplicated.
  std::vector<std::vector<Literal*>> GetInputLiteralPtrs() {
    std::vector<std::vector<Literal*>> input_literal_ptrs;
    for (int i = 0; i < inputs_.size(); ++i) {
      input_literal_ptrs.push_back({&inputs_[i], &output_init_,
                                    &input_offsets_[i], &input_sizes_[i],
                                    &output_offsets_[i], &output_sizes_[i]});
    }
    return input_literal_ptrs;
  }

 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions opts = CollectiveOpsWithFlagsBase::GetDebugOptionsForTest();
    opts.set_xla_gpu_unsupported_enable_ragged_all_to_all_decomposer(
        impl_type_ == RaggedAllToAllImplType::kDecomposer);
    opts.set_xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel(
        impl_type_ == RaggedAllToAllImplType::kOneShot);
    return opts;
  }

  // Computes ragged tensor offsets based on the sizes of the ragged rows.
  Array<int64_t> CalculateOffsetsFromSizes(const Array<int64_t>& sizes) {
    int64_t num_replicas = sizes.dim(0);
    int64_t num_updates_per_replica = sizes.dim(2);
    Array<int64_t> offsets(sizes.dimensions());
    for (int i = 0; i < num_replicas; ++i) {
      int64_t cur_offset = 0;
      for (int j = 0; j < num_replicas; ++j) {
        for (int k = 0; k < num_updates_per_replica; ++k) {
          offsets(i, j, k) = cur_offset;
          cur_offset += sizes(i, j, k);
        }
      }
    }
    return offsets;
  }

  // Fill the input and output tensors with random data. An all-to-all is
  // effectively a transpose. We generate a chunk of random data for each update
  // of each pair of replicas and write the chunk starting from the (i, j, k)
  // offset of the input tensor and starting from the (j, i, k) offset of the
  // output tensor.
  void FillWithRandomData(std::vector<Array<float>>& input_data,
                          std::vector<Array<float>>& output_data,
                          const Array<int64_t>& input_offsets,
                          const Array<int64_t>& output_offsets,
                          const Array<int64_t>& input_sizes) {
    int64_t num_replicas = input_sizes.dim(0);
    int64_t num_updates_per_replica = input_sizes.dim(2);
    std::vector<int64_t> start_indices(input_data[0].num_dimensions());
    std::vector<int64_t> chunk_sizes{input_data[0].dimensions().begin(),
                                     input_data[0].dimensions().end()};

    for (int i = 0; i < num_replicas; ++i) {
      for (int j = 0; j < num_replicas; ++j) {
        for (int k = 0; k < num_updates_per_replica; ++k) {
          chunk_sizes[0] = input_sizes(i, j, k);

          Array<float> chunk_data(chunk_sizes);
          chunk_data.FillRandomUniform(
              1, 127,
              /*seed=*/(i * num_replicas + j) * num_updates_per_replica + k);

          start_indices[0] = input_offsets(i, j, k);
          input_data[i].UpdateSlice(chunk_data, start_indices);

          start_indices[0] = output_offsets(i, j, k);
          output_data[j].UpdateSlice(chunk_data, start_indices);
        }
      }
    }
  }

  // Returns a literal for the given parameter of the given replica.
  absl::StatusOr<Literal> GetParameterLiteral(HloModule* module,
                                              int64_t parameter_index,
                                              int64_t replica_id,
                                              const Array<int64_t>& data) {
    HloInstruction* param =
        module->entry_computation()->parameter_instruction(parameter_index);

    int64_t num_replicas = data.dim(0);
    int64_t num_updates_per_replica = data.dim(2);
    Array<int64_t> replica_slice =
        data.Slice({replica_id, 0, 0},
                   {replica_id + 1, num_replicas, num_updates_per_replica});
    replica_slice.Reshape({num_replicas * num_updates_per_replica});
    return LiteralUtil::CreateFromArray(replica_slice)
        .Convert(param->shape().element_type());
  }

  // Literates for the input and output data, offset, and size parameters of
  // the ragged-all-to-all. Each vector contains one literal per replica.
  std::vector<Literal> inputs_;
  std::vector<Literal> input_offsets_;
  std::vector<Literal> input_sizes_;

  std::vector<Literal> expected_outputs_;
  std::vector<Literal> output_offsets_;
  std::vector<Literal> output_sizes_;

  Literal output_init_;

  RaggedAllToAllImplType impl_type_;
};

class RaggedAllToAllTest : public RaggedAllToAllTestBase,
                           public ::testing::WithParamInterface<
                               std::tuple<bool, RaggedAllToAllImplType>> {
 public:
  RaggedAllToAllTest()
      : RaggedAllToAllTestBase(std::get<0>(GetParam()),
                               std::get<1>(GetParam())) {}
};

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[4] parameter(0)
    output = f32[4] parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ROOT ra2a = f32[4] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleReplicatedStr, kNumReplicas));

  TF_ASSERT_OK(CreateRandomTestData(module.get(),
                                    /*input_sizes=*/{/*replica_0=*/{1, 1},
                                                     /*replica_1=*/{3, 1}}));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;

  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_CommandBuffer) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[4] parameter(0)
    output = f32[4] parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ROOT ra2a = f32[4] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleReplicatedStr, kNumReplicas));

  // Verify correctness of ragged-all-to-all when command buffers for
  // collectives are enabled.
  // As of Dec 2025, ragged-all-to-all command is not implemented, so this test
  // verifies that we don't try to accidentally create a command buffer and
  // crash.
  DebugOptions& debug_options =
      module->mutable_config().mutable_debug_options();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);
  debug_options.set_xla_gpu_graph_min_graph_size(1);

  ASSERT_OK(CreateRandomTestData(module.get(),
                                 /*input_sizes=*/{/*replica_0=*/{1, 1},
                                                  /*replica_1=*/{3, 1}}));

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;

  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_S4) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = s4[4,2]{1,0:E(4)} parameter(0)
    output = s4[4,2]{1,0:E(4)} parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ROOT ra2a = s4[4,2]{1,0:E(4)} ragged-all-to-all(input, output, 
      input_offsets, send_sizes, output_offsets, recv_sizes), 
      replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleReplicatedStr, kNumReplicas));

  TF_ASSERT_OK(CreateRandomTestData(module.get(),
                                    /*input_sizes=*/{/*replica_0=*/{1, 1},
                                                     /*replica_1=*/{3, 1}}));

  // `CreateRandomTestData` calculates sizes and offsets metadata, but fill
  // input and expected output literals with random floats. We need to manually
  // override them with s4 values.
  inputs_[0] = LiteralUtil::CreateR2<s4>(
      {{s4(1), s4(1)}, {s4(2), s4(2)}, {s4(0), s4(0)}, {s4(0), s4(0)}});
  inputs_[1] = LiteralUtil::CreateR2<s4>(
      {{s4(3), s4(3)}, {s4(4), s4(4)}, {s4(5), s4(5)}, {s4(6), s4(6)}});

  expected_outputs_[0] = LiteralUtil::CreateR2<s4>(
      {{s4(1), s4(1)}, {s4(3), s4(3)}, {s4(4), s4(4)}, {s4(5), s4(5)}});
  expected_outputs_[1] = LiteralUtil::CreateR2<s4>(
      {{s4(2), s4(2)}, {s4(6), s4(6)}, {s4(-1), s4(-1)}, {s4(-1), s4(-1)}});

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  // Check that ragged-all-to-all on S4 was converted to S8.
  // Skip this check for decomposer test, because there ragged-all-to-all was
  // lowered to all-to-all.
  if (std::get<1>(GetParam()) != RaggedAllToAllImplType::kDecomposer) {
    HloInstruction* ragged_all_to_all = FindInstruction(
        execution_result.optimized_module, HloOpcode::kRaggedAllToAll);
    ASSERT_NE(ragged_all_to_all, nullptr);
    EXPECT_EQ(ragged_all_to_all->shape().element_type(), S8);
  }

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_InputBufferLargerThanOutput) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[32] parameter(0)
    output = f32[16] parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ROOT ra2a = f32[16] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleReplicatedStr, kNumReplicas));

  TF_ASSERT_OK(CreateRandomTestData(module.get(),
                                    /*input_sizes=*/{/*replica_0=*/{8, 5},
                                                     /*replica_1=*/{4, 3}}));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_OutputBufferLargerThanInput) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[16] parameter(0)
    output = f32[32] parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ROOT ra2a = f32[32] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleReplicatedStr, kNumReplicas));

  TF_ASSERT_OK(CreateRandomTestData(module.get(),
                                    /*input_sizes=*/{/*replica_0=*/{4, 12},
                                                     /*replica_1=*/{5, 11}}));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_MultipleUpdates) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[8] parameter(0)
    output = f32[8] parameter(1)
    input_offsets = s32[4] parameter(2)
    send_sizes = s32[4] parameter(3)
    output_offsets = s32[4] parameter(4)
    recv_sizes = s32[4] parameter(5)
    ROOT ra2a = f32[8] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleReplicatedStr, config));

  TF_ASSERT_OK(CreateRandomTestData(
      module.get(), /*input_sizes=*/{/*replica_0=*/{{1, 2}, {2, 1}},
                                     /*replica_1=*/{{3, 1}, {1, 1}}}));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_MultiDimData) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = bf16[16, 5, 32] parameter(0)
    output = bf16[16, 5, 32] parameter(1)
    input_offsets = s64[2] parameter(2)
    send_sizes = s64[2] parameter(3)
    output_offsets = s64[2] parameter(4)
    recv_sizes = s64[2] parameter(5)
    ROOT ra2a = bf16[16, 5, 32] ragged-all-to-all(input, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleReplicatedStr, kNumReplicas));

  TF_ASSERT_OK(CreateRandomTestData(module.get(),
                                    /*input_sizes=*/{/*replica_0=*/{4, 7},
                                                     /*replica_1=*/{2, 5}}));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_Degenerate) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module

  ENTRY entry {
    input = f32[4] parameter(0)
    output = f32[4] parameter(1)
    input_offsets = s32[1] parameter(2)
    send_sizes = s32[1] parameter(3)
    output_offsets = s32[1] parameter(4)
    recv_sizes = s32[1] parameter(5)
    ROOT ra2a = f32[4] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{0},{1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleReplicatedStr, kNumReplicas));

  TF_ASSERT_OK(CreateRandomTestData(module.get(),
                                    /*input_sizes=*/{/*replica_0=*/{1},
                                                     /*replica_1=*/{3}}));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllTest, RaggedAllToAll_2GPUs_NonDefaultLayout) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module

  ENTRY entry {
    input = f32[16,4,8]{0,2,1} parameter(0)
    output = f32[16,4,8]{0,1,2} parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ROOT ra2a = f32[16,4,8]{0,1,2} ragged-all-to-all(input, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,1}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleReplicatedStr, kNumReplicas));

  auto ragged_all_to_all =
      FindInstruction(module.get(), HloOpcode::kRaggedAllToAll);
  EXPECT_THAT(ragged_all_to_all, NotNull());

  TF_ASSERT_OK(CreateRandomTestData(module.get(),
                                    /*input_sizes=*/{/*replica_0=*/{4, 7},
                                                     /*replica_1=*/{2, 5}}));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllTest,
       RaggedAllToAll_2GPUs_DevicesInReplicaGroupInReverseOrder) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[4] parameter(0)
    output = f32[4] parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ROOT ra2a = f32[4] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), replica_groups={{1,0}}
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleReplicatedStr, kNumReplicas));

  TF_ASSERT_OK(CreateRandomTestData(module.get(),
                                    /*input_sizes=*/{/*replica_0=*/{1, 1},
                                                     /*replica_1=*/{3, 1}}));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[0], results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[1], results[1]));
}

TEST_P(RaggedAllToAllTest, RaggedAllToAll_8GPUs) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[512, 5, 32] parameter(0)
    output = f32[512, 5, 32] parameter(1)
    input_offsets = s32[32] parameter(2)
    send_sizes = s32[32] parameter(3)
    output_offsets = s32[32] parameter(4)
    recv_sizes = s32[32] parameter(5)
    ROOT ra2a = f32[512, 5, 32] ragged-all-to-all(input, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,1,2,3,4,5,6,7}}
  })";

  const int64_t kNumReplicas = 8;
  const int64_t kNumUpdatesPerReplica = 4;
  if (hlo_runner_->device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << hlo_runner_->device_count() << " available)";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleReplicatedStr, kNumReplicas));

  Array<int64_t> input_sizes(
      {kNumReplicas, kNumReplicas, kNumUpdatesPerReplica});
  input_sizes.FillRandomUniform(0, 10);

  TF_ASSERT_OK(CreateRandomTestData(module.get(), input_sizes));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[i], results[i]));
  }
}

TEST_P(RaggedAllToAllTest, RaggedAllToAll_8GPUs_2ReplicasPerGroups) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[512, 5, 32] parameter(0)
    output = f32[512, 5, 32] parameter(1)
    input_offsets = s32[32] parameter(2)
    send_sizes = s32[32] parameter(3)
    output_offsets = s32[32] parameter(4)
    recv_sizes = s32[32] parameter(5)
    ROOT ra2a = f32[512, 5, 32] ragged-all-to-all(input, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,4},{1,5},{2,6},{3,7}}
  })";

  const int64_t kNumReplicas = 8;
  const int64_t kNumReplicasPerGroup = 2;
  const int64_t kNumUpdatesPerReplica = 16;
  if (hlo_runner_->device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << hlo_runner_->device_count() << " available)";
  }

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleReplicatedStr, kNumReplicas));

  Array<int64_t> input_sizes(
      {kNumReplicas, kNumReplicasPerGroup, kNumUpdatesPerReplica});
  input_sizes.FillRandomUniform(0, 10);

  TF_ASSERT_OK(CreateRandomTestData(module.get(), input_sizes));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[i], results[i]));
  }
}

TEST_P(RaggedAllToAllTest, RaggedAllToAll_8GPUs_4ReplicasPerGroups) {
  absl::string_view kModuleReplicatedStr = R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[512, 5, 32] parameter(0)
    output = f32[512, 5, 32] parameter(1)
    input_offsets = s32[32] parameter(2)
    send_sizes = s32[32] parameter(3)
    output_offsets = s32[32] parameter(4)
    recv_sizes = s32[32] parameter(5)
    ROOT ra2a = f32[512, 5, 32] ragged-all-to-all(input, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,1,2,3},{4,5,6,7}}
  })";

  const int64_t kNumReplicas = 8;
  const int64_t kNumReplicasPerGroup = 4;
  const int64_t kNumUpdatesPerReplica = 8;
  if (hlo_runner_->device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << hlo_runner_->device_count() << " available)";
  }

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleReplicatedStr, kNumReplicas));

  Array<int64_t> input_sizes(
      {kNumReplicas, kNumReplicasPerGroup, kNumUpdatesPerReplica});
  input_sizes.FillRandomUniform(0, 10);

  TF_ASSERT_OK(CreateRandomTestData(module.get(), input_sizes));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[i], results[i]));
  }
}

std::string RaggedAllToAllImplTypeName(
    RaggedAllToAllImplType ragged_all_to_all_impl_type) {
  switch (ragged_all_to_all_impl_type) {
    case RaggedAllToAllImplType::kNccl:
      return "nccl";
    case RaggedAllToAllImplType::kDecomposer:
      return "decomposer";
    case RaggedAllToAllImplType::kOneShot:
      return "one_shot";
    default:
      LOG(FATAL) << "Unknown ragged all-to-all implementation type.";
  }
}

INSTANTIATE_TEST_SUITE_P(
    RaggedAllToAllTest, RaggedAllToAllTest,
    ::testing::Combine(::testing::Bool(),
                       ::testing::Values(RaggedAllToAllImplType::kNccl,
                                         RaggedAllToAllImplType::kDecomposer,
                                         RaggedAllToAllImplType::kOneShot)),
    [](const ::testing::TestParamInfo<std::tuple<bool, RaggedAllToAllImplType>>&
           info) {
      return absl::StrCat(std::get<0>(info.param) ? "async" : "sync", "_",
                          RaggedAllToAllImplTypeName(std::get<1>(info.param)));
    });

class RaggedAllToAllMultiHostDecomposerTest
    : public RaggedAllToAllTestBase,
      public ::testing::WithParamInterface<std::tuple<int64_t, int64_t>> {
 public:
  RaggedAllToAllMultiHostDecomposerTest()
      : RaggedAllToAllTestBase(/*enable_async=*/false,
                               /*impl_type=*/RaggedAllToAllImplType::kOneShot) {
  }

 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        RaggedAllToAllTestBase::GetDebugOptionsForTest();
    debug_options
        .set_xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer(
            true);
    return debug_options;
  }
};

TEST_P(RaggedAllToAllMultiHostDecomposerTest, RaggedAllToAll_2GPUs_SliceSize1) {
  auto [num_input_rows, num_output_rows] = GetParam();

  std::string kModuleReplicatedStr =
      absl::Substitute(R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[$0,5,32] parameter(0)
    output = f32[$1,5,32] parameter(1)
    input_offsets = s32[32] parameter(2)
    send_sizes = s32[32] parameter(3)
    output_offsets = s32[32] parameter(4)
    recv_sizes = s32[32] parameter(5)
    ROOT ra2a = f32[$1,5,32] ragged-all-to-all(input, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,1}}
  })",
                       num_input_rows, num_output_rows);

  const int64_t kNumReplicas = 2;
  const int64_t kNumUpdatesPerReplica = 16;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);

  config.mutable_debug_options()
      .set_xla_gpu_unsupported_override_fast_interconnect_slice_size(1);

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleReplicatedStr, config));

  Array<int64_t> input_sizes(
      {kNumReplicas, kNumReplicas, kNumUpdatesPerReplica});
  input_sizes.FillRandomUniform(0, 10);

  TF_ASSERT_OK(CreateRandomTestData(module.get(), input_sizes));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[i], results[i]));
  }
}

TEST_P(RaggedAllToAllMultiHostDecomposerTest,
       RaggedAllToAll_8GPUs_SliceSize4_ShuffledReplicaGroups) {
  auto [num_input_rows, num_output_rows] = GetParam();

  std::string kModuleReplicatedStr =
      absl::Substitute(R"(
  HloModule module

  ENTRY entry {
    input = f32[$0,5,32] parameter(0)
    output = f32[$1,5,32] parameter(1)
    input_offsets = s32[32] parameter(2)
    send_sizes = s32[32] parameter(3)
    output_offsets = s32[32] parameter(4)
    recv_sizes = s32[32] parameter(5)
    ROOT ra2a = f32[$1,5,32] ragged-all-to-all(input, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,2,4,6,1,3,5,7}}
  })",
                       num_input_rows, num_output_rows);

  const int64_t kNumReplicas = 8;
  const int64_t kNumUpdatesPerReplica = 4;
  if (hlo_runner_->device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << hlo_runner_->device_count() << " available)";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleReplicatedStr, kNumReplicas));

  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_unsupported_override_fast_interconnect_slice_size(4);

  Array<int64_t> input_sizes(
      {kNumReplicas, kNumReplicas, kNumUpdatesPerReplica});
  input_sizes.FillRandomUniform(0, 10);

  TF_ASSERT_OK(CreateRandomTestData(module.get(), input_sizes));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[i], results[i]));
  }
}

TEST_P(RaggedAllToAllMultiHostDecomposerTest, RaggedAllToAll_8GPUs_SliceSize4) {
  auto [num_input_rows, num_output_rows] = GetParam();

  std::string kModuleReplicatedStr =
      absl::Substitute(R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[$0,5,32] parameter(0)
    output = f32[$1,5,32] parameter(1)
    input_offsets = s32[32] parameter(2)
    send_sizes = s32[32] parameter(3)
    output_offsets = s32[32] parameter(4)
    recv_sizes = s32[32] parameter(5)
    ROOT ra2a = f32[$1,5,32] ragged-all-to-all(input, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,1,2,3,4,5,6,7}}
  })",
                       num_input_rows, num_output_rows);

  const int64_t kNumReplicas = 8;
  const int64_t kNumUpdatesPerReplica = 4;
  if (hlo_runner_->device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << hlo_runner_->device_count() << " available)";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleReplicatedStr, kNumReplicas));

  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_unsupported_override_fast_interconnect_slice_size(4);

  Array<int64_t> input_sizes(
      {kNumReplicas, kNumReplicas, kNumUpdatesPerReplica});
  input_sizes.FillRandomUniform(0, 16);

  TF_ASSERT_OK(CreateRandomTestData(module.get(), input_sizes));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[i], results[i]));
  }
}

TEST_P(RaggedAllToAllMultiHostDecomposerTest,
       RaggedAllToAll_8GPUs_SliceSize4_2ReplicaGroups) {
  auto [num_input_rows, num_output_rows] = GetParam();

  std::string kModuleReplicatedStr =
      absl::Substitute(R"(
  HloModule module, num_partitions=1

  ENTRY entry {
    input = f32[$0,5,32] parameter(0)
    output = f32[$1,5,32] parameter(1)
    input_offsets = s32[32] parameter(2)
    send_sizes = s32[32] parameter(3)
    output_offsets = s32[32] parameter(4)
    recv_sizes = s32[32] parameter(5)
    ROOT ra2a = f32[$1,5,32] ragged-all-to-all(input, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,2,4,6},{1,3,5,7}}
  })",
                       num_input_rows, num_output_rows);

  const int64_t kNumReplicas = 8;
  const int64_t kNumReplicasPerGroup = 4;
  const int64_t kNumUpdatesPerReplica = 8;
  if (hlo_runner_->device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << hlo_runner_->device_count() << " available)";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           kModuleReplicatedStr, kNumReplicas));

  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_unsupported_override_fast_interconnect_slice_size(4);

  Array<int64_t> input_sizes(
      {kNumReplicas, kNumReplicasPerGroup, kNumUpdatesPerReplica});
  input_sizes.FillRandomUniform(0, 10);

  TF_ASSERT_OK(CreateRandomTestData(module.get(), input_sizes));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), GetInputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_outputs_[i], results[i]));
  }
}

INSTANTIATE_TEST_SUITE_P(
    RaggedAllToAllMultiHostDecomposerTest,
    RaggedAllToAllMultiHostDecomposerTest,
    ::testing::Values(std::make_tuple(512, 4096), std::make_tuple(4096, 512)),
    [](const ::testing::TestParamInfo<std::tuple<int64_t, int64_t>>& info) {
      if (std::get<0>(info.param) > std::get<1>(info.param)) {
        return absl::StrCat("combine_", std::get<0>(info.param), "_",
                            std::get<1>(info.param));
      }
      return absl::StrCat("dispatch_", std::get<0>(info.param), "_",
                          std::get<1>(info.param));
    });

}  // namespace
}  // namespace xla
