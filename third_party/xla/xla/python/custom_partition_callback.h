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
#ifndef XLA_PYTHON_CUSTOM_PARTITION_CALLBACK_H_
#define XLA_PYTHON_CUSTOM_PARTITION_CALLBACK_H_

#include <memory>
#include <optional>
#include <string>
#include <tuple>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_custom_partitioner_extension.h"
#include "xla/service/custom_call_sharding_helper.h"

namespace jax {

struct PartitionScratch {
  std::vector<std::string> strings;
  std::vector<JAX_CustomCallPartitioner_aval> op_args_storage;
};
PartitionScratch PopulateArgs(JAX_CustomCallPartitioner_Partition_Args* args,
                              const xla::HloInstruction* instruction);
absl::StatusOr<std::tuple<
    std::vector<xla::Shape>, std::vector<std::optional<xla::HloSharding>>,
    xla::Shape, std::optional<xla::HloSharding>, absl::string_view>>
ReadArgs(JAX_CustomCallPartitioner_Partition_Args* args);
void PopulateResults(
    absl::StatusOr<std::tuple<std::string, std::vector<xla::HloSharding>,
                              xla::HloSharding>>
        results,
    JAX_CustomCallPartitioner_Partition_Args* args);
absl::StatusOr<
    std::tuple<std::string, std::vector<xla::HloSharding>, xla::HloSharding>>
ConsumeResults(JAX_CustomCallPartitioner_Partition_Args* args);

absl::StatusOr<std::tuple<std::vector<xla::Shape>,
                          std::vector<std::optional<xla::HloSharding>>,
                          xla::Shape, absl::string_view>>
ReadArgs(JAX_CustomCallPartitioner_InferShardingFromOperands_Args* args);
PartitionScratch PopulateArgs(
    JAX_CustomCallPartitioner_InferShardingFromOperands_Args* args,
    const xla::HloInstruction* instruction);
void PopulateResults(
    absl::StatusOr<std::optional<xla::HloSharding>> result,
    JAX_CustomCallPartitioner_InferShardingFromOperands_Args* args);
absl::StatusOr<std::optional<xla::HloSharding>> ConsumeResults(
    JAX_CustomCallPartitioner_InferShardingFromOperands_Args* args);

absl::StatusOr<std::tuple<xla::HloSharding, xla::Shape, absl::string_view>>
ReadArgs(JAX_CustomCallPartitioner_PropagateUserSharding_Args* args);
PartitionScratch PopulateArgs(
    JAX_CustomCallPartitioner_PropagateUserSharding_Args* args,
    const xla::HloInstruction* instruction, const xla::HloSharding& sharding);
void PopulateResults(
    absl::StatusOr<xla::HloSharding> result,
    JAX_CustomCallPartitioner_PropagateUserSharding_Args* args);
absl::StatusOr<xla::HloSharding> ConsumeResults(
    JAX_CustomCallPartitioner_PropagateUserSharding_Args* args);

// Wraps c-api callbacks with the custom-call partitioner.
std::unique_ptr<xla::CustomCallPartitioner> CreateCApiCustomCallPartitioner(
    JAX_CustomCallPartitioner_Callbacks* c_fns);

}  // namespace jax

#endif  // XLA_PYTHON_CUSTOM_PARTITION_CALLBACK_H_
