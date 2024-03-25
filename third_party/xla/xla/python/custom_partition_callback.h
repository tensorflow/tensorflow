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
#include <string_view>
#include <tuple>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/service/custom_call_sharding_helper.h"

extern "C" {

struct JAX_CustomCallPartitioner_string {
  const char* data;
  size_t size;
};

struct JAX_CustomCallPartitioner_aval {
  JAX_CustomCallPartitioner_string shape;
  bool has_sharding;
  JAX_CustomCallPartitioner_string sharding;
};

// General callback information containing api versions, the result error
// message and the cleanup function to free any temporary memory that is backing
// the results. Arguments are always owned by the caller, and results are owned
// by the cleanup_fn. These should never be used directly. Args and results
// should be serialized via the PopulateArgs, ReadArgs, PopulateResults,
// ConsumeResults functions defined below.
struct JAX_CustomCallPartitioner_version_and_error {
  int64_t api_version;
  void* data;  // out
  // cleanup_fn cleans up any returned results. The caller must finish with all
  // uses by the point the cleanup is called.
  void (*cleanup_fn)(void* data);  // out
  bool has_error;
  PJRT_Error_Code code;                        // out
  JAX_CustomCallPartitioner_string error_msg;  // out
};

struct JAX_CustomCallPartitioner_Partition_Args {
  JAX_CustomCallPartitioner_version_and_error header;

  size_t num_args;
  JAX_CustomCallPartitioner_aval* op_args;
  JAX_CustomCallPartitioner_aval op_result;
  JAX_CustomCallPartitioner_string backend_config;

  // out
  JAX_CustomCallPartitioner_string mlir_module;
  JAX_CustomCallPartitioner_string* args_sharding;
  JAX_CustomCallPartitioner_string result_sharding;
};

struct JAX_CustomCallPartitioner_InferShardingFromOperands_Args {
  JAX_CustomCallPartitioner_version_and_error header;

  size_t num_args;
  JAX_CustomCallPartitioner_aval* op_args;
  JAX_CustomCallPartitioner_string result_shape;
  JAX_CustomCallPartitioner_string backend_config;

  bool has_result_sharding;
  JAX_CustomCallPartitioner_string result_sharding;
};

struct JAX_CustomCallPartitioner_PropagateUserSharding_Args {
  JAX_CustomCallPartitioner_version_and_error header;

  JAX_CustomCallPartitioner_string backend_config;

  JAX_CustomCallPartitioner_string result_shape;

  JAX_CustomCallPartitioner_string result_sharding;  // inout
};

struct JAX_CustomCallPartitioner_Callbacks {
  int64_t version;
  void* private_data;
  void (*dtor)(JAX_CustomCallPartitioner_Callbacks* data);
  void (*partition)(JAX_CustomCallPartitioner_Callbacks* data,
                    JAX_CustomCallPartitioner_Partition_Args* args);
  void (*infer_sharding)(
      JAX_CustomCallPartitioner_Callbacks* data,
      JAX_CustomCallPartitioner_InferShardingFromOperands_Args* args);
  void (*propagate_user_sharding)(
      JAX_CustomCallPartitioner_Callbacks* data,
      JAX_CustomCallPartitioner_PropagateUserSharding_Args* args);
  bool can_side_effecting_have_replicated_sharding;
};

}  // extern "C"

namespace jax {

struct PartitionScratch {
  std::vector<std::string> strings;
  std::vector<JAX_CustomCallPartitioner_aval> op_args_storage;
};
PartitionScratch PopulateArgs(JAX_CustomCallPartitioner_Partition_Args* args,
                              const xla::HloInstruction* instruction);
absl::StatusOr<std::tuple<
    std::vector<xla::Shape>, std::vector<std::optional<xla::HloSharding>>,
    xla::Shape, std::optional<xla::HloSharding>, std::string_view>>
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
                          xla::Shape, std::string_view>>
ReadArgs(JAX_CustomCallPartitioner_InferShardingFromOperands_Args* args);
PartitionScratch PopulateArgs(
    JAX_CustomCallPartitioner_InferShardingFromOperands_Args* args,
    const xla::HloInstruction* instruction);
void PopulateResults(
    absl::StatusOr<std::optional<xla::HloSharding>> result,
    JAX_CustomCallPartitioner_InferShardingFromOperands_Args* args);
absl::StatusOr<std::optional<xla::HloSharding>> ConsumeResults(
    JAX_CustomCallPartitioner_InferShardingFromOperands_Args* args);

absl::StatusOr<std::tuple<xla::HloSharding, xla::Shape, std::string_view>>
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
