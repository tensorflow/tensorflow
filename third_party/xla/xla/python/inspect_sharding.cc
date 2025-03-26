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

#include "xla/python/inspect_sharding.h"

#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "xla/service/custom_call_sharding_helper.h"
#include "xla/service/spmd/spmd_partitioner_util.h"
#include "xla/xla_data.pb.h"

namespace jax {

using xla::HloInstruction;
using xla::HloSharding;

void InspectShardingSetError(JAX_InspectSharding_Callback_Args* args,
                             std::string msg) {
  auto* tmp_error = new std::string(std::move(msg));
  args->error_txt = tmp_error->c_str();
  args->error_scratch = tmp_error;
  args->free_error = +[](JAX_InspectSharding_Callback_Args* args) {
    delete reinterpret_cast<std::string*>(args->error_scratch);
  };
}
std::optional<xla::HloSharding> InspectShardingReadArgs(
    JAX_InspectSharding_Callback_Args* args) {
  xla::OpSharding proto;
  if (args->sharding_spec_size > std::numeric_limits<int>::max() ||
      !proto.ParseFromArray(args->sharding_spec, args->sharding_spec_size)) {
    InspectShardingSetError(args,
                            "inspect_sharding: error parsing OpShardingProto");
    return std::nullopt;
  }
  auto result = xla::HloSharding::FromProto(std::move(proto));
  if (!result.ok()) {
    InspectShardingSetError(args, std::string(result.status().message()));
  }
  return std::move(*result);
}

class InspectShardingCallPartitioner : public xla::CustomCallPartitioner {
 public:
  absl::Status Partition(xla::spmd::SpmdPartitioningVisitor* partitioner,
                         HloInstruction* instruction) const override {
    const HloInstruction* operand = instruction->operand(0);
    if (!operand->has_sharding()) {
      return xla::Internal(
          "Inspect sharding called but no sharding is available.");
    }
    std::string sharding_spec =
        operand->sharding().ToProto().SerializeAsString();
    JAX_InspectSharding_Callback_Args args;
    args.sharding_spec = sharding_spec.data();
    args.sharding_spec_size = sharding_spec.size();
    args.error_txt = nullptr;
    const auto& str = instruction->raw_backend_config_string();
    if (str.size() != sizeof(JAX_InspectSharding_Callback)) {
      return xla::Internal("Invalid config string for inspect sharding.");
    }
    JAX_InspectSharding_Callback cb;
    memcpy(&cb, str.data(), sizeof(JAX_InspectSharding_Callback));
    cb.call(cb.data, &args);
    if (args.error_txt) {
      auto result =
          xla::Internal("Error calling inspect_sharding: %s", args.error_txt);
      args.free_error(&args);
      return result;
    }
    partitioner->SetPartitionedHlo(
        instruction,
        partitioner->GetPartitionedHlo(instruction->mutable_operand(0)));
    return absl::OkStatus();
  }
  HloSharding PropagateUserSharding(
      const HloInstruction* instruction, const HloInstruction* user,
      const HloSharding& sharding) const override {
    return sharding;
  }
  std::optional<HloSharding> InferShardingFromOperands(
      const HloInstruction* instruction) const override {
    const HloInstruction* operand = instruction->operand(0);
    if (!operand->has_sharding()) {
      return std::nullopt;
    }
    return operand->sharding();
  }
  bool IsCustomCallShardable(const HloInstruction* instruction) const override {
    return true;
  }
  bool CanSideEffectingHaveReplicatedSharding() const override { return true; }
};

namespace {
struct Registerer {
  Registerer() {
    RegisterCustomCallPartitioner(
        "InspectSharding",
        std::make_unique<jax::InspectShardingCallPartitioner>());
  }
};
Registerer custom_call_registerer;
}  // namespace

}  // namespace jax
