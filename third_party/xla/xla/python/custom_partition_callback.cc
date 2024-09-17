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
#include "xla/python/custom_partition_callback.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/client/xla_computation.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/c/pjrt_c_api_custom_partitioner_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/service/call_inliner.h"
#include "xla/service/custom_call_sharding_helper.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/spmd/spmd_partitioner_util.h"
#include "xla/util.h"

namespace xla {

absl::StatusOr<HloInstruction*> InlineHloComputation(
    HloInstruction* instruction, HloComputation* computation,
    HloComputation::Builder* builder, std::vector<HloInstruction*> operands,
    std::function<int64_t()> new_channel, const std::string& suffix) {
  HloCloneContext context(instruction->GetModule(), suffix);

  absl::flat_hash_map<HloInstruction*, HloInstruction*> replacements;
  auto resolve = [&](HloInstruction* inst) -> absl::StatusOr<HloInstruction*> {
    auto it = replacements.find(inst);
    if (it == replacements.end()) {
      return absl::InternalError(
          absl::StrCat("Could not find mapping for: ", inst->ToString()));
    }
    return it->second;
  };

  for (auto* inst : computation->MakeInstructionPostOrder()) {
    if (inst->opcode() == HloOpcode::kParameter) {
      replacements.emplace(inst, operands[inst->parameter_number()]);
    } else {
      std::vector<HloInstruction*> new_operands;
      new_operands.reserve(inst->operand_count());
      for (HloInstruction* operand : inst->mutable_operands()) {
        TF_ASSIGN_OR_RETURN(auto* new_operand, resolve(operand));
        new_operands.push_back(new_operand);
      }
      auto* new_inst = builder->AddInstruction(
          inst->CloneWithNewOperands(inst->shape(), new_operands, &context));
      HloChannelInstruction* channel_instr =
          DynCast<HloChannelInstruction>(new_inst);
      if (channel_instr && channel_instr->channel_id().has_value()) {
        new_inst->set_channel_id(new_channel());
      }
      replacements.emplace(inst, new_inst);
    }
  }
  return resolve(computation->root_instruction());
}

class CApiCustomCallPartitioner : public xla::CustomCallPartitioner {
 public:
  explicit CApiCustomCallPartitioner(JAX_CustomCallPartitioner_Callbacks* c_fns)
      : c_fns_(c_fns) {}
  ~CApiCustomCallPartitioner() override { c_fns_->dtor(c_fns_); }
  absl::Status Partition(spmd::SpmdPartitioningVisitor* partitioner,
                         HloInstruction* instruction) const override {
    JAX_CustomCallPartitioner_Partition_Args args;
    auto scratch = jax::PopulateArgs(&args, instruction);
    c_fns_->partition(c_fns_, &args);

    XlaComputation computation;
    std::vector<HloSharding> arg_shardings;
    std::optional<HloSharding> result_sharding;
    std::string mlir_module;
    TF_ASSIGN_OR_RETURN(std::tie(mlir_module, arg_shardings, result_sharding),
                        jax::ConsumeResults(&args));
    TF_RETURN_IF_ERROR(ParseMlirModuleStringAndConvertToXlaComputation(
        mlir_module, computation, /*use_tuple_args=*/false,
        /*return_tuple=*/false));
    auto hlo_module_config =
        xla::HloModule::CreateModuleConfigFromProto(
            computation.proto(), xla::DefaultDebugOptionsIgnoringFlags())
            .value();
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                        xla::HloModule::CreateFromProto(computation.proto(),
                                                        hlo_module_config));
    std::vector<HloInstruction*> operands;
    operands.reserve(instruction->operand_count());
    if (arg_shardings.size() != instruction->operand_count()) {
      return xla::Internal(
          "Shardings returned from partitioning %s must match: %d vs %d",
          instruction->ToString(), arg_shardings.size(),
          instruction->operand_count());
    }
    for (size_t i = 0; i < instruction->operand_count(); ++i) {
      operands.push_back(
          partitioner->GetPartitionedHlo(instruction->mutable_operand(i))
              .Reshard(arg_shardings[i])
              .hlo());
    }

    // The custom call module does not go through the main compiler pipeline,
    // so inline all calls here explicitly, since some targets require it.
    HloPassPipeline pipeline("custom-call-inliner");
    pipeline.AddPass<CallInliner>();
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module.get(), {}).status());

    TF_ASSIGN_OR_RETURN(
        auto* partitioned_hlo,
        InlineHloComputation(
            instruction, hlo_module->entry_computation(),
            partitioner->builder(), operands,
            [partitioner]() { return partitioner->NewChannel(); },
            "_custom_call_lowering_rule"));
    partitioned_hlo->set_sharding(result_sharding.value());

    spmd::PartitionedHlo result_partitioned =
        spmd::PartitionedHlo(partitioned_hlo, instruction->shape(),
                             partitioner->MakePartitioningState())
            .Reshard(instruction->sharding());

    partitioner->SetPartitionedHlo(instruction, result_partitioned);
    return absl::OkStatus();
  }
  HloSharding PropagateUserSharding(
      const HloInstruction* instruction, const HloInstruction* user,
      const HloSharding& sharding) const override {
    JAX_CustomCallPartitioner_PropagateUserSharding_Args args;
    auto scratch = jax::PopulateArgs(&args, instruction, sharding);
    c_fns_->propagate_user_sharding(c_fns_, &args);
    auto status_or_result = jax::ConsumeResults(&args);
    TF_CHECK_OK(status_or_result.status());
    return *status_or_result;
  }
  std::optional<HloSharding> InferShardingFromOperands(
      const HloInstruction* instruction) const override {
    JAX_CustomCallPartitioner_InferShardingFromOperands_Args args;
    auto scratch = jax::PopulateArgs(&args, instruction);
    c_fns_->infer_sharding(c_fns_, &args);
    auto status_or_result = jax::ConsumeResults(&args);
    TF_CHECK_OK(status_or_result.status());
    return *status_or_result;
  }
  bool IsCustomCallShardable(const HloInstruction* instruction) const override {
    return true;
  }
  bool CanSideEffectingHaveReplicatedSharding() const override {
    return c_fns_->can_side_effecting_have_replicated_sharding;
  }

  JAX_CustomCallPartitioner_Callbacks* c_fns_;
};

}  // namespace xla

namespace jax {

namespace {

void SetCAPIString(JAX_CustomCallPartitioner_string& out, std::string result,
                   std::vector<std::string>& scratch) {
  scratch.push_back(std::move(result));
  out.data = scratch.back().data();
  out.size = scratch.back().size();
}

std::string_view ToStringView(JAX_CustomCallPartitioner_string data) {
  return std::string_view(data.data, data.size);
}

void SetCAPIAval(JAX_CustomCallPartitioner_aval& result,
                 const xla::HloInstruction* inst,
                 std::vector<std::string>& scratch) {
  SetCAPIString(result.shape, inst->shape().SerializeAsString(), scratch);
  if (inst->has_sharding()) {
    result.has_sharding = true;
    SetCAPIString(result.sharding,
                  inst->sharding().ToProto().SerializeAsString(), scratch);
  } else {
    result.has_sharding = false;
  }
}

}  // namespace

struct ResultScratch {
  absl::Status status;
  std::vector<std::string> strings;
  std::vector<JAX_CustomCallPartitioner_string> op_args_sharding_storage;
};

absl::StatusOr<xla::HloSharding> ReadHloSharding(
    JAX_CustomCallPartitioner_string data) {
  xla::OpSharding proto;
  if (data.size > std::numeric_limits<int>::max() ||
      !proto.ParseFromArray(data.data, data.size)) {
    return absl::InternalError(
        "custom_call_sharding.cc: error parsing OpShardingProto");
  }
  return xla::HloSharding::FromProto(std::move(proto));
}

absl::StatusOr<xla::Shape> ReadHloShape(JAX_CustomCallPartitioner_string data) {
  xla::ShapeProto proto;
  if (data.size > std::numeric_limits<int>::max() ||
      !proto.ParseFromArray(data.data, data.size)) {
    return absl::InternalError(
        "custom_call_sharding.cc: error parsing xla::Shape");
  }
  return xla::Shape(proto);
}

bool PopulateErrorHeader(JAX_CustomCallPartitioner_version_and_error& header,
                         absl::Status status) {
  header.has_error = !status.ok();
  if (header.has_error) {
    auto* status_copy = new absl::Status(status);
    header.data = status_copy;
    header.cleanup_fn = reinterpret_cast<void (*)(void*)>(
        +[](absl::Status* data) { delete data; });
    header.code = pjrt::StatusCodeToPjrtErrorCode(status_copy->code());
    header.error_msg.data = status_copy->message().data();
    header.error_msg.size = status_copy->message().size();
  }
  return header.has_error;
}

absl::Status ConsumeHeader(
    JAX_CustomCallPartitioner_version_and_error& header) {
  if (header.has_error) {
    return absl::Status(pjrt::PjrtErrorCodeToStatusCode(header.code),
                        ToStringView(header.error_msg));
  }
  return absl::OkStatus();
}

void PopulateResults(
    absl::StatusOr<std::tuple<std::string, std::vector<xla::HloSharding>,
                              xla::HloSharding>>
        results,
    JAX_CustomCallPartitioner_Partition_Args* args) {
  if (PopulateErrorHeader(args->header, results.status())) {
    return;
  }
  auto* scratch = new ResultScratch;
  args->header.data = scratch;
  args->header.cleanup_fn = reinterpret_cast<void (*)(void*)>(
      +[](ResultScratch* data) { delete data; });
  auto& [mlir_module, shardings, result_shardings] = *results;
  scratch->strings.reserve(2 + args->num_args);
  SetCAPIString(args->mlir_module, std::move(mlir_module), scratch->strings);
  SetCAPIString(args->result_sharding,
                result_shardings.ToProto().SerializeAsString(),
                scratch->strings);
  scratch->op_args_sharding_storage.resize(args->num_args);
  for (size_t i = 0; i < args->num_args; ++i) {
    SetCAPIString(scratch->op_args_sharding_storage[i],
                  shardings[i].ToProto().SerializeAsString(), scratch->strings);
  }
  args->args_sharding = scratch->op_args_sharding_storage.data();
}

absl::StatusOr<
    std::tuple<std::string, std::vector<xla::HloSharding>, xla::HloSharding>>
ConsumeResults(JAX_CustomCallPartitioner_Partition_Args* args) {
  absl::Cleanup cleanup = [args] {
    args->header.cleanup_fn(args->header.data);
  };
  TF_RETURN_IF_ERROR(ConsumeHeader(args->header));
  TF_ASSIGN_OR_RETURN(auto result_sharding,
                      ReadHloSharding(args->result_sharding));
  std::vector<xla::HloSharding> arg_shardings;
  arg_shardings.reserve(args->num_args);
  for (size_t i = 0; i < args->num_args; ++i) {
    TF_ASSIGN_OR_RETURN(auto arg_sharding,
                        ReadHloSharding(args->args_sharding[i]));
    arg_shardings.push_back(std::move(arg_sharding));
  }
  return std::tuple<std::string, std::vector<xla::HloSharding>,
                    xla::HloSharding>(
      std::string(ToStringView(args->mlir_module)), std::move(arg_shardings),
      std::move(result_sharding));
}

PartitionScratch PopulateArgs(JAX_CustomCallPartitioner_Partition_Args* args,
                              const xla::HloInstruction* instruction) {
  args->header.api_version = 0;
  args->header.data = nullptr;
  args->header.cleanup_fn = nullptr;
  PartitionScratch scratch;
  scratch.op_args_storage.resize(instruction->operand_count());
  scratch.strings.reserve(instruction->operand_count() * 2 + 2);
  size_t i = 0;
  for (xla::HloInstruction* operand : instruction->operands()) {
    SetCAPIAval(scratch.op_args_storage[i], operand, scratch.strings);
    ++i;
  }
  args->num_args = instruction->operand_count();
  args->op_args = scratch.op_args_storage.data();
  SetCAPIAval(args->op_result, instruction, scratch.strings);
  args->backend_config.data = instruction->raw_backend_config_string().data();
  args->backend_config.size = instruction->raw_backend_config_string().size();
  return scratch;
}

absl::StatusOr<std::tuple<
    std::vector<xla::Shape>, std::vector<std::optional<xla::HloSharding>>,
    xla::Shape, std::optional<xla::HloSharding>, std::string_view>>
ReadArgs(JAX_CustomCallPartitioner_Partition_Args* args) {
  std::vector<xla::Shape> shapes;
  std::vector<std::optional<xla::HloSharding>> shardings;
  shapes.reserve(args->num_args);
  shardings.reserve(args->num_args);
  for (size_t i = 0; i < args->num_args; ++i) {
    TF_ASSIGN_OR_RETURN(auto shape, ReadHloShape(args->op_args[i].shape));
    shapes.push_back(shape);
    if (args->op_args[i].has_sharding) {
      TF_ASSIGN_OR_RETURN(auto sharding,
                          ReadHloSharding(args->op_args[i].sharding));
      shardings.push_back(std::move(sharding));
    } else {
      shardings.push_back(std::nullopt);
    }
  }

  TF_ASSIGN_OR_RETURN(auto result_shape, ReadHloShape(args->op_result.shape));
  std::optional<xla::HloSharding> result_sharding;
  if (args->op_result.has_sharding) {
    TF_ASSIGN_OR_RETURN(result_sharding,
                        ReadHloSharding(args->op_result.sharding));
  }
  return std::tuple<std::vector<xla::Shape>,
                    std::vector<std::optional<xla::HloSharding>>, xla::Shape,
                    std::optional<xla::HloSharding>, std::string_view>(
      std::move(shapes), std::move(shardings), std::move(result_shape),
      std::move(result_sharding), ToStringView(args->backend_config));
}

absl::StatusOr<std::tuple<std::vector<xla::Shape>,
                          std::vector<std::optional<xla::HloSharding>>,
                          xla::Shape, std::string_view>>
ReadArgs(JAX_CustomCallPartitioner_InferShardingFromOperands_Args* args) {
  std::vector<xla::Shape> shapes;
  std::vector<std::optional<xla::HloSharding>> shardings;
  shapes.reserve(args->num_args);
  shardings.reserve(args->num_args);
  for (size_t i = 0; i < args->num_args; ++i) {
    TF_ASSIGN_OR_RETURN(auto shape, ReadHloShape(args->op_args[i].shape));
    shapes.push_back(shape);
    if (args->op_args[i].has_sharding) {
      TF_ASSIGN_OR_RETURN(auto sharding,
                          ReadHloSharding(args->op_args[i].sharding));
      shardings.push_back(std::move(sharding));
    } else {
      shardings.push_back(std::nullopt);
    }
  }

  TF_ASSIGN_OR_RETURN(auto result_shape, ReadHloShape(args->result_shape));
  return std::tuple<std::vector<xla::Shape>,
                    std::vector<std::optional<xla::HloSharding>>, xla::Shape,
                    std::string_view>(std::move(shapes), std::move(shardings),
                                      std::move(result_shape),
                                      ToStringView(args->backend_config));
}

PartitionScratch PopulateArgs(
    JAX_CustomCallPartitioner_InferShardingFromOperands_Args* args,
    const xla::HloInstruction* instruction) {
  args->header.api_version = 0;
  args->header.data = nullptr;
  args->header.cleanup_fn = nullptr;
  PartitionScratch scratch;
  scratch.op_args_storage.resize(instruction->operand_count());
  scratch.strings.reserve(instruction->operand_count() * 2 + 2);
  size_t i = 0;
  for (xla::HloInstruction* operand : instruction->operands()) {
    SetCAPIAval(scratch.op_args_storage[i], operand, scratch.strings);
    ++i;
  }
  args->num_args = instruction->operand_count();
  args->op_args = scratch.op_args_storage.data();
  SetCAPIString(args->result_shape, instruction->shape().SerializeAsString(),
                scratch.strings);
  args->backend_config.data = instruction->raw_backend_config_string().data();
  args->backend_config.size = instruction->raw_backend_config_string().size();
  return scratch;
}

void PopulateResults(
    absl::StatusOr<std::optional<xla::HloSharding>> result,
    JAX_CustomCallPartitioner_InferShardingFromOperands_Args* args) {
  if (PopulateErrorHeader(args->header, result.status())) {
    return;
  }
  args->has_result_sharding = result->has_value();
  if (result->has_value()) {
    auto* data = new std::string((*result)->ToProto().SerializeAsString());
    args->header.data = data;
    args->header.cleanup_fn = reinterpret_cast<void (*)(void*)>(
        +[](std::string* data) { delete data; });
    args->result_sharding.data = data->data();
    args->result_sharding.size = data->size();
  } else {
    args->header.cleanup_fn = +[](void*) {};
  }
}
absl::StatusOr<std::optional<xla::HloSharding>> ConsumeResults(
    JAX_CustomCallPartitioner_InferShardingFromOperands_Args* args) {
  absl::Cleanup cleanup = [args] {
    args->header.cleanup_fn(args->header.data);
  };
  TF_RETURN_IF_ERROR(ConsumeHeader(args->header));
  if (!args->has_result_sharding) {
    return std::nullopt;
  }
  return ReadHloSharding(args->result_sharding);
}

absl::StatusOr<std::tuple<xla::HloSharding, xla::Shape, std::string_view>>
ReadArgs(JAX_CustomCallPartitioner_PropagateUserSharding_Args* args) {
  TF_ASSIGN_OR_RETURN(auto shape, ReadHloShape(args->result_shape));
  TF_ASSIGN_OR_RETURN(auto sharding, ReadHloSharding(args->result_sharding));
  return std::tuple<xla::HloSharding, xla::Shape, std::string_view>(
      std::move(sharding), std::move(shape),
      ToStringView(args->backend_config));
}
PartitionScratch PopulateArgs(
    JAX_CustomCallPartitioner_PropagateUserSharding_Args* args,
    const xla::HloInstruction* instruction, const xla::HloSharding& sharding) {
  args->header.api_version = 0;
  args->header.data = nullptr;
  args->header.cleanup_fn = nullptr;
  PartitionScratch scratch;
  scratch.strings.reserve(2);
  SetCAPIString(args->result_sharding, sharding.ToProto().SerializeAsString(),
                scratch.strings);
  SetCAPIString(args->result_shape, instruction->shape().SerializeAsString(),
                scratch.strings);
  args->backend_config.data = instruction->raw_backend_config_string().data();
  args->backend_config.size = instruction->raw_backend_config_string().size();
  return scratch;
}

void PopulateResults(
    absl::StatusOr<xla::HloSharding> result,
    JAX_CustomCallPartitioner_PropagateUserSharding_Args* args) {
  if (PopulateErrorHeader(args->header, result.status())) {
    return;
  }
  auto* data = new std::string(result->ToProto().SerializeAsString());
  args->header.data = data;
  args->header.cleanup_fn = reinterpret_cast<void (*)(void*)>(
      +[](std::string* data) { delete data; });
  args->result_sharding.data = data->data();
  args->result_sharding.size = data->size();
}
absl::StatusOr<xla::HloSharding> ConsumeResults(
    JAX_CustomCallPartitioner_PropagateUserSharding_Args* args) {
  absl::Cleanup cleanup = [args] {
    args->header.cleanup_fn(args->header.data);
  };
  TF_RETURN_IF_ERROR(ConsumeHeader(args->header));
  return ReadHloSharding(args->result_sharding);
}

std::unique_ptr<xla::CustomCallPartitioner> CreateCApiCustomCallPartitioner(
    JAX_CustomCallPartitioner_Callbacks* c_fns) {
  return std::make_unique<xla::CApiCustomCallPartitioner>(c_fns);
}

}  // namespace jax
