/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/rng_expander.h"

#include <cstdint>
#include <iterator>
#include <random>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/builder/lib/prng.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/side_effect_util.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

int64_t GetNumberOf32bitUnits(const Shape& shape) {
  int64_t bit_width = primitive_util::BitWidth(shape.element_type());
  CHECK(bit_width == 32 || bit_width == 64);
  int64_t num_elems = ShapeUtil::ElementsIn(shape);
  return num_elems * (bit_width / 32);
}

absl::StatusOr<HloInstruction*> ConvertSmallFpRngToF32Rng(HloInstruction* rng) {
  CHECK_EQ(rng->opcode(), HloOpcode::kRng);
  PrimitiveType primitive_type = rng->shape().element_type();
  CHECK(primitive_type == F16 || primitive_type == BF16);

  std::vector<HloInstruction*> new_operands;
  absl::c_transform(rng->operands(), std::back_inserter(new_operands),
                    [&](HloInstruction* operand) {
                      CHECK_EQ(operand->shape().element_type(), primitive_type);
                      return MakeConvertToHlo(operand, F32);
                    });

  Shape shape = ShapeUtil::ChangeElementType(rng->shape(), F32);
  HloComputation* computation = rng->parent();
  HloCloneContext context(computation->parent());
  HloInstruction* new_rng = computation->AddInstruction(
      rng->CloneWithNewOperands(shape, new_operands, &context));
  RETURN_IF_ERROR(new_rng->CopyAllControlDepsFrom(rng));

  RETURN_IF_ERROR(
      rng->ReplaceAllUsesWith(MakeConvertToHlo(new_rng, primitive_type)));
  RETURN_IF_ERROR(rng->DropAllControlDeps());

  // Since rng is a side-effecting instruction, we can't rely on DCE to remove
  // it.
  RETURN_IF_ERROR(computation->RemoveInstruction(rng));

  return new_rng;
}

absl::StatusOr<HloComputation*> GetComputationForRng(HloInstruction* rng) {
  XlaBuilder builder("rng");
  const Shape u64_shape = ShapeUtil::MakeShape(xla::U64, {});
  const Shape u128_shape = ShapeUtil::MakeShape(xla::U64, {2});
  const Shape& result_shape = rng->shape();

  XlaOp key = Parameter(&builder, 0, u64_shape, "key");
  XlaOp state = Parameter(&builder, 1, u128_shape, "state");
  XlaOp a_or_mean =
      Parameter(&builder, 2, rng->operand(0)->shape(), "a_or_mean");
  XlaOp b_or_sigma =
      Parameter(&builder, 3, rng->operand(1)->shape(), "b_or_sigma");

  auto generator = [](xla::XlaOp key, xla::XlaOp state,
                      const xla::Shape& shape) {
    return PhiloxBitGenerator(key, state, shape);
  };

  XlaOp result;
  if (rng->random_distribution() == RNG_NORMAL) {
    result =
        NormalFloatingPointDistribution(key, state, generator, result_shape)
            .value;
    // Transform standard normal distribution to normal distribution with the
    // given mean and standard deviation.
    result = a_or_mean + (b_or_sigma * result);
  } else {
    CHECK_EQ(rng->random_distribution(), RNG_UNIFORM);
    if (primitive_util::IsFloatingPointType(result_shape.element_type())) {
      result = UniformFloatingPointDistribution(
                   key, state, generator, a_or_mean, b_or_sigma, result_shape)
                   .value;
    } else {
      result = UniformIntDistribution(key, state, generator, a_or_mean,
                                      b_or_sigma, result_shape)
                   .value;
    }
  }

  ASSIGN_OR_RETURN(XlaComputation xla_computation, builder.Build());
  return XlaComputationToHloComputation(xla_computation, rng->GetModule());
}

}  // namespace

bool RngExpander::InstructionMatchesPattern(HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kRng;
}

absl::StatusOr<HloInstruction*> RngExpander::ExpandInstruction(
    HloInstruction* rng) {
  VLOG(2) << "Expand rng instruction " << rng->ToString();
  PrimitiveType old_primitive_type = rng->shape().element_type();
  if (primitive_util::BitWidth(old_primitive_type) < 32) {
    ASSIGN_OR_RETURN(rng, ConvertSmallFpRngToF32Rng(rng));
  }
  HloComputation*& rng_computation = expanded_rng_instructions_[std::make_tuple(
      rng->random_distribution(), rng->shape(), rng->operand(0)->shape(),
      rng->operand(1)->shape())];
  if (!rng_computation) {
    ASSIGN_OR_RETURN(rng_computation, GetComputationForRng(rng));
  }
  HloComputation* computation = rng->parent();

  HloInstruction* seed_u64 =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          ShapeUtil::MakeShape(U64, {}), {}, "GetRngSeed"));
  FrontendAttributes attributes;
  attributes.mutable_map()->emplace(kXlaCseSafeZeroOperandAttr, "true");
  seed_u64->set_frontend_attributes(attributes);

  // Spells out the computation for the Philox key:
  // Key = (Seed ^ ((OpID * 0x9E3779B97F4A7C15) >> 32)) ^ ReplicaId ^
  // (PartitionId << 32)
  HloInstruction* op_id_c = MakeR0ConstantHlo<uint64_t>(
      computation, static_cast<uint64_t>(rng->unique_id()));
  HloInstruction* multiplier =
      MakeR0ConstantHlo<uint64_t>(computation, 0x9E3779B97F4A7C15ULL);
  ASSIGN_OR_RETURN(HloInstruction * mix_mul,
                   MakeBinaryHlo(HloOpcode::kMultiply, op_id_c, multiplier));
  HloInstruction* shift_32 = MakeR0ConstantHlo<uint64_t>(computation, 32);
  ASSIGN_OR_RETURN(
      HloInstruction * mix_shifted,
      MakeBinaryHlo(HloOpcode::kShiftRightLogical, mix_mul, shift_32));

  // SeedU64 ^ ((OpIDU64 * ...) >> 32)
  ASSIGN_OR_RETURN(HloInstruction * mixed_seed,
                   MakeBinaryHlo(HloOpcode::kXor, seed_u64, mix_shifted));

  // Multi-Device Support: Combine the RNG seed with the ReplicaID and
  // PartitionID only when configured, avoiding explicit PartitionId in SPMD
  // mode which aborts partitioning.
  const auto& config = rng->GetModule()->config();
  HloInstruction* replica_u64;
  if (config.replica_count() > 1) {
    HloInstruction* replica_id = computation->AddInstruction(
        HloInstruction::CreateReplicaId(ShapeUtil::MakeShape(U32, {})));
    replica_u64 = MakeConvertToHlo(replica_id, U64);
  } else {
    replica_u64 = MakeR0ConstantHlo<uint64_t>(computation, 0);
  }

  HloInstruction* partition_u64;
  if (!config.use_spmd_partitioning() && config.num_partitions() > 1) {
    HloInstruction* partition_id = computation->AddInstruction(
        HloInstruction::CreatePartitionId(ShapeUtil::MakeShape(U32, {})));
    partition_u64 = MakeConvertToHlo(partition_id, U64);
  } else {
    partition_u64 = MakeR0ConstantHlo<uint64_t>(computation, 0);
  }

  // Key = (mixed_seed ^ replica_u64) ^ (partition_u64 << 32)
  // `mixed_seed` combines the global seed with a hash of the operation ID.
  // `replica_u64` is the ReplicaId (or 0 if single-replica).
  // `partition_u64` is the PartitionId (or 0 if single-partition or
  // using SPMD).
  ASSIGN_OR_RETURN(HloInstruction * key_with_repl,
                   MakeBinaryHlo(HloOpcode::kXor, mixed_seed, replica_u64));

  ASSIGN_OR_RETURN(
      HloInstruction * part_shifted,
      MakeBinaryHlo(HloOpcode::kShiftLeft, partition_u64, shift_32));

  ASSIGN_OR_RETURN(HloInstruction * key,
                   MakeBinaryHlo(HloOpcode::kXor, key_with_repl, part_shifted));

  const Shape u128_shape = ShapeUtil::MakeShape(xla::U64, {2});
  HloInstruction* state =
      computation->AddInstruction(HloInstruction::CreateRngGetAndUpdateState(
          u128_shape, GetNumberOf32bitUnits(rng->shape())));

  VLOG(2) << "Rng key " << key->ToString();
  VLOG(2) << "Rng state " << state->ToString();

  HloInstruction* new_rng =
      computation->AddInstruction(HloInstruction::CreateCall(
          rng->shape(),
          {key, state, rng->mutable_operand(0), rng->mutable_operand(1)},
          rng_computation));

  RETURN_IF_ERROR(new_rng->CopyAllControlDepsFrom(rng));

  RETURN_IF_ERROR(rng->ReplaceAllUsesWith(new_rng));
  RETURN_IF_ERROR(rng->DropAllControlDeps());

  // Since rng is a side-effecting instruction, we can't rely on DCE to remove
  // it.
  RETURN_IF_ERROR(computation->RemoveInstruction(rng));

  // Returns nullptr to OpExpanderPass::Run to indicate the old rng instruction
  // has been replaced with the new rng instruction.
  return nullptr;
}

}  // namespace xla
