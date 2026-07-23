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
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/builder/lib/prng.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
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

// Builds a 64-bit key by mixing a per-op seed with the replica and partition
// IDs.
//
// The key is created as:
// (Seed ^ ((OpID * PRIME_CONSTANT) >> 32)) ^ ReplicaId ^ (PartitionId << 32)
//
// The rationale behind this: We want to generate a unique key for each RNG
// call that is stable as long as the HLO Graph does not change, and produces
// different keys on different devices (matching TPU behavior).
// With parallel compilation, we can no longer rely on a global counter, so we
// instead explicitly mix the op id with the replica and partition id.
//
// The multiplication and shift is a way to diffuse the individual components
// into the key and increase the entropy (as opposed to e.g. just XOR'ing all
// bits) - otherwise, ReplicaId and PartitionId would often clash.
//
// Params:
//   builder: The XlaBuilder to use for the key generation.
//   op_seed: The pre-calculated seed (OpID * PRIME_CONSTANT) >> 32) as
//     constant.
//   config: The HloModuleConfig to use for the key generation.
//
// Returns:
//   The 64-bit key to use for the RNG.
XlaOp BuildPhiloxKey(XlaBuilder* builder, XlaOp op_seed,
                     const HloModuleConfig& config) {
  FrontendAttributes attributes;
  attributes.mutable_map()->emplace(kXlaCseSafeZeroOperandAttr, "true");
  builder->SetFrontendAttributes(attributes);
  XlaOp seed_u64 =
      CustomCall(builder, "GetRngSeed", {}, ShapeUtil::MakeShape(U64, {}));
  builder->ClearFrontendAttributes();

  XlaOp replica_u64;
  bool has_replica = config.replica_count() > 1;
  if (has_replica) {
    XlaOp replica_id = ReplicaId(builder);
    replica_u64 = ConvertElementType(replica_id, U64);
  }

  XlaOp partition_u64;
  bool has_partition =
      !config.use_spmd_partitioning() && config.num_partitions() > 1;
  if (has_partition) {
    XlaOp partition_id = internal::XlaBuilderFriend::BuildPartitionId(
        builder, ShapeUtil::MakeShape(U32, {}));
    partition_u64 = ConvertElementType(partition_id, U64);
    XlaOp shift_32 = ConstantR0<uint64_t>(builder, 32);
    partition_u64 = ShiftLeft(partition_u64, shift_32);
  }

  XlaOp mixed_seed = Xor(seed_u64, op_seed);
  if (has_replica && has_partition) {
    XlaOp replica_and_partition = Xor(replica_u64, partition_u64);
    return Xor(mixed_seed, replica_and_partition);
  }
  if (has_replica) {
    return Xor(mixed_seed, replica_u64);
  }
  if (has_partition) {
    return Xor(mixed_seed, partition_u64);
  }
  return mixed_seed;
}

absl::StatusOr<HloComputation*> GetComputationForRng(HloInstruction* rng) {
  XlaBuilder builder("rng");
  const Shape u64_shape = ShapeUtil::MakeShape(xla::U64, {});
  const Shape u128_shape = ShapeUtil::MakeShape(xla::U64, {2});
  const Shape& result_shape = rng->shape();

  XlaOp op_seed = Parameter(&builder, 0, u64_shape, "op_seed");
  XlaOp state = Parameter(&builder, 1, u128_shape, "state");
  XlaOp a_or_mean =
      Parameter(&builder, 2, rng->operand(0)->shape(), "a_or_mean");
  XlaOp b_or_sigma =
      Parameter(&builder, 3, rng->operand(1)->shape(), "b_or_sigma");

  XlaOp key = BuildPhiloxKey(&builder, op_seed, rng->GetModule()->config());

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
  auto key_tuple =
      std::make_tuple(rng->random_distribution(), rng->shape(),
                      rng->operand(0)->shape(), rng->operand(1)->shape());
  auto [it, inserted] =
      expanded_rng_instructions_.try_emplace(key_tuple, nullptr);
  HloComputation* rng_computation = it->second;
  if (inserted) {
    ASSIGN_OR_RETURN(rng_computation, GetComputationForRng(rng));
    it->second = rng_computation;
  }
  HloComputation* computation = rng->parent();

  // The unique op_seed is passed as op_seed constant, and the key calculation
  // is performed inside the cached rng_computation to reduce caller graph size.
  // We pre-calculate a seed with higher entropy to reduce the chance of
  // collisions.
  uint64_t op_id = static_cast<uint64_t>(rng->unique_id());
  uint64_t op_seed = (op_id * 0x9E3779B97F4A7C15ULL) >> 32;
  HloInstruction* op_seed_hlo =
      MakeR0ConstantHlo<uint64_t>(computation, op_seed);

  const Shape u128_shape = ShapeUtil::MakeShape(xla::U64, {2});
  HloInstruction* state =
      computation->AddInstruction(HloInstruction::CreateRngGetAndUpdateState(
          u128_shape, GetNumberOf32bitUnits(rng->shape())));

  VLOG(2) << "Rng op_seed " << op_seed_hlo->ToString();
  VLOG(2) << "Rng state " << state->ToString();

  HloInstruction* new_rng = computation->AddInstruction(
      HloInstruction::CreateCall(rng->shape(),
                                 {op_seed_hlo, state, rng->mutable_operand(0),
                                  rng->mutable_operand(1)},
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
