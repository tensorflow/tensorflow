/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/rng_expander.h"

#include "tensorflow/compiler/xla/client/lib/prng.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"

namespace xla {

namespace {

int64 GlobalRandomValue() {
  static auto* mu = new tensorflow::mutex();
  static std::mt19937_64 rng{42};
  tensorflow::mutex_lock l(*mu);
  return rng();
}

int64 GetNumberOf32bitUnits(const Shape& shape) {
  int64 bit_width = primitive_util::BitWidth(shape.element_type());
  CHECK(bit_width == 32 || bit_width == 64);
  int64 num_elems = ShapeUtil::ElementsIn(shape);
  return num_elems * (bit_width / 32);
}

StatusOr<HloInstruction*> ConvertSmallFpRngToF32Rng(HloInstruction* rng) {
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
  TF_RETURN_IF_ERROR(new_rng->CopyAllControlDepsFrom(rng));

  TF_RETURN_IF_ERROR(
      rng->ReplaceAllUsesWith(MakeConvertToHlo(new_rng, primitive_type)));
  TF_RETURN_IF_ERROR(rng->DropAllControlDeps());

  // Since rng is a side-effecting instruction, we can't rely on DCE to remove
  // it.
  TF_RETURN_IF_ERROR(computation->RemoveInstruction(rng));

  return new_rng;
}

StatusOr<HloComputation*> GetComputationForRng(HloInstruction* rng) {
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

  TF_ASSIGN_OR_RETURN(XlaComputation xla_computation, builder.Build());

  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      xla_computation.GetProgramShape());
  HloModuleConfig config(program_shape);
  TF_ASSIGN_OR_RETURN(auto new_module, HloModule::CreateFromProto(
                                           xla_computation.proto(), config));
  HloModule* module = rng->parent()->parent();
  HloCloneContext context(module);
  return module->DeepCloneComputation(new_module->entry_computation(),
                                      &context);
}

}  // namespace

bool RngExpander::InstructionMatchesPattern(HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kRng;
}

StatusOr<HloInstruction*> RngExpander::ExpandInstruction(HloInstruction* rng) {
  VLOG(2) << "Expand rng instruction " << rng->ToString();
  PrimitiveType old_primitive_type = rng->shape().element_type();
  if (primitive_util::BitWidth(old_primitive_type) < 32) {
    TF_ASSIGN_OR_RETURN(rng, ConvertSmallFpRngToF32Rng(rng));
  }
  HloComputation*& rng_computation = expanded_rng_instructions_[std::make_tuple(
      rng->random_distribution(), rng->shape(), rng->operand(0)->shape(),
      rng->operand(1)->shape())];
  if (!rng_computation) {
    TF_ASSIGN_OR_RETURN(rng_computation, GetComputationForRng(rng));
  }
  HloComputation* computation = rng->parent();

  // A random number generated by the per module random number generator.
  int64 module_random_value = rng->GetModule()->RandomNew64();

  // A value specified by the configuration or generated by a global random
  // number generator.
  int64 module_config_seed = rng->parent()->parent()->config().seed();
  int64 global_random_value =
      module_config_seed != 0 ? module_config_seed : GlobalRandomValue();

  // Construct the key using the two random values above.
  HloInstruction* key = MakeR0ConstantHlo<uint64>(
      computation, module_random_value ^ global_random_value);

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

  TF_RETURN_IF_ERROR(new_rng->CopyAllControlDepsFrom(rng));

  TF_RETURN_IF_ERROR(rng->ReplaceAllUsesWith(new_rng));
  TF_RETURN_IF_ERROR(rng->DropAllControlDeps());

  // Since rng is a side-effecting instruction, we can't rely on DCE to remove
  // it.
  TF_RETURN_IF_ERROR(computation->RemoveInstruction(rng));

  // Returns nullptr to OpExpanderPass::Run to indicate the old rng instruction
  // has been replaced with the new rng instruction.
  return nullptr;
}

}  // namespace xla
