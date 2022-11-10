/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/rng_bit_generator_expander.h"

#include "tensorflow/compiler/xla/client/lib/prng.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/lib/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

XlaOp GetPhiloxStateOp(XlaOp input_state, const Shape& state_shape) {
  if (state_shape.dimensions(0) >= 3) {
    return Slice(input_state, {1}, {3}, {1});
  }
  return Rev(input_state, {0});
}

XlaOp GetPhiloxOutputStateOp(XlaOp output_state, const Shape& state_shape) {
  if (state_shape.dimensions(0) < 3) {
    output_state = Slice(output_state, {0}, {1}, {1});
  }
  return output_state;
}

}  // namespace

bool RngBitGeneratorExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kRngBitGenerator;
}

StatusOr<HloComputation*> RngBitGeneratorExpander::GetGeneratorComputation(
    const Shape& data_shape, const Shape& state_shape,
    RandomAlgorithm algorithm, HloModule* module) {
  RngGeneratorKey cache_key{data_shape, state_shape, algorithm, module};
  auto it = computation_cache_.find(cache_key);
  if (it != computation_cache_.end()) {
    return it->second;
  }

  XlaBuilder builder("rng");
  XlaOp state_param = Parameter(&builder, 0, state_shape, "state");
  XlaOp key_op = Reshape(Slice(state_param, {0}, {1}, {1}), {});
  RngOutput output;
  switch (algorithm) {
    case RandomAlgorithm::RNG_THREE_FRY:
      output = ThreeFryBitGenerator(key_op, Slice(state_param, {1}, {2}, {1}),
                                    data_shape);
      break;
    case RandomAlgorithm::RNG_PHILOX:
      output = PhiloxBitGenerator(
          key_op, GetPhiloxStateOp(state_param, state_shape), data_shape);
      output.state = GetPhiloxOutputStateOp(output.state, state_shape);
      break;
    default:
      return Unimplemented("Unsupported random algorthm: %s",
                           RandomAlgorithm_Name(algorithm));
  }

  XlaOp final_state =
      ConcatInDim(&builder, {Reshape(key_op, {1}), output.state}, 0);
  Tuple(&builder, {final_state, output.value});
  TF_ASSIGN_OR_RETURN(XlaComputation xla_computation, builder.Build());

  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      xla_computation.GetProgramShape());
  HloModuleConfig config(program_shape);
  TF_ASSIGN_OR_RETURN(auto new_module, HloModule::CreateFromProto(
                                           xla_computation.proto(), config));
  HloCloneContext context(module);
  HloComputation* new_computation =
      module->DeepCloneComputation(new_module->entry_computation(), &context);
  computation_cache_.emplace(cache_key, new_computation);
  return new_computation;
}

StatusOr<HloInstruction*> RngBitGeneratorExpander::ExpandInstruction(
    HloInstruction* hlo) {
  HloRngBitGeneratorInstruction* rng = Cast<HloRngBitGeneratorInstruction>(hlo);
  RandomAlgorithm algorithm = rng->algorithm();
  if (algorithm == RandomAlgorithm::RNG_DEFAULT) {
    algorithm = default_algorithm_;
  }

  HloModule* module = hlo->GetModule();
  const Shape& data_shape = rng->shape().tuple_shapes(1);
  const Shape& state_shape = rng->operand(0)->shape();
  TF_ASSIGN_OR_RETURN(
      HloComputation * generator_computation,
      GetGeneratorComputation(data_shape, state_shape, algorithm, module));
  return hlo->parent()->AddInstruction(HloInstruction::CreateCall(
      ShapeUtil::MakeTupleShapeWithPtrs({&state_shape, &data_shape}),
      {hlo->mutable_operand(0)}, generator_computation));
}

}  // namespace xla
