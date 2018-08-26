/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_REDUCE_PRECISION_INSERTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_REDUCE_PRECISION_INSERTION_H_

#include "tensorflow/compiler/xla/service/buffer_liveness.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/core/lib/gtl/flatmap.h"

namespace xla {

// HLO pass which inserts reduce-precision instructions into the HLO graph, for
// purposes of experimenting with the effects of reduced-precision storage of
// intermediate values.
class ReducePrecisionInsertion : public HloPassInterface {
  using InstructionFilterFunction = std::function<bool(const HloInstruction*)>;

 public:
  // The exponent_bits and mantissa_bits arguments specify the parameters of
  // the instructions to insert.  The instructions will be inserted after each
  // instruction with an opcode for which the instruction_filter_function
  // function returns true and the output type is F32.
  explicit ReducePrecisionInsertion(
      const int exponent_bits, const int mantissa_bits,
      const HloReducePrecisionOptions::Location location,
      const InstructionFilterFunction& instruction_filter_function)
      : exponent_bits_(exponent_bits),
        mantissa_bits_(mantissa_bits),
        location_(location),
        instruction_filter_function_(instruction_filter_function) {}

  // Version of the constructor that takes an HloReducePrecisionOptions proto
  // rather than explicitly-enumerated parameters, for convenience when
  // creating passes based on DebugOptions.
  explicit ReducePrecisionInsertion(
      const HloReducePrecisionOptions& reduce_precision_options)
      : exponent_bits_(reduce_precision_options.exponent_bits()),
        mantissa_bits_(reduce_precision_options.mantissa_bits()),
        location_(reduce_precision_options.location()),
        instruction_filter_function_(
            make_filter_function(reduce_precision_options)) {}

  ~ReducePrecisionInsertion() override{};

  absl::string_view name() const override {
    return "reduce-precision-insertion";
  }

  // Run the pass on the given module. Returns whether the module was changed
  // (reduce-precision instructions were inserted).
  StatusOr<bool> Run(HloModule* module) override;

  // Convert between the (inconvenient) xla.proto HloReducePrecisionOptions
  // representation and InstructionFilterFunction functions.
  static InstructionFilterFunction make_filter_function(
      const HloReducePrecisionOptions& reduce_precision_options);
  static HloReducePrecisionOptions make_options_proto(
      const HloReducePrecisionOptions::Location location,
      const int exponent_bits, const int mantissa_bits,
      const std::function<bool(HloOpcode)>& opcode_filter_function,
      const std::vector<string>& opname_substring_list = {});

  // Enumeration to control which passes should be added.
  enum class PassTiming { BEFORE_OPTIMIZATION, AFTER_FUSION };

  // Add ReducePrecisionInsertion passes to an HloPassPipeline based on the list
  // of HloReducePrecisionOptions in a DebugOptions proto.  Returns true if any
  // passes were added.
  static bool AddPasses(HloPassPipeline* pipeline,
                        const DebugOptions& debug_options,
                        const PassTiming pass_timing);

 private:
  // Select the instructions that should have reduce-precision operations
  // attached to them.
  std::vector<HloInstruction*> instructions_to_modify(
      const HloComputation* computation);

  // Insert a reduce-precision operation into the graph on the output of the
  // given instruction.
  StatusOr<bool> insert_after(HloInstruction* instruction);

  // Insert reduce-precision operations into the graph on the inputs of the
  // given instructions.  (For fusion instructions, the operations will be
  // inserted inside the fusion computation, on the outputs of the relevant
  // input parameters.)
  StatusOr<bool> insert_on_inputs(
      const std::vector<HloInstruction*>& instructions);

  // Insert reduce-precision operations into the graph on the outputs of the
  // given instructions.  (For fusion instructions, the operations will be
  // inserted inside the fusion computation as a new root.)
  StatusOr<bool> insert_on_outputs(
      const std::vector<HloInstruction*>& instructions);

  // Is this shape valid for inserting a reduce-precision operation?
  bool is_valid_shape(const Shape& shape) {
    // For now, ReducePrecision is only implemented for F32 arrays, so this
    // ignores instructions that produce other data.  In particular, this
    // currently ignores instructions producing tuples, even if those tuples
    // contain F32 arrays inside them.  The assumption is that in most cases
    // equivalent behavior can be obtained by adding ReducePrecision
    // instructions after the instructions that pull the F32 arrays out of
    // the tuples.
    //
    // TODO(b/64093391): Remove the IsScalar check once this won't cause
    // failures on the GPU backend if the ReducePrecision instruction ends up
    // inserted between a scalar constant and the init_value argument of a
    // Reduce operation.
    return shape.element_type() == PrimitiveType::F32 &&
           !ShapeUtil::IsScalar(shape);
  }

  // Is this instruction one such that following or preceding it with a new
  // reduce-precision operation will be redundant?
  bool is_redundant(const HloInstruction* instruction) {
    return instruction->opcode() == HloOpcode::kReducePrecision &&
           instruction->exponent_bits() <= exponent_bits_ &&
           instruction->mantissa_bits() <= mantissa_bits_;
  }

  // Parameters for the precision reduction to be added.
  const int exponent_bits_;
  const int mantissa_bits_;

  // Pass "timing" parameter.  This also controls aspects of how the pass
  // selects locations to insert instructions.
  const HloReducePrecisionOptions::Location location_;

  // User-provided Function to determine whether a given instruction should
  // have a reduce-precision instruction inserted in its output stream.
  const InstructionFilterFunction instruction_filter_function_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_REDUCE_PRECISION_INSERTION_H_
