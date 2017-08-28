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
      const HloReducePrecisionOptions::PassTiming pass_timing,
      const InstructionFilterFunction& instruction_filter_function)
      : exponent_bits_(exponent_bits),
        mantissa_bits_(mantissa_bits),
        pass_timing_(pass_timing),
        instruction_filter_function_(instruction_filter_function) {}

  // Version of the constructor that takes an HloReducePrecisionOptions proto
  // rather than explicitly-enumerated parameters, for convenience when
  // creating passes based on DebugOptions.
  explicit ReducePrecisionInsertion(
      const HloReducePrecisionOptions& reduce_precision_options)
      : exponent_bits_(reduce_precision_options.exponent_bits()),
        mantissa_bits_(reduce_precision_options.mantissa_bits()),
        pass_timing_(reduce_precision_options.pass_timing()),
        instruction_filter_function_(
            make_filter_function(reduce_precision_options)) {}

  ~ReducePrecisionInsertion() override{};

  tensorflow::StringPiece name() const override {
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
      const HloReducePrecisionOptions::PassTiming pass_timing,
      const int exponent_bits, const int mantissa_bits,
      const std::function<bool(HloOpcode)>& opcode_filter_function,
      const std::vector<string>& opname_substring_list = {});

  // Add ReducePrecisionInsertion passes to an HloPassPipeline based on the list
  // of HloReducePrecisionOptions in a DebugOptions proto.  Returns true if any
  // passes were added.
  static bool AddPasses(
      HloPassPipeline* pipeline, const DebugOptions& debug_options,
      const HloReducePrecisionOptions::PassTiming pass_timing);

 private:
  // Select the instructions that should be suffixed with reduce-precision
  // operators.
  std::vector<HloInstruction*> instructions_to_suffix(
      const HloComputation* computation);

  // Parameters for the precision reduction to be added.
  const int exponent_bits_;
  const int mantissa_bits_;

  // Pass "timing" parameter.  This also controls aspects of how the pass
  // selects locations to insert instructions.
  const HloReducePrecisionOptions::PassTiming pass_timing_;

  // Function to determine (from the opcode) whether a given instruction should
  // have a reduce-precision instruction inserted in its output stream.
  const InstructionFilterFunction instruction_filter_function_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_REDUCE_PRECISION_INSERTION_H_
