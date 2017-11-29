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
#include "tensorflow/core/lib/gtl/flatmap.h"

namespace xla {

// HLO pass which inserts reduce-precision instructions into the HLO graph, for
// purposes of experimenting with the effects of reduced-precision storage of
// intermediate values.
class ReducePrecisionInsertion : public HloPassInterface {
  using OpcodeFilterFunction = std::function<bool(HloOpcode)>;

 public:
  // The exponent_bits and mantissa_bits arguments specify the parameters of
  // the instructions to insert.  The instructions will be inserted after each
  // instruction with an opcode for which the should_reduce_output_precision
  // function returns true and the output type is F32.
  explicit ReducePrecisionInsertion(
      const int exponent_bits, const int mantissa_bits,
      const OpcodeFilterFunction& should_reduce_output_precision)
      : exponent_bits_(exponent_bits),
        mantissa_bits_(mantissa_bits),
        should_reduce_output_precision_(should_reduce_output_precision) {}
  ~ReducePrecisionInsertion() override{};

  tensorflow::StringPiece name() const override {
    return "reduce-precision-insertion";
  }

  // Run the pass on the given module. Returns whether the module was changed
  // (reduce-precision instructions were inserted).
  StatusOr<bool> Run(HloModule* module) override;

 private:
  // Parameters for the precision reduction to be added.
  const int exponent_bits_;
  const int mantissa_bits_;

  // Function to determine (from the opcode) whether a given instruction should
  // have a reduce-precision instruction inserted in its output stream.
  const OpcodeFilterFunction& should_reduce_output_precision_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_REDUCE_PRECISION_INSERTION_H_
