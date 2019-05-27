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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_RECOMPUTE_INSTRUCTIONS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_RECOMPUTE_INSTRUCTIONS_H_

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;
class HloInstruction;

namespace poplarplugin {

// Look through the instructions in all the computations in a given module and
// attempt to identify particular patterns in them. If the patterns are found
// replace them with a more optimal pattern.
// Patterns currently replaced.
// X->Conv->Norm->Non-linearity.
// |    |    |     |    (Memory Connections to backward gradient instructions)
// |  Group Norm < Non-linearity Gradient
//
// This is optimized by recomputing the Conv->Norm->Non-linearity chain inplace
// right before we execute the backward connections, thus saving some of the
// memory from the backward connection.
//
// If we couldn't identify that pattern from an instruction we still try to get
// just the Conv->Norm connection on its own and recompute them along with the
// backprop instructions, leaving the Non-linearity in the forward pass.
class RecomputeInstructions : public HloModulePass {
 public:
  RecomputeInstructions(bool recompute_norm_inputs,
                        CompilerAnnotations& annotations);

  ~RecomputeInstructions() override = default;

  absl::string_view name() const override {
    return "non-linearity-recomputation";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  bool allow_recompute_;
  CompilerAnnotations& annotations_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
