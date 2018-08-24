/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_CONSTANT_SINKING_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_CONSTANT_SINKING_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Sinks while loop invariant values that happen to be constants into the while
// loop body.  This is probably not a win in isolation but may unlock further
// optimizations like constant folding.
//
//   state = (..., const, ...)
//   while (pred(state)) {
//     (..., v, ...) = state
//     use(v)
//     state = (..., v, ...)
//   }
//
// =>
//
//   state = (..., const, ...)
//   while (pred(state)) {
//     (..., v, ...) = state
//     use(const)
//     state = (..., v, ...)
//   }
//
// Note that it leaves the `v` in place to keep that component of the state
// tuple trivially loop invariant.  WhileLoopSimplifier will later get rid of
// `v`.
//
// We only sink into while loop bodies, but this can be extended to transform
// conditions as well.
//
// TODO(b/79121449):  We should also sink broadcasts of constants.
class WhileLoopConstantSinking : public HloPassInterface {
 public:
  ~WhileLoopConstantSinking() override = default;

  absl::string_view name() const override {
    return "while-loop-invariant-code-motion";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  StatusOr<bool> TrySinkingConstantsIntoWhileBody(HloInstruction* while_instr);
};
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_CONSTANT_SINKING_H_
