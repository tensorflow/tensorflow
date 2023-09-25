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

#ifndef XLA_SERVICE_GATHER_EXPANDER_H_
#define XLA_SERVICE_GATHER_EXPANDER_H_

#include "xla/service/op_expander_pass.h"

namespace xla {

// This pass rewrites gather operations into (roughly) while loops of dynamic
// slices.
//
// This pass can be used two ways:
//
//  - kEliminateAllGathers: For backends that don't support gather, this pass
//    can convert every gather to a loop.
//
//  - kEliminateSimpleGathers: For backends that *do* support gather, this pass
//    can strength-reduce "simple" gathers -- specifically, gathers that can be
//    represented without a loop -- to dyanmic-slices.
//
// Note that even in kEliminateSimpleGathers mode, this pass may still expand a
// gather into a loop (with a trip-count of 1).  It's up to other simplification
// passes to remove the loop.
//
class GatherExpander : public OpExpanderPass {
 public:
  enum Mode {
    kEliminateAllGathers,
    kEliminateSimpleGathers,
  };

  explicit GatherExpander(Mode m) : mode_(m) {}

  absl::string_view name() const override { return "gather_expander"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* gather_inst) override;

 private:
  Mode mode_;
};

}  // namespace xla

#endif  // XLA_SERVICE_GATHER_EXPANDER_H_
