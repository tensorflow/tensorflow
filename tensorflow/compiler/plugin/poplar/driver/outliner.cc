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

#include "tensorflow/compiler/plugin/poplar/driver/outliner.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

class OutlineCandidateFinder : public DfsHloVisitorWithDefault {
 public:
  explicit OutlineCandidateFinder() {}

  Status DefaultAction(HloInstruction*) override {
    return Status::OK();
  }

  Status HandleConvolution(
          HloInstruction* convolution,
          HloInstruction* lhs_instruction,
          HloInstruction* rhs_instruction,
          const Window& window) override {
    candidates_.push_back(convolution);
    return Status::OK();
  }

  Status HandleDot(HloInstruction* dot, HloInstruction* lhs,
                   HloInstruction* rhs) override {
    candidates_.push_back(dot);
    return Status::OK();
  }

  std::vector<HloInstruction*> candidates_;
};

StatusOr<bool> Outliner::Run(HloModule* module) {
  OutlineCandidateFinder visitor;
  TF_RETURN_IF_ERROR(
          module->entry_computation()->root_instruction()->Accept(&visitor));

  for (HloInstruction* top : visitor.candidates_) {
    std::vector<HloInstruction*> to_replace;
    to_replace.push_back(top);
    if (top->user_count() == 1) {
      for (HloInstruction* inst = top->users()[0];
           inst->user_count() == 1 && inst->IsElementwise() &&
           to_replace.size() < max_outlined_instructions_;
           inst = inst->users()[0]) {
        to_replace.push_back(inst);
      }
    }

    module->OutlineExpressionFromComputation(to_replace,
                                             top->name(),
                                             module->entry_computation());
  }
  return visitor.candidates_.size() > 0;
}

}  // namespace xla
