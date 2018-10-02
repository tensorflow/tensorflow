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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CALL_INLINER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CALL_INLINER_H_

#include <deque>

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// For every kCall operation in the main computation, we inline the body of the
// called function, and proceed recursively.
class CallInliner : public HloModulePass {
 public:
  using InlinedInstructionMap =
      std::unordered_map<HloInstruction*, HloInstruction*>;

  // Inlines one call instruction.  Returns a mapping from the original
  // instructions to their inlined versions.
  static StatusOr<InlinedInstructionMap> Inline(HloInstruction* call);

  ~CallInliner() override = default;
  absl::string_view name() const override { return "CallInliner"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CALL_INLINER_H_
