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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CONDITIONAL_CODE_MOTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CONDITIONAL_CODE_MOTION_H_

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// HLO pass that moves identical ops out of conditional.
// - The definition of identical are the shape of the operands are identical
// and their properties are identical.
// - Currently, only some types of instructions is supported.
// TODO(b/154283721): relax non-sharable operand constraint and avoid copies in
// the new root.
// - Only the identical ops that won't share operands with other ops will
// be moved out of conditional.
class ConditionalCodeMotion : public HloModulePass {
 public:
  // If is_layout_sensitive is true, then the hoist process preserves layout
  // during identical comparison. Otherwise, layout is ignored.
  explicit ConditionalCodeMotion(bool is_layout_sensitive = true)
      : is_layout_sensitive_(is_layout_sensitive) {}
  absl::string_view name() const override { return "conditional-code-motion"; }
  StatusOr<bool> Run(HloModule* module) override;

 private:
  const bool is_layout_sensitive_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CONDITIONAL_CODE_MOTION_H_
