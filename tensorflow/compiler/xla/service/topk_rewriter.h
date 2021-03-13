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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_TOPK_REWRITER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_TOPK_REWRITER_H_

#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
// This pass pattern-matches soups of HLOs executing a TopK operation and
// replaces them with a TopK CustomCall when the given values are supported by
// the CustomCall and it is more efficient to use that implementation.
class TopkRewriter : public HloModulePass {
 public:
  explicit TopkRewriter(std::function<bool(const HloSortInstruction*, int64)>
                            is_profitable_to_convert)
      : is_profitable_to_convert_(std::move(is_profitable_to_convert)) {}

  absl::string_view name() const override { return "topk-rewriter"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  // Predicate that returns true if a sort instruction is profitable to be
  // converted into a custom call.
  std::function<bool(const HloSortInstruction*, int64)>
      is_profitable_to_convert_;
};
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_TOPK_REWRITER_H_
