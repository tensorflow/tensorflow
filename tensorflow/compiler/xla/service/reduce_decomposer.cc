
/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/reduce_decomposer.h"

#include <functional>
#include <utility>

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {

namespace {

class ReduceDecomposerVisitor : public DfsHloRewriteVisitor {
 public:
  explicit ReduceDecomposerVisitor(
      std::function<bool(const HloInstruction*)> custom_layout_allowed)
      : custom_layout_allowed_(std::move(custom_layout_allowed)) {}

  Status HandleReduce(HloInstruction* reduce) override {
    if (custom_layout_allowed_ && custom_layout_allowed_(reduce)) {
      return OkStatus();
    }

    Shape shape = reduce->shape();
    HloInstruction* operand = reduce->mutable_operand(0);
    Shape operand_shape = operand->shape();
    Shape expected_shape =
        ShapeUtil::DeleteDimensions(reduce->dimensions(), operand_shape);

    if (expected_shape != shape) {
      // Decompose it into a reduction to expected_shape and a copy.
      TF_ASSIGN_OR_RETURN(auto r_prime,
                          MakeReduceHlo(operand, reduce->mutable_operand(1),
                                        reduce->dimensions(),
                                        reduce->called_computations()[0]));
      TF_RET_CHECK(r_prime->shape() == expected_shape);
      auto copy = MakeCopyHlo(r_prime, shape);
      TF_RETURN_IF_ERROR(ReplaceInstruction(reduce, copy));
      return OkStatus();
    }

    return OkStatus();
  }

 private:
  std::function<bool(const HloInstruction*)> custom_layout_allowed_;
};

}  // namespace

StatusOr<bool> ReduceDecomposer::Run(HloModule* module) {
  return ReduceDecomposerVisitor{custom_layout_allowed_}.RunOnModule(module);
}

}  // namespace xla
