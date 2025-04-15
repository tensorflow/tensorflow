/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_BATCHNORM_EXPANDER_H_
#define XLA_SERVICE_BATCHNORM_EXPANDER_H_

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// A pass which rewrites batch norm operations into more operations. Breaking a
// big operation into smaller operations helps leverage our generic fusion
// logic.
class BatchNormExpander : public HloModulePass {
 public:
  // When use_fusion is set, a multi-output fusion node is created.
  explicit BatchNormExpander(bool rewrite_training_op = false,
                             bool rewrite_inference_op = false,
                             bool rewrite_grad_op = false)
      : rewrite_training_op_(rewrite_training_op),
        rewrite_inference_op_(rewrite_inference_op),
        rewrite_grad_op_(rewrite_grad_op) {}
  ~BatchNormExpander() override = default;
  absl::string_view name() const override { return "batchnorm_expander"; }

  // Run operation expander on the given computation. Returns whether the
  // computation was changed.
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  bool rewrite_training_op_;
  bool rewrite_inference_op_;
  bool rewrite_grad_op_;
};

}  // namespace xla

#endif  // XLA_SERVICE_BATCHNORM_EXPANDER_H_
