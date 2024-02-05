/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_EIGH_EXPANDER_H_
#define XLA_SERVICE_EIGH_EXPANDER_H_

#include "absl/container/flat_hash_map.h"
#include "xla/client/xla_builder.h"
#include "xla/service/op_expander_pass.h"

namespace xla {

class EighExpander : public OpExpanderPass {
 public:
  absl::string_view name() const override { return "eigh_expander"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;

  virtual XlaOp BuildEigh(XlaOp a, bool lower, int64_t max_iter, float tol,
                          bool sort_eigenvalues);

  Status SortByEigenvalues(XlaOp& v, XlaOp& w);

 private:
  // Mapping from op signatures to existing computations.
  absl::flat_hash_map<std::string, HloComputation*> computation_cache_;
};

}  // namespace xla

#endif  // XLA_SERVICE_EIGH_EXPANDER_H_
