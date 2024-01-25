/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_RESHAPE_DECOMPOSER_H_
#define XLA_SERVICE_RESHAPE_DECOMPOSER_H_

#include "xla/service/hlo_pass_interface.h"

namespace xla {

// Decomposes a reshape which does not satisfy the ReshapeIsBitcast precondition
// into a bitcast and a copy (physical transposition). Tries to create only one
// transposition, but when it's not possible, creates two.
//
// Postcondition: All reshapes are turned into bitcasts.
class ReshapeDecomposer : public HloModulePass {
 public:
  absl::string_view name() const override { return "reshape-decomposer"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_RESHAPE_DECOMPOSER_H_
