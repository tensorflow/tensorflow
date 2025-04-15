/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_REDUCE_SCATTER_REASSOCIATE_H_
#define XLA_SERVICE_REDUCE_SCATTER_REASSOCIATE_H_

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// A pass that reassociates reduce-scatter feeding into compatible elementwise
// operations. As an example: add(reduce-scatter(x), reduce-scatter(y)) will be
// replaced with reduce_scatter(add(x,y)).
//
//  i.e., reassociating the reduce-scatter operation.

class ReduceScatterReassociate : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "reduce-scatter-reassociate";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_REDUCE_SCATTER_REASSOCIATE_H_
