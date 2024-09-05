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

#ifndef XLA_SERVICE_ALL_REDUCE_REASSOCIATE_H_
#define XLA_SERVICE_ALL_REDUCE_REASSOCIATE_H_

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// A pass that reassociates all-reduce feeding into compatible elementwise
// operations. As an example: add(all-reduce(x), all-reduce(y)) will be replaced
// with all-reduce(add(x,y)). Mathematically, this is replacing
//   add(x0, x1, ... xk) + add(y0, y1, ... yk) with
//   add((x0+y0), (x1+y), ... (xk+yk)
//
//  i.e., reassociating the reduction operation.

class AllReduceReassociate : public HloModulePass {
 public:
  explicit AllReduceReassociate(bool reassociate_converted_ar = false)
      : reassociate_converted_ar_(reassociate_converted_ar) {}

  absl::string_view name() const override { return "all-reduce-reassociate"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Specify whether we should reassociate allreduces that are consumed by
  // Convert nodes.
  bool reassociate_converted_ar_;
};

}  // namespace xla

#endif  // XLA_SERVICE_ALL_REDUCE_REASSOCIATE_H_
