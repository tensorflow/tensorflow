/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_COLLECTIVE_TRANSFORMATION_REORDERER_H_
#define XLA_SERVICE_COLLECTIVE_TRANSFORMATION_REORDERER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// Transforms
//  -- all-gather + reshape into reshape + all-gather and
//  -- reshape + all-reduce into all-reduce + reshape.
// Both transformations require that there are no other users affected, i.e.,
// reshape user count should be 1.
// all-gather transformation requires the reshape to only change the shape of
// the all-gather shards, i.e., not reshaping across the all-gather dimension.
// all-reduce transformation requires all-reduce to be not layout constrained.

// all-gather + reshape example:

// input = [C_0, C_1, ..., C_i, ..., C_{n-1}, C_n] ...
// all-gather = [C_0, C_1, ..., P*C_i, ... C_{n-1}, C_n] all-gather(input)
// reshape = [D_0, D_1, ..., P*D_j, ..., D_{m-1}, D_m] reshape(all-gather)

// can be transformed to:

// input = [C_0, C_1, ..., C_i, ..., C_{n-1}, C_n] ...
// reshape = [D_0, D_1, ..., D_j, ..., D_{m-1}, D_m] reshape(input)
// all-gather = [D_0, D_1, ..., P*D_j, ... D_{m-1}, D_m] all-gather(input)

// if and only if C_0 * C_1 * ... * C_{i-1} = D_0 * D_1 * ... * D_{j-1}
// and C_{i+1} * ... * C_{n-1} * C_n = D_{j+1} * ... * D_{m-1} * D_{m}.

class CollectiveTransformationReorder : public HloModulePass {
 public:
  CollectiveTransformationReorder() = default;
  ~CollectiveTransformationReorder() override = default;
  absl::string_view name() const override {
    return "collective-transformation-reorderer";
  }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::StatusOr<bool> ReorderAllGatherTransformations(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);
  absl::StatusOr<bool> ReorderAllReduceTransformations(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);
};

}  // namespace xla

#endif  // XLA_SERVICE_COLLECTIVE_TRANSFORMATION_REORDERER_H_
