/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COLLECTIVE_TRANSFORMATION_REORDERER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COLLECTIVE_TRANSFORMATION_REORDERER_H_

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Transforms all-gather + reshape into reshape + all-gather when the reshape
// only changes the shape of the all-gather shards, i.e., it does not reshape
// across the all-gather dimension.

// Generally speaking,

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
    static constexpr absl::string_view kName =
        "collective-transformation-reorderer";
    return kName;
  }
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  StatusOr<bool> ReorderAllGatherTransformations(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COLLECTIVE_TRANSFORMATION_REORDERER_H_
