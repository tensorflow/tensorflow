/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REDUCTION_SPLITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REDUCTION_SPLITTER_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Splits a reduce op into two consecutive reduce ops if
// * the reduce dimensions are not contiguous and
// * at least one reduce dimension is large (i.e. corresponds to a large input
//   shape dimension).
//
// Reductions with non-contiguous dimensions are emitted as simple element-wise
// loops. This is inefficient when reducing large input shape dimensions.
// Splitting such reductions allows using more efficient reduction emitters.
//
// This pass splits reduce ops into two consecutive reduce ops. Run it to a
// fixpoint to split reduce ops along multiple large dimensions.
//
// Precondition: ReductionDimensionGrouper has been run and adjacent reduce
// dimentsions have been grouped. Reduction layouts have been normalized.

class ReductionSplitter : public HloModulePass {
 public:
  absl::string_view name() const override { return "reduction-splitter"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REDUCTION_SPLITTER_H_
