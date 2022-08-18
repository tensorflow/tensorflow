/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSION_BITCAST_LIFT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSION_BITCAST_LIFT_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Lift "rank-reducing bitcast" inside fusions to outside the fusion.
//
// Bitcast instruction are no-op, so we can duplicate them for free.
// "rank-reducing bitcast" is defined to be a bitcast whose output have
// a lower rank than the input.  Inside a fusion, we lift the
// "rank-reducing bitcast" out of the fusion as this allow using simpler,
// so faster, indexing.
class FusionBitcastLift : public HloModulePass {
 public:
  absl::string_view name() const override { return "fusion_bitcast_lift"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSION_BITCAST_LIFT_H_
