/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_SCATTER_EXPANDER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_SCATTER_EXPANDER_H_

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/scatter_expander.h"

namespace xla {

// Legalizes scatters on the GPU.
class GpuScatterExpander : public ScatterExpander {
 public:
  // Although we pass kEliminateAllScatters, we override this behavior in
  // InstructionMatchesPattern and select only some scatters to expand.
  GpuScatterExpander() : ScatterExpander(kEliminateAllScatters) {}

  absl::string_view name() const override { return "gpu_scatter_expander"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* inst) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_SCATTER_EXPANDER_H_
