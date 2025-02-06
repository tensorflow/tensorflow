/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_CUDNN_CUSTOM_CALL_CONVERTER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_CUDNN_CUSTOM_CALL_CONVERTER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Converts custom calls with kCuDnnFusionKind backend config to
// fusions with the same backend config. Frameworks can pass computations
// outlined this way through StableHLO; after the conversion they can be
// processed by XLA using the existing pipeline for custom fusions.
class CuDnnCustomCallConverter : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "cudnn-custom-call-converter";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_CUDNN_CUSTOM_CALL_CONVERTER_H_
