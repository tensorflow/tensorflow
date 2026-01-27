/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_LAYOUT_ANALYSIS_H_
#define XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_LAYOUT_ANALYSIS_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

class HostOffloadingLayoutAnalysis : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "host-offloading-layout-analysis";
  }

  // This static method provides an API better named than "Run".
  static absl::StatusOr<bool> NeedsLayoutConversion(HloModule* module) {
    HostOffloadingLayoutAnalysis pass;
    return pass.Run(module);
  }

  // Returns true if the shape has padding due to tiling.
  // This function is useful when the HloModule has no tiling information, yet
  // we have it from shapes coming from buffers, e.g. TpuBuffer's.
  static bool ShapeHasPadding(const Shape& shape);

 protected:
  // This method does not modify the module; it purely informs the caller
  // whether device<->host layout conversion (i.e., (de)linearization of input
  // and result buffers) can be safely skipped.
  // Note: the pass is conservative in that it can return true for some cases
  // that might not need layout conversion. This is OK because performing layout
  // conversion is always correct, despite its performance impact.
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) final;
};

}  // namespace xla

#endif  // XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_LAYOUT_ANALYSIS_H_
