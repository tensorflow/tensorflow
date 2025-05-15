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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_COMBINER_ANNOTATOR_H_
#define XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_COMBINER_ANNOTATOR_H_

#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Annotates collective operations with metadata used by collective combiners.
class CollectiveCombinerAnnotator : public HloModulePass {
 public:
  CollectiveCombinerAnnotator(se::DeviceDescription device_info,
                              int64_t pointer_size)
      : device_info_(std::move(device_info)), pointer_size_(pointer_size) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  absl::string_view name() const override {
    return "collective-combiner-annotator";
  }

 private:
  se::DeviceDescription device_info_;
  int64_t pointer_size_;
};

// Returns true if `instr` is a combinable sync collective. False otherwise.
bool IsCombinableSyncCollective(const HloInstruction& instr);

// Returns true if module contains any combinable sync collective. False
// otherwise.
bool ContainsCombinableSyncCollective(const HloModule& module);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_COMBINER_ANNOTATOR_H_
