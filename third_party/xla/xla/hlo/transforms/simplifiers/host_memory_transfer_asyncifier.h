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
#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_HOST_MEMORY_TRANSFER_ASYNCIFIER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_HOST_MEMORY_TRANSFER_ASYNCIFIER_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

/*
This pass finds copies between the host memory and device memory and converts
them into the async ops. This includes, but is not limited to:
 - device to host DynamicUpdateSlice
 - host to device DynamicSlice
* The examples below are not yet supported *
 - host to device DynamicUpdateSlice
 - device to host DynamicSlice
 - host to device Copy
 - device to host Copy
*/
class HostMemoryTransferAsyncifier : public HloModulePass {
 public:
  explicit HostMemoryTransferAsyncifier(int64_t host_memory_space_color)
      : kHostMemorySpaceColor(host_memory_space_color) {}
  ~HostMemoryTransferAsyncifier() override = default;

  absl::string_view name() const override {
    return "host-memory-transfer-asyncifier";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const int64_t kHostMemorySpaceColor;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_HOST_MEMORY_TRANSFER_ASYNCIFIER_H_
