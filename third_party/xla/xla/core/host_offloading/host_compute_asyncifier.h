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

#ifndef XLA_CORE_HOST_OFFLOADING_HOST_COMPUTE_ASYNCIFIER_H_
#define XLA_CORE_HOST_OFFLOADING_HOST_COMPUTE_ASYNCIFIER_H_

#include <functional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Converts call instructions that execute on the host into async start/done
// instructions.
class HostComputeAsyncifier : public HloModulePass {
 public:
  explicit HostComputeAsyncifier(std::function<bool(HloInstruction*)>
                                     backend_config_device_type_is_host_fn)
      : backend_config_device_type_is_host_fn_(
            std::move(backend_config_device_type_is_host_fn)) {}

  absl::string_view name() const override { return "host_compute_asyncifier"; }

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Function that returns true if the instruction should be executed on the
  // host. Implementation of the function is device specific.
  std::function<bool(HloInstruction*)> backend_config_device_type_is_host_fn_;
};

}  // namespace xla

#endif  // XLA_CORE_HOST_OFFLOADING_HOST_COMPUTE_ASYNCIFIER_H_
