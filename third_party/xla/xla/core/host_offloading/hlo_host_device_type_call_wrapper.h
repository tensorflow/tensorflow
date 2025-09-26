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

#ifndef XLA_CORE_HOST_OFFLOADING_HLO_HOST_DEVICE_TYPE_CALL_WRAPPER_H_
#define XLA_CORE_HOST_OFFLOADING_HLO_HOST_DEVICE_TYPE_CALL_WRAPPER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Wraps host instructions annotated as host compute into calls and annotates
// the device type as host.
class HloHostDeviceTypeCallWrapper : public HloModulePass {
 public:
  struct Options {
    // Function that sets the device type of the instruction to host.
    // Implementation of the function is device specific.
    std::function<absl::Status(HloInstruction*)> set_backend_config_fn;
    // Function that clears the device type from the backend config.
    std::function<absl::Status(HloInstruction*)>
        clear_backend_config_device_type;
  };

  explicit HloHostDeviceTypeCallWrapper(const Options& options)
      : options_(options) {}

  absl::string_view name() const override {
    return "hlo_host_device_type_call_wrapper";
  }

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // Materializes constants on the host computation to avoid unnecessary device
  // to host transfers.
  //
  // Returns an an updated call instruction/computation that does not
  // contain constant operands/parameters.
  static absl::StatusOr<HloCallInstruction*>
  MaterializeConstantsOnHostComputation(HloCallInstruction* call);

  // Removes tuple parameter/operands from the call instruction.
  //
  // Returns an an updated call instruction/computation that does not
  // contain tuple parameters/operands.
  static absl::StatusOr<HloCallInstruction*> RemoveTupleParameters(
      HloCallInstruction* call);

 private:
  Options options_;
};

}  // namespace xla

#endif  // XLA_CORE_HOST_OFFLOADING_HLO_HOST_DEVICE_TYPE_CALL_WRAPPER_H_
