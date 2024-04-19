/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SHARDING_FORMAT_PICKER_H_
#define XLA_SERVICE_SHARDING_FORMAT_PICKER_H_

#include "xla/service/hlo_pass_interface.h"

namespace xla {

// Test-only pass to transform the HloSharding format of all the instructions in
// a module to the selected format.
class ShardingFormatPicker : public HloModulePass {
 public:
  enum class ShardingType {
    kV1,            // Converts all HloSharding to V1 format.
    kBestEffortV2,  // Best effort to convert all HloSharding to V2 format.
  };
  explicit ShardingFormatPicker(ShardingType sharding_type)
      : sharding_type_(sharding_type) {}
  absl::string_view name() const override { return "sharding-format-picker"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const ShardingType sharding_type_;
};

}  // namespace xla

#endif  // XLA_SERVICE_SHARDING_FORMAT_PICKER_H_
