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

#ifndef XLA_SERVICE_UNIQUE_CHANNEL_ID_ENFORCER_H_
#define XLA_SERVICE_UNIQUE_CHANNEL_ID_ENFORCER_H_

#include "xla/service/hlo_pass_interface.h"

namespace xla {
// A pass which enforces that every collective
// must have a unique channel id.
class UniqueChannelIdEnforcer : public HloModulePass {
 public:
  explicit UniqueChannelIdEnforcer(bool assert_unique_channel_ids = false)
      : assert_unique_channel_ids_(assert_unique_channel_ids) {}

  absl::string_view name() const override {
    return "unique-channel-id-enforcer";
  }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  bool assert_unique_channel_ids_;
};

}  // namespace xla

#endif  // XLA_SERVICE_UNIQUE_CHANNEL_ID_ENFORCER_H_
