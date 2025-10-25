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

#include "xla/hlo/pass/hlo_pass_interface.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace xla {

absl::StatusOr<bool> HloPassInterface::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return RunImpl(module, execution_threads);
}

absl::StatusOr<bool> HloPassInterface::Run(
    std::unique_ptr<HloModule>& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return RunImpl(module, execution_threads);
}

}  // namespace xla
