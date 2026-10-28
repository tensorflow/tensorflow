/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/pass/hlo_pass_options.h"

#include <initializer_list>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace xla {

HloPassOptions::HloPassOptions(std::initializer_list<Option> options) {
  for (const Option& option : options) {
    CHECK(options_.try_emplace(option.name, option.value).second)
        << "Duplicate pass option '" << option.name << "'";
  }
}

absl::Status HloPassOptions::CheckAllConsumed() const {
  // Order only affects which of several unknown options gets reported.
  // NOLINTNEXTLINE(google*-custom-deterministic-iteration-order)
  for (const auto& [key, _] : options_) {
    if (!consumed_keys_.contains(key)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown pass option '", key, "'"));
    }
  }
  return absl::OkStatus();
}

}  // namespace xla
