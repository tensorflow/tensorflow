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
#include "absl/strings/string_view.h"
#include "xla/hlo/pass/hlo_pipeline.pb.h"

namespace xla {

HloPassOptions::HloPassOptions(std::initializer_list<Option> options) {
  for (const Option& option : options) {
    CHECK(options_.try_emplace(option.name, option.value).second)
        << "Duplicate pass option '" << option.name << "'";
  }
}

namespace {

absl::string_view TypeName(const HloPassOptionValue& value) {
  switch (value.value_case()) {
    case HloPassOptionValue::kBoolVal:
      return "bool";
    case HloPassOptionValue::kIntVal:
      return "int64";
    case HloPassOptionValue::VALUE_NOT_SET:
      return "unset";
  }
}

}  // namespace

absl::Status HloPassOptions::CheckAllConsumed() const {
  // Order only affects which of several unconsumed options gets reported.
  // NOLINTNEXTLINE(google*-custom-deterministic-iteration-order)
  for (const auto& [key, value] : options_) {
    if (!consumed_keys_.contains(key)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Pass option '", key, "' (", TypeName(value),
                       ") was not consumed: unknown name or wrong type"));
    }
  }
  return absl::OkStatus();
}

HloPassOptions::ProtoMap HloPassOptions::ToProto() const {
  return ProtoMap(options_.begin(), options_.end());
}

}  // namespace xla
