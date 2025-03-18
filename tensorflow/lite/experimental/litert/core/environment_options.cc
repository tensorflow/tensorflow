// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/core/environment_options.h"

#include <string>
#include <utility>

#include "tensorflow/lite/experimental/litert/c/litert_any.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

LiteRtEnvironmentOptionsT::LiteRtEnvironmentOptionsT(
    LiteRtEnvironmentOptionsT&& other)
    : options_(std::move(other.options_)),
      string_option_values_(std::move(other.string_option_values_)) {
  // Update the string pointers in case they have changed when moving the
  // container. This can happen because of small string optimization.
  RefreshStringOptionValuePointers();
}

LiteRtEnvironmentOptionsT& LiteRtEnvironmentOptionsT::operator=(
    LiteRtEnvironmentOptionsT&& other) {
  options_ = std::move(other.options_);
  string_option_values_ = std::move(other.string_option_values_);
  // Update the string pointers in case they have changed when moving the
  // container. This can happen because of small string optimization.
  RefreshStringOptionValuePointers();
  return *this;
}

void LiteRtEnvironmentOptionsT::RefreshStringOptionValuePointers() {
  for (const auto& [tag, value] : string_option_values_) {
    options_[tag].str_value = value.c_str();
  }
}

litert::Expected<LiteRtAny> LiteRtEnvironmentOptionsT::GetOption(
    LiteRtEnvOptionTag tag) const {
  if (auto it = options_.find(tag); it != options_.end()) {
    return it->second;
  }
  return litert::Error(kLiteRtStatusErrorNotFound,
                       "Option was not set for this environment.");
}

litert::Expected<void> LiteRtEnvironmentOptionsT::SetOption(
    LiteRtEnvOption option) {
  if (option.value.type == kLiteRtAnyTypeString) {
    auto [string_it, _] = string_option_values_.insert_or_assign(
        option.tag, option.value.str_value);
    LiteRtAny value{/*type=*/kLiteRtAnyTypeString};
    value.str_value = string_it->second.c_str();
    options_[option.tag] = value;
  } else {
    options_[option.tag] = option.value;
  }
  return {};
}
