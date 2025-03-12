// Copyright 2024 Google LLC.
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

#include "tensorflow/lite/experimental/litert/core/environment.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_any.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

litert::Expected<LiteRtEnvironmentT::Ptr> LiteRtEnvironmentT::CreateWithOptions(
    absl::Span<const LiteRtEnvOption> options) {
  LITERT_LOG(LITERT_INFO, "Creating LiteRT environment with options");
  auto env = std::make_unique<LiteRtEnvironmentT>();
  // Copy the options into the environment.
  for (auto& option : options) {
    if (option.value.type == kLiteRtAnyTypeString) {
      std::string str_copy = std::string(option.value.str_value);
      env->string_options_.push_back(std::move(str_copy));
      LiteRtAny litert_any;
      litert_any.type = kLiteRtAnyTypeString;
      litert_any.str_value = env->string_options_.back().c_str();
      env->options_[option.tag] = litert_any;
    } else {
      env->options_[option.tag] = option.value;
    }
  }

  return env;
}
