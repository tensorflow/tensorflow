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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_ENVIRONMENT_OPTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_ENVIRONMENT_OPTIONS_H_

#include <string>
#include <unordered_map>

#include "tensorflow/lite/experimental/litert/c/litert_any.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

class LiteRtEnvironmentOptionsT {
 public:
  LiteRtEnvironmentOptionsT() = default;

  LiteRtEnvironmentOptionsT(LiteRtEnvironmentOptionsT&& other);
  LiteRtEnvironmentOptionsT& operator=(LiteRtEnvironmentOptionsT&& other);

  litert::Expected<LiteRtAny> GetOption(LiteRtEnvOptionTag tag) const;
  litert::Expected<void> SetOption(LiteRtEnvOption option);

 private:
  void RefreshStringOptionValuePointers();

  std::unordered_map<LiteRtEnvOptionTag, LiteRtAny> options_;
  std::unordered_map<LiteRtEnvOptionTag, std::string> string_option_values_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_ENVIRONMENT_OPTIONS_H_
