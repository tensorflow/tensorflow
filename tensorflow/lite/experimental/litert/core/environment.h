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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_ENVIRONMENT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_ENVIRONMENT_H_

#include <any>
#include <map>
#include <optional>

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert::internal {

// A singleton class that contains global LiteRT environment options.
class Environment {
 public:
  // Create the singleton environment instance with options. Returns an error if
  // the instance already exists, in which case the specified options have no
  // effect.
  static Expected<void> CreateWithOptions(
      absl::Span<const LiteRtEnvOption> options);

  // Return the envirnment instance and, if not yet created, creates one with no
  // options.
  static Expected<Environment*> Instance();

  // Destroy the environment instance.
  static void Destroy();

  std::optional<LiteRtAny> GetOption(LiteRtEnvOptionTag tag) const {
    auto i = options_.find(tag);
    if (i != options_.end()) {
      return i->second;
    } else {
      return std::nullopt;
    }
  }

 private:
  std::map<LiteRtEnvOptionTag, LiteRtAny> options_;

  static Environment* the_instance_;
};

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_ENVIRONMENT_H_
