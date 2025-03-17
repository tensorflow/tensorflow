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

#include <memory>
#include <optional>

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_any.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/core/accelerator.h"
#include "tensorflow/lite/experimental/litert/core/environment_options.h"

// A singleton class that contains global LiteRT environment options.
class LiteRtEnvironmentT {
 public:
  using Ptr = std::unique_ptr<LiteRtEnvironmentT>;

  LiteRtEnvironmentT() = default;
  // Create an environment instance with options.
  static litert::Expected<Ptr> CreateWithOptions(
      absl::Span<const LiteRtEnvOption> options);

  ~LiteRtEnvironmentT() = default;

  std::optional<LiteRtAny> GetOption(LiteRtEnvOptionTag tag) const {
    auto opt = options_.GetOption(tag);
    return opt.HasValue() ? std::optional<LiteRtAny>(opt.Value())
                          : std::nullopt;
  }

  LiteRtEnvironmentOptionsT& GetOptions() { return options_; }
  const LiteRtEnvironmentOptionsT& GetOptions() const { return options_; }

  litert::internal::AcceleratorRegistry& GetAcceleratorRegistry() {
    return accelerators_;
  }

 private:
  litert::internal::AcceleratorRegistry accelerators_;
  LiteRtEnvironmentOptionsT options_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_ENVIRONMENT_H_
