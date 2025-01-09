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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_ENVIRONMENT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_ENVIRONMENT_H_

#include <any>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_any.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert {

class Environment {
 public:
  enum class OptionTag {
    CompilerPluginLibraryPath = kLiteRtEnvOptionTagCompilerPluginLibraryPath,
    DispatchLibraryPath = kLiteRtEnvOptionTagDispatchLibraryPath,
  };

  struct Option {
    OptionTag tag;
    std::any value;
  };

  static Expected<void> Create(absl::Span<const Option> options) {
    auto c_options = ConvertOptions(options);
    if (!c_options) {
      return c_options.Error();
    }
    if (auto status =
            LiteRtEnvironmentCreate(c_options->size(), c_options->data());
        status != kLiteRtStatusOk) {
      return Error(status);
    } else {
      return {};
    }
  }

  static void Destroy() { LiteRtEnvironmentDestroy(); }

 private:
  static Expected<std::vector<LiteRtEnvOption>> ConvertOptions(
      absl::Span<const Option> options) {
    std::vector<LiteRtEnvOption> c_options;
    c_options.reserve(options.size());

    for (auto& option : options) {
      auto litert_any = ToLiteRtAny(option.value);
      if (!litert_any) {
        return litert_any.Error();
      }

      LiteRtEnvOption c_option = {
          /*.tag=*/static_cast<LiteRtEnvOptionTag>(option.tag),
          /*.value=*/*litert_any,
      };
      c_options.push_back(c_option);
    }

    return c_options;
  }
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_ENVIRONMENT_H_
