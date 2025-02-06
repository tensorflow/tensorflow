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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_DISPATCH_DISPATCH_DELEGATE_OPTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_DISPATCH_DISPATCH_DELEGATE_OPTIONS_H_

#include <any>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_any.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_any.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/core/environment.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"

class LiteRtDispatchDelegateOptions {
 public:
  explicit LiteRtDispatchDelegateOptions(LiteRtEnvironmentT& environment) {
    auto option = environment.GetOption(kLiteRtEnvOptionTagDispatchLibraryPath);
    if (!option.has_value()) {
      return;
    }

    if (option->type != kLiteRtAnyTypeString) {
      LITERT_LOG(LITERT_WARNING,
                 "Ingoring option kLiteRtEnvOptionTagDispatchLibraryPath due "
                 "to invalid value");
      return;
    }

    LiteRtDispatchOption dispatch_option = {
        /*.name=*/kDispatchOptionSharedLibraryDir,
        /*.value=*/*option,
    };
    AddOption(dispatch_option);
  }

  // Push a new dispatch option.
  void AddOption(LiteRtDispatchOption option) { options_.push_back(option); }

  // Get all dispatch options.
  const std::vector<LiteRtDispatchOption>& GetDispatchOptions() const {
    return options_;
  }

  // Find a dispatch option under the given name if it exists.
  litert::Expected<std::any> FindDispatchOption(absl::string_view name) const {
    for (const auto& option : options_) {
      if (option.name != name) {
        continue;
      }
      return litert::ToStdAny(option.value);
    }
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument);
  }

 private:
  std::vector<LiteRtDispatchOption> options_;
};

//
// Common options
//

static constexpr absl::string_view kAllocBase = "alloc_base";

inline void AddAllocBaseOption(const void* alloc_base,
                               LiteRtDispatchDelegateOptions& opts) {
  LiteRtAny opt;
  opt.type = kLiteRtAnyTypeVoidPtr;
  opt.ptr_value = alloc_base;
  opts.AddOption(LiteRtDispatchOption{kAllocBase.data(), opt});
}

inline litert::Expected<const void*> FindAllocBase(
    const LiteRtDispatchDelegateOptions& opts) {
  auto alloc_base = opts.FindDispatchOption(kAllocBase);
  if (!alloc_base) {
    return alloc_base.Error();
  }
  return std::any_cast<const void*>(*alloc_base);
}

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_DISPATCH_DISPATCH_DELEGATE_OPTIONS_H_
