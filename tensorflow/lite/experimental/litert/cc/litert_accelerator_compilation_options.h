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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_ACCELERATOR_COMPILATION_OPTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_ACCELERATOR_COMPILATION_OPTIONS_H_

#include <cassert>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_accelerator_compilation_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_handle.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"

namespace litert {

class AcceleratorCompilationOptions
    : public internal::Handle<LiteRtAcceleratorCompilationOptions,
                              LiteRtDestroyAcceleratorCompilationOptions> {
 public:
  AcceleratorCompilationOptions() = default;

  // Parameter `owned` indicates if the created AcceleratorCompilationOptions
  // object should take ownership of the provided `options` handle.
  explicit AcceleratorCompilationOptions(
      LiteRtAcceleratorCompilationOptions options, bool owned = true)
      : internal::Handle<LiteRtAcceleratorCompilationOptions,
                         LiteRtDestroyAcceleratorCompilationOptions>(options,
                                                                     owned) {}

  static Expected<AcceleratorCompilationOptions> Create(
      const LiteRtApiVersion& payload_version,
      const std::string& payload_identifier, void* payload_data,
      void (*payload_destructor)(void* payload_data)) {
    LiteRtAcceleratorCompilationOptions options;
    LITERT_RETURN_IF_ERROR(LiteRtCreateAcceleratorCompilationOptions(
        &payload_version, payload_identifier.c_str(), payload_data,
        payload_destructor, &options));
    return AcceleratorCompilationOptions(options);
  }

  Expected<LiteRtApiVersion> GetVersion() const {
    LiteRtApiVersion payload_version;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetAcceleratorCompilationOptionsVersion(Get(), &payload_version));
    return payload_version;
  }

  Expected<absl::string_view> GetIdentifier() const {
    const char* payload_identifier;
    LITERT_RETURN_IF_ERROR(LiteRtGetAcceleratorCompilationOptionsIdentifier(
        Get(), &payload_identifier));
    return absl::string_view(payload_identifier);
  }

  template <typename T>
  Expected<T*> GetData() const {
    void* payload_data;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetAcceleratorCompilationOptionsData(Get(), &payload_data));
    return reinterpret_cast<T*>(payload_data);
  }

  template <typename T>
  Expected<std::pair<LiteRtApiVersion, T*>> FindData(
      const std::string& payload_identifier) {
    LiteRtApiVersion payload_version;
    void* payload_data;
    LITERT_RETURN_IF_ERROR(LiteRtFindAcceleratorCompilationOptionsData(
        Get(), payload_identifier.c_str(), &payload_version, &payload_data));
    return std::make_pair(payload_version, reinterpret_cast<T*>(payload_data));
  }

  Expected<AcceleratorCompilationOptions> Next() {
    auto h = Get();
    LITERT_RETURN_IF_ERROR(LiteRtGetNextAcceleratorCompilationOptions(&h));
    return AcceleratorCompilationOptions(h, /*owned=*/false);
  }

  Expected<void> Append(AcceleratorCompilationOptions&& appended_options) {
    auto h = Get();
    LITERT_RETURN_IF_ERROR(LiteRtAppendAcceleratorCompilationOptions(
        &h, appended_options.Release()));
    if (h != Get()) {
      // If appending a new linked list item has changed the linked list head
      // pointer, then we need to reflect that as the new handle. Note that
      // should happen only if the previous handle was null.
      assert(!Get());
      *this = AcceleratorCompilationOptions(h);
    }
    return {};
  }

  Expected<void> Pop() {
    auto h = Get();
    LITERT_RETURN_IF_ERROR(LiteRtPopAcceleratorCompilationOptions(&h));
    if (h != Get()) {
      // If popping the last item has changed the linked list head pointer, then
      // we release the current handle since it has been already destructed by
      // the pop call, and then use the new head pointer as the new handle.
      (void)Release();
      *this = AcceleratorCompilationOptions(h);
    }
    return {};
  }
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_ACCELERATOR_COMPILATION_OPTIONS_H_
