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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_DISPATCH_DISPATCH_DELEGATE_OPTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_DISPATCH_DISPATCH_DELEGATE_OPTIONS_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_dispatch_delegate.h"

class LiteRtDispatchDelegateOptions {
 public:
  // Information about NPU binary, including the NPU binary bytecode and the
  // name of the entry-point function.
  struct ExecInfo {
    absl::Span<const uint8_t> bytecode;
    std::optional<std::string> function_name;
  };

  void AddOption(absl::string_view key, absl::string_view value) {
    options_[std::string{key}] = std::string{value};
  }

  std::optional<std::string> GetOption(const std::string& key) const {
    if (auto iter = options_.find(key); iter != options_.end()) {
      return iter->second;
    }
    return {};
  }

  void SetSharedLibraryDir(absl::string_view dir) { shared_library_dir_ = dir; }

  const std::optional<std::string>& GetSharedLibraryDir() const {
    return shared_library_dir_;
  }

  // Store a given ExecInfo object and associated it to a given tag.
  void AddExecInfo(absl::string_view exec_tag, ExecInfo&& exec_info) {
    exec_infos_[std::string{exec_tag}] = std::move(exec_info);
  }

  // Retrieve the ExecInfo object associated with a given tag.
  absl::StatusOr<ExecInfo> GetExecInfo(const std::string& exec_tag) const {
    if (auto iter = exec_infos_.find(exec_tag); iter != exec_infos_.end()) {
      return iter->second;
    }
    return absl::NotFoundError("ExecInfo not found");
  }

 private:
  // Options are stored as (key, value) pairs.
  std::map<std::string, std::string> options_;
  // ExecInfos are stored as (tag, ExecInfo) pairs.
  std::map<std::string, ExecInfo> exec_infos_;
  std::optional<std::string> shared_library_dir_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_DISPATCH_DISPATCH_DELEGATE_OPTIONS_H_
