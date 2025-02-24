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

#include "tensorflow/lite/experimental/litert/compiler/plugin/compiler_flags.h"

#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"

namespace {
static constexpr absl::string_view kPairChar = "=";
static constexpr absl::string_view kDelim = ",";
}  // namespace

namespace litert::internal {

void CompilerFlags::Clear() {
  keys_.clear();
  values_.clear();
}

void CompilerFlags::Push(std::string key, std::string value) {
  keys_.push_back(std::move(key));
  values_.push_back(std::move(value));
}

LiteRtStatus CompilerFlags::SetPluginFlags(
    LiteRtCompilerPlugin handle,
    decltype(LiteRtCompilerPluginSetFlags) set_flags) const {
  std::vector<const char*> keys(keys_.size());
  std::vector<const char*> values(values_.size());
  for (auto i = 0; i < keys_.size(); ++i) {
    keys[i] = keys_[i].c_str();
    values[i] = values_[i].c_str();
  }
  return set_flags(handle, keys.size(), keys.data(), values.data());
}

Expected<CompilerFlags> ParseCompilerFlags(absl::string_view flags_str) {
  using KeyVal = std::pair<std::string, std::string>;

  CompilerFlags result;
  if (flags_str.empty()) {
    return result;
  }

  for (const auto flag : absl::StrSplit(flags_str, kDelim)) {
    KeyVal key_value = absl::StrSplit(flag, absl::MaxSplits(kPairChar, 1));
    result.Push(std::move(key_value.first), std::move(key_value.second));
  }

  return result;
}

}  // namespace litert::internal

std::ostream& operator<<(std::ostream& os,
                         const litert::internal::CompilerFlags& flags) {
  for (auto i = 0; i < flags.keys_.size(); ++i) {
    os << flags.keys_[i];
    const auto& value = flags.values_[i];
    if (!value.empty()) {
      os << kPairChar << value;
    }
    if (i < flags.keys_.size() - 1) {
      os << kDelim;
    }
  }
  return os;
}
