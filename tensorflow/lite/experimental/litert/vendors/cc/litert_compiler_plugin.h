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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_LITERT_COMPILER_PLUGIN_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_LITERT_COMPILER_PLUGIN_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"

namespace litert {

// Deleter for incomplete compiler plugin type.
struct LiteRtCompilerPluginDeleter {
  void operator()(LiteRtCompilerPlugin plugin) {
    if (plugin != nullptr) {
      LiteRtDestroyCompilerPlugin(plugin);
    }
  }
};

// Smart pointer wrapper for incomplete plugin type.
using PluginPtr =
    std::unique_ptr<LiteRtCompilerPluginT, LiteRtCompilerPluginDeleter>;

// Initialize a plugin via c-api and wrap result in smart pointer.
inline PluginPtr CreatePlugin() {
  LiteRtCompilerPlugin plugin;
  LITERT_CHECK_STATUS_OK(LiteRtCreateCompilerPlugin(&plugin));
  return PluginPtr(plugin);
}

class CompilerFlags {
 public:
  CompilerFlags() = default;

  void Clear() {
    keys_.clear();
    values_.clear();
  }

  void Push(std::string key, std::string value = "") {
    keys_.push_back(std::move(key));
    values_.push_back(std::move(value));
  }

  LiteRtStatus SetPluginFlags(
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

 private:
  std::vector<std::string> keys_;
  std::vector<std::string> values_;
};

}  // namespace litert
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_LITERT_COMPILER_PLUGIN_H_
