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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_COMPILER_PLUGIN_COMPILER_FLAGS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_COMPILER_PLUGIN_COMPILER_FLAGS_H_

#include <ostream>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"

namespace litert::internal {
class CompilerFlags;
}

// For logging.
std::ostream& operator<<(std::ostream& os,
                         const litert::internal::CompilerFlags& flags);

namespace litert::internal {

class CompilerFlags {
 public:
  CompilerFlags() = default;

  // Clears all flags.
  void Clear();

  // Pushes a new flag to the end of the list.
  void Push(std::string key, std::string value = "");

  // Sets the flags on the given plugin.
  LiteRtStatus SetPluginFlags(
      LiteRtCompilerPlugin handle,
      decltype(LiteRtCompilerPluginSetFlags) set_flags) const;

 private:
  friend std::ostream& ::operator<<(std::ostream& os,
                                    const CompilerFlags& flags);

  std::vector<std::string> keys_;
  std::vector<std::string> values_;
};

// Parses a comma-separated (no space) list of compiler flags. Flags may be
// key-value pairs in the format of "key=value", or just "key". E.g.
// "key1=value1,key2".
Expected<CompilerFlags> ParseCompilerFlags(absl::string_view flags_str);

}  // namespace litert::internal

// For logging.

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_COMPILER_PLUGIN_COMPILER_FLAGS_H_
