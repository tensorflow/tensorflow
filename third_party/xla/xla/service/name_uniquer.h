/* Copyright 2017 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_SERVICE_NAME_UNIQUER_H_
#define XLA_SERVICE_NAME_UNIQUER_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/types.h"

namespace xla {

// Simple stateful class that helps generate "unique" names. To use it, simply
// call GetUniqueName as many times as needed. The names returned by
// GetUniqueName are guaranteed to be distinct for this instance of the class.
// Note that the names will be sanitized to match regexp
// "[a-zA-Z_][a-zA-Z0-9_.-]*".
class NameUniquer {
 public:
  // The separator must contain allowed characters only: "[a-zA-Z0-9_.-]".
  explicit NameUniquer(const std::string& separator = "__");

  // Get a sanitized unique name in a string, with an optional prefix for
  // convenience.
  std::string GetUniqueName(absl::string_view prefix = "");

  // Sanitizes and returns the name. Unallowed characters will be replaced with
  // '_'. The result will match the regexp "[a-zA-Z_][a-zA-Z0-9_.-]*".
  static std::string GetSanitizedName(absl::string_view name);

 private:
  // Used to track and generate new identifiers for the same instruction name
  // root.
  class SequentialIdGenerator {
   public:
    SequentialIdGenerator() = default;

    // Tries to register id as used identifier. If id is not already used, the
    // id itself will be returned. Otherwise a new one will be generated, and
    // returned.
    int64_t RegisterId(int64_t id) {
      if (used_.insert(id).second) {
        return id;
      }
      while (!used_.insert(next_).second) {
        ++next_;
      }
      return next_++;
    }

   private:
    // The next identifier to be tried.
    int64_t next_ = 0;

    // Set of all the identifiers which has been used.
    absl::flat_hash_set<int64_t> used_;
  };

  // The string to use to separate the prefix of the name from the uniquing
  // integer value.
  std::string separator_;

  // Map from name prefix to the generator data structure which tracks used
  // identifiers and generates new ones.
  absl::flat_hash_map<std::string, SequentialIdGenerator> generated_names_;

  NameUniquer(const NameUniquer&) = delete;
  NameUniquer& operator=(const NameUniquer&) = delete;
};

}  // namespace xla

#endif  // XLA_SERVICE_NAME_UNIQUER_H_
