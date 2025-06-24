/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_TSL_UTIL_FIXED_OPTION_SET_FLAG_H_
#define XLA_TSL_UTIL_FIXED_OPTION_SET_FLAG_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

namespace xla {

// Controls FixedOptionSetFlagParser's behavior.
struct FixedOptionSetFlagParserConfig {
  // If true, allows aliases for flag options. The first option listed for a
  // given name takes precedence when unparsing.
  bool allow_aliases = false;
  // Whether the flag values are case sensitive. It's a bad practice to have
  // case-insensitive flag values. DO NOT SET THIS FIELD TO FALSE IN NEW CODE.
  bool case_sensitive_do_not_use_in_new_code = true;
};

// A parser for a flag of type T that takes a fixed set of options. This makes
// it easier and safer to define flags that take a fixed set of options.
// Requires T to support equality comparison, hashing, and conversion to
// std::string via absl::StrCat.
//
// Example usage:
//
// enum class Foo {
//   kBar,
//   kBaz,
// };
//
// static const FixedOptionSetFlagParser<Foo>& GetFooParser() {
//   static const auto& parser = GetFixedOptionSetFlagParser<Foo>({
//       {"bar", Foo::kBar, "Optional description of bar."},
//       {"baz", Foo::kBaz, "Optional description of baz."},
//   });
//   return parser;
// }
//
// bool AbslParseFlag(absl::string_view text, Foo* foo, std::string* error) {
//   return GetFooParser().Parse(text, foo, error);
// }
//
// std::string AbslUnparseFlag(Foo foo) { return GetFooParser().Unparse(foo); }
//
// Compared with implementing AbslParseFlag and AbslUnparseFlag manually, this
// class provides the following benefits:
//
// - We only need to define the mapping between options and values once, and
//   the two directions are guaranteed to be consistent.
// - The parser validates the flag options, so it's impossible to have
//   duplicate names or values in the mapping.
//
// This class is thread-safe.
template <typename T>
class FixedOptionSetFlagParser {
 public:
  // Stores the name, value, and description of one option of a flag of type T.
  struct FlagOption {
    std::string name;
    T value;
    std::string description;
  };

  // Creates a parser for a flag of type T that takes a fixed set of options.
  // The options must be valid, i.e., there must be no duplicate names or
  // values.
  explicit FixedOptionSetFlagParser(
      const std::vector<FlagOption>& options,
      const FixedOptionSetFlagParserConfig& config)
      : options_(ValidateFlagOptionsOrDie(options, config)),
        case_sensitive_(config.case_sensitive_do_not_use_in_new_code) {}

  // Parses the flag from the given text. Returns true if the text is
  // valid, and sets the value to the corresponding option. Otherwise, returns
  // false and sets the error message.
  [[nodiscard]] bool Parse(absl::string_view text, T* value,
                           std::string* error) const {
    for (const auto& option : options_) {
      if ((case_sensitive_ && text == option.name) ||
          (!case_sensitive_ && absl::EqualsIgnoreCase(text, option.name))) {
        *value = option.value;
        return true;
      }
    }
    *error = absl::StrCat(
        "Unrecognized flag option: ", text, ". Valid options are: ",
        absl::StrJoin(options_, ", ",
                      [](std::string* out, const FlagOption& option) {
                        absl::StrAppend(out, option.name);
                        if (!option.description.empty()) {
                          absl::StrAppend(out, " (", option.description, ")");
                        }
                      }),
        ".");
    return false;
  }

  // Unparses the flag value to the corresponding option name. If the value is
  // not one of the options, returns the string representation of the value.
  [[nodiscard]] std::string Unparse(const T& value) const {
    for (const auto& option : options_) {
      if (option.value == value) {
        return std::string(option.name);
      }
    }
    return absl::StrCat(value);
  }

 private:
  // Validates the flag options and returns them. Dies if the options are not
  // valid.
  static std::vector<FlagOption> ValidateFlagOptionsOrDie(
      const std::vector<FlagOption>& options,
      const FixedOptionSetFlagParserConfig& config) {
    // Check that the same name or value is not used multiple times.
    absl::flat_hash_set<std::string> names;
    absl::flat_hash_set<T> values;
    for (const auto& option : options) {
      CHECK(!names.contains(option.name))
          << "Duplicate flag option name: " << option.name;
      names.insert(option.name);

      if (!config.allow_aliases) {
        CHECK(!values.contains(option.value))
            << "Duplicate flag option value: " << absl::StrCat(option.value);
        values.insert(option.value);
      }
    }
    return options;
  }

  const std::vector<FlagOption> options_;
  const bool case_sensitive_ = true;
};

// Returns the parser for a flag of type T that takes a fixed set of options.
// The options must be valid, i.e., there must be no duplicate names or values.
// The returned parser is guaranteed to be alive for the lifetime of the
// program.
//
// For each T, the caller must call this function exactly once to get the
// parser, and then use the parser to define the AbslParseFlag and
// AbslUnparseFlag functions for T.
template <typename T>
[[nodiscard]] const FixedOptionSetFlagParser<T>& GetFixedOptionSetFlagParser(
    const std::vector<typename FixedOptionSetFlagParser<T>::FlagOption>&
        options,
    const FixedOptionSetFlagParserConfig& config = {}) {
  // Per Google C++ style guide, we use a function-local static
  // variable to ensure that the parser is only created once and never
  // destroyed. We cannot use absl::NoDestructor here because it is not
  // available in the version of Abseil that openxla uses.
  static const auto* const parser =
      new FixedOptionSetFlagParser<T>(options, config);
  return *parser;
}

}  // namespace xla

#endif  // XLA_TSL_UTIL_FIXED_OPTION_SET_FLAG_H_
