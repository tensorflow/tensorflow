/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/compiler/mlir/lite/tools/command_line_flags.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/match.h"

// TODO(b/321735756): Remove this file and copy original once common library is
// implemented.

// LINT.IfChange

namespace mlir {
namespace {

template <typename T>
std::string ToString(T val) {
  std::ostringstream stream;
  stream << val;
  return stream.str();
}

template <>
std::string ToString(bool val) {
  return val ? "true" : "false";
}

template <>
std::string ToString(const std::string& val) {
  return val;
}

bool ParseFlag(const std::string& arg, int argv_position,
               const std::string& flag, bool positional,
               const std::function<bool(const std::string&, int argv_position)>&
                   parse_func,
               bool* value_parsing_ok) {
  if (positional) {
    *value_parsing_ok = parse_func(arg, argv_position);
    return true;
  }
  *value_parsing_ok = true;
  std::string flag_prefix = "--" + flag + "=";
  if (!absl::StartsWith(arg, flag_prefix)) {
    return false;
  }
  bool has_value = arg.size() >= flag_prefix.size();
  *value_parsing_ok = has_value;
  if (has_value) {
    *value_parsing_ok =
        parse_func(arg.substr(flag_prefix.size()), argv_position);
  }
  return true;
}

template <typename T>
bool ParseFlag(const std::string& flag_value, int argv_position,
               const std::function<void(const T&, int)>& hook) {
  std::istringstream stream(flag_value);
  T read_value;
  stream >> read_value;
  if (!stream.eof() && !stream.good()) {
    return false;
  }
  hook(read_value, argv_position);
  return true;
}

template <>
bool ParseFlag(const std::string& flag_value, int argv_position,
               const std::function<void(const bool&, int)>& hook) {
  if (flag_value != "true" && flag_value != "false" && flag_value != "0" &&
      flag_value != "1") {
    return false;
  }

  hook(flag_value == "true" || flag_value == "1", argv_position);
  return true;
}

template <typename T>
bool ParseFlag(const std::string& flag_value, int argv_position,
               const std::function<void(const std::string&, int)>& hook) {
  hook(flag_value, argv_position);
  return true;
}
}  // namespace

#define CONSTRUCTOR_IMPLEMENTATION(flag_T, default_value_T, flag_enum_val)     \
  Flag::Flag(const char* name,                                                 \
             const std::function<void(const flag_T& /*flag_val*/,              \
                                      int /*argv_position*/)>& hook,           \
             default_value_T default_value, const std::string& usage_text,     \
             FlagType flag_type)                                               \
      : name_(name),                                                           \
        type_(flag_enum_val),                                                  \
        value_hook_([hook](const std::string& flag_value, int argv_position) { \
          return ParseFlag<flag_T>(flag_value, argv_position, hook);           \
        }),                                                                    \
        default_for_display_(ToString<default_value_T>(default_value)),        \
        usage_text_(usage_text),                                               \
        flag_type_(flag_type) {}

CONSTRUCTOR_IMPLEMENTATION(int32_t, int32_t, TYPE_INT32)
CONSTRUCTOR_IMPLEMENTATION(int64_t, int64_t, TYPE_INT64)
CONSTRUCTOR_IMPLEMENTATION(float, float, TYPE_FLOAT)
CONSTRUCTOR_IMPLEMENTATION(bool, bool, TYPE_BOOL)
CONSTRUCTOR_IMPLEMENTATION(std::string, const std::string&, TYPE_STRING)

#undef CONSTRUCTOR_IMPLEMENTATION

bool Flag::Parse(const std::string& arg, int argv_position,
                 bool* value_parsing_ok) const {
  return ParseFlag(
      arg, argv_position, name_, flag_type_ == kPositional,
      [&](const std::string& read_value, int argv_position) {
        return value_hook_(read_value, argv_position);
      },
      value_parsing_ok);
}


/*static*/ bool Flags::Parse(int* argc, const char** argv,
                             const std::vector<Flag>& flag_list) {
  bool result = true;
  std::vector<bool> unknown_argvs(*argc, true);
  // Record the list of flags that have been processed. key is the flag's name
  // and the value is the corresponding argv index if there's one, or -1 when
  // the argv list doesn't contain this flag.
  std::unordered_map<std::string, int> processed_flags;

  // Stores indexes of flag_list in a sorted order.
  std::vector<int> sorted_idx(flag_list.size());
  std::iota(std::begin(sorted_idx), std::end(sorted_idx), 0);
  std::sort(sorted_idx.begin(), sorted_idx.end(), [&flag_list](int a, int b) {
    return flag_list[a].GetFlagType() < flag_list[b].GetFlagType();
  });
  int positional_count = 0;

  for (int idx = 0; idx < sorted_idx.size(); ++idx) {
    const Flag& flag = flag_list[sorted_idx[idx]];

    const auto it = processed_flags.find(flag.name_);
    if (it != processed_flags.end()) {
#ifndef NDEBUG
      // Only log this in debug builds.
      LOG(WARNING) << "Duplicate flags: " << flag.name_;
#endif
      if (it->second != -1) {
        bool value_parsing_ok;
        flag.Parse(argv[it->second], it->second, &value_parsing_ok);
        if (!value_parsing_ok) {
          LOG(ERROR) << "Failed to parse flag '" << flag.name_
                     << "' against argv '" << argv[it->second] << "'";
          result = false;
        }
        continue;
      } else if (flag.flag_type_ == Flag::kRequired) {
        LOG(ERROR) << "Required flag not provided: " << flag.name_;
        // If the required flag isn't found, we immediately stop the whole flag
        // parsing.
        result = false;
        break;
      }
    }

    // Parses positional flags.
    if (flag.flag_type_ == Flag::kPositional) {
      if (++positional_count >= *argc) {
        LOG(ERROR) << "Too few command line arguments.";
        return false;
      }
      bool value_parsing_ok;
      flag.Parse(argv[positional_count], positional_count, &value_parsing_ok);
      if (!value_parsing_ok) {
        LOG(ERROR) << "Failed to parse positional flag: " << flag.name_;
        return false;
      }
      unknown_argvs[positional_count] = false;
      processed_flags[flag.name_] = positional_count;
      continue;
    }

    // Parse other flags.
    bool was_found = false;
    for (int i = positional_count + 1; i < *argc; ++i) {
      if (!unknown_argvs[i]) continue;
      bool value_parsing_ok;
      was_found = flag.Parse(argv[i], i, &value_parsing_ok);
      if (!value_parsing_ok) {
        LOG(ERROR) << "Failed to parse flag '" << flag.name_
                   << "' against argv '" << argv[i] << "'";
        result = false;
      }
      if (was_found) {
        unknown_argvs[i] = false;
        processed_flags[flag.name_] = i;
        break;
      }
    }

    // If the flag is found from the argv (i.e. the flag name appears in argv),
    // continue to the next flag parsing.
    if (was_found) continue;

    // The flag isn't found, do some bookkeeping work.
    processed_flags[flag.name_] = -1;
    if (flag.flag_type_ == Flag::kRequired) {
      LOG(ERROR) << "Required flag not provided: " << flag.name_;
      result = false;
      // If the required flag isn't found, we immediately stop the whole flag
      // parsing by breaking the outer-loop (i.e. the 'sorted_idx'-iteration
      // loop).
      break;
    }
  }

  int dst = 1;  // Skip argv[0]
  for (int i = 1; i < *argc; ++i) {
    if (unknown_argvs[i]) {
      argv[dst++] = argv[i];
    }
  }
  *argc = dst;
  return result && (*argc < 2 || std::strcmp(argv[1], "--help") != 0);
}

}  // namespace mlir

// LINT.ThenChange(//tensorflow/lite/tools/command_line_flags.cc)
