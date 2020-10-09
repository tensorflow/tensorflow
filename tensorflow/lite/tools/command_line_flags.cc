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

#include "tensorflow/lite/tools/command_line_flags.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace {

template <typename T>
std::string ToString(T val) {
  std::ostringstream stream;
  stream << val;
  return stream.str();
}

bool ParseFlag(const std::string& arg, const std::string& flag, bool positional,
               const std::function<bool(const std::string&)>& parse_func,
               bool* value_parsing_ok) {
  if (positional) {
    *value_parsing_ok = parse_func(arg);
    return true;
  }
  *value_parsing_ok = true;
  std::string flag_prefix = "--" + flag + "=";
  if (arg.find(flag_prefix) != 0) {
    return false;
  }
  bool has_value = arg.size() >= flag_prefix.size();
  *value_parsing_ok = has_value;
  if (has_value) {
    *value_parsing_ok = parse_func(arg.substr(flag_prefix.size()));
  }
  return true;
}

template <typename T>
bool ParseFlag(const std::string& flag_value,
               const std::function<void(const T&)>& hook) {
  std::istringstream stream(flag_value);
  T read_value;
  stream >> read_value;
  if (!stream.eof() && !stream.good()) {
    return false;
  }
  hook(read_value);
  return true;
}

bool ParseBoolFlag(const std::string& flag_value,
                   const std::function<void(const bool&)>& hook) {
  if (flag_value != "true" && flag_value != "false" && flag_value != "0" &&
      flag_value != "1") {
    return false;
  }

  hook(flag_value == "true" || flag_value == "1");
  return true;
}
}  // namespace

Flag::Flag(const char* name, const std::function<void(const int32_t&)>& hook,
           int32_t default_value, const std::string& usage_text,
           FlagType flag_type)
    : name_(name),
      type_(TYPE_INT32),
      value_hook_([hook](const std::string& flag_value) {
        return ParseFlag<int32_t>(flag_value, hook);
      }),
      default_for_display_(ToString(default_value)),
      usage_text_(usage_text),
      flag_type_(flag_type) {}

Flag::Flag(const char* name, const std::function<void(const int64_t&)>& hook,
           int64_t default_value, const std::string& usage_text,
           FlagType flag_type)
    : name_(name),
      type_(TYPE_INT64),
      value_hook_([hook](const std::string& flag_value) {
        return ParseFlag<int64_t>(flag_value, hook);
      }),
      default_for_display_(ToString(default_value)),
      usage_text_(usage_text),
      flag_type_(flag_type) {}

Flag::Flag(const char* name, const std::function<void(const float&)>& hook,
           float default_value, const std::string& usage_text,
           FlagType flag_type)
    : name_(name),
      type_(TYPE_FLOAT),
      value_hook_([hook](const std::string& flag_value) {
        return ParseFlag<float>(flag_value, hook);
      }),
      default_for_display_(ToString(default_value)),
      usage_text_(usage_text),
      flag_type_(flag_type) {}

Flag::Flag(const char* name, const std::function<void(const bool&)>& hook,
           bool default_value, const std::string& usage_text,
           FlagType flag_type)
    : name_(name),
      type_(TYPE_BOOL),
      value_hook_([hook](const std::string& flag_value) {
        return ParseBoolFlag(flag_value, hook);
      }),
      default_for_display_(default_value ? "true" : "false"),
      usage_text_(usage_text),
      flag_type_(flag_type) {}

Flag::Flag(const char* name,
           const std::function<void(const std::string&)>& hook,
           const std::string& default_value, const std::string& usage_text,
           FlagType flag_type)
    : name_(name),
      type_(TYPE_STRING),
      value_hook_([hook](const std::string& flag_value) {
        hook(flag_value);
        return true;
      }),
      default_for_display_(default_value),
      usage_text_(usage_text),
      flag_type_(flag_type) {}

bool Flag::Parse(const std::string& arg, bool* value_parsing_ok) const {
  return ParseFlag(arg, name_, flag_type_ == kPositional, value_hook_,
                   value_parsing_ok);
}

std::string Flag::GetTypeName() const {
  switch (type_) {
    case TYPE_INT32:
      return "int32";
    case TYPE_INT64:
      return "int64";
    case TYPE_FLOAT:
      return "float";
    case TYPE_BOOL:
      return "bool";
    case TYPE_STRING:
      return "string";
  }

  return "unknown";
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
      TFLITE_LOG(WARN) << "Duplicate flags: " << flag.name_;
#endif
      if (it->second != -1) {
        bool value_parsing_ok;
        flag.Parse(argv[it->second], &value_parsing_ok);
        if (!value_parsing_ok) {
          TFLITE_LOG(ERROR) << "Failed to parse flag '" << flag.name_
                            << "' against argv '" << argv[it->second] << "'";
          result = false;
        }
        continue;
      } else if (flag.flag_type_ == Flag::kRequired) {
        TFLITE_LOG(ERROR) << "Required flag not provided: " << flag.name_;
        // If the required flag isn't found, we immediately stop the whole flag
        // parsing.
        result = false;
        break;
      }
    }

    // Parses positional flags.
    if (flag.flag_type_ == Flag::kPositional) {
      if (++positional_count >= *argc) {
        TFLITE_LOG(ERROR) << "Too few command line arguments.";
        return false;
      }
      bool value_parsing_ok;
      flag.Parse(argv[positional_count], &value_parsing_ok);
      if (!value_parsing_ok) {
        TFLITE_LOG(ERROR) << "Failed to parse positional flag: " << flag.name_;
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
      was_found = flag.Parse(argv[i], &value_parsing_ok);
      if (!value_parsing_ok) {
        TFLITE_LOG(ERROR) << "Failed to parse flag '" << flag.name_
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
      TFLITE_LOG(ERROR) << "Required flag not provided: " << flag.name_;
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

/*static*/ std::string Flags::Usage(const std::string& cmdline,
                                    const std::vector<Flag>& flag_list) {
  // Stores indexes of flag_list in a sorted order.
  std::vector<int> sorted_idx(flag_list.size());
  std::iota(std::begin(sorted_idx), std::end(sorted_idx), 0);
  std::sort(sorted_idx.begin(), sorted_idx.end(), [&flag_list](int a, int b) {
    return flag_list[a].GetFlagType() < flag_list[b].GetFlagType();
  });
  // Counts number of positional flags will be shown.
  int positional_count = 0;
  std::ostringstream usage_text;
  usage_text << "usage: " << cmdline;
  // Prints usage for positional flag.
  for (int i = 0; i < sorted_idx.size(); ++i) {
    const Flag& flag = flag_list[sorted_idx[i]];
    if (flag.flag_type_ == Flag::kPositional) {
      positional_count++;
      usage_text << " <" << flag.name_ << ">";
    } else {
      usage_text << " <flags>";
      break;
    }
  }
  usage_text << "\n";

  // Finds the max number of chars of the name column in the usage message.
  int max_name_width = 0;
  std::vector<std::string> name_column(flag_list.size());
  for (int i = 0; i < sorted_idx.size(); ++i) {
    const Flag& flag = flag_list[sorted_idx[i]];
    if (flag.flag_type_ != Flag::kPositional) {
      name_column[i] += "--";
      name_column[i] += flag.name_;
      name_column[i] += "=";
      name_column[i] += flag.default_for_display_;
    } else {
      name_column[i] += flag.name_;
    }
    if (name_column[i].size() > max_name_width) {
      max_name_width = name_column[i].size();
    }
  }

  if (positional_count > 0) {
    usage_text << "Where:\n";
  }
  for (int i = 0; i < sorted_idx.size(); ++i) {
    const Flag& flag = flag_list[sorted_idx[i]];
    if (i == positional_count) {
      usage_text << "Flags:\n";
    }
    auto type_name = flag.GetTypeName();
    usage_text << "\t";
    usage_text << std::left << std::setw(max_name_width) << name_column[i];
    usage_text << "\t" << type_name << "\t";
    usage_text << (flag.flag_type_ != Flag::kOptional ? "required"
                                                      : "optional");
    usage_text << "\t" << flag.usage_text_ << "\n";
  }
  return usage_text.str();
}

/*static*/ std::string Flags::ArgsToString(int argc, const char** argv) {
  std::string args;
  for (int i = 1; i < argc; ++i) {
    args.append(argv[i]);
    if (i != argc - 1) args.append(" ");
  }
  return args;
}

}  // namespace tflite
