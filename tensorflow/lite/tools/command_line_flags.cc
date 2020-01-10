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
#include <utility>
#include <vector>

#include "tensorflow/core/platform/logging.h"

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
  return ParseFlag(arg, name_, flag_type_ == POSITIONAL, value_hook_,
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
  std::vector<bool> unknown_flags(*argc, true);
  // Stores indexes of flag_list in a sorted order.
  std::vector<int> sorted_idx(flag_list.size());
  std::iota(std::begin(sorted_idx), std::end(sorted_idx), 0);
  std::sort(sorted_idx.begin(), sorted_idx.end(), [&flag_list](int a, int b) {
    return flag_list[a].GetFlagType() < flag_list[b].GetFlagType();
  });
  int positional_count = 0;

  for (int i = 0; i < sorted_idx.size(); ++i) {
    const Flag& flag = flag_list[sorted_idx[i]];
    // Parses positional flags.
    if (flag.flag_type_ == Flag::POSITIONAL) {
      if (++positional_count >= *argc) {
        LOG(ERROR) << "Too few command line arguments";
        return false;
      }
      bool value_parsing_ok;
      flag.Parse(argv[positional_count], &value_parsing_ok);
      if (!value_parsing_ok) {
        LOG(ERROR) << "Failed to parse positional flag: " << flag.name_;
        return false;
      }
      unknown_flags[positional_count] = false;
      continue;
    }

    // Parse other flags.
    bool was_found = false;
    for (int i = positional_count + 1; i < *argc; ++i) {
      if (!unknown_flags[i]) continue;
      bool value_parsing_ok;
      was_found = flag.Parse(argv[i], &value_parsing_ok);
      if (!value_parsing_ok) {
        LOG(ERROR) << "Failed to parse flag: " << flag.name_;
        result = false;
      }
      if (was_found) {
        unknown_flags[i] = false;
        break;
      }
    }
    // Check if required flag not found.
    if (flag.flag_type_ == Flag::REQUIRED && !was_found) {
      LOG(ERROR) << "Required flag not provided: " << flag.name_;
      result = false;
      break;
    }
  }

  int dst = 1;  // Skip argv[0]
  for (int i = 1; i < *argc; ++i) {
    if (unknown_flags[i]) {
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
    if (flag.flag_type_ == Flag::POSITIONAL) {
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
    if (flag.flag_type_ != Flag::POSITIONAL) {
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
    usage_text << (flag.flag_type_ != Flag::OPTIONAL ? "required" : "optional");
    usage_text << "\t" << flag.usage_text_ << "\n";
  }
  return usage_text.str();
}  // namespace tflite

}  // namespace tflite
