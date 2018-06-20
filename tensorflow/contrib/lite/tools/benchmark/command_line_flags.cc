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

#include "tensorflow/contrib/lite/tools/benchmark/command_line_flags.h"

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace tflite {
namespace {

template <typename T>
std::string ToString(T val) {
  std::ostringstream stream;
  stream << val;
  return stream.str();
}

bool ParseFlag(const std::string& arg, const std::string& flag,
               const std::function<bool(const std::string&)>& parse_func,
               bool* value_parsing_ok) {
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
bool ParseFlag(const std::string& flag_value, T* value) {
  std::istringstream stream(flag_value);
  T read_value;
  stream >> read_value;
  if (!stream.eof() && !stream.good()) {
    return false;
  }
  *value = read_value;
  return true;
}

bool ParseBoolFlag(const std::string& flag_value, bool* value) {
  if (flag_value != "true" && flag_value != "false") {
    return false;
  }

  *value = (flag_value == "true");
  return true;
}

bool ParseStringFlag(const std::string& flag_value, std::string* value) {
  *value = flag_value;
  return true;
}

}  // namespace

Flag::Flag(const char* name, int32_t* dst, const std::string& usage_text)
    : name_(name),
      type_(TYPE_INT32),
      value_hook_([dst](const std::string& flag_value) {
        return ParseFlag<int32_t>(flag_value, dst);
      }),
      default_for_display_(ToString(*dst)),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, int64_t* dst, const std::string& usage_text)
    : name_(name),
      type_(TYPE_INT64),
      value_hook_([dst](const std::string& flag_value) {
        return ParseFlag<int64_t>(flag_value, dst);
      }),
      default_for_display_(ToString(*dst)),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, float* dst, const std::string& usage_text)
    : name_(name),
      type_(TYPE_FLOAT),
      value_hook_([dst](const std::string& flag_value) {
        return ParseFlag<float>(flag_value, dst);
      }),
      default_for_display_(ToString(*dst)),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, bool* dst, const std::string& usage_text)
    : name_(name),
      type_(TYPE_BOOL),
      value_hook_([dst](const std::string& flag_value) {
        return ParseBoolFlag(flag_value, dst);
      }),
      default_for_display_((*dst) ? "true" : "false"),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, std::string* dst, const std::string& usage_text)
    : name_(name),
      type_(TYPE_STRING),
      value_hook_([dst](const std::string& flag_value) {
        return ParseStringFlag(flag_value, dst);
      }),
      default_for_display_(*dst),
      usage_text_(usage_text) {}

bool Flag::Parse(const std::string& arg, bool* value_parsing_ok) const {
  return ParseFlag(arg, name_, value_hook_, value_parsing_ok);
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
  std::vector<const char*> unknown_flags;
  for (int i = 1; i < *argc; ++i) {
    if (std::string(argv[i]) == "--") {
      while (i < *argc) {
        unknown_flags.push_back(argv[i]);
        ++i;
      }
      break;
    }

    bool was_found = false;
    for (const Flag& flag : flag_list) {
      bool value_parsing_ok;
      was_found = flag.Parse(argv[i], &value_parsing_ok);
      if (!value_parsing_ok) {
        result = false;
      }
      if (was_found) {
        break;
      }
    }
    if (!was_found) {
      unknown_flags.push_back(argv[i]);
    }
  }
  int dst = 1;  // Skip argv[0]
  for (auto f : unknown_flags) {
    argv[dst++] = f;
  }
  argv[dst++] = nullptr;
  *argc = unknown_flags.size() + 1;
  return result && (*argc < 2 || std::strcmp(argv[1], "--help") != 0);
}

/*static*/ std::string Flags::Usage(const std::string& cmdline,
                                    const std::vector<Flag>& flag_list) {
  std::ostringstream usage_text;
  usage_text << "usage: " << cmdline << "\n";
  if (!flag_list.empty()) {
    usage_text << "Flags:\n";
  }

  for (const Flag& flag : flag_list) {
    auto type_name = flag.GetTypeName();
    usage_text << "\t";
    usage_text << "--" << flag.name_ << "=" << flag.default_for_display_;
    usage_text << "\t" << type_name << "\t" << flag.usage_text_ << "\n";
  }
  return usage_text.str();
}

}  // namespace tflite
