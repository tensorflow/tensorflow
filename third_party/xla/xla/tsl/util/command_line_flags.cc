/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/util/command_line_flags.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/logging.h"

namespace tsl {
namespace {

bool ParseStringFlag(absl::string_view arg, absl::string_view flag,
                     absl::FunctionRef<bool(std::string)> hook,
                     bool* value_parsing_ok) {
  *value_parsing_ok = true;
  if (absl::ConsumePrefix(&arg, "--") && absl::ConsumePrefix(&arg, flag) &&
      absl::ConsumePrefix(&arg, "=")) {
    *value_parsing_ok = hook(std::string(arg));
    return true;
  }

  return false;
}

template <typename T>
bool ParseIntFlag(absl::string_view arg, absl::string_view flag,
                  absl::FunctionRef<bool(T)> hook, bool* value_parsing_ok) {
  static_assert(std::numeric_limits<T>::is_integer);
  *value_parsing_ok = true;
  if (absl::ConsumePrefix(&arg, "--") && absl::ConsumePrefix(&arg, flag) &&
      absl::ConsumePrefix(&arg, "=")) {
    T parsed_int;
    if (absl::SimpleAtoi(arg, &parsed_int)) {
      *value_parsing_ok = hook(parsed_int);
    } else {
      LOG(ERROR) << "Couldn't interpret value " << arg << " for flag " << flag
                 << ".";
      *value_parsing_ok = false;
    }
    return true;
  }

  return false;
}

bool ParseBoolFlag(absl::string_view arg, absl::string_view flag,
                   absl::FunctionRef<bool(bool)> hook, bool* value_parsing_ok) {
  *value_parsing_ok = true;
  if (absl::ConsumePrefix(&arg, "--") && absl::ConsumePrefix(&arg, flag)) {
    if (arg.empty()) {
      *value_parsing_ok = hook(true);
      return true;
    }
    // It's probably another argument name which begins with the name of this.
    if (!absl::ConsumePrefix(&arg, "=")) {
      return false;
    }
    if (absl::EqualsIgnoreCase(arg, "true") || arg == "1") {
      *value_parsing_ok = hook(true);
      return true;
    }
    if (absl::EqualsIgnoreCase(arg, "false") || arg == "0") {
      *value_parsing_ok = hook(false);
      return true;
    }
    LOG(ERROR) << "Couldn't interpret value " << arg << " for flag " << flag
               << ".";
    *value_parsing_ok = false;
    return true;
  }

  return false;
}

bool ParseFloatFlag(absl::string_view arg, absl::string_view flag,
                    absl::FunctionRef<bool(float)> hook,
                    bool* value_parsing_ok) {
  *value_parsing_ok = true;
  if (absl::ConsumePrefix(&arg, "--") && absl::ConsumePrefix(&arg, flag) &&
      absl::ConsumePrefix(&arg, "=")) {
    float parsed_float;
    if (absl::SimpleAtof(arg, &parsed_float)) {
      *value_parsing_ok = hook(parsed_float);
    } else {
      LOG(ERROR) << "Couldn't interpret value " << arg << " for flag " << flag
                 << ".";
      *value_parsing_ok = false;
    }
    return true;
  }

  return false;
}

}  // namespace

Flag::Flag(const char* name, int32_t* dst, absl::string_view usage_text,
           bool* dst_updated)
    : name_(name),
      type_(TYPE_INT32),
      int32_hook_([dst, dst_updated](int32_t value) {
        *dst = value;
        if (dst_updated) {
          *dst_updated = true;
        }
        return true;
      }),
      int32_default_for_display_(*dst),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, int64_t* dst, absl::string_view usage_text,
           bool* dst_updated)
    : name_(name),
      type_(TYPE_INT64),
      int64_hook_([dst, dst_updated](int64_t value) {
        *dst = value;
        if (dst_updated) {
          *dst_updated = true;
        }
        return true;
      }),
      int64_default_for_display_(*dst),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, float* dst, absl::string_view usage_text,
           bool* dst_updated)
    : name_(name),
      type_(TYPE_FLOAT),
      float_hook_([dst, dst_updated](float value) {
        *dst = value;
        if (dst_updated) {
          *dst_updated = true;
        }
        return true;
      }),
      float_default_for_display_(*dst),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, bool* dst, absl::string_view usage_text,
           bool* dst_updated)
    : name_(name),
      type_(TYPE_BOOL),
      bool_hook_([dst, dst_updated](bool value) {
        *dst = value;
        if (dst_updated) {
          *dst_updated = true;
        }
        return true;
      }),
      bool_default_for_display_(*dst),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, std::string* dst, absl::string_view usage_text,
           bool* dst_updated)
    : name_(name),
      type_(TYPE_STRING),
      string_hook_([dst, dst_updated](std::string value) {
        *dst = std::move(value);
        if (dst_updated) {
          *dst_updated = true;
        }
        return true;
      }),
      string_default_for_display_(*dst),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, std::function<bool(int32_t)> int32_hook,
           int32_t default_value_for_display, absl::string_view usage_text)
    : name_(name),
      type_(TYPE_INT32),
      int32_hook_(std::move(int32_hook)),
      int32_default_for_display_(default_value_for_display),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, std::function<bool(int64_t)> int64_hook,
           int64_t default_value_for_display, absl::string_view usage_text)
    : name_(name),
      type_(TYPE_INT64),
      int64_hook_(std::move(int64_hook)),
      int64_default_for_display_(default_value_for_display),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, std::function<bool(float)> float_hook,
           float default_value_for_display, absl::string_view usage_text)
    : name_(name),
      type_(TYPE_FLOAT),
      float_hook_(std::move(float_hook)),
      float_default_for_display_(default_value_for_display),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, std::function<bool(bool)> bool_hook,
           bool default_value_for_display, absl::string_view usage_text)
    : name_(name),
      type_(TYPE_BOOL),
      bool_hook_(std::move(bool_hook)),
      bool_default_for_display_(default_value_for_display),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, std::function<bool(std::string)> string_hook,
           std::string default_value_for_display, absl::string_view usage_text)
    : name_(name),
      type_(TYPE_STRING),
      string_hook_(std::move(string_hook)),
      string_default_for_display_(std::move(default_value_for_display)),
      usage_text_(usage_text) {}

bool Flag::Parse(absl::string_view arg, bool* value_parsing_ok) const {
  bool result = false;
  if (type_ == TYPE_INT32) {
    result = ParseIntFlag<int32_t>(arg, name_, int32_hook_, value_parsing_ok);
  } else if (type_ == TYPE_INT64) {
    result = ParseIntFlag<int64_t>(arg, name_, int64_hook_, value_parsing_ok);
  } else if (type_ == TYPE_BOOL) {
    result = ParseBoolFlag(arg, name_, bool_hook_, value_parsing_ok);
  } else if (type_ == TYPE_STRING) {
    result = ParseStringFlag(arg, name_, string_hook_, value_parsing_ok);
  } else if (type_ == TYPE_FLOAT) {
    result = ParseFloatFlag(arg, name_, float_hook_, value_parsing_ok);
  }
  return result;
}

/*static*/ bool Flags::Parse(int* argc, char** argv,
                             absl::Span<const Flag> flag_list) {
  bool result = true;
  std::vector<char*> unknown_flags;
  for (int i = 1; i < *argc; ++i) {
    absl::string_view arg = argv[i];
    if (arg == "--") {
      while (i < *argc) {
        unknown_flags.push_back(argv[i]);
        ++i;
      }
      break;
    }

    bool was_found = false;
    for (const Flag& flag : flag_list) {
      bool value_parsing_ok;
      was_found = flag.Parse(arg, &value_parsing_ok);
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
  // Passthrough any extra flags.
  absl::c_copy(unknown_flags, argv + 1);
  *argc = unknown_flags.size() + 1;
  argv[*argc] = nullptr;
  return result && (*argc < 2 || absl::string_view(argv[1]) != "--help");
}

/*static*/ bool Flags::Parse(std::vector<std::string>& flags,
                             absl::Span<const Flag> flag_list) {
  bool result = true;
  std::vector<std::string> unknown_flags;
  for (auto& flag : flags) {
    for (const Flag& flag_object : flag_list) {
      bool value_parsing_ok;
      bool was_found = flag_object.Parse(flag, &value_parsing_ok);
      if (!value_parsing_ok) {
        result = false;
      }
      // Clear parsed flags, these empty entries are removed later.
      if (was_found) {
        flag.clear();
        break;
      }
    }
  }
  auto IsEmpty = [](absl::string_view flag) { return flag.empty(); };
  flags.erase(std::remove_if(flags.begin(), flags.end(), IsEmpty), flags.end());
  return result;
}

/*static*/ std::string Flags::Usage(absl::string_view cmdline,
                                    absl::Span<const Flag> flag_list) {
  std::string usage_text;
  if (!flag_list.empty()) {
    absl::StrAppendFormat(&usage_text, "usage: %s\nFlags:\n", cmdline);
  } else {
    absl::StrAppendFormat(&usage_text, "usage: %s\n", cmdline);
  }
  for (const Flag& flag : flag_list) {
    const char* type_name = "";
    std::string flag_string;
    if (flag.type_ == Flag::TYPE_INT32) {
      type_name = "int32";
      flag_string = absl::StrFormat("--%s=%d", flag.name_,
                                    flag.int32_default_for_display_);
    } else if (flag.type_ == Flag::TYPE_INT64) {
      type_name = "int64";
      flag_string = absl::StrFormat("--%s=%d", flag.name_,
                                    flag.int64_default_for_display_);
    } else if (flag.type_ == Flag::TYPE_BOOL) {
      type_name = "bool";
      flag_string = absl::StrFormat("--%s=%v", flag.name_,
                                    flag.bool_default_for_display_);
    } else if (flag.type_ == Flag::TYPE_STRING) {
      type_name = "string";
      flag_string = absl::StrFormat("--%s=\"%s\"", flag.name_,
                                    flag.string_default_for_display_);
    } else if (flag.type_ == Flag::TYPE_FLOAT) {
      type_name = "float";
      flag_string = absl::StrFormat("--%s=%f", flag.name_,
                                    flag.float_default_for_display_);
    }
    absl::StrAppendFormat(&usage_text, "\t%-33s\t%s\t%s\n", flag_string,
                          type_name, flag.usage_text_);
  }
  return usage_text;
}

}  // namespace tsl
