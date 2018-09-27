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

#include <string>
#include <vector>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace {

bool ParseStringFlag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                     const std::function<bool(string)>& hook,
                     bool* value_parsing_ok) {
  *value_parsing_ok = true;
  if (str_util::ConsumePrefix(&arg, "--") &&
      str_util::ConsumePrefix(&arg, flag) &&
      str_util::ConsumePrefix(&arg, "=")) {
    *value_parsing_ok = hook(string(arg));
    return true;
  }

  return false;
}

bool ParseInt32Flag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                    const std::function<bool(int32)>& hook,
                    bool* value_parsing_ok) {
  *value_parsing_ok = true;
  if (str_util::ConsumePrefix(&arg, "--") &&
      str_util::ConsumePrefix(&arg, flag) &&
      str_util::ConsumePrefix(&arg, "=")) {
    char extra;
    int32 parsed_int32;
    if (sscanf(arg.data(), "%d%c", &parsed_int32, &extra) != 1) {
      LOG(ERROR) << "Couldn't interpret value " << arg << " for flag " << flag
                 << ".";
      *value_parsing_ok = false;
    } else {
      *value_parsing_ok = hook(parsed_int32);
    }
    return true;
  }

  return false;
}

bool ParseInt64Flag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                    const std::function<bool(int64)>& hook,
                    bool* value_parsing_ok) {
  *value_parsing_ok = true;
  if (str_util::ConsumePrefix(&arg, "--") &&
      str_util::ConsumePrefix(&arg, flag) &&
      str_util::ConsumePrefix(&arg, "=")) {
    char extra;
    int64_t parsed_int64;
    if (sscanf(arg.data(), "%ld%c", &parsed_int64, &extra) != 1) {
      LOG(ERROR) << "Couldn't interpret value " << arg << " for flag " << flag
                 << ".";
      *value_parsing_ok = false;
    } else {
      *value_parsing_ok = hook(parsed_int64);
    }
    return true;
  }

  return false;
}

bool ParseBoolFlag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                   const std::function<bool(bool)>& hook,
                   bool* value_parsing_ok) {
  *value_parsing_ok = true;
  if (str_util::ConsumePrefix(&arg, "--") &&
      str_util::ConsumePrefix(&arg, flag)) {
    if (arg.empty()) {
      *value_parsing_ok = hook(true);
      return true;
    }

    if (arg == "=true") {
      *value_parsing_ok = hook(true);
      return true;
    } else if (arg == "=false") {
      *value_parsing_ok = hook(false);
      return true;
    } else {
      LOG(ERROR) << "Couldn't interpret value " << arg << " for flag " << flag
                 << ".";
      *value_parsing_ok = false;
      return true;
    }
  }

  return false;
}

bool ParseFloatFlag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                    const std::function<bool(float)>& hook,
                    bool* value_parsing_ok) {
  *value_parsing_ok = true;
  if (str_util::ConsumePrefix(&arg, "--") &&
      str_util::ConsumePrefix(&arg, flag) &&
      str_util::ConsumePrefix(&arg, "=")) {
    char extra;
    float parsed_float;
    if (sscanf(arg.data(), "%f%c", &parsed_float, &extra) != 1) {
      LOG(ERROR) << "Couldn't interpret value " << arg << " for flag " << flag
                 << ".";
      *value_parsing_ok = false;
    } else {
      *value_parsing_ok = hook(parsed_float);
    }
    return true;
  }

  return false;
}

}  // namespace

Flag::Flag(const char* name, tensorflow::int32* dst, const string& usage_text)
    : name_(name),
      type_(TYPE_INT32),
      int32_hook_([dst](int32 value) {
        *dst = value;
        return true;
      }),
      int32_default_for_display_(*dst),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, tensorflow::int64* dst, const string& usage_text)
    : name_(name),
      type_(TYPE_INT64),
      int64_hook_([dst](int64 value) {
        *dst = value;
        return true;
      }),
      int64_default_for_display_(*dst),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, float* dst, const string& usage_text)
    : name_(name),
      type_(TYPE_FLOAT),
      float_hook_([dst](float value) {
        *dst = value;
        return true;
      }),
      float_default_for_display_(*dst),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, bool* dst, const string& usage_text)
    : name_(name),
      type_(TYPE_BOOL),
      bool_hook_([dst](bool value) {
        *dst = value;
        return true;
      }),
      bool_default_for_display_(*dst),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, string* dst, const string& usage_text)
    : name_(name),
      type_(TYPE_STRING),
      string_hook_([dst](string value) {
        *dst = std::move(value);
        return true;
      }),
      string_default_for_display_(*dst),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, std::function<bool(int32)> int32_hook,
           int32 default_value_for_display, const string& usage_text)
    : name_(name),
      type_(TYPE_INT32),
      int32_hook_(std::move(int32_hook)),
      int32_default_for_display_(default_value_for_display),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, std::function<bool(int64)> int64_hook,
           int64 default_value_for_display, const string& usage_text)
    : name_(name),
      type_(TYPE_INT64),
      int64_hook_(std::move(int64_hook)),
      int64_default_for_display_(default_value_for_display),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, std::function<bool(float)> float_hook,
           float default_value_for_display, const string& usage_text)
    : name_(name),
      type_(TYPE_FLOAT),
      float_hook_(std::move(float_hook)),
      float_default_for_display_(default_value_for_display),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, std::function<bool(bool)> bool_hook,
           bool default_value_for_display, const string& usage_text)
    : name_(name),
      type_(TYPE_BOOL),
      bool_hook_(std::move(bool_hook)),
      bool_default_for_display_(default_value_for_display),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, std::function<bool(string)> string_hook,
           string default_value_for_display, const string& usage_text)
    : name_(name),
      type_(TYPE_STRING),
      string_hook_(std::move(string_hook)),
      string_default_for_display_(std::move(default_value_for_display)),
      usage_text_(usage_text) {}

bool Flag::Parse(string arg, bool* value_parsing_ok) const {
  bool result = false;
  if (type_ == TYPE_INT32) {
    result = ParseInt32Flag(arg, name_, int32_hook_, value_parsing_ok);
  } else if (type_ == TYPE_INT64) {
    result = ParseInt64Flag(arg, name_, int64_hook_, value_parsing_ok);
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
                             const std::vector<Flag>& flag_list) {
  bool result = true;
  std::vector<char*> unknown_flags;
  for (int i = 1; i < *argc; ++i) {
    if (string(argv[i]) == "--") {
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
  // Passthrough any extra flags.
  int dst = 1;  // Skip argv[0]
  for (char* f : unknown_flags) {
    argv[dst++] = f;
  }
  argv[dst++] = nullptr;
  *argc = unknown_flags.size() + 1;
  return result && (*argc < 2 || strcmp(argv[1], "--help") != 0);
}

/*static*/ string Flags::Usage(const string& cmdline,
                               const std::vector<Flag>& flag_list) {
  string usage_text;
  if (!flag_list.empty()) {
    strings::Appendf(&usage_text, "usage: %s\nFlags:\n", cmdline.c_str());
  } else {
    strings::Appendf(&usage_text, "usage: %s\n", cmdline.c_str());
  }
  for (const Flag& flag : flag_list) {
    const char* type_name = "";
    string flag_string;
    if (flag.type_ == Flag::TYPE_INT32) {
      type_name = "int32";
      flag_string = strings::Printf("--%s=%d", flag.name_.c_str(),
                                    flag.int32_default_for_display_);
    } else if (flag.type_ == Flag::TYPE_INT64) {
      type_name = "int64";
      flag_string = strings::Printf(
          "--%s=%lld", flag.name_.c_str(),
          static_cast<long long>(flag.int64_default_for_display_));
    } else if (flag.type_ == Flag::TYPE_BOOL) {
      type_name = "bool";
      flag_string =
          strings::Printf("--%s=%s", flag.name_.c_str(),
                          flag.bool_default_for_display_ ? "true" : "false");
    } else if (flag.type_ == Flag::TYPE_STRING) {
      type_name = "string";
      flag_string = strings::Printf("--%s=\"%s\"", flag.name_.c_str(),
                                    flag.string_default_for_display_.c_str());
    } else if (flag.type_ == Flag::TYPE_FLOAT) {
      type_name = "float";
      flag_string = strings::Printf("--%s=%f", flag.name_.c_str(),
                                    flag.float_default_for_display_);
    }
    strings::Appendf(&usage_text, "\t%-33s\t%s\t%s\n", flag_string.c_str(),
                     type_name, flag.usage_text_.c_str());
  }
  return usage_text;
}

}  // namespace tensorflow
