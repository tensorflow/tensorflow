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
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace {

bool ParseStringFlag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                     string* dst, bool* value_parsing_ok) {
  *value_parsing_ok = true;
  if (arg.Consume("--") && arg.Consume(flag) && arg.Consume("=")) {
    *dst = arg.ToString();
    return true;
  }

  return false;
}

bool ParseInt32Flag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                    tensorflow::int32* dst, bool* value_parsing_ok) {
  *value_parsing_ok = true;
  if (arg.Consume("--") && arg.Consume(flag) && arg.Consume("=")) {
    char extra;
    if (sscanf(arg.data(), "%d%c", dst, &extra) != 1) {
      LOG(ERROR) << "Couldn't interpret value " << arg << " for flag " << flag
                 << ".";
      *value_parsing_ok = false;
    }
    return true;
  }

  return false;
}

bool ParseInt64Flag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                    tensorflow::int64* dst, bool* value_parsing_ok) {
  *value_parsing_ok = true;
  if (arg.Consume("--") && arg.Consume(flag) && arg.Consume("=")) {
    char extra;
    if (sscanf(arg.data(), "%lld%c", dst, &extra) != 1) {
      LOG(ERROR) << "Couldn't interpret value " << arg << " for flag " << flag
                 << ".";
      *value_parsing_ok = false;
    }
    return true;
  }

  return false;
}

bool ParseBoolFlag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
                   bool* dst, bool* value_parsing_ok) {
  *value_parsing_ok = true;
  if (arg.Consume("--") && arg.Consume(flag)) {
    if (arg.empty()) {
      *dst = true;
      return true;
    }

    if (arg == "=true") {
      *dst = true;
      return true;
    } else if (arg == "=false") {
      *dst = false;
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

}  // namespace

Flag::Flag(const char* name, tensorflow::int32* dst, const string& usage_text)
    : name_(name), type_(TYPE_INT), int_value_(dst), usage_text_(usage_text) {}

Flag::Flag(const char* name, tensorflow::int64* dst, const string& usage_text)
    : name_(name),
      type_(TYPE_INT64),
      int64_value_(dst),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, bool* dst, const string& usage_text)
    : name_(name),
      type_(TYPE_BOOL),
      bool_value_(dst),
      usage_text_(usage_text) {}

Flag::Flag(const char* name, string* dst, const string& usage_text)
    : name_(name),
      type_(TYPE_STRING),
      string_value_(dst),
      usage_text_(usage_text) {}

bool Flag::Parse(string arg, bool* value_parsing_ok) const {
  bool result = false;
  if (type_ == TYPE_INT) {
    result = ParseInt32Flag(arg, name_, int_value_, value_parsing_ok);
  } else if (type_ == TYPE_INT64) {
    result = ParseInt64Flag(arg, name_, int64_value_, value_parsing_ok);
  } else if (type_ == TYPE_BOOL) {
    result = ParseBoolFlag(arg, name_, bool_value_, value_parsing_ok);
  } else if (type_ == TYPE_STRING) {
    result = ParseStringFlag(arg, name_, string_value_, value_parsing_ok);
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
    if (flag.type_ == Flag::TYPE_INT) {
      type_name = "int32";
      flag_string =
          strings::Printf("--%s=%d", flag.name_.c_str(), *flag.int_value_);
    } else if (flag.type_ == Flag::TYPE_INT64) {
      type_name = "int64";
      flag_string = strings::Printf("--%s=%lld", flag.name_.c_str(),
                                    static_cast<long long>(*flag.int64_value_));
    } else if (flag.type_ == Flag::TYPE_BOOL) {
      type_name = "bool";
      flag_string = strings::Printf("--%s=%s", flag.name_.c_str(),
                                    *flag.bool_value_ ? "true" : "false");
    } else if (flag.type_ == Flag::TYPE_STRING) {
      type_name = "string";
      flag_string = strings::Printf("--%s=\"%s\"", flag.name_.c_str(),
                                    flag.string_value_->c_str());
    }
    strings::Appendf(&usage_text, "\t%-33s\t%s\t%s\n", flag_string.c_str(),
                     type_name, flag.usage_text_.c_str());
  }
  return usage_text;
}

}  // namespace tensorflow
