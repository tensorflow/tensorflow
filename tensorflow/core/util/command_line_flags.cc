/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/logging.h"

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

Flag::Flag(const char* name, tensorflow::int32* dst)
    : name_(name), type_(TYPE_INT), int_value_(dst) {}

Flag::Flag(const char* name, bool* dst)
    : name_(name), type_(TYPE_BOOL), bool_value_(dst) {}

Flag::Flag(const char* name, string* dst)
    : name_(name), type_(TYPE_STRING), string_value_(dst) {}

bool Flag::Parse(string arg, bool* value_parsing_ok) const {
  bool result = false;
  if (type_ == TYPE_INT) {
    result = ParseInt32Flag(arg, name_, int_value_, value_parsing_ok);
  } else if (type_ == TYPE_BOOL) {
    result = ParseBoolFlag(arg, name_, bool_value_, value_parsing_ok);
  } else if (type_ == TYPE_STRING) {
    result = ParseStringFlag(arg, name_, string_value_, value_parsing_ok);
  }
  return result;
}

bool ParseFlags(int* argc, char** argv, const std::vector<Flag>& flag_list) {
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
  return result;
}

}  // namespace tensorflow
