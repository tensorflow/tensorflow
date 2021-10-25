/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "ral/ral_logging.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <iterator>
#include <string>
#include <unordered_map>

namespace mlir {
namespace disc_ral {

namespace internal {

int ParseInteger(const char* str, size_t size) {
  // Ideally we would use env_var / safe_strto64, but it is
  // hard to use here without pulling in a lot of dependencies,
  // so we use std:istringstream instead
  std::string integer_str(str, size);
  std::istringstream ss(integer_str);
  int level = 0;
  ss >> level;
  return level;
}

// Parse log level (int) from environment variable (char*)
int LogLevelStrToInt(const char* disc_env_var_val) {
  if (disc_env_var_val == nullptr) {
    return 0;
  }
  return ParseInteger(disc_env_var_val, strlen(disc_env_var_val));
}

// Using StringPiece breaks Windows build.
struct StringData {
  struct Hasher {
    size_t operator()(const StringData& sdata) const {
      // For dependency reasons, we cannot use hash.h here. Use DBJHash instead.
      size_t hash = 5381;
      const char* data = sdata.data;
      for (const char* top = data + sdata.size; data < top; ++data) {
        hash = ((hash << 5) + hash) + (*data);
      }
      return hash;
    }
  };

  StringData() = default;
  StringData(const char* data, size_t size) : data(data), size(size) {}

  bool operator==(const StringData& rhs) const {
    return size == rhs.size && memcmp(data, rhs.data, size) == 0;
  }

  const char* data = nullptr;
  size_t size = 0;
};

using VmoduleMap = std::unordered_map<StringData, int, StringData::Hasher>;

// Returns a mapping from module name to VLOG level, derived from the
// DISC_CPP_VMODULE environment variable; ownership is transferred to the
// caller.
VmoduleMap* VmodulesMapFromEnv() {
  // The value of the env var is supposed to be of the form:
  //    "foo=1,bar=2,baz=3"
  const char* env = getenv("DISC_CPP_VMODULE");
  if (env == nullptr) {
    // If there is no DISC_CPP_VMODULE configuration (most common case), return
    // nullptr so that the ShouldVlogModule() API can fast bail out of it.
    return nullptr;
  }
  // The memory returned by getenv() can be invalidated by following getenv() or
  // setenv() calls. And since we keep references to it in the VmoduleMap in
  // form of StringData objects, make a copy of it.
  const char* env_data = strdup(env);
  VmoduleMap* result = new VmoduleMap();
  while (true) {
    const char* eq = strchr(env_data, '=');
    if (eq == nullptr) {
      break;
    }
    const char* after_eq = eq + 1;

    // Comma either points at the next comma delimiter, or at a null terminator.
    // We check that the integer we parse ends at this delimiter.
    const char* comma = strchr(after_eq, ',');
    const char* new_env_data;
    if (comma == nullptr) {
      comma = strchr(after_eq, '\0');
      new_env_data = comma;
    } else {
      new_env_data = comma + 1;
    }
    (*result)[StringData(env_data, eq - env_data)] =
        ParseInteger(after_eq, comma - after_eq);
    env_data = new_env_data;
  }
  return result;
}

int MinLogLevelFromEnv() {
  const char* disc_env_var_val = getenv("DISC_CPP_MIN_LOG_LEVEL");
  return LogLevelStrToInt(disc_env_var_val);
}

int MinVLogLevelFromEnv() {
  const char* disc_env_var_val = getenv("DISC_CPP_MIN_VLOG_LEVEL");
  return LogLevelStrToInt(disc_env_var_val);
}

LogMessage::LogMessage(const char* fname, int line, int severity)
    : fname_(fname), line_(line), severity_(severity) {}

LogMessage& LogMessage::AtLocation(const char* fname, int line) {
  fname_ = fname;
  line_ = line;
  return *this;
}

LogMessage::~LogMessage() {
  // Read the min log level once during the first call to logging.
  static int min_log_level = MinLogLevelFromEnv();
  if (severity_ >= min_log_level) {
    GenerateLogMessage();
  }
}

void LogMessage::GenerateLogMessage() {
  time_t result = time(nullptr);
  const size_t time_buffer_size = 30;
  char time_buffer[time_buffer_size];
  strftime(time_buffer, time_buffer_size, "%Y-%m-%d %H:%M:%S",
           localtime(&result));
  fprintf(stderr, "%s: %c %s:%d] %s\n", time_buffer, "IWEF"[severity_], fname_,
          line_, str().c_str());
}

std::string LogMessage::GetFilterStringForTesting(int min_log_level) {
  return (severity_ >= min_log_level) ? str() : "";
}

int LogMessage::MinVLogLevel() {
  static int min_vlog_level = MinVLogLevelFromEnv();
  return min_vlog_level;
}

bool LogMessage::VmoduleActivated(const char* fname, int level) {
  if (level <= MinVLogLevel()) {
    return true;
  }
  static VmoduleMap* vmodules = VmodulesMapFromEnv();
  if (DISC_PREDICT_TRUE(vmodules == nullptr)) {
    return false;
  }
  const char* last_slash = strrchr(fname, '/');
  const char* module_start = last_slash == nullptr ? fname : last_slash + 1;
  const char* dot_after = strchr(module_start, '.');
  const char* module_limit =
      dot_after == nullptr ? strchr(fname, '\0') : dot_after;
  StringData module(module_start, module_limit - module_start);
  auto it = vmodules->find(module);
  return it != vmodules->end() && it->second >= level;
}

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, FATAL) {}

LogMessageFatal::~LogMessageFatal() {
  // abort() ensures we don't return (we promised we would not via
  // ATTRIBUTE_NORETURN).
  GenerateLogMessage();
  abort();
}

}  // namespace internal
}  // namespace disc_ral
}  // namespace mlir
