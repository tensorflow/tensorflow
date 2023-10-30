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

#include "tsl/platform/default/logging.h"

#include <stdlib.h>
#include <string.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/base/no_destructor.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace tsl {
void UpdateLogVerbosityIfDefined(const char* env_var) {}
}  // namespace tsl

namespace {

// Returns a mapping from module name to VLOG level, derived from the
// TF_CPP_VMODULE environment variable; ownership is transferred to the caller.
std::unordered_map<std::string, int> VmodulesMapFromEnv() {
  std::unordered_map<std::string, int> result;

  const char* env = getenv("TF_CPP_VMODULE");
  if (env == nullptr) {
    return result;
  }

  // The value of the env var is supposed to be of the form:
  //    "foo=1,bar=2,baz=3"
  for (absl::string_view s : absl::StrSplit(env, ',', absl::SkipEmpty())) {
    std::vector<absl::string_view> kv = absl::StrSplit(s, '=');
    int level;
    if (kv.size() < 2 || !absl::SimpleAtoi(kv[1], &level)) continue;
    result[std::string(kv[0])] = level;
  }
  return result;
}

void SetVmoduleFromEnv() {
  std::unordered_map<std::string, int> vmodules = VmodulesMapFromEnv();
  for (const auto& [filename, level] : vmodules) {
    absl::SetVLogLevel(filename, level);
  }
}

// Parse log level (int64) from environment variable (char*)
int64_t LogLevelFromEnv(const char* tf_env_var_val) {
  int64_t level;
  if (tf_env_var_val == nullptr || !absl::SimpleAtoi(tf_env_var_val, &level)) {
    return 0;
  }
  return level >= 0 ? level : -level;
}

class VlogFileSink : public absl::LogSink {
 public:
  explicit VlogFileSink(FILE* fp) : fp_(fp) { absl::AddLogSink(this); }

  // Neither copyable nor movable.
  VlogFileSink(const VlogFileSink&) = delete;
  VlogFileSink& operator=(const VlogFileSink&) = delete;

  void Send(const absl::LogEntry& entry) final {
    fprintf(fp_, "%s", entry.text_message_with_prefix_and_newline_c_str());
  }

  void Flush() final {
    fflush(stderr);
    fflush(fp_);
  }

  ~VlogFileSink() {
    absl::RemoveLogSink(this);
    fclose(fp_);
  }

 private:
  FILE* fp_;
};

// If the environment variable TF_CPP_VLOG_FILENAME is set, all LOG
// and VLOG calls are redirected from stderr to the given file.
//
// Note: the name of the environment variable is confusing, because both LOG and
// VLOG (i.e. not just VLOG) messages are redirected. We have decided to keep
// the confusing name in order not to break existing users.
void SetLogFilenameFromEnv() {
  const char* env = getenv("TF_CPP_VLOG_FILENAME");
  if (env == nullptr) return;
  FILE* fp = fopen(env, "w");
  if (fp == nullptr) return;

  static const absl::NoDestructor<VlogFileSink> file_sink(fp);
  // Now that the file sink is registered, stop logging to stderr.
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfinity);
}

static int init = []() {
  absl::InitializeLog();
  absl::SetMinLogLevel(static_cast<absl::LogSeverityAtLeast>(
      LogLevelFromEnv(getenv("TF_CPP_MIN_LOG_LEVEL"))));
  absl::SetGlobalVLogLevel(LogLevelFromEnv(getenv("TF_CPP_MAX_VLOG_LEVEL")));
  SetLogFilenameFromEnv();
  SetVmoduleFromEnv();
  return 0;
}();  // call the lambda.

}  // namespace
