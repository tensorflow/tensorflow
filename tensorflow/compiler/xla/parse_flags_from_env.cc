/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// This module exports ParseFlagsFromEnvAndDieIfUnknown(), which allows other
// modules to parse flags from an environtment variable, or a file named by the
// environment variable.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/parse_flags_from_env.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {

static const char kWS[] = " \t\r\n";  // whitespace

// The following struct represents an argv[]-style array, parsed
// from data gleaned from the environment.
//
// As usual, an anonymous namespace is advisable to avoid
// constructor/destructor collisions with other "private" types
// in the same named namespace.
namespace {

// Functor which deletes objects by calling `free`.  Necessary to free strdup'ed
// strings created by AppendToEnvArgv.
struct FreeDeleter {
  void operator()(char* ptr) { free(ptr); }
};

struct EnvArgv {
  EnvArgv() : initialized(false), argc(0) {}
  bool initialized;         // whether the other fields have been set.
  int argc;                 // elements used in argv[]
  std::vector<char*> argv;  // flag arguments parsed from environment string.
  // saved values from argv[] to avoid leaks
  std::vector<std::unique_ptr<char, FreeDeleter>> argv_save;
};
}  // anonymous namespace

// Append the string s0[0, .., s0len-1] concatenated with s1[0, .., s1len-1] as
// a newly allocated nul-terminated string to the array *a.  If s0==nullptr, a
// nullptr is appended without increasing a->argc.
static void AppendToEnvArgv(const char* s0, size_t s0len, const char* s1,
                            size_t s1len, EnvArgv* a) {
  if (s0 == nullptr) {
    a->argv.push_back(nullptr);
    a->argv_save.push_back(nullptr);
  } else {
    string s = string(s0, s0len) + string(s1, s1len);
    char* str = strdup(s.c_str());
    a->argv.push_back(str);
    a->argv_save.emplace_back(str);
    a->argc++;
  }
}

// Like s.find_first_of(x, pos), but return s.size() when find_first_of() would
// return string::npos.  This avoids if-statements elsewhere.
static size_t FindFirstOf(const string& s, const char* x, size_t pos) {
  size_t result = s.find_first_of(x, pos);
  return result == string::npos ? s.size() : result;
}

// Like s.find_first_not_of(x, pos), but return s.size() when
// find_first_not_of() would return string::npos.  This avoids if-statements
// elsewhere.
static size_t FindFirstNotOf(const string& s, const char* x, size_t pos) {
  size_t result = s.find_first_not_of(x, pos);
  return result == string::npos ? s.size() : result;
}

// Given a string containing flags, parse them into the XLA command line flags.
// The parse is best effort, and gives up on the first syntax error.
static void ParseArgvFromString(const string& flag_str, EnvArgv* a) {
  size_t b = FindFirstNotOf(flag_str, kWS, 0);
  while (b != flag_str.size() && flag_str[b] == '-') {
    // b is the index of the start of a flag.
    // Set e to the index just past the end of the flag.
    size_t e = b;
    while (e != flag_str.size() && isascii(flag_str[e]) &&
           (strchr("-_", flag_str[e]) != nullptr ||
            absl::ascii_isalnum(flag_str[e]))) {
      e++;
    }
    if (e != flag_str.size() && flag_str[e] == '=' &&
        e + 1 != flag_str.size() && strchr("'\"", flag_str[e + 1]) != nullptr) {
      // A flag of the form  --flag="something in double or single quotes"
      int c;
      e++;  // point just past '='
      size_t eflag = e;
      char quote = flag_str[e];
      e++;  // point just past quote
      // Put in value the string with quotes removed.
      string value;
      for (; e != flag_str.size() && (c = flag_str[e]) != quote; e++) {
        if (quote == '"' && c == '\\' && e + 1 != flag_str.size()) {
          // Handle backslash in double quoted strings.  They are literal in
          // single-quoted strings.
          e++;
          c = flag_str[e];
        }
        value += c;
      }
      if (e != flag_str.size()) {  // skip final " or '
        e++;
      }
      AppendToEnvArgv(flag_str.data() + b, eflag - b, value.data(),
                      value.size(), a);
    } else {  // A flag without a quoted value.
      e = FindFirstOf(flag_str, kWS, e);
      AppendToEnvArgv(flag_str.data() + b, e - b, "", 0, a);
    }
    b = FindFirstNotOf(flag_str, kWS, e);
  }
}

// Call ParseArgvFromString(..., a) on a string derived from the setting of the
// environment variable `envvar`, or a file it points to.
static void SetArgvFromEnv(absl::string_view envvar, EnvArgv* a) {
  if (!a->initialized) {
    static const char kDummyArgv[] = "<argv[0]>";
    AppendToEnvArgv(kDummyArgv, strlen(kDummyArgv), nullptr, 0,
                    a);  // dummy argv[0]
    const char* env = getenv(string(envvar).c_str());
    if (env == nullptr || env[0] == '\0') {
      // nothing
    } else if (env[strspn(env, kWS)] == '-') {  // flags in env var value
      ParseArgvFromString(env, a);
    } else {  // assume it's a file name
      FILE* fp = fopen(env, "r");
      if (fp != nullptr) {
        string str;
        char buf[512];
        int n;
        while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
          str.append(buf, n);
        }
        fclose(fp);
        ParseArgvFromString(str, a);
      }
    }
    AppendToEnvArgv(nullptr, 0, nullptr, 0, a);  // add trailing nullptr to *a.
    a->initialized = true;
  }
}

// The simulated argv[] parsed from the environment, one for each different
// environment variable we've seen.
static std::unordered_map<string, EnvArgv>& EnvArgvs() {
  static auto* env_argvs = new std::unordered_map<string, EnvArgv>();
  return *env_argvs;
}

// Used to protect accesses to env_argvs.
static tensorflow::mutex env_argv_mu(tensorflow::LINKER_INITIALIZED);

bool ParseFlagsFromEnvAndDieIfUnknown(
    absl::string_view envvar, const std::vector<tensorflow::Flag>& flag_list) {
  tensorflow::mutex_lock lock(env_argv_mu);
  auto* env_argv = &EnvArgvs()[string(envvar)];
  SetArgvFromEnv(envvar, env_argv);  // a no-op if already initialized
  bool result =
      tensorflow::Flags::Parse(&env_argv->argc, &env_argv->argv[0], flag_list);

  // There's always at least one unparsed argc, namely the fake argv[0].
  if (result && env_argv->argc != 1) {
    // Skip the first argv, which is the fake argv[0].
    auto unknown_flags = absl::MakeSpan(env_argv->argv);
    unknown_flags.remove_prefix(1);

    // Some flags are set on XLA_FLAGS, others on TF_XLA_FLAGS.  If we find an
    // unrecognized flag, suggest the alternative.
    string alternate_envvar;
    if (envvar == "TF_XLA_FLAGS") {
      alternate_envvar = "XLA_FLAGS";
    } else if (envvar == "XLA_FLAGS") {
      alternate_envvar = "TF_XLA_FLAGS";
    }
    string did_you_mean;
    if (!alternate_envvar.empty()) {
      did_you_mean = absl::StrFormat(
          "\nPerhaps you meant to specify these on the %s envvar?",
          alternate_envvar);
    }

    LOG(FATAL) << "Unknown flag" << (unknown_flags.size() > 1 ? "s" : "")
               << " in " << envvar << ": " << absl::StrJoin(unknown_flags, " ")
               << did_you_mean;
    return false;
  }
  return result;
}

// Testing only.
//
// Resets the env_argv struct so that subsequent calls to
// ParseFlagsFromEnvAndDieIfUnknown() will parse the environment variable (or
// the file it points to) anew, and set *pargc, and *pargv to point to the
// internal locations of the argc and argv constructed from the environment.
void ResetFlagsFromEnvForTesting(absl::string_view envvar, int** pargc,
                                 std::vector<char*>** pargv) {
  tensorflow::mutex_lock lock(env_argv_mu);
  EnvArgvs().erase(string(envvar));
  auto& env_argv = EnvArgvs()[string(envvar)];
  *pargc = &env_argv.argc;
  *pargv = &env_argv.argv;
}

}  // namespace xla
