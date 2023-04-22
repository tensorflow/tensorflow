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

// Test for the platform_strings.h header file.

#include "tensorflow/core/platform/platform_strings.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/str_util.h"

// Embed the platform strings in this binary.
TF_PLATFORM_STRINGS()

// A vector of strings.
typedef std::vector<std::string> string_vec;

// Append to *found the strings within the named file with the platform_strings
// magic prefix, and return true; or return false on error.

// Print the platform strings embedded in the binary file_name and return 0,
// or on error return 2.
static int PrintStrings(const std::string file_name) {
  int rc = 0;
  string_vec str;
  if (!tensorflow::GetPlatformStrings(file_name, &str)) {
    for (int i = 0; i != str.size(); i++) {
      printf("%s\n", str[i].c_str());
    }
  } else {
    perror(file_name.c_str());
    rc = 2;
  }
  return rc;
}

// Return whether str[] contains a string with prefix "macro_name="; if so,
// set *pvalue to the suffix.
static bool GetValue(const string_vec &str, const std::string &macro_name,
                     std::string *pvalue) {
  std::string nam_eq = macro_name + "=";
  int i = 0;
  while (i != str.size() && !absl::StartsWith(str[i], nam_eq)) {
    i++;
  }
  bool found = (i != str.size());
  if (found) {
    *pvalue = str[i].substr(nam_eq.size());
  }
  return found;
}

// If macro_name[] is not equal to value[], check that str[] contains the
// string "macro_name=value".  Otherwise, check that str[] does not contain any
// string starting with macro_name=".
static void CheckStr(const string_vec &str, const std::string &macro_name,
                     const std::string &value) {
  std::string value_from_str;
  if (GetValue(str, macro_name, &value_from_str)) {
    if (value != value_from_str) {
      // Output everything found, to aid debugging.
      LOG(ERROR) << "===== value=" << value
                 << "  value_from_str=" << value_from_str;
      for (int i = 0; i != str.size(); i++) {
        LOG(ERROR) << "% " << str[i];
      }
      LOG(ERROR) << "=====";
    }
    CHECK_EQ(value, value_from_str) << " " << macro_name << ": bad value";
  } else {
    // If the string is not found, we expect value to be macro_name.
    if (value != macro_name) {
      // Output everything found, to aid debugging.
      LOG(ERROR) << "===== value=" << value << "  macro_name=" << macro_name;
      for (int i = 0; i != str.size(); i++) {
        LOG(ERROR) << "% " << str[i];
      }
      LOG(ERROR) << "=====";
    }
    CHECK_EQ(value, macro_name) << " " << macro_name << ": not found in binary";
  }
}

// Helper for AS_STR(), below, to perform macro expansion.
#define AS_STR_1_(x) #x

// Yield x after macro expansion as a nul-terminated constant string.
#define AS_STR(x) AS_STR_1_(x)

// Run the test, and return 0 on success, 2 otherwise.
static int RunTest(const std::string &binary_name) {
  int rc = 0;
  string_vec str;

  if (!tensorflow::GetPlatformStrings(binary_name, &str)) {
    CheckStr(str, "__linux__", AS_STR(__linux__));
    CheckStr(str, "_WIN32", AS_STR(_WIN32));
    CheckStr(str, "__APPLE__", AS_STR(__APPLE__));
    CheckStr(str, "__x86_64__", AS_STR(__x86_64__));
    CheckStr(str, "__aarch64__", AS_STR(__aarch64__));
    CheckStr(str, "__powerpc64__", AS_STR(__powerpc64__));
    CheckStr(str, "TF_PLAT_STR_VERSION", TF_PLAT_STR_VERSION_);
  } else {
    perror(binary_name.c_str());
    rc = 2;
  }

  return rc;
}

int main(int argc, char *argv[]) {
  tensorflow::Env *env = tensorflow::Env::Default();
  static const char usage[] = "usage: platform_strings_test [file...]";
  int rc = 0;
  tensorflow::port::InitMain(usage, &argc, &argv);
  if (argc == 1) {
    printf("rc=%d\n", PrintStrings(env->GetExecutablePath()));
    rc = RunTest(env->GetExecutablePath());
  } else {
    for (int argn = 1; argn != argc; argn++) {
      rc |= PrintStrings(argv[argn]);
    }
  }
  return rc;
}
