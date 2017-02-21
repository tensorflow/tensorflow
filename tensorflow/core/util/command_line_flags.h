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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_UTIL_COMMAND_LINE_FLAGS_H
#define THIRD_PARTY_TENSORFLOW_CORE_UTIL_COMMAND_LINE_FLAGS_H

#include <string>
#include <vector>
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// N.B. This library is for INTERNAL use only.
//
// This is a simple command-line argument parsing module to help us handle
// parameters for C++ binaries. The recommended way of using it is with local
// variables and an initializer list of Flag objects, for example:
//
// int some_int = 10;
// bool some_switch = false;
// string some_name = "something";
// std::vector<tensorFlow::Flag> flag_list = {
//   Flag("some_int", &some_int, "an integer that affects X"),
//   Flag("some_switch", &some_switch, "a bool that affects Y"),
//   Flag("some_name", &some_name, "a string that affects Z")
// };
// // Get usage message before ParseFlags() to capture default values.
// string usage = Flag::Usage(argv[0], flag_list);
// bool parsed_values_ok = Flags::Parse(&argc, argv, flag_list);
//
// tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
// if (argc != 1 || !parsed_values_ok) {
//    ...output usage and error message...
// }
//
// The argc and argv values are adjusted by the Parse function so all that
// remains is the program name (at argv[0]) and any unknown arguments fill the
// rest of the array. This means you can check for flags that weren't understood
// by seeing if argv is greater than 1.
// The result indicates if there were any errors parsing the values that were
// passed to the command-line switches. For example, --some_int=foo would return
// false because the argument is expected to be an integer.
//
// NOTE: Unlike gflags-style libraries, this library is intended to be
// used in the `main()` function of your binary. It does not handle
// flag definitions that are scattered around the source code.

// A description of a single command line flag, holding its name, type, usage
// text, and a pointer to the corresponding variable.
class Flag {
 public:
  Flag(const char* name, int32* dst1, const string& usage_text);
  Flag(const char* name, int64* dst1, const string& usage_text);
  Flag(const char* name, bool* dst, const string& usage_text);
  Flag(const char* name, string* dst, const string& usage_text);

 private:
  friend class Flags;

  bool Parse(string arg, bool* value_parsing_ok) const;

  string name_;
  enum { TYPE_INT, TYPE_INT64, TYPE_BOOL, TYPE_STRING } type_;
  int* int_value_;
  int64* int64_value_;
  bool* bool_value_;
  string* string_value_;
  string usage_text_;
};

class Flags {
 public:
  // Parse the command line represented by argv[0, ..., (*argc)-1] to find flag
  // instances matching flags in flaglist[].  Update the variables associated
  // with matching flags, and remove the matching arguments from (*argc, argv).
  // Return true iff all recognized flag values were parsed correctly, and the
  // first remaining argument is not "--help".
  static bool Parse(int* argc, char** argv, const std::vector<Flag>& flag_list);

  // Return a usage message with command line cmdline, and the
  // usage_text strings in flag_list[].
  static string Usage(const string& cmdline,
                      const std::vector<Flag>& flag_list);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_UTIL_COMMAND_LINE_FLAGS_H
