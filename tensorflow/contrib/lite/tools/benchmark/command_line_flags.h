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

#ifndef TENSORFLOW_CONTRIB_LITE_TOOLS_BENCHMARK_COMMAND_LINE_FLAGS_H_
#define TENSORFLOW_CONTRIB_LITE_TOOLS_BENCHMARK_COMMAND_LINE_FLAGS_H_

#include <functional>
#include <string>
#include <vector>

namespace tflite {
// A simple command-line argument parsing module.
// Dependency free simplified port of core/util/command_line_flags.
// This class is written for benchmarks and uses inefficient string
// concatenation. This was written to avoid dependency on tensorflow/core/util
// which transitively brings in a lot of other dependencies that are not
// necessary for tflite benchmarking code.
// The recommended way of using it is with local variables and an initializer
// list of Flag objects, for example:
//
// int some_int = 10;
// bool some_switch = false;
// std::string some_name = "something";
//
// std::vector<tensorFlow::Flag> flag_list = {
//   Flag::CreateFlag("some_int", &some_int, "an integer that affects X"),
//   Flag::CreateFlag("some_switch", &some_switch, "a bool that affects Y"),
//   Flag::CreateFlag("some_name", &some_name, "a string that affects Z")
// };
// // Get usage message before ParseFlags() to capture default values.
// std::string usage = Flag::Usage(argv[0], flag_list);
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
  template <typename T>
  static Flag CreateFlag(const char* name, T* val, const char* usage) {
    return Flag(name, [val](const T& v) { *val = v; }, *val, usage);
  }

  Flag(const char* name, const std::function<void(const int32_t&)>& hook,
       int32_t default_value, const std::string& usage_text);
  Flag(const char* name, const std::function<void(const int64_t&)>& hook,
       int64_t default_value, const std::string& usage_text);
  Flag(const char* name, const std::function<void(const float&)>& hook,
       float default_value, const std::string& usage_text);
  Flag(const char* name, const std::function<void(const bool&)>& hook,
       bool default_value, const std::string& usage_text);
  Flag(const char* name, const std::function<void(const std::string&)>& hook,
       const std::string& default_value, const std::string& usage_text);

 private:
  friend class Flags;

  bool Parse(const std::string& arg, bool* value_parsing_ok) const;

  std::string name_;
  enum {
    TYPE_INT32,
    TYPE_INT64,
    TYPE_BOOL,
    TYPE_STRING,
    TYPE_FLOAT,
  } type_;

  std::string GetTypeName() const;

  std::function<bool(const std::string&)> value_hook_;
  std::string default_for_display_;

  std::string usage_text_;
};

class Flags {
 public:
  // Parse the command line represented by argv[0, ..., (*argc)-1] to find flag
  // instances matching flags in flaglist[].  Update the variables associated
  // with matching flags, and remove the matching arguments from (*argc, argv).
  // Return true iff all recognized flag values were parsed correctly, and the
  // first remaining argument is not "--help".
  static bool Parse(int* argc, const char** argv,
                    const std::vector<Flag>& flag_list);

  // Return a usage message with command line cmdline, and the
  // usage_text strings in flag_list[].
  static std::string Usage(const std::string& cmdline,
                           const std::vector<Flag>& flag_list);
};

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_TOOLS_BENCHMARK_COMMAND_LINE_FLAGS_H_
