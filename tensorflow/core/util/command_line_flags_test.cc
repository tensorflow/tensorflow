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

#include <ctype.h>
#include <vector>

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace {
// The returned array is only valid for the lifetime of the input vector.
// We're using const casting because we need to pass in an argv-style array of
// char* pointers for the API, even though we know they won't be altered.
std::vector<char *> CharPointerVectorFromStrings(
    const std::vector<string> &strings) {
  std::vector<char *> result;
  for (const string &string : strings) {
    result.push_back(const_cast<char *>(string.c_str()));
  }
  return result;
}
}

TEST(CommandLineFlagsTest, BasicUsage) {
  int some_int = 10;
  int64 some_int64 = 21474836470;  // max int32 is 2147483647
  bool some_switch = false;
  string some_name = "something";
  int argc = 5;
  std::vector<string> argv_strings = {
      "program_name", "--some_int=20", "--some_int64=214748364700",
      "--some_switch", "--some_name=somethingelse"};
  std::vector<char *> argv_array = CharPointerVectorFromStrings(argv_strings);
  bool parsed_ok =
      Flags::Parse(&argc, argv_array.data(),
                   {Flag("some_int", &some_int, "some int"),
                    Flag("some_int64", &some_int64, "some int64"),
                    Flag("some_switch", &some_switch, "some switch"),
                    Flag("some_name", &some_name, "some name")});
  EXPECT_EQ(true, parsed_ok);
  EXPECT_EQ(20, some_int);
  EXPECT_EQ(214748364700, some_int64);
  EXPECT_EQ(true, some_switch);
  EXPECT_EQ("somethingelse", some_name);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, BadIntValue) {
  int some_int = 10;
  int argc = 2;
  std::vector<string> argv_strings = {"program_name", "--some_int=notanumber"};
  std::vector<char *> argv_array = CharPointerVectorFromStrings(argv_strings);
  bool parsed_ok = Flags::Parse(&argc, argv_array.data(),
                                {Flag("some_int", &some_int, "some int")});

  EXPECT_EQ(false, parsed_ok);
  EXPECT_EQ(10, some_int);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, BadBoolValue) {
  bool some_switch = false;
  int argc = 2;
  std::vector<string> argv_strings = {"program_name", "--some_switch=notabool"};
  std::vector<char *> argv_array = CharPointerVectorFromStrings(argv_strings);
  bool parsed_ok =
      Flags::Parse(&argc, argv_array.data(),
                   {Flag("some_switch", &some_switch, "some switch")});

  EXPECT_EQ(false, parsed_ok);
  EXPECT_EQ(false, some_switch);
  EXPECT_EQ(argc, 1);
}

// Return whether str==pat, but allowing any whitespace in pat
// to match zero or more whitespace characters in str.
static bool MatchWithAnyWhitespace(const string &str, const string &pat) {
  bool matching = true;
  int pat_i = 0;
  for (int str_i = 0; str_i != str.size() && matching; str_i++) {
    if (isspace(str[str_i])) {
      matching = (pat_i != pat.size() && isspace(pat[pat_i]));
    } else {
      while (pat_i != pat.size() && isspace(pat[pat_i])) {
        pat_i++;
      }
      matching = (pat_i != pat.size() && str[str_i] == pat[pat_i++]);
    }
  }
  while (pat_i != pat.size() && isspace(pat[pat_i])) {
    pat_i++;
  }
  return (matching && pat_i == pat.size());
}

TEST(CommandLineFlagsTest, UsageString) {
  int some_int = 10;
  int64 some_int64 = 21474836470;  // max int32 is 2147483647
  bool some_switch = false;
  string some_name = "something";
  const string tool_name = "some_tool_name";
  string usage = Flags::Usage(tool_name + "<flags>",
                              {Flag("some_int", &some_int, "some int"),
                               Flag("some_int64", &some_int64, "some int64"),
                               Flag("some_switch", &some_switch, "some switch"),
                               Flag("some_name", &some_name, "some name")});
  // Match the usage message, being sloppy about whitespace.
  const char *expected_usage =
      " usage: some_tool_name <flags>\n"
      "Flags:\n"
      "--some_int=10 int32 some int\n"
      "--some_int64=21474836470 int64 some int64\n"
      "--some_switch=false bool some switch\n"
      "--some_name=\"something\" string some name\n";
  ASSERT_EQ(MatchWithAnyWhitespace(usage, expected_usage), true);

  // Again but with no flags.
  usage = Flags::Usage(tool_name, {});
  ASSERT_EQ(MatchWithAnyWhitespace(usage, " usage: some_tool_name\n"), true);
}
}  // namespace tensorflow
