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
  result.reserve(strings.size());
  for (const string &string : strings) {
    result.push_back(const_cast<char *>(string.c_str()));
  }
  return result;
}
}  // namespace

TEST(CommandLineFlagsTest, BasicUsage) {
  int some_int32_set_directly = 10;
  int some_int32_set_via_hook = 20;
  int64_t some_int64_set_directly = 21474836470;  // max int32 is 2147483647
  int64_t some_int64_set_via_hook = 21474836479;  // max int32 is 2147483647
  bool some_switch_set_directly = false;
  bool some_switch_set_via_hook = true;
  bool some_switch_set_capitalized = false;
  bool some_switch_set_by_number = false;
  string some_name_set_directly = "something_a";
  string some_name_set_via_hook = "something_b";
  float some_float_set_directly = -23.23f;
  float some_float_set_via_hook = -25.23f;
  std::vector<string> argv_strings = {"program_name",
                                      "--some_int32_set_directly=20",
                                      "--some_int32_set_via_hook=50",
                                      "--some_int64_set_directly=214748364700",
                                      "--some_int64_set_via_hook=214748364710",
                                      "--some_switch_set_directly",
                                      "--some_switch_set_via_hook=false",
                                      "--some_switch_set_capitalized=True",
                                      "--some_switch_set_by_number=1",
                                      "--some_name_set_directly=somethingelse",
                                      "--some_name_set_via_hook=anythingelse",
                                      "--some_float_set_directly=42.0",
                                      "--some_float_set_via_hook=43.0"};
  int argc = argv_strings.size();
  std::vector<char *> argv_array = CharPointerVectorFromStrings(argv_strings);
  bool parsed_ok = Flags::Parse(
      &argc, argv_array.data(),
      {
          Flag("some_int32_set_directly", &some_int32_set_directly,
               "some int32 set directly"),
          Flag(
              "some_int32_set_via_hook",
              [&](int32_t value) {
                some_int32_set_via_hook = value;
                return true;
              },
              some_int32_set_via_hook, "some int32 set via hook"),
          Flag("some_int64_set_directly", &some_int64_set_directly,
               "some int64 set directly"),
          Flag(
              "some_int64_set_via_hook",
              [&](int64_t value) {
                some_int64_set_via_hook = value;
                return true;
              },
              some_int64_set_via_hook, "some int64 set via hook"),
          Flag("some_switch_set_directly", &some_switch_set_directly,
               "some switch set directly"),
          Flag(
              "some_switch_set_via_hook",
              [&](bool value) {
                some_switch_set_via_hook = value;
                return true;
              },
              some_switch_set_via_hook, "some switch set via hook"),
          Flag("some_switch_set_capitalized", &some_switch_set_capitalized,
               "some switch set capitalized"),
          Flag("some_switch_set_by_number", &some_switch_set_by_number,
               "some switch set by number"),
          Flag("some_name_set_directly", &some_name_set_directly,
               "some name set directly"),
          Flag(
              "some_name_set_via_hook",
              [&](string value) {
                some_name_set_via_hook = std::move(value);
                return true;
              },
              some_name_set_via_hook, "some name set via hook"),
          Flag("some_float_set_directly", &some_float_set_directly,
               "some float set directly"),
          Flag(
              "some_float_set_via_hook",
              [&](float value) {
                some_float_set_via_hook = value;
                return true;
              },
              some_float_set_via_hook, "some float set via hook"),
      });

  EXPECT_EQ(true, parsed_ok);
  EXPECT_EQ(20, some_int32_set_directly);
  EXPECT_EQ(50, some_int32_set_via_hook);
  EXPECT_EQ(214748364700, some_int64_set_directly);
  EXPECT_EQ(214748364710, some_int64_set_via_hook);
  EXPECT_EQ(true, some_switch_set_directly);
  EXPECT_EQ(false, some_switch_set_via_hook);
  EXPECT_EQ(true, some_switch_set_capitalized);
  EXPECT_EQ(true, some_switch_set_by_number);
  EXPECT_EQ("somethingelse", some_name_set_directly);
  EXPECT_EQ("anythingelse", some_name_set_via_hook);
  EXPECT_NEAR(42.0f, some_float_set_directly, 1e-5f);
  EXPECT_NEAR(43.0f, some_float_set_via_hook, 1e-5f);
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

TEST(CommandLineFlagsTest, BadFloatValue) {
  float some_float = -23.23f;
  int argc = 2;
  std::vector<string> argv_strings = {"program_name",
                                      "--some_float=notanumber"};
  std::vector<char *> argv_array = CharPointerVectorFromStrings(argv_strings);
  bool parsed_ok =
      Flags::Parse(&argc, argv_array.data(),
                   {Flag("some_float", &some_float, "some float")});

  EXPECT_EQ(false, parsed_ok);
  EXPECT_NEAR(-23.23f, some_float, 1e-5f);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, FailedInt32Hook) {
  int argc = 2;
  std::vector<string> argv_strings = {"program_name", "--some_int32=200"};
  std::vector<char *> argv_array = CharPointerVectorFromStrings(argv_strings);
  bool parsed_ok =
      Flags::Parse(&argc, argv_array.data(),
                   {Flag(
                       "some_int32", [](int32_t value) { return false; }, 30,
                       "some int32")});

  EXPECT_EQ(false, parsed_ok);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, FailedInt64Hook) {
  int argc = 2;
  std::vector<string> argv_strings = {"program_name", "--some_int64=200"};
  std::vector<char *> argv_array = CharPointerVectorFromStrings(argv_strings);
  bool parsed_ok =
      Flags::Parse(&argc, argv_array.data(),
                   {Flag(
                       "some_int64", [](int64_t value) { return false; }, 30,
                       "some int64")});

  EXPECT_EQ(false, parsed_ok);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, FailedFloatHook) {
  int argc = 2;
  std::vector<string> argv_strings = {"program_name", "--some_float=200.0"};
  std::vector<char *> argv_array = CharPointerVectorFromStrings(argv_strings);
  bool parsed_ok =
      Flags::Parse(&argc, argv_array.data(),
                   {Flag("some_float", [](float value) { return false; }, 30.0f,
                         "some float")});

  EXPECT_EQ(false, parsed_ok);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, FailedBoolHook) {
  int argc = 2;
  std::vector<string> argv_strings = {"program_name", "--some_switch=true"};
  std::vector<char *> argv_array = CharPointerVectorFromStrings(argv_strings);
  bool parsed_ok =
      Flags::Parse(&argc, argv_array.data(),
                   {Flag("some_switch", [](bool value) { return false; }, false,
                         "some switch")});

  EXPECT_EQ(false, parsed_ok);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, FailedStringHook) {
  int argc = 2;
  std::vector<string> argv_strings = {"program_name", "--some_name=true"};
  std::vector<char *> argv_array = CharPointerVectorFromStrings(argv_strings);
  bool parsed_ok = Flags::Parse(
      &argc, argv_array.data(),
      {Flag("some_name", [](string value) { return false; }, "", "some name")});

  EXPECT_EQ(false, parsed_ok);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, RepeatedStringHook) {
  int argc = 3;
  std::vector<string> argv_strings = {"program_name", "--some_name=this",
                                      "--some_name=that"};
  std::vector<char *> argv_array = CharPointerVectorFromStrings(argv_strings);
  int call_count = 0;
  bool parsed_ok = Flags::Parse(&argc, argv_array.data(),
                                {Flag("some_name",
                                      [&call_count](string value) {
                                        call_count++;
                                        return true;
                                      },
                                      "", "some name")});

  EXPECT_EQ(true, parsed_ok);
  EXPECT_EQ(argc, 1);
  EXPECT_EQ(call_count, 2);
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
  int64_t some_int64 = 21474836470;  // max int32 is 2147483647
  bool some_switch = false;
  string some_name = "something";
  // Don't test float in this case, because precision is hard to predict and
  // match against, and we don't want a franky test.
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

namespace {
template <typename T, typename ExpectationFun>
void PrefixTestTempl(ExpectationFun expectation_fun, const T &value0,
                     const T &value1, string str0, string str1) {
  int argc = 3;
  std::vector<string> argv_strings = {
      "program_name",
      "--hello" + str0,
      "--hello_world" + str1,
  };
  std::vector<char *> argv_array = CharPointerVectorFromStrings(argv_strings);

  T hello{};
  T hello_world{};
  bool parsed_ok = Flags::Parse(
      &argc, argv_array.data(),
      {
          Flag("hello", &hello, "usage of hello"),
          Flag("hello_world", &hello_world, "usage of hello world"),
      });

  EXPECT_EQ(true, parsed_ok);
  expectation_fun(value0, hello);
  expectation_fun(value1, hello_world);
  EXPECT_EQ(argc, 1);
}
}  // namespace

TEST(CommandLineFlagsTest, OneArgumentIsAPrefixOfAnother) {
  auto expect_eq = [](auto a, auto b) { EXPECT_EQ(a, b); };
  auto expect_near = [](auto a, auto b) { EXPECT_NEAR(a, b, 1e-5f); };

  PrefixTestTempl<int32_t>(expect_eq, 1, 2, "=1", "=2");
  PrefixTestTempl<int64_t>(expect_eq, 1, 2, "=1", "=2");
  PrefixTestTempl<bool>(expect_eq, false, true, "=false", "=true");
  PrefixTestTempl<bool>(expect_eq, false, true, "=false", "");
  PrefixTestTempl<bool>(expect_eq, true, false, "=true", "=false");
  PrefixTestTempl<bool>(expect_eq, true, false, "", "=false");
  PrefixTestTempl<string>(expect_eq, "a", "b", "=a", "=b");
  PrefixTestTempl<float>(expect_near, 0.1f, 0.2f, "=0.1", "=0.2");
}

}  // namespace tensorflow
