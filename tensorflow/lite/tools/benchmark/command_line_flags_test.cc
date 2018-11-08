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

#include "tensorflow/lite/tools/benchmark/command_line_flags.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

TEST(CommandLineFlagsTest, BasicUsage) {
  int some_int32 = 10;
  int64_t some_int64 = 21474836470;  // max int32 is 2147483647
  bool some_switch = false;
  std::string some_name = "something_a";
  float some_float = -23.23f;
  bool some_bool = false;
  bool some_numeric_bool = true;
  const char* argv_strings[] = {"program_name",
                                "--some_int32=20",
                                "--some_int64=214748364700",
                                "--some_switch=true",
                                "--some_name=somethingelse",
                                "--some_float=42.0",
                                "--some_bool=true",
                                "--some_numeric_bool=0"};
  int argc = 8;
  bool parsed_ok = Flags::Parse(
      &argc, reinterpret_cast<const char**>(argv_strings),
      {
          Flag::CreateFlag("some_int32", &some_int32, "some int32"),
          Flag::CreateFlag("some_int64", &some_int64, "some int64"),
          Flag::CreateFlag("some_switch", &some_switch, "some switch"),
          Flag::CreateFlag("some_name", &some_name, "some name"),
          Flag::CreateFlag("some_float", &some_float, "some float"),
          Flag::CreateFlag("some_bool", &some_bool, "some bool"),
          Flag::CreateFlag("some_numeric_bool", &some_numeric_bool,
                           "some numeric bool"),
      });

  EXPECT_EQ(true, parsed_ok);
  EXPECT_EQ(20, some_int32);
  EXPECT_EQ(214748364700, some_int64);
  EXPECT_EQ(true, some_switch);
  EXPECT_EQ("somethingelse", some_name);
  EXPECT_NEAR(42.0f, some_float, 1e-5f);
  EXPECT_TRUE(some_bool);
  EXPECT_FALSE(some_numeric_bool);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, EmptyStringFlag) {
  int argc = 2;
  std::string some_string = "invalid";
  const char* argv_strings[] = {"program_name", "--some_string="};
  bool parsed_ok = Flags::Parse(
      &argc, reinterpret_cast<const char**>(argv_strings),
      {Flag::CreateFlag("some_string", &some_string, "some string")});

  EXPECT_EQ(true, parsed_ok);
  EXPECT_EQ(some_string, "");
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, BadIntValue) {
  int some_int = 10;
  int argc = 2;
  const char* argv_strings[] = {"program_name", "--some_int=notanumber"};
  bool parsed_ok =
      Flags::Parse(&argc, reinterpret_cast<const char**>(argv_strings),
                   {Flag::CreateFlag("some_int", &some_int, "some int")});

  EXPECT_EQ(false, parsed_ok);
  EXPECT_EQ(10, some_int);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, BadBoolValue) {
  bool some_switch = false;
  int argc = 2;
  const char* argv_strings[] = {"program_name", "--some_switch=notabool"};
  bool parsed_ok = Flags::Parse(
      &argc, reinterpret_cast<const char**>(argv_strings),
      {Flag::CreateFlag("some_switch", &some_switch, "some switch")});

  EXPECT_EQ(false, parsed_ok);
  EXPECT_EQ(false, some_switch);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, BadFloatValue) {
  float some_float = -23.23f;
  int argc = 2;
  const char* argv_strings[] = {"program_name", "--some_float=notanumber"};
  bool parsed_ok =
      Flags::Parse(&argc, reinterpret_cast<const char**>(argv_strings),
                   {Flag::CreateFlag("some_float", &some_float, "some float")});

  EXPECT_EQ(false, parsed_ok);
  EXPECT_NEAR(-23.23f, some_float, 1e-5f);
  EXPECT_EQ(argc, 1);
}

// Return whether str==pat, but allowing any whitespace in pat
// to match zero or more whitespace characters in str.
static bool MatchWithAnyWhitespace(const std::string& str,
                                   const std::string& pat) {
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
  std::string some_name = "something";
  // Don't test float in this case, because precision is hard to predict and
  // match against, and we don't want a flakey test.
  const std::string tool_name = "some_tool_name";
  std::string usage = Flags::Usage(
      tool_name + " <flags>",
      {Flag::CreateFlag("some_int", &some_int, "some int"),
       Flag::CreateFlag("some_int64", &some_int64, "some int64"),
       Flag::CreateFlag("some_switch", &some_switch, "some switch"),
       Flag::CreateFlag("some_name", &some_name, "some name")});
  // Match the usage message, being sloppy about whitespace.
  const char* expected_usage =
      " usage: some_tool_name <flags>\n"
      "Flags:\n"
      "--some_int=10\tint32\tsome int\n"
      "--some_int64=21474836470\tint64\tsome int64\n"
      "--some_switch=false\tbool\tsome switch\n"
      "--some_name=something\tstring\tsome name\n";
  ASSERT_EQ(MatchWithAnyWhitespace(usage, expected_usage), true) << usage;

  // Again but with no flags.
  usage = Flags::Usage(tool_name, {});
  ASSERT_EQ(MatchWithAnyWhitespace(usage, " usage: some_tool_name\n"), true)
      << usage;
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
