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

#include "tensorflow/lite/tools/command_line_flags.h"

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/tools/tool_params.h"

namespace tflite {
namespace {

TEST(CommandLineFlagsTest, BasicUsage) {
  int some_int32 = 10;
  int some_int1 = 8;  // Not provided via arguments, the value should remain.
  int some_int2 = 9;  // Required flag.
  int64_t some_int64 = 21474836470;  // max int32 is 2147483647
  bool some_switch = false;
  std::string some_name = "something_a";
  float some_float = -23.23f;
  float float_1 = -23.23f;  // positional flag.
  bool some_bool = false;
  bool some_numeric_bool = true;
  const char* argv_strings[] = {"program_name",
                                "12.2",
                                "--some_int32=20",
                                "--some_int2=5",
                                "--some_int64=214748364700",
                                "--some_switch=true",
                                "--some_name=somethingelse",
                                "--some_float=42.0",
                                "--some_bool=true",
                                "--some_numeric_bool=0"};
  int argc = 10;
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
          Flag::CreateFlag("some_int1", &some_int1, "some int"),
          Flag::CreateFlag("some_int2", &some_int2, "some int",
                           Flag::kRequired),
          Flag::CreateFlag("float_1", &float_1, "some float",
                           Flag::kPositional),
      });

  EXPECT_TRUE(parsed_ok);
  EXPECT_EQ(20, some_int32);
  EXPECT_EQ(8, some_int1);
  EXPECT_EQ(5, some_int2);
  EXPECT_EQ(214748364700, some_int64);
  EXPECT_TRUE(some_switch);
  EXPECT_EQ("somethingelse", some_name);
  EXPECT_NEAR(42.0f, some_float, 1e-5f);
  EXPECT_NEAR(12.2f, float_1, 1e-5f);
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

  EXPECT_TRUE(parsed_ok);
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

  EXPECT_FALSE(parsed_ok);
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

  EXPECT_FALSE(parsed_ok);
  EXPECT_FALSE(some_switch);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, BadFloatValue) {
  float some_float = -23.23f;
  int argc = 2;
  const char* argv_strings[] = {"program_name", "--some_float=notanumber"};
  bool parsed_ok =
      Flags::Parse(&argc, reinterpret_cast<const char**>(argv_strings),
                   {Flag::CreateFlag("some_float", &some_float, "some float")});

  EXPECT_FALSE(parsed_ok);
  EXPECT_NEAR(-23.23f, some_float, 1e-5f);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, RequiredFlagNotFound) {
  float some_float = -23.23f;
  int argc = 2;
  const char* argv_strings[] = {"program_name", "--flag=12"};
  bool parsed_ok = Flags::Parse(
      &argc, reinterpret_cast<const char**>(argv_strings),
      {Flag::CreateFlag("some_flag", &some_float, "", Flag::kRequired)});

  EXPECT_FALSE(parsed_ok);
  EXPECT_NEAR(-23.23f, some_float, 1e-5f);
  EXPECT_EQ(argc, 2);
}

TEST(CommandLineFlagsTest, NoArguments) {
  float some_float = -23.23f;
  int argc = 1;
  const char* argv_strings[] = {"program_name"};
  bool parsed_ok = Flags::Parse(
      &argc, reinterpret_cast<const char**>(argv_strings),
      {Flag::CreateFlag("some_flag", &some_float, "", Flag::kRequired)});

  EXPECT_FALSE(parsed_ok);
  EXPECT_NEAR(-23.23f, some_float, 1e-5f);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, NotEnoughArguments) {
  float some_float = -23.23f;
  int argc = 1;
  const char* argv_strings[] = {"program_name"};
  bool parsed_ok = Flags::Parse(
      &argc, reinterpret_cast<const char**>(argv_strings),
      {Flag::CreateFlag("some_flag", &some_float, "", Flag::kPositional)});

  EXPECT_FALSE(parsed_ok);
  EXPECT_NEAR(-23.23f, some_float, 1e-5f);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, PositionalFlagFailed) {
  float some_float = -23.23f;
  int argc = 2;
  const char* argv_strings[] = {"program_name", "string"};
  bool parsed_ok = Flags::Parse(
      &argc, reinterpret_cast<const char**>(argv_strings),
      {Flag::CreateFlag("some_flag", &some_float, "", Flag::kPositional)});

  EXPECT_FALSE(parsed_ok);
  EXPECT_NEAR(-23.23f, some_float, 1e-5f);
  EXPECT_EQ(argc, 2);
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
  int some_int2 = 4;
  // Don't test float in this case, because precision is hard to predict and
  // match against, and we don't want a flakey test.
  const std::string tool_name = "some_tool_name";
  std::string usage = Flags::Usage(
      tool_name,
      {Flag::CreateFlag("some_int", &some_int, "some int"),
       Flag::CreateFlag("some_int64", &some_int64, "some int64"),
       Flag::CreateFlag("some_switch", &some_switch, "some switch"),
       Flag::CreateFlag("some_name", &some_name, "some name", Flag::kRequired),
       Flag::CreateFlag("some_int2", &some_int2, "some int",
                        Flag::kPositional)});
  // Match the usage message, being sloppy about whitespace.
  const char* expected_usage =
      " usage: some_tool_name <some_int2> <flags>\n"
      "Where:\n"
      "some_int2\tint32\trequired\tsome int\n"
      "Flags:\n"
      "--some_name=something\tstring\trequired\tsome name\n"
      "--some_int=10\tint32\toptional\tsome int\n"
      "--some_int64=21474836470\tint64\toptional\tsome int64\n"
      "--some_switch=false\tbool\toptional\tsome switch\n";
  ASSERT_EQ(MatchWithAnyWhitespace(usage, expected_usage), true) << usage;

  // Again but with no flags.
  usage = Flags::Usage(tool_name, {});
  ASSERT_EQ(MatchWithAnyWhitespace(usage, " usage: some_tool_name\n"), true)
      << usage;
}

// When there are duplicate args, the flag value and the parsing result will be
// based on the 1st arg.
TEST(CommandLineFlagsTest, DuplicateArgsParsableValues) {
  int some_int = -23;
  int argc = 3;
  const char* argv_strings[] = {"program_name", "--some_int=1", "--some_int=2"};
  bool parsed_ok =
      Flags::Parse(&argc, reinterpret_cast<const char**>(argv_strings),
                   {Flag::CreateFlag("some_int", &some_int, "some int")});

  EXPECT_TRUE(parsed_ok);
  EXPECT_EQ(1, some_int);
  EXPECT_EQ(argc, 2);
  EXPECT_EQ("--some_int=2", argv_strings[1]);
}

TEST(CommandLineFlagsTest, DuplicateArgsBadValueAppearFirst) {
  int some_int = -23;
  int argc = 3;
  const char* argv_strings[] = {"program_name", "--some_int=value",
                                "--some_int=1"};
  bool parsed_ok =
      Flags::Parse(&argc, reinterpret_cast<const char**>(argv_strings),
                   {Flag::CreateFlag("some_int", &some_int, "some int")});

  EXPECT_FALSE(parsed_ok);
  EXPECT_EQ(-23, some_int);
  EXPECT_EQ(argc, 2);
  EXPECT_EQ("--some_int=1", argv_strings[1]);
}

TEST(CommandLineFlagsTest, DuplicateArgsBadValueAppearSecondly) {
  int some_int = -23;
  int argc = 3;
  // Although the 2nd arg has non-parsable int value, the flag 'some_int' value
  // could still be set and the parsing result is ok.
  const char* argv_strings[] = {"program_name", "--some_int=1",
                                "--some_int=value"};
  bool parsed_ok =
      Flags::Parse(&argc, reinterpret_cast<const char**>(argv_strings),
                   {Flag::CreateFlag("some_int", &some_int, "some int")});

  EXPECT_TRUE(parsed_ok);
  EXPECT_EQ(1, some_int);
  EXPECT_EQ(argc, 2);
  EXPECT_EQ("--some_int=value", argv_strings[1]);
}

// When there are duplicate flags, all of them will be checked against the arg
// list.
TEST(CommandLineFlagsTest, DuplicateFlags) {
  int some_int1 = -23;
  int some_int2 = -23;
  int argc = 2;
  const char* argv_strings[] = {"program_name", "--some_int=1"};
  bool parsed_ok =
      Flags::Parse(&argc, reinterpret_cast<const char**>(argv_strings),
                   {Flag::CreateFlag("some_int", &some_int1, "some int1"),
                    Flag::CreateFlag("some_int", &some_int2, "some int2")});

  EXPECT_TRUE(parsed_ok);
  EXPECT_EQ(1, some_int1);
  EXPECT_EQ(1, some_int2);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, DuplicateFlagsNotFound) {
  int some_int1 = -23;
  int some_int2 = -23;
  int argc = 2;
  const char* argv_strings[] = {"program_name", "--some_float=1.0"};
  bool parsed_ok = Flags::Parse(
      &argc, reinterpret_cast<const char**>(argv_strings),
      {Flag::CreateFlag("some_int", &some_int1, "some int1", Flag::kOptional),
       Flag::CreateFlag("some_int", &some_int2, "some int2", Flag::kRequired)});

  EXPECT_FALSE(parsed_ok);
  EXPECT_EQ(-23, some_int1);
  EXPECT_EQ(-23, some_int2);
  EXPECT_EQ(argc, 2);
}

TEST(CommandLineFlagsTest, DuplicateFlagNamesButDifferentTypes) {
  int some_int = -23;
  bool some_bool = true;
  int argc = 2;
  const char* argv_strings[] = {"program_name", "--some_val=20"};
  // In this case, the 2nd 'some_val' flag of bool type will cause a no-ok
  // parsing result.
  bool parsed_ok =
      Flags::Parse(&argc, reinterpret_cast<const char**>(argv_strings),
                   {Flag::CreateFlag("some_val", &some_int, "some val-int"),
                    Flag::CreateFlag("some_val", &some_bool, "some val-bool")});

  EXPECT_FALSE(parsed_ok);
  EXPECT_EQ(20, some_int);
  EXPECT_TRUE(some_bool);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, DuplicateFlagsAndArgs) {
  int some_int1 = -23;
  int some_int2 = -23;
  int argc = 3;
  const char* argv_strings[] = {"program_name", "--some_int=1", "--some_int=2"};
  bool parsed_ok = Flags::Parse(
      &argc, reinterpret_cast<const char**>(argv_strings),
      {Flag::CreateFlag("some_int", &some_int1, "flag1: bind with some_int1"),
       Flag::CreateFlag("some_int", &some_int2, "flag2: bind with some_int2")});

  // Note, when there're duplicate args, the flag value and the parsing result
  // will be based on the 1st arg (i.e. --some_int=1). And both duplicate flags
  // (i.e. flag1 and flag2) are checked, thus resulting their associated values
  // (some_int1 and some_int2) being set to 1.
  EXPECT_TRUE(parsed_ok);
  EXPECT_EQ(1, some_int1);
  EXPECT_EQ(1, some_int2);
  EXPECT_EQ(argc, 2);
}

TEST(CommandLineFlagsTest, ArgsToString) {
  int argc = 3;
  const char* argv_strings[] = {"program_name", "--some_int=1", "--some_int=2"};
  std::string args =
      Flags::ArgsToString(argc, reinterpret_cast<const char**>(argv_strings));
  EXPECT_EQ("--some_int=1 --some_int=2", args);
}

TEST(CommandLineFlagsTest, ArgvPositions) {
  tools::ToolParams params;
  params.AddParam("some_int", tools::ToolParam::Create<int>(13));
  params.AddParam("some_float", tools::ToolParam::Create<float>(17.0f));
  params.AddParam("some_bool", tools::ToolParam::Create<bool>(true));

  const char* argv_strings[] = {"program_name", "--some_float=42.0",
                                "--some_bool=false", "--some_int=5"};
  int argc = 4;
  tools::ToolParams* const params_ptr = &params;
  bool parsed_ok = Flags::Parse(
      &argc, reinterpret_cast<const char**>(argv_strings),
      {
          Flag(
              "some_int",
              // NOLINT because of needing templating both trivial and complex
              // types for a Flag.
              [params_ptr](const int& val, int argv_position) {  // NOLINT
                params_ptr->Set<int>("some_int", val, argv_position);
              },
              13, "some int", Flag::kOptional),
          Flag(
              "some_float",
              [params_ptr](const float& val, int argv_position) {  // NOLINT
                params_ptr->Set<float>("some_float", val, argv_position);
              },
              17.0f, "some float", Flag::kOptional),
          Flag(
              "some_bool",
              [params_ptr](const bool& val, int argv_position) {  // NOLINT
                params_ptr->Set<bool>("some_bool", val, argv_position);
              },
              true, "some bool", Flag::kOptional),
      });

  EXPECT_TRUE(parsed_ok);
  EXPECT_EQ(5, params.Get<int>("some_int"));
  EXPECT_NEAR(42.0f, params.Get<float>("some_float"), 1e-5f);
  EXPECT_FALSE(params.Get<bool>("some_bool"));

  // The position of a parameter depends on the ordering of the associated flag
  // specfied in the argv (i.e. 'argv_strings' above), not as the ordering of
  // the flag in the flag list that's passed to Flags::Parse above.
  EXPECT_EQ(3, params.GetPosition<int>("some_int"));
  EXPECT_EQ(1, params.GetPosition<float>("some_float"));
  EXPECT_EQ(2, params.GetPosition<bool>("some_bool"));

  EXPECT_EQ(argc, 1);
}

}  // namespace
}  // namespace tflite
