/* Copyright 2017 The OpenXLA Authors.

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

// Test for parse_flags_from_env.cc

#include "xla/parse_flags_from_env.h"

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/command_line_flags.h"

namespace xla {

// Test that XLA flags can be set from the environment.
// Failure messages are accompanied by the text in msg[].
static void TestParseFlagsFromEnv(const char* msg) {
  // Initialize module under test.
  int* pargc;
  std::vector<char*>* pargv;
  ResetFlagsFromEnvForTesting("TF_XLA_FLAGS", &pargc, &pargv);

  // Check that actual flags can be parsed.
  bool simple = false;
  std::string with_value;
  std::string embedded_quotes;
  std::string single_quoted;
  std::string double_quoted;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("simple", &simple, ""),
      tsl::Flag("with_value", &with_value, ""),
      tsl::Flag("embedded_quotes", &embedded_quotes, ""),
      tsl::Flag("single_quoted", &single_quoted, ""),
      tsl::Flag("double_quoted", &double_quoted, ""),
  };
  ParseFlagsFromEnvAndDieIfUnknown("TF_XLA_FLAGS", flag_list);
  CHECK_EQ(*pargc, 1) << msg;
  const std::vector<char*>& argv_second = *pargv;
  CHECK_NE(argv_second[0], nullptr) << msg;
  CHECK_EQ(argv_second[1], nullptr) << msg;
  CHECK(simple) << msg;
  CHECK_EQ(with_value, "a_value") << msg;
  CHECK_EQ(embedded_quotes, "single'double\"") << msg;
  CHECK_EQ(single_quoted, "single quoted \\\\ \n \"") << msg;
  CHECK_EQ(double_quoted, "double quoted \\ \n '\"") << msg;
}

// The flags settings to test.
static const char kTestFlagString[] =
    "--simple "
    "--with_value=a_value "
    "--embedded_quotes=single'double\" "
    "--single_quoted='single quoted \\\\ \n \"' "
    "--double_quoted=\"double quoted \\\\ \n '\\\"\" ";

// Test that the environment variable is parsed correctly.
TEST(ParseFlagsFromEnv, Basic) {
  // Prepare environment.
  tsl::setenv("TF_XLA_FLAGS", kTestFlagString, true /*overwrite*/);
  TestParseFlagsFromEnv("(flags in environment variable)");
}

// Test that a file named by the environment variable is parsed correctly.
TEST(ParseFlagsFromEnv, File) {
  // environment variables where  tmp dir may be specified.
  static const char* kTempVars[] = {"TEST_TMPDIR", "TMP"};
  static const char kTempDir[] = "/tmp";  // default temp dir if all else fails.
  const char* tmp_dir = nullptr;
  for (int i = 0; i != TF_ARRAYSIZE(kTempVars) && tmp_dir == nullptr; i++) {
    tmp_dir = getenv(kTempVars[i]);
  }
  if (tmp_dir == nullptr) {
    tmp_dir = kTempDir;
  }
  std::string tmp_file =
      absl::StrFormat("%s/parse_flags_from_env.%d", tmp_dir, getpid());
  FILE* fp = fopen(tmp_file.c_str(), "w");
  CHECK_NE(fp, nullptr) << "can't write to " << tmp_file;
  for (int i = 0; kTestFlagString[i] != '\0'; i++) {
    putc(kTestFlagString[i], fp);
  }
  fflush(fp);
  CHECK_EQ(ferror(fp), 0) << "writes failed to " << tmp_file;
  fclose(fp);
  // Prepare environment.
  tsl::setenv("TF_XLA_FLAGS", tmp_file.c_str(), true /*overwrite*/);
  TestParseFlagsFromEnv("(flags in file)");
  unlink(tmp_file.c_str());
}

// Name of the test binary.
static const char* binary_name;

// Test that when we use both the environment variable and actual
// commend line flags (when the latter is possible), the latter win.
TEST(ParseFlagsFromEnv, EnvAndFlag) {
  static struct {
    const char* env;
    const char* arg;
    const char* expected_value;
  } test[] = {
      {nullptr, nullptr, "1\n"},
      {nullptr, "--int_flag=2", "2\n"},
      {"--int_flag=3", nullptr, "3\n"},
      {"--int_flag=3", "--int_flag=2", "2\n"},  // flag beats environment
  };
  for (int i = 0; i != TF_ARRAYSIZE(test); i++) {
    if (test[i].env == nullptr) {
      // Might be set from previous tests.
      tsl::unsetenv("TF_XLA_FLAGS");
    } else {
      tsl::setenv("TF_XLA_FLAGS", test[i].env, /*overwrite=*/true);
    }
    tsl::SubProcess child;
    std::vector<std::string> argv;
    argv.push_back(binary_name);
    argv.push_back("--recursing");
    if (test[i].arg != nullptr) {
      argv.push_back(test[i].arg);
    }
    child.SetProgram(binary_name, argv);
    child.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
    child.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
    CHECK(child.Start()) << "test " << i;
    std::string stdout_str;
    std::string stderr_str;
    int child_status = child.Communicate(nullptr, &stdout_str, &stderr_str);
    CHECK_EQ(child_status, 0) << "test " << i << "\nstdout\n"
                              << stdout_str << "\nstderr\n"
                              << stderr_str;
    // On windows, we get CR characters. Remove them.
    stdout_str.erase(std::remove(stdout_str.begin(), stdout_str.end(), '\r'),
                     stdout_str.end());
    CHECK_EQ(stdout_str, test[i].expected_value) << "test " << i;
  }
}

TEST(ParseFlagsFromEnv, ErrorOutOnFlagFailure) {
  const char* env = "--int_flag=3parsefailure";

  if (env == nullptr) {
    // Might be set from previous tests.
    tsl::unsetenv("TF_XLA_FLAGS");
  } else {
    tsl::setenv("TF_XLA_FLAGS", env, /*overwrite=*/true);
  }
  tsl::SubProcess child;
  std::vector<std::string> argv;
  argv.push_back(binary_name);
  argv.push_back("--recursing");
  child.SetProgram(binary_name, argv);
  child.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  child.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
  EXPECT_TRUE(child.Start());
  std::string stdout_str;
  std::string stderr_str;

  // Expecting failure.
  int child_status = child.Communicate(nullptr, &stdout_str, &stderr_str);
  EXPECT_NE(child_status, 0);
}

TEST(ParseFlagsFromEnv, ErrorOutOnUnknownFlag) {
  const char* env = "--int_flag=3 --unknown_flag=value";

  if (env == nullptr) {
    // Might be set from previous tests.
    tsl::unsetenv("TF_XLA_FLAGS");
  } else {
    tsl::setenv("TF_XLA_FLAGS", env, /*overwrite=*/true);
  }
  tsl::SubProcess child;
  std::vector<std::string> argv;
  argv.push_back(binary_name);
  argv.push_back("--recursing");
  child.SetProgram(binary_name, argv);
  child.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  child.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
  EXPECT_TRUE(child.Start());
  std::string stdout_str;
  std::string stderr_str;

  // Expecting failure.
  int child_status = child.Communicate(nullptr, &stdout_str, &stderr_str);
  EXPECT_NE(child_status, 0);
}

TEST(ParseFlagsFromEnv, UknownFlagErrorMessage) {
  const char* env =
      "--unknown_flag_1=value --int_flag=3 --unknown_flag_2=value "
      "--float_flag=3.0";

  if (env == nullptr) {
    // Might be set from previous tests.
    tsl::unsetenv("TF_XLA_FLAGS");
  } else {
    tsl::setenv("TF_XLA_FLAGS", env, /*overwrite=*/true);
  }
  tsl::SubProcess child;
  std::vector<std::string> argv;
  argv.push_back(binary_name);
  argv.push_back("--recursing");
  child.SetProgram(binary_name, argv);
  child.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  child.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
  EXPECT_TRUE(child.Start());
  std::string stdout_str;
  std::string stderr_str;

  int child_status = child.Communicate(nullptr, &stdout_str, &stderr_str);
  EXPECT_NE(child_status, 0);

  EXPECT_THAT(
      stderr_str,
      ::testing::EndsWith("Unknown flags in TF_XLA_FLAGS: "
                          "--unknown_flag_1=value --unknown_flag_2=value\n"));
}

}  // namespace xla

int main(int argc, char* argv[]) {
  // Save name of binary so that it may invoke itself.
  xla::binary_name = argv[0];
  bool recursing = false;
  int32_t int_flag = 1;
  float float_flag = 1.;
  const std::vector<tsl::Flag> flag_list = {
      tsl::Flag("recursing", &recursing,
                "Whether the binary is being invoked recursively."),
      tsl::Flag("int_flag", &int_flag, "An integer flag to test with"),
      tsl::Flag("float_flag", &float_flag, "A float flag to test with"),
  };
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  xla::ParseFlagsFromEnvAndDieIfUnknown("TF_XLA_FLAGS", flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);
  if (recursing) {
    printf("%d\n", int_flag);
    exit(0);
  }
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
