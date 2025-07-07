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

#include "xla/tsl/platform/logging.h"

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/stacktrace_handler.h"

// Make sure popen and pclose are available on Windows.
#ifdef PLATFORM_WINDOWS
#define popen _popen
#define pclose _pclose
#endif

static char* program_name;

namespace tsl {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

std::string ReadFromFilePointer(FILE* fp) {
  std::string result;
  while (!feof(fp)) {
    char buf[512];
    size_t len = fread(buf, sizeof(buf[0]), 512, fp);
    result.append(buf, len);
  }
  return result;
}

class SubcommandTest : public ::testing::Test {
 public:
  static constexpr absl::string_view kLogVLog = "log_and_vlog";

  static bool IsSubcommand(absl::string_view subcommand) {
    return subcommand == kLogVLog;
  }

  static int Run(absl::string_view subcommand) {
    CHECK_EQ(subcommand, kLogVLog);
    LOG(INFO) << "LOG INFO";
    LOG(WARNING) << "LOG WARNING";
    LOG(ERROR) << "LOG ERROR";
    LOG(INFO) << absl::StrFormat("VLOG_IS_ON(1)? %d", VLOG_IS_ON(1));
    LOG(INFO) << absl::StrFormat("VLOG_IS_ON(2)? %d", VLOG_IS_ON(2));
    LOG(INFO) << absl::StrFormat("VLOG_IS_ON(3)? %d", VLOG_IS_ON(3));
    VLOG(1) << "VLevel 1";
    VLOG(2) << "VLevel 2";
    VLOG(3) << "VLevel 3";
    return EXIT_SUCCESS;
  }

 protected:
  absl::StatusOr<std::string> CaptureOutput(absl::string_view invocation) {
    std::shared_ptr<FILE> fp(popen(invocation.data(), "r"), pclose);
    if (fp == nullptr) {
      return absl::ErrnoToStatus(
          errno, absl::StrFormat("Cannot popen '%s'", invocation));
    }
    return ReadFromFilePointer(fp.get());
  }
};

// By default, messages with severity >= INFO should be printed.
TEST_F(SubcommandTest, LogDefaultTest) {
  std::string command = absl::StrFormat("%s %s", program_name, kLogVLog);
#if defined(PLATFORM_GOOGLE)
  command += " --alsologtostderr";
#endif
  command += " 2>&1";
  TF_ASSERT_OK_AND_ASSIGN(std::string out, CaptureOutput(command));
  EXPECT_THAT(out, HasSubstr("LOG INFO"));
  EXPECT_THAT(out, HasSubstr("LOG WARNING"));
  EXPECT_THAT(out, HasSubstr("LOG ERROR"));
  EXPECT_THAT(out, HasSubstr("VLOG_IS_ON(1)? 0"));
  EXPECT_THAT(out, HasSubstr("VLOG_IS_ON(2)? 0"));
  EXPECT_THAT(out, HasSubstr("VLOG_IS_ON(3)? 0"));
}

TEST_F(SubcommandTest, MinLogLevelTest) {
  std::string command = absl::StrFormat("%s %s", program_name, kLogVLog);
#if defined(PLATFORM_GOOGLE)
  command += " --minloglevel=1 --alsologtostderr";
#elif defined(PLATFORM_WINDOWS)
  command = absl::StrFormat("set TF_CPP_MIN_LOG_LEVEL=1 && %s", command);
#else
  command = absl::StrFormat("TF_CPP_MIN_LOG_LEVEL=1 %s", command);
#endif
  command += " 2>&1";
  TF_ASSERT_OK_AND_ASSIGN(std::string out, CaptureOutput(command));
  EXPECT_THAT(out, Not(HasSubstr("LOG INFO")));
  EXPECT_THAT(out, HasSubstr("LOG WARNING"));
  EXPECT_THAT(out, HasSubstr("LOG ERROR"));
}

// By default, no VLOG messages should be printed.
TEST_F(SubcommandTest, VLogDefaultTest) {
  std::string command = absl::StrFormat("%s %s", program_name, kLogVLog);
#if defined(PLATFORM_GOOGLE)
  command += " --alsologtostderr";
#endif
  command += " 2>&1";
  TF_ASSERT_OK_AND_ASSIGN(std::string out, CaptureOutput(command));
  EXPECT_THAT(out, Not(HasSubstr("VLevel 1")));
  EXPECT_THAT(out, Not(HasSubstr("VLevel 2")));
  EXPECT_THAT(out, Not(HasSubstr("VLevel 3")));
}

TEST_F(SubcommandTest, MaxVLogLevelTest) {
  std::string command = absl::StrFormat("%s %s", program_name, kLogVLog);
#if defined(PLATFORM_GOOGLE)
  command += " --v=2 --alsologtostderr";
#elif defined(PLATFORM_WINDOWS)
  command = absl::StrFormat("set TF_CPP_MAX_VLOG_LEVEL=2 && %s", command);
#else
  command = absl::StrFormat("TF_CPP_MAX_VLOG_LEVEL=2 %s", command);
#endif
  command += " 2>&1";
  TF_ASSERT_OK_AND_ASSIGN(std::string out, CaptureOutput(command));
  EXPECT_THAT(out, HasSubstr("VLevel 1"));
  EXPECT_THAT(out, HasSubstr("VLevel 2"));
  EXPECT_THAT(out, Not(HasSubstr("VLevel 3")));
  EXPECT_THAT(out, HasSubstr("VLOG_IS_ON(1)? 1"));
  EXPECT_THAT(out, HasSubstr("VLOG_IS_ON(2)? 1"));
  EXPECT_THAT(out, HasSubstr("VLOG_IS_ON(3)? 0"));
}

TEST_F(SubcommandTest, VModuleTest) {
  std::string command = absl::StrFormat("%s %s", program_name, kLogVLog);
#if defined(PLATFORM_GOOGLE)
  command += " --vmodule=logging_test=2,shoobadooba=3 --alsologtostderr";
#elif defined(PLATFORM_WINDOWS)
  command = absl::StrFormat(
      "set TF_CPP_VMODULE=logging_test=2,shoobadooba=3 && %s", command);
#else
  command = absl::StrFormat("TF_CPP_VMODULE=logging_test=2,shoobadooba=3 %s",
                            command);
#endif
  command += " 2>&1";
  TF_ASSERT_OK_AND_ASSIGN(std::string out, CaptureOutput(command));
  EXPECT_THAT(out, HasSubstr("VLevel 1"));
  EXPECT_THAT(out, HasSubstr("VLevel 2"));
  EXPECT_THAT(out, Not(HasSubstr("VLevel 3")));
  EXPECT_THAT(out, HasSubstr("VLOG_IS_ON(1)? 1"));
  EXPECT_THAT(out, HasSubstr("VLOG_IS_ON(2)? 1"));
  EXPECT_THAT(out, HasSubstr("VLOG_IS_ON(3)? 0"));
}

}  // namespace
}  // namespace tsl

GTEST_API_ int main(int argc, char** argv) {
  tsl::testing::InstallStacktraceHandler();
  testing::InitGoogleTest(&argc, argv);
  program_name = argv[0];
  if (argc >= 2 && tsl::SubcommandTest::IsSubcommand(argv[1])) {
    return tsl::SubcommandTest::Run(argv[1]);
  }
  return RUN_ALL_TESTS();
}
