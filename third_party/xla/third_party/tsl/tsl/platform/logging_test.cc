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

#include "tsl/platform/logging.h"

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/path.h"
#include "tsl/platform/stacktrace_handler.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

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

TEST(Logging, Log) {
  LOG(INFO) << "Hello";
  LOG(INFO) << "Another log message";
  LOG(ERROR) << "Error message";
  VLOG(1) << "A VLOG message";
  VLOG(2) << "A higher VLOG message";
  DVLOG(1) << "A DVLOG message";
  DVLOG(2) << "A higher DVLOG message";
}

TEST(Logging, CheckChecks) {
  CHECK(true);
  CHECK(7 > 5);
  string a("abc");
  string b("xyz");
  CHECK_EQ(a, a);
  CHECK_NE(a, b);
  CHECK_EQ(3, 3);
  CHECK_NE(4, 3);
  CHECK_GT(4, 3);
  CHECK_GE(3, 3);
  CHECK_LT(2, 3);
  CHECK_LE(2, 3);

  DCHECK(true);
  DCHECK(7 > 5);
  DCHECK_EQ(a, a);
  DCHECK_NE(a, b);
  DCHECK_EQ(3, 3);
  DCHECK_NE(4, 3);
  DCHECK_GT(4, 3);
  DCHECK_GE(3, 3);
  DCHECK_LT(2, 3);
  DCHECK_LE(2, 3);
}

TEST(LoggingDeathTest, FailedChecks) {
  string a("abc");
  string b("xyz");
  const char* p_const = "hello there";
  const char* p_null_const = nullptr;
  char mybuf[10];
  char* p_non_const = mybuf;
  char* p_null = nullptr;
  CHECK_NOTNULL(p_const);
  CHECK_NOTNULL(p_non_const);

  ASSERT_DEATH(CHECK(false), "false");
  ASSERT_DEATH(CHECK(9 < 7), "9 < 7");
  ASSERT_DEATH(CHECK_EQ(a, b), "a == b");
  ASSERT_DEATH(CHECK_EQ(3, 4), "3 == 4");
  ASSERT_DEATH(CHECK_NE(3, 3), "3 != 3");
  ASSERT_DEATH(CHECK_GT(2, 3), "2 > 3");
  ASSERT_DEATH(CHECK_GE(2, 3), "2 >= 3");
  ASSERT_DEATH(CHECK_LT(3, 2), "3 < 2");
  ASSERT_DEATH(CHECK_LE(3, 2), "3 <= 2");
  ASSERT_DEATH(CHECK(false), "false");
  ASSERT_DEATH(printf("%s", CHECK_NOTNULL(p_null)), "Must be non NULL");
  ASSERT_DEATH(printf("%s", CHECK_NOTNULL(p_null_const)), "Must be non NULL");
#ifndef NDEBUG
  ASSERT_DEATH(DCHECK(9 < 7), "9 < 7");
  ASSERT_DEATH(DCHECK(9 < 7), "9 < 7");
  ASSERT_DEATH(DCHECK_EQ(a, b), "a == b");
  ASSERT_DEATH(DCHECK_EQ(3, 4), "3 == 4");
  ASSERT_DEATH(DCHECK_NE(3, 3), "3 != 3");
  ASSERT_DEATH(DCHECK_GT(2, 3), "2 > 3");
  ASSERT_DEATH(DCHECK_GE(2, 3), "2 >= 3");
  ASSERT_DEATH(DCHECK_LT(3, 2), "3 < 2");
  ASSERT_DEATH(DCHECK_LE(3, 2), "3 <= 2");
#endif
}

TEST(InternalLogString, Basic) {
  // Just make sure that this code compiles (we don't actually verify
  // the output)
  internal::LogString(__FILE__, __LINE__, INFO, "Hello there");
}

class TestSink : public TFLogSink {
 public:
  void Send(const TFLogEntry& entry) override {
    ss_ << entry.text_message() << std::endl;
  }

  std::string Get() const { return ss_.str(); }

 private:
  std::stringstream ss_;
};

TEST(LogSinkTest, testLogSinks) {
  const int sinks_initial_size = TFGetLogSinks().size();
  TestSink sink;

  TFAddLogSink(&sink);

  EXPECT_EQ(TFGetLogSinks().size(), sinks_initial_size + 1);

  LOG(INFO) << "Foo";
  LOG(INFO) << "Bar";
  EXPECT_EQ(sink.Get(), "Foo\nBar\n");

  TFRemoveLogSink(&sink);

  EXPECT_EQ(TFGetLogSinks().size(), sinks_initial_size);
}

std::string ReadFromFilePointer(FILE* fp) {
  std::string result;
  while (!feof(fp)) {
    char buf[512];
    size_t len = fread(buf, sizeof(buf[0]), 512, fp);
    result.append(buf, len);
  }
  return result;
}

absl::StatusOr<std::string> ReadFromFile(const std::string& filename) {
  std::shared_ptr<FILE> fp(fopen(filename.c_str(), "r"), fclose);
  if (fp == nullptr) {
    return absl::ErrnoToStatus(errno,
                               absl::StrFormat("Cannot fopen '%s'", filename));
  }
  return ReadFromFilePointer(fp.get());
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
  absl::StatusOr<std::string> CaptureOutput(const char* invocation) {
    std::shared_ptr<FILE> fp(popen(invocation, "r"), pclose);
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
  TF_ASSERT_OK_AND_ASSIGN(std::string out, CaptureOutput(command.c_str()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::string out, CaptureOutput(command.c_str()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::string out, CaptureOutput(command.c_str()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::string out, CaptureOutput(command.c_str()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::string out, CaptureOutput(command.c_str()));
  EXPECT_THAT(out, HasSubstr("VLevel 1"));
  EXPECT_THAT(out, HasSubstr("VLevel 2"));
  EXPECT_THAT(out, Not(HasSubstr("VLevel 3")));
  EXPECT_THAT(out, HasSubstr("VLOG_IS_ON(1)? 1"));
  EXPECT_THAT(out, HasSubstr("VLOG_IS_ON(2)? 1"));
  EXPECT_THAT(out, HasSubstr("VLOG_IS_ON(3)? 0"));
}

TEST_F(SubcommandTest, VLogFilenameTest) {
#if defined(PLATFORM_GOOGLE)
  constexpr bool kVLogFilenameEnvVarIsSupported = false;
#else
  constexpr bool kVLogFilenameEnvVarIsSupported = true;
#endif
  if (!kVLogFilenameEnvVarIsSupported) {
    GTEST_SKIP() << "Not supported on this platform";
  }

  std::string command = absl::StrFormat("%s %s", program_name, kLogVLog);
  std::string filename = io::GetTempFilename("logging_test");
#if defined(PLATFORM_WINDOWS)
  command = absl::StrFormat(
      "set TF_CPP_VLOG_FILENAME=%s && set TF_CPP_MAX_VLOG_LEVEL=1 && %s",
      filename, command);
#else
  command = absl::StrFormat(
      "TF_CPP_VLOG_FILENAME=%s TF_CPP_MAX_VLOG_LEVEL=1 %s", filename, command);
#endif
  command += " 2>&1";

  // All output should be in the file, not in stderr.
  TF_ASSERT_OK_AND_ASSIGN(std::string out, CaptureOutput(command.c_str()));
  EXPECT_THAT(out, Not(HasSubstr("LOG INFO")));
  EXPECT_THAT(out, Not(HasSubstr("LOG WARNING")));
  EXPECT_THAT(out, Not(HasSubstr("LOG ERROR")));
  EXPECT_THAT(out, Not(HasSubstr("VLOG_IS_ON(1)?")));
  EXPECT_THAT(out, Not(HasSubstr("VLOG_IS_ON(2)?")));
  EXPECT_THAT(out, Not(HasSubstr("VLOG_IS_ON(3)?")));
  EXPECT_THAT(out, Not(HasSubstr("VLevel 1")));
  EXPECT_THAT(out, Not(HasSubstr("VLevel 2")));
  EXPECT_THAT(out, Not(HasSubstr("VLevel 3")));

  TF_ASSERT_OK_AND_ASSIGN(std::string log_file, ReadFromFile(filename));
  EXPECT_THAT(log_file, HasSubstr("LOG INFO"));
  EXPECT_THAT(log_file, HasSubstr("LOG WARNING"));
  EXPECT_THAT(log_file, HasSubstr("LOG ERROR"));
  EXPECT_THAT(log_file, HasSubstr("VLOG_IS_ON(1)"));
  EXPECT_THAT(log_file, HasSubstr("VLOG_IS_ON(2)"));
  EXPECT_THAT(log_file, HasSubstr("VLOG_IS_ON(3)"));
  EXPECT_THAT(log_file, HasSubstr("VLevel 1"));
  EXPECT_THAT(log_file, Not(HasSubstr("VLevel 2")));
  EXPECT_THAT(log_file, Not(HasSubstr("VLevel 3")));
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
