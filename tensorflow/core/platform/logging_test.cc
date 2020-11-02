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

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

#include <sstream>
#include <vector>

namespace tensorflow {

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

struct TestSink : public TFLogSink {
  std::stringstream ss;

  void Send(const TFLogEntry& entry) override {
    ss << entry.ToString() << std::endl;
  }
};

}  // namespace tensorflow

#if defined(__linux__) && !defined(PLATFORM_POSIX_ANDROID)

#include "absl/strings/str_split.h"

#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

namespace tensorflow {

class LogSinkTest : public ::testing::Test {
 protected:
  static const size_t kBufLen = 1024;

  void SetUp() override {
    ASSERT_EQ(pipe(out_pipe_), 0);

    auto flags = fcntl(out_pipe_[0], F_GETFL); 
    flags |= O_NONBLOCK; 
    fcntl(out_pipe_[0], F_SETFL, flags);

    saved_stderr_ = dup(STDERR_FILENO);
    ASSERT_NE(saved_stderr_, -1);

    ASSERT_NE(dup2(out_pipe_[1], STDERR_FILENO), -1);
    ASSERT_EQ(close(out_pipe_[1]), 0);

    auto sinks = TFGetLogSinks();
#ifndef NO_DEFAULT_LOGGER
    ASSERT_EQ(sinks.size(), 1);
#else
    ASSERT_EQ(sinks.size(), 0);
#endif

    for(auto sink : TFGetLogSinks()) {
      TFRemoveLogSink(sink);
    }

    TFAddLogSink(&default_sink_);
  }

  void TearDown() override {
    ASSERT_NE(dup2(saved_stderr_, STDERR_FILENO), -1);
    saved_stderr_ = out_pipe_[0] = out_pipe_[1] = -1;

    for(auto sink : TFGetLogSinks()) {
      TFRemoveLogSink(sink);
    }
#ifndef NO_DEFAULT_LOGGER
    TFAddLogSink(&default_sink_);
#endif
  }

  std::string GetStdErr() {
    fflush(stderr);

    ssize_t len = read(out_pipe_[0], (void*)buf_, kBufLen);
    
    std::string ret;
    if(len > 0) {
      ret.assign(buf_, buf_ + len);
    }
    
    return ret;
  }

 private:
  int saved_stderr_ = -1;
  int out_pipe_[2] = { -1, -1 };
  char buf_[kBufLen + 1] = { '/0' };
  static TFDefaultLogSink default_sink_;
};

TFDefaultLogSink LogSinkTest::default_sink_;

TEST_F(LogSinkTest, testLogSinks) {
  // First log into the default log sink
  LOG(INFO) << "Foo";
  LOG(INFO) << "Bar";

  std::vector<std::string> lines = absl::StrSplit(GetStdErr(), '\n');
  ASSERT_EQ(lines.size(), 3);
  ASSERT_NE(lines[0].find_first_of("Foo"), std::string::npos);
  ASSERT_NE(lines[1].find_first_of("Bar"), std::string::npos);
  ASSERT_TRUE(lines[2].empty());

  // Remove the default log sink
  auto sinks = TFGetLogSinks();
  ASSERT_EQ(sinks.size(), 1);

  TFRemoveLogSink(sinks[0]);
  sinks = TFGetLogSinks();
  ASSERT_EQ(sinks.size(), 0);

  LOG(INFO) << "Foo";
  ASSERT_TRUE(GetStdErr().empty());

  static TestSink sink;
  TFAddLogSink(&sink);
  sinks = TFGetLogSinks();
  ASSERT_EQ(sinks.size(), 1);

  ASSERT_EQ(sink.ss.str(), "Foo\n");

  LOG(INFO) << "Bar";
  ASSERT_EQ(sink.ss.str(), "Foo\nBar\n");
}

}  // namespace tensorflow

#else // defined(__linux__) && !defined(PLATFORM_POSIX_ANDROID)

namespace tensorflow {

TEST(LogSinkTest, testLogSinks)
{
  // Test the default log sink as much as we can, write a `stdout` interceptor for you OS to test it further.
  auto sinks = TFGetLogSinks();
#ifdef NO_DEFAULT_LOGGER
  EXPECT_EQ(sinks.size(), 0);
#else
  EXPECT_EQ(sinks.size(), 1);
#endif

  static TestSink sink;
  TFAddLogSink(&sink);
  sinks = TFGetLogSinks();

#ifdef NO_DEFAULT_LOGGER
  EXPECT_EQ(sinks.size(), 1);
#else
  EXPECT_EQ(sinks.size(), 2);
#endif

  LOG(INFO) << "Foo";
  LOG(INFO) << "Bar";
  EXPECT_EQ(sink.ss.str(), "Foo\nBar\n");

  TFRemoveLogSink(&sink);
  sinks = TFGetLogSinks();

#ifdef NO_DEFAULT_LOGGER
  EXPECT_EQ(sinks.size(), 0);
#else
  EXPECT_EQ(sinks.size(), 1);
#endif
}

} // namespace tensorflow

#endif // defined(__linux__) && !defined(PLATFORM_POSIX_ANDROID)
