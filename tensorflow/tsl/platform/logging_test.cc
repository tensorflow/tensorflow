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

#include "tensorflow/tsl/platform/logging.h"

#include <sstream>
#include <vector>

#include "tensorflow/tsl/platform/test.h"

namespace tsl {

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

}  // namespace tsl
