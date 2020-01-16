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

#include "tensorflow/core/platform/stringprintf.h"

#include <string>

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace strings {
namespace {

TEST(PrintfTest, Empty) {
  EXPECT_EQ("", Printf("%s", string().c_str()));
  EXPECT_EQ("", Printf("%s", ""));
}

TEST(PrintfTest, Misc) {
// MSVC does not support $ format specifier.
#if !defined(_MSC_VER)
  EXPECT_EQ("123hello w", Printf("%3$d%2$s %1$c", 'w', "hello", 123));
#endif  // !_MSC_VER
}

TEST(AppendfTest, Empty) {
  string value("Hello");
  const char* empty = "";
  Appendf(&value, "%s", empty);
  EXPECT_EQ("Hello", value);
}

TEST(AppendfTest, EmptyString) {
  string value("Hello");
  Appendf(&value, "%s", "");
  EXPECT_EQ("Hello", value);
}

TEST(AppendfTest, String) {
  string value("Hello");
  Appendf(&value, " %s", "World");
  EXPECT_EQ("Hello World", value);
}

TEST(AppendfTest, Int) {
  string value("Hello");
  Appendf(&value, " %d", 123);
  EXPECT_EQ("Hello 123", value);
}

TEST(PrintfTest, Multibyte) {
  // If we are in multibyte mode and feed invalid multibyte sequence,
  // Printf should return an empty string instead of running
  // out of memory while trying to determine destination buffer size.
  // see b/4194543.

  char* old_locale = setlocale(LC_CTYPE, nullptr);
  // Push locale with multibyte mode
  setlocale(LC_CTYPE, "en_US.utf8");

  const char kInvalidCodePoint[] = "\375\067s";
  string value = Printf("%.*s", 3, kInvalidCodePoint);

  // In some versions of glibc (e.g. eglibc-2.11.1, aka GRTEv2), snprintf
  // returns error given an invalid codepoint. Other versions
  // (e.g. eglibc-2.15, aka pre-GRTEv3) emit the codepoint verbatim.
  // We test that the output is one of the above.
  EXPECT_TRUE(value.empty() || value == kInvalidCodePoint);

  // Repeat with longer string, to make sure that the dynamically
  // allocated path in StringAppendV is handled correctly.
  int n = 2048;
  char* buf = new char[n + 1];
  memset(buf, ' ', n - 3);
  memcpy(buf + n - 3, kInvalidCodePoint, 4);
  value = Printf("%.*s", n, buf);
  // See GRTEv2 vs. GRTEv3 comment above.
  EXPECT_TRUE(value.empty() || value == buf);
  delete[] buf;

  setlocale(LC_CTYPE, old_locale);
}

TEST(PrintfTest, NoMultibyte) {
  // No multibyte handling, but the string contains funny chars.
  char* old_locale = setlocale(LC_CTYPE, nullptr);
  setlocale(LC_CTYPE, "POSIX");
  string value = Printf("%.*s", 3, "\375\067s");
  setlocale(LC_CTYPE, old_locale);
  EXPECT_EQ("\375\067s", value);
}

TEST(PrintfTest, DontOverwriteErrno) {
  // Check that errno isn't overwritten unless we're printing
  // something significantly larger than what people are normally
  // printing in their badly written PLOG() statements.
  errno = ECHILD;
  string value = Printf("Hello, %s!", "World");
  EXPECT_EQ(ECHILD, errno);
}

TEST(PrintfTest, LargeBuf) {
  // Check that the large buffer is handled correctly.
  int n = 2048;
  char* buf = new char[n + 1];
  memset(buf, ' ', n);
  buf[n] = 0;
  string value = Printf("%s", buf);
  EXPECT_EQ(buf, value);
  delete[] buf;
}

}  // namespace

}  // namespace strings
}  // namespace tensorflow
