/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/lib/io/inputbuffer.h"

#include <vector>
#include "tensorflow/core/platform/env.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

static std::vector<int> BufferSizes() {
  return {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   11,
          12, 13, 14, 15, 16, 17, 18, 19, 20, 65536};
}

TEST(InputBuffer, ReadLine_Empty) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/inputbuffer_test";
  WriteStringToFile(env, fname, "");

  for (auto buf_size : BufferSizes()) {
    RandomAccessFile* file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    string line;
    io::InputBuffer in(file, buf_size);
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(InputBuffer, ReadLine1) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/inputbuffer_test";
  WriteStringToFile(env, fname, "line one\nline two\nline three\n");

  for (auto buf_size : BufferSizes()) {
    RandomAccessFile* file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    string line;
    io::InputBuffer in(file, buf_size);
    TF_CHECK_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line one");
    TF_CHECK_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_CHECK_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line three");
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
    // A second call should also return end of file
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(InputBuffer, ReadLine_NoTrailingNewLine) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/inputbuffer_test";
  WriteStringToFile(env, fname, "line one\nline two\nline three");

  for (auto buf_size : BufferSizes()) {
    RandomAccessFile* file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    string line;
    io::InputBuffer in(file, buf_size);
    TF_CHECK_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line one");
    TF_CHECK_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_CHECK_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line three");
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
    // A second call should also return end of file
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(InputBuffer, ReadLine_EmptyLines) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/inputbuffer_test";
  WriteStringToFile(env, fname, "line one\n\n\nline two\nline three");

  for (auto buf_size : BufferSizes()) {
    RandomAccessFile* file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    string line;
    io::InputBuffer in(file, buf_size);
    TF_CHECK_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line one");
    TF_CHECK_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_CHECK_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_CHECK_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_CHECK_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line three");
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
    // A second call should also return end of file
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(InputBuffer, ReadLine_CRLF) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/inputbuffer_test";
  WriteStringToFile(env, fname, "line one\r\n\r\n\r\nline two\r\nline three");

  for (auto buf_size : BufferSizes()) {
    RandomAccessFile* file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    string line;
    io::InputBuffer in(file, buf_size);
    TF_CHECK_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line one");
    TF_CHECK_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_CHECK_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_CHECK_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_CHECK_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line three");
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
    // A second call should also return end of file
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(InputBuffer, ReadNBytes) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/inputbuffer_test";
  WriteStringToFile(env, fname, "0123456789");

  for (auto buf_size : BufferSizes()) {
    RandomAccessFile* file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    string read;
    io::InputBuffer in(file, buf_size);
    EXPECT_EQ(0, in.Tell());
    TF_CHECK_OK(in.ReadNBytes(3, &read));
    EXPECT_EQ(read, "012");
    EXPECT_EQ(3, in.Tell());
    TF_CHECK_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(3, in.Tell());
    TF_CHECK_OK(in.ReadNBytes(4, &read));
    EXPECT_EQ(read, "3456");
    EXPECT_EQ(7, in.Tell());
    TF_CHECK_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(7, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "789");
    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
    TF_CHECK_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
  }
}

TEST(InputBuffer, SkipNBytes) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/inputbuffer_test";
  WriteStringToFile(env, fname, "0123456789");

  for (auto buf_size : BufferSizes()) {
    RandomAccessFile* file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    string read;
    io::InputBuffer in(file, buf_size);
    EXPECT_EQ(0, in.Tell());
    TF_CHECK_OK(in.SkipNBytes(3));
    EXPECT_EQ(3, in.Tell());
    TF_CHECK_OK(in.SkipNBytes(0));
    EXPECT_EQ(3, in.Tell());
    TF_CHECK_OK(in.ReadNBytes(2, &read));
    EXPECT_EQ(read, "34");
    EXPECT_EQ(5, in.Tell());
    TF_CHECK_OK(in.SkipNBytes(0));
    EXPECT_EQ(5, in.Tell());
    TF_CHECK_OK(in.SkipNBytes(2));
    EXPECT_EQ(7, in.Tell());
    TF_CHECK_OK(in.ReadNBytes(1, &read));
    EXPECT_EQ(read, "7");
    EXPECT_EQ(8, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.SkipNBytes(5)));
    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.SkipNBytes(5)));
    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
  }
}

}  // namespace tensorflow
