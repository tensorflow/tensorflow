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

#include "tensorflow/core/lib/io/inputbuffer.h"

#include <vector>

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

static std::vector<int> BufferSizes() {
  return {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   11,
          12, 13, 14, 15, 16, 17, 18, 19, 20, 65536};
}

TEST(InputBuffer, ReadLine_Empty) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, ""));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessFile> file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    string line;
    io::InputBuffer in(file.get(), buf_size);
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(InputBuffer, ReadLine1) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_CHECK_OK(
      WriteStringToFile(env, fname, "line one\nline two\nline three\n"));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessFile> file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    string line;
    io::InputBuffer in(file.get(), buf_size);
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
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "line one\nline two\nline three"));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessFile> file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    string line;
    io::InputBuffer in(file.get(), buf_size);
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
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_CHECK_OK(
      WriteStringToFile(env, fname, "line one\n\n\nline two\nline three"));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessFile> file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    string line;
    io::InputBuffer in(file.get(), buf_size);
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
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname,
                                 "line one\r\n\r\n\r\nline two\r\nline three"));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessFile> file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    string line;
    io::InputBuffer in(file.get(), buf_size);
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
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));

  // ReadNBytes(int64, string*).
  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessFile> file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    string read;
    io::InputBuffer in(file.get(), buf_size);
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
  // ReadNBytes(int64, char*, size_t*).
  size_t bytes_read;
  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessFile> file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    char read[5];
    io::InputBuffer in(file.get(), buf_size);

    EXPECT_EQ(0, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(3, read, &bytes_read));
    EXPECT_EQ(StringPiece(read, 3), "012");

    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, read, &bytes_read));
    EXPECT_EQ(StringPiece(read, 3), "012");

    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(4, read, &bytes_read));
    EXPECT_EQ(StringPiece(read, 4), "3456");

    EXPECT_EQ(7, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, read, &bytes_read));
    EXPECT_EQ(StringPiece(read, 4), "3456");

    EXPECT_EQ(7, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(5, read, &bytes_read)));
    EXPECT_EQ(StringPiece(read, 3), "789");

    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(5, read, &bytes_read)));
    EXPECT_EQ(StringPiece(read, 3), "789");

    EXPECT_EQ(10, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, read, &bytes_read));
    EXPECT_EQ(StringPiece(read, 3), "789");
    EXPECT_EQ(10, in.Tell());
  }
}

TEST(InputBuffer, SkipNBytes) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessFile> file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    string read;
    io::InputBuffer in(file.get(), buf_size);
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

TEST(InputBuffer, Seek) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessFile> file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    string read;
    io::InputBuffer in(file.get(), buf_size);

    TF_CHECK_OK(in.ReadNBytes(3, &read));
    EXPECT_EQ(read, "012");
    TF_CHECK_OK(in.ReadNBytes(3, &read));
    EXPECT_EQ(read, "345");

    TF_CHECK_OK(in.Seek(0));
    TF_CHECK_OK(in.ReadNBytes(3, &read));
    EXPECT_EQ(read, "012");

    TF_CHECK_OK(in.Seek(3));
    TF_CHECK_OK(in.ReadNBytes(4, &read));
    EXPECT_EQ(read, "3456");

    TF_CHECK_OK(in.Seek(4));
    TF_CHECK_OK(in.ReadNBytes(4, &read));
    EXPECT_EQ(read, "4567");

    TF_CHECK_OK(in.Seek(1 << 25));
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(1, &read)));

    EXPECT_TRUE(absl::StrContains(in.Seek(-1).ToString(), "negative position"));
  }
}

TEST(InputBuffer, ReadVarint32) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));

  // Generates data.
  std::vector<uint32> data;
  uint32 i = 0;
  for (; i < (1U << 10); i += 1) data.push_back(i);
  for (; i < (1U << 15); i += 5) data.push_back(i);
  for (; i < (1U << 31); i += 132817) data.push_back(i);
  data.push_back(std::numeric_limits<uint32>::max());

  // Writes the varints.
  {
    std::unique_ptr<WritableFile> file;
    TF_CHECK_OK(env->NewWritableFile(fname, &file));
    string varint;
    for (uint32 number : data) {
      varint.clear();
      core::PutVarint32(&varint, number);
      TF_CHECK_OK(file->Append(StringPiece(varint)));
    }
  }

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessFile> file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    io::InputBuffer in(file.get(), buf_size);
    uint32 result = 0;

    for (uint32 expected : data) {
      TF_ASSERT_OK(in.ReadVarint32(&result));
      EXPECT_EQ(expected, result);
    }
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadVarint32(&result)));
  }
}

TEST(InputBuffer, ReadVarint64) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));

  // Generates data.
  std::vector<uint64> data;
  uint64 i = 0;
  for (; i < (1U << 10); i += 1) data.push_back(i);
  for (; i < (1U << 15); i += 5) data.push_back(i);
  for (; i < (1U << 31); i += 164817) data.push_back(i);
  for (; i < (1ULL << 63); i += 16481797854795663UL) data.push_back(i);
  data.push_back(std::numeric_limits<uint64>::max());

  // Writes the varints.
  {
    std::unique_ptr<WritableFile> file;
    TF_CHECK_OK(env->NewWritableFile(fname, &file));
    string varint;
    for (uint64 number : data) {
      varint.clear();
      core::PutVarint64(&varint, number);
      TF_CHECK_OK(file->Append(StringPiece(varint)));
    }
  }

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessFile> file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    io::InputBuffer in(file.get(), buf_size);
    uint64 result = 0;

    for (uint64 expected : data) {
      TF_ASSERT_OK(in.ReadVarint64(&result));
      EXPECT_EQ(expected, result);
    }
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadVarint64(&result)));
  }
}

TEST(InputBuffer, Hint) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessFile> file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    string read;
    io::InputBuffer in(file.get(), buf_size);

    TF_CHECK_OK(in.ReadNBytes(3, &read));
    EXPECT_EQ(read, "012");
    TF_CHECK_OK(in.Hint(4));
    TF_CHECK_OK(in.ReadNBytes(3, &read));
    EXPECT_EQ(read, "345");
    TF_CHECK_OK(in.Hint(1));
    TF_CHECK_OK(in.ReadNBytes(3, &read));
    EXPECT_EQ(read, "678");

    TF_CHECK_OK(in.Seek(0));
    TF_CHECK_OK(in.Hint(7));
    TF_CHECK_OK(in.ReadNBytes(3, &read));
    EXPECT_EQ(read, "012");
    TF_CHECK_OK(in.ReadNBytes(4, &read));
    EXPECT_EQ(read, "3456");

    TF_CHECK_OK(in.Hint(2));
    TF_CHECK_OK(in.Seek(4));
    TF_CHECK_OK(in.ReadNBytes(4, &read));
    EXPECT_EQ(read, "4567");

    TF_CHECK_OK(in.Seek(0));
    TF_CHECK_OK(in.Hint(1 << 25));

    TF_CHECK_OK(in.Seek(1 << 25));
    EXPECT_TRUE(errors::IsOutOfRange(in.Hint(1)));

    EXPECT_TRUE(errors::IsInvalidArgument(in.Hint(-1)));
  }
}

}  // namespace
}  // namespace tensorflow
