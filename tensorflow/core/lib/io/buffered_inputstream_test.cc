/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/io/buffered_inputstream.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace io {
namespace {

static std::vector<int> BufferSizes() {
  return {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   11,
          12, 13, 14, 15, 16, 17, 18, 19, 20, 65536};
}

TEST(BufferedInputStream, ReadLine_Empty) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/buffered_inputstream_test";
  TF_ASSERT_OK(WriteStringToFile(env, fname, ""));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    string line;
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(BufferedInputStream, ReadLine1) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/buffered_inputstream_test";
  TF_ASSERT_OK(
      WriteStringToFile(env, fname, "line one\nline two\nline three\n"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    string line;
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line one");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line three");
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
    // A second call should also return end of file
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(BufferedInputStream, ReadLine_NoTrailingNewLine) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/buffered_inputstream_test";
  TF_ASSERT_OK(WriteStringToFile(env, fname, "line one\nline two\nline three"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    string line;
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line one");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line three");
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
    // A second call should also return end of file
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(BufferedInputStream, ReadLine_EmptyLines) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/buffered_inputstream_test";
  TF_ASSERT_OK(
      WriteStringToFile(env, fname, "line one\n\n\nline two\nline three"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    string line;
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line one");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line three");
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
    // A second call should also return end of file
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(BufferedInputStream, ReadLine_CRLF) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/buffered_inputstream_test";
  TF_ASSERT_OK(WriteStringToFile(env, fname,
                                 "line one\r\n\r\n\r\nline two\r\nline three"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    string line;
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line one");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line three");
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
    // A second call should also return end of file
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(BufferedInputStream, ReadNBytes) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/buffer_test";
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    string read;
    BufferedInputStream in(input_stream.get(), buf_size);
    EXPECT_EQ(0, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(3, &read));
    EXPECT_EQ(read, "012");
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(4, &read));
    EXPECT_EQ(read, "3456");
    EXPECT_EQ(7, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(7, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "789");
    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
  }
}

TEST(BufferedInputStream, SkipNBytes) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/buffered_inputstream_test";
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    string read;
    BufferedInputStream in(input_stream.get(), buf_size);
    EXPECT_EQ(0, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(3));
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(0));
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(2, &read));
    EXPECT_EQ(read, "34");
    EXPECT_EQ(5, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(0));
    EXPECT_EQ(5, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(2));
    EXPECT_EQ(7, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(1, &read));
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

TEST(BufferedInputStream, ReadNBytesRandomAccessFile) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/buffer_test";
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    string read;
    BufferedInputStream in(file.get(), buf_size);
    EXPECT_EQ(0, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(3, &read));
    EXPECT_EQ(read, "012");
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(4, &read));
    EXPECT_EQ(read, "3456");
    EXPECT_EQ(7, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(7, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "789");
    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
  }
}

TEST(BufferedInputStream, SkipNBytesRandomAccessFile) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/buffered_inputstream_test";
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    string read;
    BufferedInputStream in(file.get(), buf_size);
    EXPECT_EQ(0, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(3));
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(0));
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(2, &read));
    EXPECT_EQ(read, "34");
    EXPECT_EQ(5, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(0));
    EXPECT_EQ(5, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(2));
    EXPECT_EQ(7, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(1, &read));
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

TEST(BufferedInputStream, Seek) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/buffered_inputstream_test";
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    string read;
    BufferedInputStream in(input_stream.get(), buf_size);

    // Seek forward
    TF_ASSERT_OK(in.Seek(3));
    EXPECT_EQ(3, in.Tell());

    // Read 4 bytes
    TF_ASSERT_OK(in.ReadNBytes(4, &read));
    EXPECT_EQ(read, "3456");
    EXPECT_EQ(7, in.Tell());

    // Seek backwards
    TF_ASSERT_OK(in.Seek(1));
    TF_ASSERT_OK(in.ReadNBytes(4, &read));
    EXPECT_EQ(read, "1234");
    EXPECT_EQ(5, in.Tell());
  }
}

TEST(BufferedInputStream, ReadAll_Empty) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/buffered_inputstream_test";
  const string expected = "";
  TF_ASSERT_OK(WriteStringToFile(env, fname, expected));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    RandomAccessInputStream input_stream(file.get());
    string read;
    BufferedInputStream in(&input_stream, buf_size);
    string contents;
    TF_ASSERT_OK(in.ReadAll(&contents));
    EXPECT_EQ(expected, contents);
  }
}

TEST(BufferedInputStream, ReadAll_Text) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/buffered_inputstream_test";
  const string expected = "line one\nline two\nline three";
  TF_ASSERT_OK(WriteStringToFile(env, fname, expected));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    RandomAccessInputStream input_stream(file.get());
    string read;
    BufferedInputStream in(&input_stream, buf_size);
    string contents;
    TF_ASSERT_OK(in.ReadAll(&contents));
    EXPECT_EQ(expected, contents);
  }
}

void BM_BufferedReaderSmallReads(const int iters, const int buff_size,
                                 const int file_size) {
  testing::StopTiming();
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/buffered_inputstream_test";

  const string file_elem = "0123456789";
  std::unique_ptr<WritableFile> write_file;
  TF_ASSERT_OK(env->NewWritableFile(fname, &write_file));
  for (int i = 0; i < file_size; ++i) {
    TF_ASSERT_OK(write_file->Append(file_elem));
  }
  TF_ASSERT_OK(write_file->Close());

  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  string result;
  testing::StartTiming();

  for (int itr = 0; itr < iters; ++itr) {
    BufferedInputStream in(file.get(), buff_size);
    for (int64 i = 0; i < 10 * file_size; ++i) {
      TF_ASSERT_OK(in.ReadNBytes(1, &result))
          << "i: " << i << " itr: " << itr << " buff_size: " << buff_size
          << " file size: " << file_size;
    }
  }
}
BENCHMARK(BM_BufferedReaderSmallReads)
    ->ArgPair(1, 5)
    ->ArgPair(1, 1024)
    ->ArgPair(10, 5)
    ->ArgPair(10, 1024)
    ->ArgPair(1024, 1024)
    ->ArgPair(1024 * 1024, 1024)
    ->ArgPair(1024 * 1024, 1024 * 1024)
    ->ArgPair(256 * 1024 * 1024, 1024);

}  // anonymous namespace
}  // namespace io
}  // namespace tensorflow
