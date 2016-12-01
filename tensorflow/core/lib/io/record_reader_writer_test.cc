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

#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"

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

TEST(RecordReaderWriterTest, TestBasics) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/record_reader_writer_test";

  for (auto buf_size : BufferSizes()) {
    {
      std::unique_ptr<WritableFile> file;
      TF_CHECK_OK(env->NewWritableFile(fname, &file));

      io::RecordWriterOptions options;
      options.zlib_options.output_buffer_size = buf_size;
      io::RecordWriter writer(file.get(), options);
      writer.WriteRecord("abc");
      writer.WriteRecord("defg");
      TF_CHECK_OK(writer.Flush());
    }

    {
      std::unique_ptr<RandomAccessFile> read_file;
      // Read it back with the RecordReader.
      TF_CHECK_OK(env->NewRandomAccessFile(fname, &read_file));
      io::RecordReaderOptions options;
      options.zlib_options.input_buffer_size = buf_size;
      io::RecordReader reader(read_file.get(), options);
      uint64 offset = 0;
      string record;
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ("abc", record);
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ("defg", record);
    }
  }
}

TEST(RecordReaderWriterTest, TestZlib) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/record_reader_writer_zlib_test";

  for (auto buf_size : BufferSizes()) {
    // Zlib compression needs output buffer size > 1.
    if (buf_size == 1) continue;
    {
      std::unique_ptr<WritableFile> file;
      TF_CHECK_OK(env->NewWritableFile(fname, &file));

      io::RecordWriterOptions options;
      options.compression_type = io::RecordWriterOptions::ZLIB_COMPRESSION;
      options.zlib_options.output_buffer_size = buf_size;
      io::RecordWriter writer(file.get(), options);
      writer.WriteRecord("abc");
      writer.WriteRecord("defg");
      TF_CHECK_OK(writer.Flush());
    }

    {
      std::unique_ptr<RandomAccessFile> read_file;
      // Read it back with the RecordReader.
      TF_CHECK_OK(env->NewRandomAccessFile(fname, &read_file));
      io::RecordReaderOptions options;
      options.compression_type = io::RecordReaderOptions::ZLIB_COMPRESSION;
      options.zlib_options.input_buffer_size = buf_size;
      io::RecordReader reader(read_file.get(), options);
      uint64 offset = 0;
      string record;
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ("abc", record);
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ("defg", record);
    }
  }
}

}  // namespace tensorflow
