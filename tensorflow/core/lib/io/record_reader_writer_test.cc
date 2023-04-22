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

#include <zlib.h>
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

namespace {

io::RecordReaderOptions GetMatchingReaderOptions(
    const io::RecordWriterOptions& options) {
  if (options.compression_type == io::RecordWriterOptions::ZLIB_COMPRESSION) {
    return io::RecordReaderOptions::CreateRecordReaderOptions("ZLIB");
  }
  return io::RecordReaderOptions::CreateRecordReaderOptions("");
}

uint64 GetFileSize(const string& fname) {
  Env* env = Env::Default();
  uint64 fsize;
  TF_CHECK_OK(env->GetFileSize(fname, &fsize));
  return fsize;
}

void VerifyFlush(const io::RecordWriterOptions& options) {
  std::vector<string> records = {
      "abcdefghijklmnopqrstuvwxyz",
      "ZYXWVUTSRQPONMLKJIHGFEDCBA0123456789!@#$%^&*()",
      "G5SyohOL9UmXofSOOwWDrv9hoLLMYPJbG9r38t3uBRcHxHj2PdKcPDuZmKW62RIY",
      "aaaaaaaaaaaaaaaaaaaaaaaaaa",
  };

  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/record_reader_writer_flush_test";

  std::unique_ptr<WritableFile> file;
  TF_CHECK_OK(env->NewWritableFile(fname, &file));
  io::RecordWriter writer(file.get(), options);

  std::unique_ptr<RandomAccessFile> read_file;
  TF_CHECK_OK(env->NewRandomAccessFile(fname, &read_file));
  io::RecordReaderOptions read_options = GetMatchingReaderOptions(options);
  io::RecordReader reader(read_file.get(), read_options);

  EXPECT_EQ(GetFileSize(fname), 0);
  for (size_t i = 0; i < records.size(); i++) {
    uint64 start_size = GetFileSize(fname);

    // Write a new record.
    TF_EXPECT_OK(writer.WriteRecord(records[i]));
    TF_CHECK_OK(writer.Flush());
    TF_CHECK_OK(file->Flush());

    // Verify that file size has changed after file flush.
    uint64 new_size = GetFileSize(fname);
    EXPECT_GT(new_size, start_size);

    // Verify that file has all records written so far and no more.
    uint64 offset = 0;
    tstring record;
    for (size_t j = 0; j <= i; j++) {
      // Check that j'th record is written correctly.
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ(record, records[j]);
    }

    // Verify that file has no more records.
    CHECK_EQ(reader.ReadRecord(&offset, &record).code(), error::OUT_OF_RANGE);
  }
}

}  // namespace

TEST(RecordReaderWriterTest, TestFlush) {
  io::RecordWriterOptions options;
  VerifyFlush(options);
}

TEST(RecordReaderWriterTest, TestZlibSyncFlush) {
  io::RecordWriterOptions options;
  options.compression_type = io::RecordWriterOptions::ZLIB_COMPRESSION;
  // The default flush_mode is Z_NO_FLUSH and only writes to the file when the
  // buffer is full or the file is closed, which makes testing harder.
  // By using Z_SYNC_FLUSH the test can verify Flush does write out records of
  // approximately the right size at the right times.
  options.zlib_options.flush_mode = Z_SYNC_FLUSH;

  VerifyFlush(options);
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
      TF_EXPECT_OK(writer.WriteRecord("abc"));
      TF_EXPECT_OK(writer.WriteRecord("defg"));
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
      tstring record;
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ("abc", record);
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ("defg", record);

      io::RecordReader::Metadata md;
      TF_ASSERT_OK(reader.GetMetadata(&md));
      EXPECT_EQ(2, md.stats.entries);
      EXPECT_EQ(7, md.stats.data_size);
      // Two entries have 16 bytes of header/footer each.
      EXPECT_EQ(39, md.stats.file_size);
    }
  }
}

TEST(RecordReaderWriterTest, TestSkipBasic) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/record_reader_writer_skip_basic_test";

  for (auto buf_size : BufferSizes()) {
    {
      std::unique_ptr<WritableFile> file;
      TF_CHECK_OK(env->NewWritableFile(fname, &file));

      io::RecordWriterOptions options;
      options.zlib_options.output_buffer_size = buf_size;
      io::RecordWriter writer(file.get(), options);
      TF_EXPECT_OK(writer.WriteRecord("abc"));
      TF_EXPECT_OK(writer.WriteRecord("defg"));
      TF_EXPECT_OK(writer.WriteRecord("hij"));
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
      int num_skipped;
      tstring record;
      TF_CHECK_OK(reader.SkipRecords(&offset, 2, &num_skipped));
      EXPECT_EQ(2, num_skipped);
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ("hij", record);
    }
  }
}

TEST(RecordReaderWriterTest, TestSkipOutOfRange) {
  Env* env = Env::Default();
  string fname =
      testing::TmpDir() + "/record_reader_writer_skip_out_of_range_test";

  for (auto buf_size : BufferSizes()) {
    {
      std::unique_ptr<WritableFile> file;
      TF_CHECK_OK(env->NewWritableFile(fname, &file));

      io::RecordWriterOptions options;
      options.zlib_options.output_buffer_size = buf_size;
      io::RecordWriter writer(file.get(), options);
      TF_EXPECT_OK(writer.WriteRecord("abc"));
      TF_EXPECT_OK(writer.WriteRecord("defg"));
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
      int num_skipped;
      tstring record;
      Status s = reader.SkipRecords(&offset, 3, &num_skipped);
      EXPECT_EQ(2, num_skipped);
      EXPECT_EQ(error::OUT_OF_RANGE, s.code());
    }
  }
}

TEST(RecordReaderWriterTest, TestSnappy) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/record_reader_writer_snappy_test";

  for (auto buf_size : BufferSizes()) {
    // Snappy compression needs output buffer size > 1.
    if (buf_size == 1) continue;
    {
      std::unique_ptr<WritableFile> file;
      TF_CHECK_OK(env->NewWritableFile(fname, &file));

      io::RecordWriterOptions options;
      options.compression_type = io::RecordWriterOptions::SNAPPY_COMPRESSION;
      options.zlib_options.output_buffer_size = buf_size;
      io::RecordWriter writer(file.get(), options);
      TF_EXPECT_OK(writer.WriteRecord("abc"));
      TF_EXPECT_OK(writer.WriteRecord("defg"));
      TF_CHECK_OK(writer.Flush());
    }

    {
      std::unique_ptr<RandomAccessFile> read_file;
      // Read it back with the RecordReader.
      TF_CHECK_OK(env->NewRandomAccessFile(fname, &read_file));
      io::RecordReaderOptions options;
      options.compression_type = io::RecordReaderOptions::SNAPPY_COMPRESSION;
      options.zlib_options.input_buffer_size = buf_size;
      io::RecordReader reader(read_file.get(), options);
      uint64 offset = 0;
      tstring record;
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
      TF_EXPECT_OK(writer.WriteRecord("abc"));
      TF_EXPECT_OK(writer.WriteRecord("defg"));
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
      tstring record;
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ("abc", record);
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ("defg", record);
    }
  }
}

TEST(RecordReaderWriterTest, TestUseAfterClose) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/record_reader_writer_flush_close_test";

  {
    std::unique_ptr<WritableFile> file;
    TF_CHECK_OK(env->NewWritableFile(fname, &file));

    io::RecordWriterOptions options;
    options.compression_type = io::RecordWriterOptions::ZLIB_COMPRESSION;
    io::RecordWriter writer(file.get(), options);
    TF_EXPECT_OK(writer.WriteRecord("abc"));
    TF_CHECK_OK(writer.Flush());
    TF_CHECK_OK(writer.Close());

    CHECK_EQ(writer.WriteRecord("abc").code(), error::FAILED_PRECONDITION);
    CHECK_EQ(writer.Flush().code(), error::FAILED_PRECONDITION);

    // Second call to close is fine.
    TF_CHECK_OK(writer.Close());
  }
}

}  // namespace tensorflow
