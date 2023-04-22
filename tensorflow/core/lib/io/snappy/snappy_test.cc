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

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/snappy/snappy_inputbuffer.h"
#include "tensorflow/core/lib/io/snappy/snappy_inputstream.h"
#include "tensorflow/core/lib/io/snappy/snappy_outputbuffer.h"

namespace tensorflow {

static void CheckPrefixSuffix(const string& str, const string& prefix,
                              const string& suffix) {
  CHECK_GE(str.size(), prefix.size());
  CHECK_GE(str.size(), suffix.size());
  CHECK_EQ(str.substr(0, prefix.length()), prefix);
  CHECK_EQ(str.substr(str.length() - suffix.length()), suffix);
}

static string GetRecord() {
  static const string lorem_ipsum =
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
      " Fusce vehicula tincidunt libero sit amet ultrices. Vestibulum non "
      "felis augue. Duis vitae augue id lectus lacinia congue et ut purus. "
      "Donec auctor, nisl at dapibus volutpat, diam ante lacinia dolor, vel"
      "dignissim lacus nisi sed purus. Duis fringilla nunc ac lacus sagittis"
      " efficitur. Praesent tincidunt egestas eros, eu vehicula urna ultrices"
      " et. Aliquam erat volutpat. Maecenas vehicula risus consequat risus"
      " dictum, luctus tincidunt nibh imperdiet. Aenean bibendum ac erat"
      " cursus scelerisque. Cras lacinia in enim dapibus iaculis. Nunc porta"
      " felis lectus, ac tincidunt massa pharetra quis. Fusce feugiat dolor"
      " vel ligula rutrum egestas. Donec vulputate quam eros, et commodo"
      " purus lobortis sed.";
  return lorem_ipsum;
}

static string GenTestString(int copies = 1) {
  string result = "";
  for (int i = 0; i < copies; i++) {
    result += GetRecord();
  }
  return result;
}

Status TestMultipleWritesWriteFile(size_t compress_input_buf_size,
                                   size_t compress_output_buf_size,
                                   int num_writes, bool with_flush,
                                   int num_copies, bool corrupt_compressed_file,
                                   string& fname, string& data,
                                   string& expected_result) {
  Env* env = Env::Default();

  fname = testing::TmpDir() + "/snappy_buffers_test";
  data = GenTestString(num_copies);
  std::unique_ptr<WritableFile> file_writer;

  TF_RETURN_IF_ERROR(env->NewWritableFile(fname, &file_writer));
  io::SnappyOutputBuffer out(file_writer.get(), compress_input_buf_size,
                             compress_output_buf_size);

  for (int i = 0; i < num_writes; i++) {
    TF_RETURN_IF_ERROR(out.Write(StringPiece(data)));
    if (with_flush) {
      TF_RETURN_IF_ERROR(out.Flush());
    }
    strings::StrAppend(&expected_result, data);
  }
  TF_RETURN_IF_ERROR(out.Flush());
  TF_RETURN_IF_ERROR(file_writer->Flush());
  TF_RETURN_IF_ERROR(file_writer->Close());

  if (corrupt_compressed_file) {
    string corrupt_fname = testing::TmpDir() + "/snappy_buffers_test_corrupt";
    std::unique_ptr<WritableFile> corrupt_file_writer;
    TF_RETURN_IF_ERROR(
        env->NewWritableFile(corrupt_fname, &corrupt_file_writer));

    std::unique_ptr<RandomAccessFile> file_reader;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(fname, &file_reader));

    StringPiece data;
    size_t file_pos = 0;
    size_t bytes_to_read = 256;
    char* scratch = new char[bytes_to_read];
    char* buffer = new char[bytes_to_read];
    size_t buffer_size = 0;

    while ((file_reader->Read(file_pos, bytes_to_read, &data, scratch)).ok()) {
      file_pos += data.size();
      TF_CHECK_OK(
          corrupt_file_writer->Append(StringPiece(buffer, buffer_size)));
      memcpy(buffer, data.data(), data.size());
      buffer_size = data.size();
    }

    // Drop the last byte. File is now corrupt.
    TF_CHECK_OK(
        corrupt_file_writer->Append(StringPiece(buffer, buffer_size - 1)));
    TF_CHECK_OK(corrupt_file_writer->Flush());
    TF_CHECK_OK(corrupt_file_writer->Close());
    delete[] scratch;
    delete[] buffer;
    fname = corrupt_fname;
  }

  return Status::OK();
}

Status TestMultipleWrites(size_t compress_input_buf_size,
                          size_t compress_output_buf_size,
                          size_t uncompress_input_buf_size,
                          size_t uncompress_output_buf_size, int num_writes = 1,
                          bool with_flush = false, int num_copies = 1,
                          bool corrupt_compressed_file = false) {
  Env* env = Env::Default();

  string expected_result;
  string fname;
  string data;

  TF_RETURN_IF_ERROR(TestMultipleWritesWriteFile(
      compress_input_buf_size, compress_output_buf_size, num_writes, with_flush,
      num_copies, corrupt_compressed_file, fname, data, expected_result));

  std::unique_ptr<RandomAccessFile> file_reader;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(fname, &file_reader));
  io::SnappyInputBuffer in(file_reader.get(), uncompress_input_buf_size,
                           uncompress_output_buf_size);

  // Run the test twice, resetting the stream after the first attempt.
  for (int attempt = 0; attempt < 2; ++attempt) {
    string actual_result;
    for (int i = 0; i < num_writes; i++) {
      tstring decompressed_output;
      TF_RETURN_IF_ERROR(in.ReadNBytes(data.size(), &decompressed_output));
      strings::StrAppend(&actual_result, decompressed_output);
    }

    if (actual_result.compare(expected_result)) {
      return errors::DataLoss("Actual and expected results don't match.");
    }
    TF_RETURN_IF_ERROR(in.Reset());
  }

  return Status::OK();
}

Status TestMultipleWritesInputStream(
    size_t compress_input_buf_size, size_t compress_output_buf_size,
    size_t uncompress_input_buf_size, size_t uncompress_output_buf_size,
    int num_writes = 1, bool with_flush = false, int num_copies = 1,
    bool corrupt_compressed_file = false) {
  Env* env = Env::Default();

  string expected_result;
  string fname;
  string data;

  TF_RETURN_IF_ERROR(TestMultipleWritesWriteFile(
      compress_input_buf_size, compress_output_buf_size, num_writes, with_flush,
      num_copies, corrupt_compressed_file, fname, data, expected_result));

  std::unique_ptr<RandomAccessFile> file_reader;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(fname, &file_reader));
  io::RandomAccessInputStream random_input_stream(file_reader.get(), false);
  io::SnappyInputStream snappy_input_stream(&random_input_stream,
                                            uncompress_output_buf_size);

  for (int attempt = 0; attempt < 2; ++attempt) {
    string actual_result;
    for (int i = 0; i < num_writes; ++i) {
      tstring decompressed_output;
      TF_RETURN_IF_ERROR(
          snappy_input_stream.ReadNBytes(data.size(), &decompressed_output));
      strings::StrAppend(&actual_result, decompressed_output);
    }

    if (actual_result.compare(expected_result)) {
      return errors::DataLoss("Actual and expected results don't match.");
    }
    TF_RETURN_IF_ERROR(snappy_input_stream.Reset());
  }
  return Status::OK();
}

void TestTellWriteFile(size_t compress_input_buf_size,
                       size_t compress_output_buf_size,
                       size_t uncompress_input_buf_size,
                       size_t uncompress_output_buf_size, int num_copies,
                       string& fname, string& data) {
  Env* env = Env::Default();
  fname = testing::TmpDir() + "/snappy_buffers_test";
  data = GenTestString(num_copies);

  // Write the compressed file.
  std::unique_ptr<WritableFile> file_writer;
  TF_CHECK_OK(env->NewWritableFile(fname, &file_writer));
  io::SnappyOutputBuffer out(file_writer.get(), compress_input_buf_size,
                             compress_output_buf_size);
  TF_CHECK_OK(out.Write(StringPiece(data)));
  TF_CHECK_OK(out.Flush());
  TF_CHECK_OK(file_writer->Flush());
  TF_CHECK_OK(file_writer->Close());
}

void TestTell(size_t compress_input_buf_size, size_t compress_output_buf_size,
              size_t uncompress_input_buf_size,
              size_t uncompress_output_buf_size, int num_copies = 1) {
  Env* env = Env::Default();
  string data;
  string fname;

  TestTellWriteFile(compress_input_buf_size, compress_output_buf_size,
                    uncompress_input_buf_size, uncompress_output_buf_size,
                    num_copies, fname, data);

  tstring first_half(string(data, 0, data.size() / 2));
  tstring bytes_read;
  std::unique_ptr<RandomAccessFile> file_reader;
  TF_CHECK_OK(env->NewRandomAccessFile(fname, &file_reader));
  io::SnappyInputBuffer in(file_reader.get(), uncompress_input_buf_size,
                           uncompress_output_buf_size);

  // Read the first half of the uncompressed file and expect that Tell()
  // returns half the uncompressed length of the file.
  TF_CHECK_OK(in.ReadNBytes(first_half.size(), &bytes_read));
  EXPECT_EQ(in.Tell(), first_half.size());
  EXPECT_EQ(bytes_read, first_half);

  // Read the remaining half of the uncompressed file and expect that
  // Tell() points past the end of file.
  tstring second_half;
  TF_CHECK_OK(in.ReadNBytes(data.size() - first_half.size(), &second_half));
  EXPECT_EQ(in.Tell(), data.size());
  bytes_read.append(second_half);

  // Expect that the file is correctly read.
  EXPECT_EQ(bytes_read, data);
}

void TestTellInputStream(size_t compress_input_buf_size,
                         size_t compress_output_buf_size,
                         size_t uncompress_input_buf_size,
                         size_t uncompress_output_buf_size,
                         int num_copies = 1) {
  Env* env = Env::Default();
  string data;
  string fname;

  TestTellWriteFile(compress_input_buf_size, compress_output_buf_size,
                    uncompress_input_buf_size, uncompress_output_buf_size,
                    num_copies, fname, data);

  tstring first_half(string(data, 0, data.size() / 2));
  tstring bytes_read;
  std::unique_ptr<RandomAccessFile> file_reader;
  TF_CHECK_OK(env->NewRandomAccessFile(fname, &file_reader));
  io::RandomAccessInputStream random_input_stream(file_reader.get(), false);
  io::SnappyInputStream in(&random_input_stream, uncompress_output_buf_size);

  // Read the first half of the uncompressed file and expect that Tell()
  // returns half the uncompressed length of the file.
  TF_CHECK_OK(in.ReadNBytes(first_half.size(), &bytes_read));
  EXPECT_EQ(in.Tell(), first_half.size());
  EXPECT_EQ(bytes_read, first_half);

  // Read the remaining half of the uncompressed file and expect that
  // Tell() points past the end of file.
  tstring second_half;
  TF_CHECK_OK(in.ReadNBytes(data.size() - first_half.size(), &second_half));
  EXPECT_EQ(in.Tell(), data.size());
  bytes_read.append(second_half);

  // Expect that the file is correctly read.
  EXPECT_EQ(bytes_read, data);
}

static bool SnappyCompressionSupported() {
  string out;
  StringPiece in = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  return port::Snappy_Compress(in.data(), in.size(), &out);
}

TEST(SnappyBuffers, MultipleWritesWithoutFlush) {
  if (!SnappyCompressionSupported()) {
    fprintf(stderr, "Snappy disabled. Skipping test\n");
    return;
  }
  TF_CHECK_OK(TestMultipleWrites(10000, 10000, 10000, 10000, 2));
  TF_CHECK_OK(TestMultipleWritesInputStream(10000, 10000, 10000, 10000, 2));
}

TEST(SnappyBuffers, MultipleWriteCallsWithFlush) {
  if (!SnappyCompressionSupported()) {
    fprintf(stderr, "skipping compression tests\n");
    return;
  }
  TF_CHECK_OK(TestMultipleWrites(10000, 10000, 10000, 10000, 2, true));
  TF_CHECK_OK(
      TestMultipleWritesInputStream(10000, 10000, 10000, 10000, 2, true));
}

TEST(SnappyBuffers, SmallUncompressInputBuffer) {
  if (!SnappyCompressionSupported()) {
    fprintf(stderr, "skipping compression tests\n");
    return;
  }
  Status status = TestMultipleWrites(10000, 10000, 10, 10000, 2, true);
  CHECK_EQ(status.code(), error::Code::RESOURCE_EXHAUSTED);
  CheckPrefixSuffix(
      status.error_message(),
      "Input buffer(size: 10 bytes) too small. Should be larger than ",
      " bytes.");
}

TEST(SnappyBuffers, SmallUncompressInputStream) {
  if (!SnappyCompressionSupported()) {
    fprintf(stderr, "skipping compression tests\n");
    return;
  }
  CHECK_EQ(TestMultipleWritesInputStream(10000, 10000, 10000, 10, 2, true),
           errors::ResourceExhausted(
               "Output buffer(size: 10 bytes) too small. ",
               "Should be larger than ", GetRecord().size(), " bytes."));
}

TEST(SnappyBuffers, CorruptBlock) {
  if (!SnappyCompressionSupported()) {
    fprintf(stderr, "skipping compression tests\n");
    return;
  }
  Status status =
      TestMultipleWrites(10000, 10000, 700, 10000, 2, true, 1, true);
  CHECK_EQ(status.code(), error::Code::DATA_LOSS);
  CheckPrefixSuffix(status.error_message(), "Failed to read ",
                    " bytes from file. Possible data corruption.");
}

TEST(SnappyBuffers, CorruptBlockInputStream) {
  if (!SnappyCompressionSupported()) {
    fprintf(stderr, "skipping compression tests\n");
    return;
  }
  Status status =
      TestMultipleWritesInputStream(10000, 10000, 700, 10000, 2, true, 1, true);
  CHECK_EQ(status.code(), error::Code::DATA_LOSS);
  CheckPrefixSuffix(status.error_message(), "Failed to read ",
                    " bytes from file. Possible data corruption.");
}

TEST(SnappyBuffers, CorruptBlockLargeInputBuffer) {
  if (!SnappyCompressionSupported()) {
    fprintf(stderr, "skipping compression tests\n");
    return;
  }
  CHECK_EQ(TestMultipleWrites(10000, 10000, 2000, 10000, 2, true, 1, true),
           errors::OutOfRange("EOF reached"));
}

TEST(SnappyBuffers, CorruptBlockLargeInputStream) {
  if (!SnappyCompressionSupported()) {
    fprintf(stderr, "skipping compression tests\n");
    return;
  }
  Status status = TestMultipleWritesInputStream(10000, 10000, 2000, 10000, 2,
                                                true, 1, true);
  CHECK_EQ(status.code(), error::Code::DATA_LOSS);
  CheckPrefixSuffix(status.error_message(), "Failed to read ",
                    " bytes from file. Possible data corruption.");
}

TEST(SnappyBuffers, Tell) {
  if (!SnappyCompressionSupported()) {
    fprintf(stderr, "skipping compression tests\n");
    return;
  }
  TestTell(10000, 10000, 2000, 10000, 2);
}

TEST(SnappyBuffers, TellInputStream) {
  if (!SnappyCompressionSupported()) {
    fprintf(stderr, "skipping compression tests\n");
    return;
  }
  TestTellInputStream(10000, 10000, 2000, 10000, 2);
}

}  // namespace tensorflow
