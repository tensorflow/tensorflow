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

#include "absl/strings/match.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/lib/io/random_inputstream.h"
#include "xla/tsl/lib/io/zlib_compression_options.h"
#include "xla/tsl/lib/io/zlib_inputstream.h"
#include "xla/tsl/lib/io/zlib_outputbuffer.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/strcat.h"

namespace tsl {
namespace io {

static std::vector<int> InputBufferSizes() {
  return {10, 100, 200, 500, 1000, 10000};
}

static std::vector<int> OutputBufferSizes() { return {100, 200, 500, 1000}; }

static std::vector<int> NumCopies() { return {1, 50, 500}; }

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

typedef io::ZlibCompressionOptions CompressionOptions;

void TestAllCombinations(CompressionOptions input_options,
                         CompressionOptions output_options) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  for (auto file_size : NumCopies()) {
    // Write to compressed file
    string data = GenTestString(file_size);
    for (auto input_buf_size : InputBufferSizes()) {
      for (auto output_buf_size : OutputBufferSizes()) {
        std::unique_ptr<WritableFile> file_writer;
        TF_ASSERT_OK(env->NewWritableFile(fname, &file_writer));
        tstring result;

        ZlibOutputBuffer out(file_writer.get(), input_buf_size, output_buf_size,
                             output_options);
        TF_ASSERT_OK(out.Init());

        TF_ASSERT_OK(out.Append(absl::string_view(data)));
        TF_ASSERT_OK(out.Close());
        TF_ASSERT_OK(file_writer->Flush());
        TF_ASSERT_OK(file_writer->Close());

        std::unique_ptr<RandomAccessFile> file_reader;
        TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
        std::unique_ptr<RandomAccessInputStream> input_stream(
            new RandomAccessInputStream(file_reader.get()));
        ZlibInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                           input_options);
        TF_ASSERT_OK(in.ReadNBytes(data.size(), &result));
        EXPECT_EQ(result, data);
      }
    }
  }
}

TEST(ZlibBuffers, DefaultOptions) {
  TestAllCombinations(CompressionOptions::DEFAULT(),
                      CompressionOptions::DEFAULT());
}

TEST(ZlibBuffers, RawDeflate) {
  TestAllCombinations(CompressionOptions::RAW(), CompressionOptions::RAW());
}

TEST(ZlibBuffers, Gzip) {
  TestAllCombinations(CompressionOptions::GZIP(), CompressionOptions::GZIP());
}

void TestMultipleWrites(uint8 input_buf_size, uint8 output_buf_size,
                        int num_writes, bool with_flush = false) {
  Env* env = Env::Default();
  CompressionOptions input_options = CompressionOptions::DEFAULT();
  CompressionOptions output_options = CompressionOptions::DEFAULT();

  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  string data = GenTestString();
  std::unique_ptr<WritableFile> file_writer;
  string actual_result;
  string expected_result;

  TF_ASSERT_OK(env->NewWritableFile(fname, &file_writer));
  ZlibOutputBuffer out(file_writer.get(), input_buf_size, output_buf_size,
                       output_options);
  TF_ASSERT_OK(out.Init());

  for (int i = 0; i < num_writes; i++) {
    TF_ASSERT_OK(out.Append(absl::string_view(data)));
    if (with_flush) {
      TF_ASSERT_OK(out.Flush());
    }
    strings::StrAppend(&expected_result, data);
  }
  TF_ASSERT_OK(out.Close());
  TF_ASSERT_OK(file_writer->Flush());
  TF_ASSERT_OK(file_writer->Close());

  std::unique_ptr<RandomAccessFile> file_reader;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
  std::unique_ptr<RandomAccessInputStream> input_stream(
      new RandomAccessInputStream(file_reader.get()));
  ZlibInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                     input_options);

  for (int i = 0; i < num_writes; i++) {
    tstring decompressed_output;
    TF_ASSERT_OK(in.ReadNBytes(data.size(), &decompressed_output));
    strings::StrAppend(&actual_result, decompressed_output);
  }

  EXPECT_EQ(actual_result, expected_result);
}

TEST(ZlibBuffers, MultipleWritesWithoutFlush) {
  TestMultipleWrites(200, 200, 10);
}

TEST(ZlibBuffers, MultipleWriteCallsWithFlush) {
  TestMultipleWrites(200, 200, 10, true);
}

TEST(ZlibInputStream, FailsToReadIfWindowBitsAreIncompatible) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  CompressionOptions output_options = CompressionOptions::DEFAULT();
  CompressionOptions input_options = CompressionOptions::DEFAULT();
  int input_buf_size = 200, output_buf_size = 200;
  output_options.window_bits = MAX_WBITS;
  // inflate() has smaller history buffer.
  input_options.window_bits = output_options.window_bits - 1;

  string data = GenTestString(10);
  std::unique_ptr<WritableFile> file_writer;
  TF_ASSERT_OK(env->NewWritableFile(fname, &file_writer));
  tstring result;
  ZlibOutputBuffer out(file_writer.get(), input_buf_size, output_buf_size,
                       output_options);
  TF_ASSERT_OK(out.Init());

  TF_ASSERT_OK(out.Append(absl::string_view(data)));
  TF_ASSERT_OK(out.Close());
  TF_ASSERT_OK(file_writer->Flush());
  TF_ASSERT_OK(file_writer->Close());

  std::unique_ptr<RandomAccessFile> file_reader;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
  std::unique_ptr<RandomAccessInputStream> input_stream(
      new RandomAccessInputStream(file_reader.get()));
  ZlibInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                     input_options);
  absl::Status read_status = in.ReadNBytes(data.size(), &result);
  CHECK_EQ(read_status.code(), error::DATA_LOSS);
  CHECK(absl::StrContains(read_status.message(), "inflate() failed"));
}

void WriteCompressedFile(Env* env, const string& fname, int input_buf_size,
                         int output_buf_size,
                         const CompressionOptions& output_options,
                         const string& data) {
  std::unique_ptr<WritableFile> file_writer;
  TF_ASSERT_OK(env->NewWritableFile(fname, &file_writer));

  ZlibOutputBuffer out(file_writer.get(), input_buf_size, output_buf_size,
                       output_options);
  TF_ASSERT_OK(out.Init());

  TF_ASSERT_OK(out.Append(absl::string_view(data)));
  TF_ASSERT_OK(out.Close());
  TF_ASSERT_OK(file_writer->Flush());
  TF_ASSERT_OK(file_writer->Close());
}

void TestTell(CompressionOptions input_options,
              CompressionOptions output_options) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  for (auto file_size : NumCopies()) {
    string data = GenTestString(file_size);
    for (auto input_buf_size : InputBufferSizes()) {
      for (auto output_buf_size : OutputBufferSizes()) {
        // Write the compressed file.
        WriteCompressedFile(env, fname, input_buf_size, output_buf_size,
                            output_options, data);

        // Boiler-plate to set up ZlibInputStream.
        std::unique_ptr<RandomAccessFile> file_reader;
        TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
        std::unique_ptr<RandomAccessInputStream> input_stream(
            new RandomAccessInputStream(file_reader.get()));
        ZlibInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                           input_options);

        tstring first_half(string(data, 0, data.size() / 2));
        tstring bytes_read;

        // Read the first half of the uncompressed file and expect that Tell()
        // returns half the uncompressed length of the file.
        TF_ASSERT_OK(in.ReadNBytes(first_half.size(), &bytes_read));
        EXPECT_EQ(in.Tell(), first_half.size());
        EXPECT_EQ(bytes_read, first_half);

        // Read the remaining half of the uncompressed file and expect that
        // Tell() points past the end of file.
        tstring second_half;
        TF_ASSERT_OK(
            in.ReadNBytes(data.size() - first_half.size(), &second_half));
        EXPECT_EQ(in.Tell(), data.size());
        bytes_read.append(second_half);

        // Expect that the file is correctly read.
        EXPECT_EQ(bytes_read, data);
      }
    }
  }
}

void TestSkipNBytes(CompressionOptions input_options,
                    CompressionOptions output_options) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  for (auto file_size : NumCopies()) {
    string data = GenTestString(file_size);
    for (auto input_buf_size : InputBufferSizes()) {
      for (auto output_buf_size : OutputBufferSizes()) {
        // Write the compressed file.
        WriteCompressedFile(env, fname, input_buf_size, output_buf_size,
                            output_options, data);

        // Boiler-plate to set up ZlibInputStream.
        std::unique_ptr<RandomAccessFile> file_reader;
        TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
        std::unique_ptr<RandomAccessInputStream> input_stream(
            new RandomAccessInputStream(file_reader.get()));
        ZlibInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                           input_options);

        size_t data_half_size = data.size() / 2;
        string second_half(data, data_half_size, data.size() - data_half_size);

        // Skip past the first half of the file and expect Tell() returns
        // correctly.
        TF_ASSERT_OK(in.SkipNBytes(data_half_size));
        EXPECT_EQ(in.Tell(), data_half_size);

        // Expect that second half is read correctly and Tell() returns past
        // end of file after reading complete file.
        tstring bytes_read;
        TF_ASSERT_OK(in.ReadNBytes(second_half.size(), &bytes_read));
        EXPECT_EQ(bytes_read, second_half);
        EXPECT_EQ(in.Tell(), data.size());
      }
    }
  }
}

void TestSoftErrorOnDecompress(CompressionOptions input_options) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));

  input_options.soft_fail_on_error = true;

  std::unique_ptr<WritableFile> file_writer;
  TF_ASSERT_OK(env->NewWritableFile(fname, &file_writer));
  TF_ASSERT_OK(file_writer->Append("nonsense non-gzip data"));
  TF_ASSERT_OK(file_writer->Flush());
  TF_ASSERT_OK(file_writer->Close());

  // Test `ReadNBytes` returns an error.
  {
    std::unique_ptr<RandomAccessFile> file_reader;
    TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file_reader.get()));
    ZlibInputStream in(input_stream.get(), 100, 100, input_options);

    tstring unused;
    EXPECT_TRUE(errors::IsDataLoss(in.ReadNBytes(5, &unused)));
  }

  // Test `SkipNBytes` returns an error.
  {
    std::unique_ptr<RandomAccessFile> file_reader;
    TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file_reader.get()));
    ZlibInputStream in(input_stream.get(), 100, 100, input_options);

    EXPECT_TRUE(errors::IsDataLoss(in.SkipNBytes(5)));
  }
}

TEST(ZlibInputStream, TellDefaultOptions) {
  TestTell(CompressionOptions::DEFAULT(), CompressionOptions::DEFAULT());
}

TEST(ZlibInputStream, TellRawDeflate) {
  TestTell(CompressionOptions::RAW(), CompressionOptions::RAW());
}

TEST(ZlibInputStream, TellGzip) {
  TestTell(CompressionOptions::GZIP(), CompressionOptions::GZIP());
}

TEST(ZlibInputStream, SkipNBytesDefaultOptions) {
  TestSkipNBytes(CompressionOptions::DEFAULT(), CompressionOptions::DEFAULT());
}

TEST(ZlibInputStream, SkipNBytesRawDeflate) {
  TestSkipNBytes(CompressionOptions::RAW(), CompressionOptions::RAW());
}

TEST(ZlibInputStream, SkipNBytesGzip) {
  TestSkipNBytes(CompressionOptions::GZIP(), CompressionOptions::GZIP());
}

TEST(ZlibInputStream, TestSoftErrorOnDecompressDefaultOptions) {
  TestSoftErrorOnDecompress(CompressionOptions::DEFAULT());
}

TEST(ZlibInputStream, TestSoftErrorOnDecompressRaw) {
  TestSoftErrorOnDecompress(CompressionOptions::RAW());
}

TEST(ZlibInputStream, TestSoftErrorOnDecompressGzip) {
  TestSoftErrorOnDecompress(CompressionOptions::GZIP());
}

}  // namespace io
}  // namespace tsl
