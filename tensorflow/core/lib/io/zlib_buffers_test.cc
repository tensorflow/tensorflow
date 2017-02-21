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
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
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
  string fname = testing::TmpDir() + "/zlib_buffers_test";
  for (auto file_size : NumCopies()) {
    // Write to compressed file
    string data = GenTestString(file_size);
    for (auto input_buf_size : InputBufferSizes()) {
      for (auto output_buf_size : OutputBufferSizes()) {
        std::unique_ptr<WritableFile> file_writer;
        TF_CHECK_OK(env->NewWritableFile(fname, &file_writer));
        string result;

        ZlibOutputBuffer out(file_writer.get(), input_buf_size, output_buf_size,
                             output_options);
        TF_CHECK_OK(out.Init());

        TF_CHECK_OK(out.Append(StringPiece(data)));
        TF_CHECK_OK(out.Close());
        TF_CHECK_OK(file_writer->Flush());
        TF_CHECK_OK(file_writer->Close());

        std::unique_ptr<RandomAccessFile> file_reader;
        TF_CHECK_OK(env->NewRandomAccessFile(fname, &file_reader));
        std::unique_ptr<RandomAccessInputStream> input_stream(
            new RandomAccessInputStream(file_reader.get()));
        ZlibInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                           input_options);
        TF_EXPECT_OK(in.ReadNBytes(data.size(), &result));
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

  string fname = testing::TmpDir() + "/zlib_buffers_test";
  string data = GenTestString();
  std::unique_ptr<WritableFile> file_writer;
  string actual_result;
  string expected_result;

  TF_CHECK_OK(env->NewWritableFile(fname, &file_writer));
  ZlibOutputBuffer out(file_writer.get(), input_buf_size, output_buf_size,
                       output_options);
  TF_CHECK_OK(out.Init());

  for (int i = 0; i < num_writes; i++) {
    TF_CHECK_OK(out.Append(StringPiece(data)));
    if (with_flush) {
      TF_CHECK_OK(out.Flush());
    }
    strings::StrAppend(&expected_result, data);
  }
  TF_CHECK_OK(out.Close());
  TF_CHECK_OK(file_writer->Flush());
  TF_CHECK_OK(file_writer->Close());

  std::unique_ptr<RandomAccessFile> file_reader;
  TF_CHECK_OK(env->NewRandomAccessFile(fname, &file_reader));
  std::unique_ptr<RandomAccessInputStream> input_stream(
      new RandomAccessInputStream(file_reader.get()));
  ZlibInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                     input_options);

  for (int i = 0; i < num_writes; i++) {
    string decompressed_output;
    TF_EXPECT_OK(in.ReadNBytes(data.size(), &decompressed_output));
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
  string fname = testing::TmpDir() + "/zlib_buffers_test";
  CompressionOptions output_options = CompressionOptions::DEFAULT();
  CompressionOptions input_options = CompressionOptions::DEFAULT();
  int input_buf_size = 200, output_buf_size = 200;
  output_options.window_bits = MAX_WBITS;
  // inflate() has smaller history buffer.
  input_options.window_bits = output_options.window_bits - 1;

  string data = GenTestString(10);
  std::unique_ptr<WritableFile> file_writer;
  TF_CHECK_OK(env->NewWritableFile(fname, &file_writer));
  string result;
  ZlibOutputBuffer out(file_writer.get(), input_buf_size, output_buf_size,
                       output_options);
  TF_CHECK_OK(out.Init());

  TF_CHECK_OK(out.Append(StringPiece(data)));
  TF_CHECK_OK(out.Close());
  TF_CHECK_OK(file_writer->Flush());
  TF_CHECK_OK(file_writer->Close());

  std::unique_ptr<RandomAccessFile> file_reader;
  TF_CHECK_OK(env->NewRandomAccessFile(fname, &file_reader));
  std::unique_ptr<RandomAccessInputStream> input_stream(
      new RandomAccessInputStream(file_reader.get()));
  ZlibInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                     input_options);
  Status read_status = in.ReadNBytes(data.size(), &result);
  CHECK_EQ(read_status.code(), error::DATA_LOSS);
  CHECK(read_status.error_message().find("inflate() failed") != string::npos);
}

}  // namespace io
}  // namespace tensorflow
