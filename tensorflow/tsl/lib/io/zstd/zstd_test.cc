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

#include <zstd.h>

#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/lib/io/random_inputstream.h"
#include "tensorflow/tsl/lib/io/zstd/zstd_compression_options.h"
#include "tensorflow/tsl/lib/io/zstd/zstd_inputstream.h"
#include "tensorflow/tsl/lib/io/zstd/zstd_outputbuffer.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace io {

static std::vector<int> InputBufferSizes() {
  return {10, 100, 200, 500, 1000, 10000, 256 << 10};
}

static std::vector<int> OutputBufferSizes() {
  return {100, 200, 500, 1000, 256 << 10};
}

static std::vector<int> NumCopies() { return {1, 50, 500, 5000}; }

static std::vector<int> NumThreads() { return {1, 2, 4, 8, 16, 32}; };

static std::vector<int> Strategies() { return {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; };

static string GetRecord() {
  static const string el_dorado =
      "Gaily bedight,"
      "A gallant knight,"
      "In sunshine and in shadow, Had journeyed long, Singing a song,"
      "In search of Eldorado."
      "But he grew old— This knight so bold— And o'er his heart a shadow— Fell"
      "as he found No spot of ground That looked like Eldorado."
      "And,"
      "as his strength Failed him at length,"
      "He met a pilgrim shadow— 'Shadow,' said he,"
      "'Where can it be— This land of Eldorado ?'"
      "'Over the Mountains Of the Moon, Down the Valley of the Shadow, Ride,"
      "boldly ride,' The shade replied,— 'If you seek for Eldorado!'"
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut "
      "condimentum, mauris sit amet euismod iaculis, sem dolor maximus turpis, "
      "id cursus magna mauris eu lacus. Vivamus tincidunt ex vitae dolor "
      "mattis sollicitudin. Nunc nec lacinia lacus. Maecenas sapien nulla, "
      "volutpat eu maximus fermentum, sollicitudin id est. Nunc venenatis, "
      "tortor eu pretium dignissim, enim leo elementum ex, nec fermentum ex "
      "ipsum vitae velit. Ut commodo nunc vel nisi fringilla rhoncus. Cras ac "
      "diam sapien. Etiam vel velit nec purus molestie gravida non a sem. "
      "Pellentesque vulputate finibus eros sit amet placerat.";
  return el_dorado;
}

static string GenTestString(int copies = 1) {
  string result = "";
  for (int i = 0; i < copies; i++) {
    result += GetRecord();
  }
  return result;
}

typedef io::ZstdCompressionOptions CompressionOptions;

void WriteCompressedFile(Env* env, const string& fname, int input_buf_size,
                         int output_buf_size,
                         const CompressionOptions& output_options,
                         const string& data) {
  std::unique_ptr<WritableFile> file_writer;
  TF_ASSERT_OK(env->NewWritableFile(fname, &file_writer));

  ZstdOutputBuffer out(file_writer.get(), input_buf_size, output_buf_size,
                       output_options);

  TF_ASSERT_OK(out.Append(StringPiece(data)));
  TF_ASSERT_OK(out.Close());
  TF_ASSERT_OK(file_writer->Flush());
  TF_ASSERT_OK(file_writer->Close());
}

void ReadCompressedFile(Env* env, const string& fname, int input_buf_size,
                        int output_buf_size,
                        const CompressionOptions& input_options,
                        const string& data) {
  tstring result;
  std::unique_ptr<RandomAccessFile> file_reader;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
  std::unique_ptr<RandomAccessInputStream> input_stream(
      new RandomAccessInputStream(file_reader.get()));

  ZstdInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                     input_options);
  TF_ASSERT_OK(in.ReadNBytes(data.size(), &result));
  EXPECT_EQ(result, data);
}

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
        for (auto strategy : Strategies()) {
          output_options.compression_strategy = strategy;
          WriteCompressedFile(env, fname, input_buf_size, output_buf_size,
                              output_options, data);
          ReadCompressedFile(env, fname, input_buf_size, output_buf_size,
                             input_options, data);
        }
      }
    }
  }
}

TEST(ZstdBuffers, DefaultOptions) {
  TestAllCombinations(CompressionOptions::DEFAULT(),
                      CompressionOptions::DEFAULT());
}

void TestCompressionMultiThread() {
  const size_t c_input_buf_size = ZSTD_CStreamInSize();
  const size_t c_output_buf_size = ZSTD_CStreamOutSize();
  const size_t d_input_buf_size = ZSTD_DStreamInSize();
  const size_t d_output_buf_size = ZSTD_DStreamOutSize();

  Env* env = Env::Default();
  string fname;
  CompressionOptions input_options = CompressionOptions::DEFAULT();
  CompressionOptions output_options = CompressionOptions::DEFAULT();

  ASSERT_TRUE(env->LocalTempFilename(&fname));
  for (auto file_size : NumCopies()) {
    for (auto num_threads : NumThreads()) {
      string data = GenTestString(file_size);
      WriteCompressedFile(env, fname, c_input_buf_size, c_output_buf_size,
                          output_options, data);
      ReadCompressedFile(env, fname, d_input_buf_size, d_output_buf_size,
                         input_options, data);
    }
  }
}

TEST(ZstdBuffers, MultiThread) { TestCompressionMultiThread(); }

void TestMultipleWritesZstd(uint8 input_buf_size, uint8 output_buf_size,
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
  ZstdOutputBuffer out(file_writer.get(), input_buf_size, output_buf_size,
                       output_options);

  for (int i = 0; i < num_writes; i++) {
    TF_ASSERT_OK(out.Append(StringPiece(data)));
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
  ZstdInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                     input_options);

  for (int i = 0; i < num_writes; i++) {
    tstring decompressed_output;
    TF_ASSERT_OK(in.ReadNBytes(data.size(), &decompressed_output));
    strings::StrAppend(&actual_result, decompressed_output);
  }

  EXPECT_EQ(actual_result, expected_result);
}

TEST(ZstdBuffers, MultipleWritesWithoutFlush) {
  TestMultipleWritesZstd(200, 200, 10);
}

TEST(ZstdBuffers, MultipleWritesWithFlush) {
  TestMultipleWritesZstd(200, 200, 10, true);
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

        // Boiler-plate to set up ZstdInputStream.
        std::unique_ptr<RandomAccessFile> file_reader;
        TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
        std::unique_ptr<RandomAccessInputStream> input_stream(
            new RandomAccessInputStream(file_reader.get()));
        ZstdInputStream in(input_stream.get(), input_buf_size, output_buf_size,
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

TEST(ZstdInputStream, TellDefaultOptions) {
  TestTell(CompressionOptions::DEFAULT(), CompressionOptions::DEFAULT());
}

}  // namespace io
}  // namespace tensorflow
