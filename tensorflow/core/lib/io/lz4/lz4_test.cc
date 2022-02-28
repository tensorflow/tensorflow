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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/lz4/lz4_compression_options.h"
#include "tensorflow/core/lib/io/lz4/lz4_inputstream.h"
#include "tensorflow/core/lib/io/lz4/lz4_outputbuffer.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

#include <lz4frame.h>

namespace tensorflow {
namespace io {

static std::vector<int> InputBufferSizes() {
  return {10, 100, 200, 500, 1000, 10000, 256 << 10};
}

static std::vector<int> OutputBufferSizes() {
  return {100, 200, 500, 1000, 256 << 10};
}

static std::vector<int> NumCopies() { return {1, 50, 500, 5000}; }

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

typedef io::Lz4CompressionOptions CompressionOptions;

void WriteCompressedFile(Env* env, const string& fname, int input_buf_size,
                         int output_buf_size,
                         const CompressionOptions& output_options,
                         const string& data) {
  std::unique_ptr<WritableFile> file_writer;
  TF_ASSERT_OK(env->NewWritableFile(fname, &file_writer));

  Lz4OutputBuffer out(file_writer.get(), input_buf_size, output_buf_size,
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

  Lz4InputStream in(input_stream.get(), input_buf_size, output_buf_size,
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
        WriteCompressedFile(env, fname, input_buf_size, output_buf_size,
                            output_options, data);
        ReadCompressedFile(env, fname, input_buf_size, output_buf_size,
                           input_options, data);
      }
    }
  }
}

TEST(Lz4Frame, DefaultOptions) {}

}  // namespace io
}  // namespace tensorflow
