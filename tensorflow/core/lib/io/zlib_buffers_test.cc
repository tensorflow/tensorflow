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

#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputbuffer.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"

#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {

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

static string GenTestString(uint copies) {
  string result = "";
  for (int i = 0; i < copies; i++) {
    result += GetRecord();
  }
  return result;
}

TEST(ZlibBuffers, ) {
  Env* env = Env::Default();
  io::ZlibCompressionOptions zlib_input_options = io::ZlibCompressionOptions();
  //  zlib_input_options.flush_mode = Z_NO_FLUSH;
  io::ZlibCompressionOptions zlib_output_options = io::ZlibCompressionOptions();
  zlib_output_options.flush_mode = Z_NO_FLUSH;
  string fname = testing::TmpDir() + "/zlib_buffers_test";
  for (auto file_size : NumCopies()) {
    // Write to compressed file
    string data = GenTestString(file_size);
    for (auto input_buf_size : InputBufferSizes()) {
      for (auto output_buf_size : OutputBufferSizes()) {
        WritableFile* file_writer;
        TF_CHECK_OK(env->NewWritableFile(fname, &file_writer));
        string result;

        io::ZlibOutputBuffer out(file_writer, input_buf_size, output_buf_size,
                                 zlib_output_options);

        TF_CHECK_OK(out.Write(StringPiece(data)));
        TF_CHECK_OK(out.Close());
        TF_CHECK_OK(file_writer->Flush());
        TF_CHECK_OK(file_writer->Close());

        RandomAccessFile* file_reader;
        TF_CHECK_OK(env->NewRandomAccessFile(fname, &file_reader));
        io::ZlibInputBuffer in(file_reader, input_buf_size, output_buf_size,
                               zlib_input_options);
        TF_CHECK_OK(in.ReadNBytes(data.size(), &result));
        EXPECT_EQ(result, data);

        delete file_reader;
        delete file_writer;
      }
    }
  }
}

}  // namespace tensorflow
