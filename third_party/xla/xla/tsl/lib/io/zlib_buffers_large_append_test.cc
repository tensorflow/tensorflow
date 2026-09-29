/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <zlib.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/io/random_inputstream.h"
#include "xla/tsl/lib/io/zlib_compression_options.h"
#include "xla/tsl/lib/io/zlib_inputstream.h"
#include "xla/tsl/lib/io/zlib_outputbuffer.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/tstring.h"

namespace tsl {
namespace io {
namespace {

// A single Append larger than 4 GiB used to be silently truncated to
// data.size() % 4 GiB bytes because zlib's avail_in is 32-bit.
// This test needs ~4.5 GiB of memory for the input string.
TEST(ZlibBuffers, AppendLargerThan4GB) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));

  constexpr size_t kDataSize = (1ULL << 32) + (1ULL << 20);
  constexpr absl::string_view kPattern =
      "The quick brown fox jumps over the lazy dog. ";
  std::string data;
  data.reserve(kDataSize);
  data.append(kPattern);
  while (data.size() < kDataSize) {
    data.append(data.data(), std::min(data.size(), kDataSize - data.size()));
  }

  ZlibCompressionOptions output_options = ZlibCompressionOptions::GZIP();
  output_options.compression_level = Z_BEST_SPEED;

  std::unique_ptr<WritableFile> file_writer;
  ASSERT_OK(env->NewWritableFile(fname, &file_writer));
  ZlibOutputBuffer out(file_writer.get(), output_options.input_buffer_size,
                       output_options.output_buffer_size, output_options);
  ASSERT_OK(out.Init());
  ASSERT_OK(out.Append(absl::string_view(data)));
  ASSERT_OK(out.Close());
  ASSERT_OK(file_writer->Flush());
  ASSERT_OK(file_writer->Close());

  ZlibCompressionOptions input_options = ZlibCompressionOptions::GZIP();
  std::unique_ptr<RandomAccessFile> file_reader;
  ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
  std::unique_ptr<RandomAccessInputStream> input_stream(
      new RandomAccessInputStream(file_reader.get()));
  ZlibInputStream in(input_stream.get(), input_options.input_buffer_size,
                     input_options.output_buffer_size, input_options);

  // Read back in chunks so peak memory stays close to the input string.
  constexpr size_t kChunkSize = 1 << 24;
  size_t total_read = 0;
  tstring chunk;
  while (total_read < kDataSize) {
    size_t bytes_to_read = std::min(kChunkSize, kDataSize - total_read);
    absl::Status read_status = in.ReadNBytes(bytes_to_read, &chunk);
    ASSERT_TRUE(read_status.ok())
        << "read failed after " << total_read << " bytes: " << read_status;
    ASSERT_EQ(chunk.size(), bytes_to_read);
    // Compare without ASSERT_EQ on the strings to avoid logging the 16 MiB
    // operands on mismatch.
    ASSERT_TRUE(absl::string_view(chunk) ==
                absl::string_view(data).substr(total_read, bytes_to_read))
        << "content mismatch at offset " << total_read;
    total_read += bytes_to_read;
  }
  // The decompressed stream must end exactly at kDataSize bytes.
  EXPECT_TRUE(absl::IsOutOfRange(in.ReadNBytes(1, &chunk)));
  EXPECT_OK(env->DeleteFile(fname));
}

}  // namespace
}  // namespace io
}  // namespace tsl
