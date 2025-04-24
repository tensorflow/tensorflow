/* Copyright 2025 The OpenXLA Authors.

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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/lib/io/mock_input_stream_interface.h"
#include "xla/tsl/lib/io/zstd/zstd_input_stream.h"
#include "xla/tsl/lib/io/zstd/zstd_output_buffer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/mock_writable_file.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/tstring.h"

namespace tsl::io {
namespace {
using ::testing::_;
using ::testing::Invoke;

tstring GeneratePseudorandomData(size_t input_size) {
  tstring input_string;
  input_string.resize_uninitialized(input_size);

  std::seed_seq seed_seq{{0, 12, 34, 56, 78}};
  std::mt19937 gen{seed_seq};
  std::uniform_int_distribution<int> distrib(std::numeric_limits<char>::min(),
                                             std::numeric_limits<char>::max());
  std::generate(input_string.data(), input_string.data() + input_string.size(),
                [&]() { return distrib(gen); });

  return input_string;
}

absl::StatusOr<std::string> Compress(absl::string_view input,
                                     size_t write_block_size) {
  std::string compressed_string;
  {
    tsl::MockWritableFile output_file{};
    EXPECT_CALL(output_file, Append(_))
        .WillRepeatedly(Invoke([&compressed_string](absl::string_view data) {
          compressed_string.append(data.data(), data.size());
          return absl::OkStatus();
        }));

    TF_ASSIGN_OR_RETURN(auto compressed_output_buffer,
                        ZstdOutputBuffer::Create(&output_file, 3));

    absl::string_view remaining_input_string = input;

    while (!remaining_input_string.empty()) {
      TF_RETURN_IF_ERROR(compressed_output_buffer->Append(
          remaining_input_string.substr(0, write_block_size)));
      remaining_input_string.remove_prefix(
          std::min(write_block_size, remaining_input_string.size()));
    }
  }

  return compressed_string;
}

absl::StatusOr<std::string> Decompress(absl::string_view compressed_data,
                                       size_t read_block_size) {
  MockInputStreamInterface input_stream;
  absl::Span<const char> remaining_compressed_data = compressed_data;
  EXPECT_CALL(input_stream, ReadNBytes(_, _))
      .WillRepeatedly(
          Invoke([&](int64_t bytes_to_read, tsl::tstring* output_buffer) {
            size_t bytes_read = std::min<size_t>(
                remaining_compressed_data.size(), bytes_to_read);
            output_buffer->resize_uninitialized(bytes_read);
            absl::c_copy(remaining_compressed_data.subspan(0, bytes_read),
                         output_buffer->data());
            remaining_compressed_data.remove_prefix(bytes_read);
            if (bytes_read < bytes_to_read) {
              // The InputStreamInterface contract wants us to return an
              // OutOfRange error when we couldn't read the full requested
              // amount.
              return absl::OutOfRangeError("Out of range");
            }
            return absl::OkStatus();
          }));

  TF_ASSIGN_OR_RETURN(auto uncompressed_input_stream,
                      ZstdInputStream::Create(&input_stream));

  std::string uncompressed_data;

  while (true) {
    tstring block;
    absl::Status read_result =
        uncompressed_input_stream->ReadNBytes(read_block_size, &block);
    uncompressed_data.append(block.begin(), block.end());
    if (absl::IsOutOfRange(read_result)) {
      break;
    }
    TF_RETURN_IF_ERROR(read_result);
  }

  return uncompressed_data;
}

using ZStdTest = ::testing::TestWithParam<std::tuple<size_t, size_t, size_t>>;
TEST_P(ZStdTest, CompressAndDecompress) {
  const auto [input_size, read_block_size, write_block_size] = GetParam();

  tsl::tstring input_data = GeneratePseudorandomData(input_size);

  TF_ASSERT_OK_AND_ASSIGN(std::string compressed_data,
                          Compress(input_data, write_block_size));

  TF_ASSERT_OK_AND_ASSIGN(std::string uncompressed_data,
                          Decompress(compressed_data, read_block_size));

  EXPECT_EQ(input_data, uncompressed_data);
}

constexpr std::array<size_t, 4> kInputSizes = {7777, 8192, 16384, 32768};
constexpr std::array<size_t, 4> kReadBlockSizes = {512, 777, 1024, 2048};
constexpr std::array<size_t, 4> kWriteBlockSizes = {512, 777, 1024, 2048};

INSTANTIATE_TEST_SUITE_P(
    ZStdTest_BlockSizeSweep, ZStdTest,
    ::testing::Combine(::testing::ValuesIn(kInputSizes),
                       ::testing::ValuesIn(kReadBlockSizes),
                       ::testing::ValuesIn(kWriteBlockSizes)),
    ([](const ::testing::TestParamInfo<std::tuple<size_t, size_t, size_t>>&
            param_info) {
      const auto& [input_size, read_block_size, write_block_size] =
          param_info.param;
      return absl::StrFormat("inp%d_rb%d_wb%d", input_size, read_block_size,
                             write_block_size);
    }));

}  // namespace
}  // namespace tsl::io
