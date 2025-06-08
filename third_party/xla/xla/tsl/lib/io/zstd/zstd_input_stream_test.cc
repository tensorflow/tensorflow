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

#include "xla/tsl/lib/io/zstd/zstd_input_stream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/lib/io/mock_input_stream_interface.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/tstring.h"

namespace {
using ::testing::_;
using ::testing::AnyOf;
using ::testing::Invoke;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

constexpr absl::string_view kShortUncompressedString = "hello\n";
// Generated with `echo hello | zstd | xxd -ps`.
constexpr absl::string_view kShortCompressedString =
    "28b52ffd045831000068656c6c6f0a5388bd91";

constexpr absl::string_view kLongUncompressedString =
    "fQ4Uz61vz4ToOjb7U3HfY1VHuII0O4gdAh7S42LgWNKVpWkm2ITywyn4zNFH67EzgIwyLZqhmL"
    "10RwAL7NM91mf6SWzg/"
    "OdfzFhqwD6rnMn6IavFm3+vzAslWP5LZPQJIOjuOm6wq7d+9FTk5oem0Glmd+"
    "S4CDX18eDURqyDOvtM7b0icKXvRZhvtplmnY+"
    "gqTkvujOXuGjpj1xatR6gfRYZCPHCZgMqmF6cEk6oJKhXKe/"
    "dy+dyW87iaoo13GqnCckBdGvCwzNRSgQs8SLkurwtcntR95ZjnWmebBZ/"
    "TjtQUdu64Yo6rMSyI9p1rHJbhHxnA82uEZrt0dzbM6MocBl4/KW/"
    "mun+DqMmw9RuZcbBI8mk+Z8lAD3ZABTZ1CmfHYLT8k6PlL9ZNpBpzS2AyHcg0GwIk9SfW4bTB/"
    "bK+YvfYumXVMGNNYZPs+gcuAkTvTMmImlSEtcTXS2/"
    "xrS57rOFLBgYySOz04JF3DUHQVwGlmxNQ3pMSnpbPVrQCa1yesgt5yeaUxii77GKFs3EgZA12W"
    "DYOTjo7g1BS+"
    "jTx8sRtG5B6Cxm5eBgZLVwpadSohif5w9ATVR2agrgFZ38Ny7fFSswDiNl0OtQS3VDUNlYA3Pf"
    "p/IgHYLqk5lnCzK7YKfG1AM/"
    "LqZcJIJiOtFO2TWHzBhSa8YWoOVbT8eCHBfjhLcXLEEfBLhx5QkKPTBnkGfjNib6FZZlFw2lAu"
    "hZgLRJ/2t6p6Q7bUi2UU2c67a6ub8+Q7Uzg7R6o/"
    "qUcKQONvYiqRs7GAvL10bfRLVj6UCqkRysAnn7frxEQGxfAlFJHGSzUQVBK5IOySpS5rtIGKqU"
    "4mJXNrY9v8Fj70/JMOiw9BU63UxLgQ78hSbI2nahLcr7DhUNBy/"
    "pAEDjRE3b3FC2sVYA975MFgxhXAOosUj0YA66VO38hdDKzLN9n8JQT24Lzqjb9jsfH7/YgNsl/"
    "0GaGiC1pv0pZ6Z+8ZcRy0kQfnYpI1yyBvfX4rUYi7YGzfIFhO11+";

constexpr absl::string_view kLongCompressedString =
    "28b52ffd0458f5180056bfc61b10c707afacd498bd70536e1cb62f0659a2"
    "b1eb0d04000000004005bf00bf00be005372b06cc1ecccc03610b6a13858"
    "84d20ab01c058c5c525dec7b8812403ee4c8cfb9b8501f677b4e32413be5"
    "19f5dc4f84ab9ab6f15c7a6e2658a369881d8b672db25dee8b0ecc418820"
    "a5230c6a3a19ea815d504f6a08437769ae6af2a955814de0af45304861e7"
    "9e894495e642c5576c7d7944bba5749306adf8f7a0e3a47a3458d62f87e6"
    "8ce3524bd75f365dfd1bd0392565b7b2dab4ed55888a0e5ad0ca06ff4294"
    "ae978ee12b9a515532051e2149255f174f8df5947b3ef8fd291b19db0b0a"
    "c476edb036d9b25a4d271adca82f4aa48062dc5a6787ffb924eded4bf7d6"
    "05d211bc310674a95aa998377ba8238f9f80053eccb2090309aa1c927348"
    "5862f178a23220bb06ade8c9f823653b4877b74f9bf9e64a051c65bc2ba4"
    "48153abcf7383e899c94f104aa32d5238e5ee61a5ac5dc885865b5185885"
    "923e238b714cbc0d46fc7a808c30ce5a53a518a0dea36d2abbc35ece860a"
    "970ddf169f5ee11ff1a00ed6d32f0d17af634cbf327a750cbcfd8ecb6e1a"
    "9a27c3838ed0ea0605cbb9feefaf6f0423cabcd0f9887c1f627e62df78cb"
    "63a40870d7f7a5eca4082da5de4b5c34b3939cd0f5f10416ba2d75793d24"
    "007011c6ab3cc987820e8c27821a3b089942573e3735f0104d8a35335005"
    "46ef5e05e61c136985ddbaa1355d50f70aa95f152d29489c30cdbe084cba"
    "2dd68c3433231aee24b2db199378e9bfbf8070a0e734f4429a25db190a2d"
    "0d5efe3f8745ad21fe2c53ccc14ced9d44f71cbfb3a7a785d100391cff01"
    "092c5953efa6597f8583bdc41f3ed847010ac47168a55491f7147633fc5e"
    "8540ce8f6d1f551baee4861a36377a86f0dcb7c6dc1e90391526a348e1ec"
    "8745a748f86cc4d3bd2f05f1b600b336d9755b91a164120de4dc098df383"
    "59cd2194f597b0a5374e08c20824256202e0ce08e0b485fd645f19f577bc"
    "cac28a546257327dfb57fd25fc7b1adad4b04061f45362e6267c33424055"
    "3ed79aaaf85dbab6b3b56ab11e70fbee6ea20252ea90f5b401ea97ecebea"
    "c03687bad0c74ff0e248d138aed9993accb9663867da668d9f1b00b493a5"
    "20";

class TestInputStream : public tsl::io::MockInputStreamInterface {
 public:
  static absl::StatusOr<std::unique_ptr<TestInputStream>> Create(
      absl::string_view compressed_hex_data) {
    std::string input_data = absl::HexStringToBytes(compressed_hex_data);
    std::unique_ptr<TestInputStream> input{
        new TestInputStream{std::move(input_data)}};
    EXPECT_CALL(*input, ReadNBytes(_, _))
        .WillRepeatedly(
            Invoke([input = input.get()](int64_t bytes_to_read,
                                         tsl::tstring* output_buffer) {
              size_t bytes_read = std::min<size_t>(
                  input->remaining_data_.size(), bytes_to_read);
              output_buffer->resize_uninitialized(bytes_read);
              std::copy(input->remaining_data_.begin(),
                        input->remaining_data_.begin() + bytes_read,
                        output_buffer->data());
              input->remaining_data_.remove_prefix(bytes_read);
              if (bytes_read < bytes_to_read) {
                return absl::OutOfRangeError("Out of range");
              }
              return absl::OkStatus();
            }));

    return input;
  }

 private:
  explicit TestInputStream(std::string input_buffer)
      : input_buffer_(std::move(input_buffer)),
        remaining_data_(input_buffer_.data(), input_buffer_.size()) {}
  std::string input_buffer_;
  absl::Span<const char> remaining_data_;
};

TEST(ZstdOutputBufferTest, ShortTestString) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestInputStream> input_stream,
                          TestInputStream::Create(kShortCompressedString));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tsl::io::ZstdInputStream> zstd_input_stream,
      tsl::io::ZstdInputStream::Create(input_stream.get()));
  tsl::tstring output;
  EXPECT_THAT(
      zstd_input_stream->ReadNBytes(kShortUncompressedString.size(), &output),
      IsOk());
  EXPECT_EQ(output, kShortUncompressedString);
}

TEST(ZstdOutputBufferTest, LongTestString) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestInputStream> input_stream,
                          TestInputStream::Create(kLongCompressedString));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<tsl::io::ZstdInputStream> zstd_input_stream,
      tsl::io::ZstdInputStream::Create(input_stream.get()));
  std::string output;
  int iterations = 0;
  while (output.size() < kLongUncompressedString.size()) {
    // We choose the block size small enough so that we at least go through 2
    // iterations of this loop. Otherwise we wouldn't be testing the streaming
    // functionality.
    constexpr size_t kBlockSize = 512;
    tsl::tstring block;
    absl::Status read_status =
        zstd_input_stream->ReadNBytes(kBlockSize, &block);
    EXPECT_THAT(read_status,
                AnyOf(IsOk(), StatusIs(absl::StatusCode::kOutOfRange)));
    output.insert(output.end(), block.begin(), block.end());
    ++iterations;
  }
  EXPECT_EQ(output, kLongUncompressedString);

  // This assertion makes sure we have at least two calls to `ReadNBytes`.
  // Otherwise we wouldn't be testing the state management in ZstdInputStream.
  EXPECT_GE(iterations, 2);
}

}  // namespace
