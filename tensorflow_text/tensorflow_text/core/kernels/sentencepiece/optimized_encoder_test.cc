// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow_text/core/kernels/sentencepiece/optimized_encoder.h"

#include <fstream>

#include "file/base/path.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "src/sentencepiece.proto.h"
#include "src/sentencepiece_processor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow_text/core/kernels/sentencepiece/double_array_trie_builder.h"
#include "tensorflow_text/core/kernels/sentencepiece/encoder_config_generated.h"
#include "tensorflow_text/core/kernels/sentencepiece/model_converter.h"

namespace tensorflow {
namespace text {
namespace sentencepiece {

namespace internal {

absl::Status TFReadFileToString(const std::string& filepath,
                                std::string* data) {
  return tensorflow::ReadFileToString(tensorflow::Env::Default(), filepath,
                                      data);
}

absl::Status StdReadFileToString(const std::string& filepath,
                                 std::string* data) {
  std::ifstream infile(filepath);
  if (!infile.is_open()) {
    return absl::NotFoundError(
        absl::StrFormat("Error when opening %s", filepath));
  }
  std::string contents((std::istreambuf_iterator<char>(infile)),
                       (std::istreambuf_iterator<char>()));
  data->append(contents);
  infile.close();
  return absl::OkStatus();
}
}  // namespace internal

namespace {

static char kConfigFilePath[] =
    "/tensorflow_text/python/ops/test_data/"
    "fast_sentencepiece.model";

TEST(OptimizedEncoder, NormalizeStringWhitestpaces) {
  flatbuffers::FlatBufferBuilder builder(1024);
  EncoderConfigBuilder ecb(builder);
  ecb.add_remove_extra_whitespaces(true);
  ecb.add_add_dummy_prefix(true);
  ecb.add_escape_whitespaces(true);
  FinishEncoderConfigBuffer(builder, ecb.Finish());
  const EncoderConfig* config = GetEncoderConfig(builder.GetBufferPointer());
  {
    const auto result = NormalizeString("x  y", *config);
    const auto res_string = std::get<0>(result);
    const auto offsets = std::get<1>(result);
    EXPECT_EQ(res_string, "\xe2\x96\x81x\xe2\x96\x81y");
    EXPECT_THAT(offsets, ::testing::ElementsAre(0, 0, 0, 0, 1, 1, 1, 3));
  }
  {
    const auto result = NormalizeString("\tx  y\n", *config);
    const auto res_string = std::get<0>(result);
    const auto offsets = std::get<1>(result);
    EXPECT_EQ(res_string, "\xe2\x96\x81x\xe2\x96\x81y");
    EXPECT_THAT(offsets, ::testing::ElementsAre(0, 0, 0, 1, 2, 2, 2, 4));
  }
}

TEST(OptimizedEncoder, NormalizeStringReplacement) {
  flatbuffers::FlatBufferBuilder builder(1024);
  const std::vector<std::string> norm_prefixes = {"A", "AA", "AAA", "AAAA"};
  const char norm_replacements[] = "A1\0A2\0A3\0A4";
  const auto trie_vector =
      builder.CreateVector(BuildTrie(norm_prefixes, {0, 3, 6, 9}));
  const auto norm_r = builder.CreateVector<int8_t>(
      reinterpret_cast<const signed char*>(norm_replacements),
      sizeof(norm_replacements));
  TrieBuilder trie_builder(builder);
  trie_builder.add_nodes(trie_vector);
  const auto norm_p = trie_builder.Finish();
  EncoderConfigBuilder ecb(builder);
  ecb.add_remove_extra_whitespaces(false);
  ecb.add_normalized_prefixes(norm_p);
  ecb.add_normalized_replacements(norm_r);
  FinishEncoderConfigBuffer(builder, ecb.Finish());
  const EncoderConfig* config = GetEncoderConfig(builder.GetBufferPointer());
  {
    const auto result = NormalizeString("ABAABAAABAAAA", *config);
    const auto res_string = std::get<0>(result);
    const auto offsets = std::get<1>(result);
    EXPECT_EQ(res_string, "A1BA2BA3BA4");
    EXPECT_THAT(offsets,
                ::testing::ElementsAre(0, 0, 1, 2, 2, 4, 5, 5, 8, 9, 9));
  }
}

TEST(OptimizedEncoder, NormalizeStringWhitespacesRemove) {
  flatbuffers::FlatBufferBuilder builder(1024);
  const std::vector<std::string> norm_prefixes = {"A", "AA", "AAA", "AAAA",
                                                  "X"};
  const char norm_replacements[] = "A1\0A2\0A3\0A4\0 ";
  const auto trie_vector =
      builder.CreateVector(BuildTrie(norm_prefixes, {0, 3, 6, 9, 12}));
  const auto norm_r = builder.CreateVector<int8_t>(
      reinterpret_cast<const signed char*>(norm_replacements),
      sizeof(norm_replacements));
  TrieBuilder trie_builder(builder);
  trie_builder.add_nodes(trie_vector);
  const auto norm_p = trie_builder.Finish();
  EncoderConfigBuilder ecb(builder);
  ecb.add_remove_extra_whitespaces(true);
  ecb.add_normalized_prefixes(norm_p);
  ecb.add_normalized_replacements(norm_r);
  FinishEncoderConfigBuffer(builder, ecb.Finish());
  const EncoderConfig* config = GetEncoderConfig(builder.GetBufferPointer());
  {
    const auto result = NormalizeString("XXABAABAAABAAAA", *config);
    const auto res_string = std::get<0>(result);
    const auto offsets = std::get<1>(result);
    EXPECT_EQ(res_string, " A1BA2BA3BA4");
    EXPECT_THAT(offsets,
                ::testing::ElementsAre(0, 2, 2, 3, 4, 4, 6, 7, 7, 10, 11, 11));
  }
}

TEST(OptimizedEncoder, ConfigConverter) {
  std::string config;
  auto status = internal::TFReadFileToString(
      file::JoinPath(::testing::SrcDir(), kConfigFilePath), &config);
  ASSERT_TRUE(status.ok());

  ::sentencepiece::SentencePieceProcessor processor;
  ASSERT_TRUE(processor.LoadFromSerializedProto(config).ok());
  const auto converted_model = ConvertSentencepieceModel(config);
  const std::string test_string("Hello world!\\xF0\\x9F\\x8D\\x95");
  const auto encoded =
      EncodeString(test_string, converted_model.data(), false, false, false);
  ASSERT_EQ(encoded.codes.size(), encoded.offsets.size());

  ::sentencepiece::SentencePieceText reference_encoded;
  ASSERT_TRUE(processor.Encode(test_string, &reference_encoded).ok());
  EXPECT_EQ(encoded.codes.size(), reference_encoded.pieces_size());
  for (int i = 0; i < encoded.codes.size(); ++i) {
    EXPECT_EQ(encoded.codes[i], reference_encoded.pieces(i).id());
    EXPECT_EQ(encoded.offsets[i], reference_encoded.pieces(i).begin());
  }
}

}  // namespace
}  // namespace sentencepiece
}  // namespace text
}  // namespace tensorflow
