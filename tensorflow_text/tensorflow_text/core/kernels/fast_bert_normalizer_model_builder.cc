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

#include "tensorflow_text/core/kernels/fast_bert_normalizer_model_builder.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "icu4c/source/common/unicode/errorcode.h"
#include "icu4c/source/common/unicode/normalizer2.h"
#include "icu4c/source/common/unicode/utf.h"
#include "icu4c/source/common/unicode/utf8.h"
#include "re2/re2.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow_text/core/kernels/darts_clone_trie_builder.h"
#include "tensorflow_text/core/kernels/fast_bert_normalizer.h"
#include "tensorflow_text/core/kernels/fast_bert_normalizer_model_generated.h"

namespace tensorflow {
namespace text {
namespace {
// Adapted from CaseFoldUTF8Op::Compute() in
// https://github.com/tensorflow/text/blob/master/tensorflow_text/core/kernels/normalize_kernels.cc.
absl::StatusOr<std::string> case_fold_utf8(absl::string_view input) {
  std::string output_text;
  icu::ErrorCode icu_error;
  const icu::Normalizer2* nfkc_cf =
      icu::Normalizer2::getNFKCCasefoldInstance(icu_error);
  if (!icu_error.isSuccess()) {
    return absl::InternalError(
        "Could not retrieve ICU NFKC_CaseFold normalizer");
  }
  icu::StringByteSink<std::string> byte_sink(&output_text);
  nfkc_cf->normalizeUTF8(0, icu::StringPiece(input.data(), input.size()),
                         byte_sink, nullptr, icu_error);
  if (!icu_error.isSuccess()) {
    return absl::InternalError(
        absl::StrCat("Could not normalize input string: ", input));
  }
  return output_text;
}

// Adapted from NormalizeUTF8Op::Compute() in
// https://github.com/tensorflow/text/blob/master/tensorflow_text/core/kernels/normalize_kernels.cc.
absl::StatusOr<std::string> normalize_utf8_nfd(absl::string_view input) {
  icu::ErrorCode icu_error;
  const icu::Normalizer2* normalizer =
      icu::Normalizer2::getNFDInstance(icu_error);
  if (!icu_error.isSuccess()) {
    return absl::InternalError(absl::StrCat(
        icu_error.errorName(), ": Could not retrieve ICU NFD normalizer"));
  }
  std::string output_text;
  icu::StringByteSink<std::string> byte_sink(&output_text);
  normalizer->normalizeUTF8(0, icu::StringPiece(input.data(), input.size()),
                            byte_sink, nullptr, icu_error);
  if (!icu_error.isSuccess()) {
    return absl::InternalError(absl::StrCat(
        icu_error.errorName(), ": Could not normalize input string: ", input));
  }
  return output_text;
}

// Returns all valid Unicode codepoints.
std::vector<char32_t> AllValidUnicodeCodePoints() {
  std::vector<char32_t> ret;
  // The maximum codepoint in Unicode is 0x0010FFFF.
  for (char32_t cp = 0; cp <= 0x0010FFFF; ++cp) {
    if (!U_IS_UNICODE_CHAR(cp)) {
      continue;
    }
    ret.push_back(cp);
  }
  return ret;
}

// Calls the original methods as in BertTokenizer (e.g., icu lib, etc.) to
// normalize the input. Based on
// https://github.com/tensorflow/text/blob/master/tensorflow_text/python/ops/bert_tokenizer.py.
absl::StatusOr<std::string> OriginalNormalizeText(
    absl::string_view input, bool lower_case_nfd_strip_accents) {
  static const RE2* const kMnRegex = new RE2("\\p{Mn}");
  static const RE2* const kControlRegex = new RE2("\\p{Cc}|\\p{Cf}");
  std::string output_text = std::string(input);
  // Lowercase and strip accents (if option is set)
  if (lower_case_nfd_strip_accents) {
    SH_ASSIGN_OR_RETURN(output_text, case_fold_utf8(output_text));
    SH_ASSIGN_OR_RETURN(output_text, normalize_utf8_nfd(output_text));
    RE2::GlobalReplace(&output_text, *kMnRegex, "");
  }

  // Replace control characters with spaces.
  RE2::GlobalReplace(&output_text, *kControlRegex, " ");

  return output_text;
}
}  // namespace

absl::StatusOr<std::string> BuildFastBertNormalizerModelAndExportToFlatBuffer(
    bool lower_case_nfd_strip_accents) {
  const auto& text_normalizer =
      FastBertNormalizerFactory::GetInstance(lower_case_nfd_strip_accents);
  flatbuffers::FlatBufferBuilder builder;
  const auto array = builder.CreateVector(text_normalizer.GetTrieData());
  const auto mapped_string_pool = builder.CreateVector(
      std::vector<uint8_t>(text_normalizer.GetMappedValuePool().begin(),
                           text_normalizer.GetMappedValuePool().end()));
  auto text_normalizer_model = CreateFastBertNormalizerModel(
      builder, lower_case_nfd_strip_accents, array,
      text_normalizer.GetDataForCodepointZero(), mapped_string_pool);
  builder.Finish(text_normalizer_model);
  return std::string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                     builder.GetSize());
}

/*static*/ absl::Status FastBertNormalizerFactory::BuildFastBertNormalizer(
    bool lower_case_nfd_strip_accents, std::vector<uint32_t>& trie_data,
    int& data_for_codepoint_zero, std::string& mapped_value_string_pool) {
  // Prepare the string keys and the encoded values.
  std::vector<std::string> keys;
  std::vector<int> values;
  mapped_value_string_pool = "";
  data_for_codepoint_zero = 0;
  // Memorize and reuse normalized strings.
  absl::flat_hash_map<std::string, int> norm_string_to_pool_offset;

  for (const auto cp : AllValidUnicodeCodePoints()) {
    // Get the utf8 view of the codepoint.
    char buf[4];
    int len = 0;
    U8_APPEND_UNSAFE(buf, len, cp);
    const absl::string_view cp_view(buf, len);
    // Normalize.
    SH_ASSIGN_OR_RETURN(
        auto cp_norm,
        OriginalNormalizeText(cp_view, lower_case_nfd_strip_accents));
    int data = 0;
    if (cp_norm != cp_view) {
      // The mapped value is different from the input.
      data |= text_norm::kIsNormalizedStringDifferentMask;
      // Encode the mapped value into `data`.
      if (!cp_norm.empty()) {
        const auto itr = norm_string_to_pool_offset.find(cp_norm);
        int current_offset = 0;
        if (itr == norm_string_to_pool_offset.end()) {
          if (cp_norm.size() >
              text_norm::kMaximumUtf8LengthOfNormalizedString) {
            LOG(ERROR) << "The length of mapped value exceeds the maximum "
                          "supported. Codepoint: "
                       << uint32_t{cp}
                       << ". Mapped value length: " << cp_norm.size()
                       << ". Maximum supported length: "
                       << text_norm::kMaximumUtf8LengthOfNormalizedString;
          }
          current_offset = mapped_value_string_pool.size();
          if (current_offset > text_norm::kMaximumOffsetOfNormalizedString) {
            LOG(ERROR) << "The offset of mapped value exceeds the maximum "
                          "supported. Codepoint: "
                       << uint32_t{cp}
                       << ". Mapped value offset: " << current_offset
                       << ". Maximum supported length: "
                       << text_norm::kMaximumOffsetOfNormalizedString;
          }
          norm_string_to_pool_offset[cp_norm] = current_offset;
          absl::StrAppend(&mapped_value_string_pool, cp_norm);
        } else {
          current_offset = norm_string_to_pool_offset[cp_norm];
        }
        data |= cp_norm.size();
        data |= (current_offset
                 << text_norm::kBitsToEncodeUtf8LengthOfNormalizedString);
      }
    }
    // Store the encoded data.
    if (cp == 0) {
      data_for_codepoint_zero = data;
      // Skip encoding it into the trie since Darts_clone cannot encode the
      // empty string.
      continue;
    }
    if (data == 0) {
      // Data is not set when normalizing the codepoint doesn't change it. These
      // characters aren't encoded to save space.
      continue;
    }
    // Key is the utf8 view; value is the encoded data.
    keys.emplace_back(buf, len);
    values.push_back(data);
  }
  // Build the trie.
  SH_ASSIGN_OR_RETURN(trie_data, trie_utils::BuildDartsCloneTrie(keys, values));
  LOG(INFO) << "CharacterSet built (lower_case_nfd_strip_accents="
            << lower_case_nfd_strip_accents
            << "). Trie data size (int32): " << trie_data.size()
            << ". Normalized string pool size (byte): "
            << mapped_value_string_pool.size();
  return absl::OkStatus();
}

FastBertNormalizerFactory::FastBertNormalizerFactory(
    bool lower_case_nfd_strip_accents) {
  auto status =
      BuildFastBertNormalizer(lower_case_nfd_strip_accents, trie_data_,
                              data_for_codepoint_zero_, mapped_value_pool_);
  if (!status.ok()) {
    // Should never happen since the same code must have passed the unit tests.
    LOG(ERROR) << "Unexpected error. Failed to build the data for "
                  "FastBertNormalizer. Error message: "
               << status.message();
    return;
  }
  auto char_set_recognizer_mapper = FastBertNormalizer::Create(
      trie_data_.data(), data_for_codepoint_zero_, mapped_value_pool_.data());
  if (!char_set_recognizer_mapper.ok()) {
    // Should never happen since the same code must have passed the unit tests.
    LOG(ERROR) << "Unexpected error: Failed to initialize "
                  "FastBertNormalizer from the data.";
    return;
  }
  char_set_normalizer_ = std::make_unique<FastBertNormalizer>(
      *std::move(char_set_recognizer_mapper));
}
}  // namespace text
}  // namespace tensorflow
