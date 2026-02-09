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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_BERT_NORMALIZER_MODEL_BUILDER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_BERT_NORMALIZER_MODEL_BUILDER_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow_text/core/kernels/fast_bert_normalizer.h"

namespace tensorflow {
namespace text {

// Builds a FastBertNormalizer model in flatbuffer format.
//
// Args:
//  * lower_case_nfd_strip_accents: If true, a preprocessing step is added to
//  lowercase the text, apply NFD normalization, and strip accents characters.
//
// Returns:
//  The bytes of the flatbuffer that stores the model.
absl::StatusOr<std::string> BuildFastBertNormalizerModelAndExportToFlatBuffer(
    bool lower_case_nfd_strip_accents);

/// A singleton class to initialize FastBertNormalizer and also to
/// own the data for it.
class FastBertNormalizerFactory {
 public:
  // Returns the singleton instance.
  //
  // Args:
  //   lower_case_nfd_strip_accents: bool
  //    - If true, it first lowercases the text, applies NFD normalization,
  //    strips accents characters, and then replaces control characters with
  //    whitespaces.
  //    - If false, it only replaces control characters with whitespaces.
  static const FastBertNormalizerFactory& GetInstance(
      bool lower_case_nfd_strip_accents) {
    if (lower_case_nfd_strip_accents) {
      return GetInstanceLowerCase();
    } else {
      return GetInstanceNoLowerCase();
    }
  }

  const FastBertNormalizer* GetNormalizer() const {
    return char_set_normalizer_.get();
  }

  const std::vector<uint32_t>& GetTrieData() const { return trie_data_; }

  int GetDataForCodepointZero() const { return data_for_codepoint_zero_; }

  absl::string_view GetMappedValuePool() const { return mapped_value_pool_; }

 private:
  FastBertNormalizerFactory(bool lower_case_nfd_strip_accents);

  // Returns a singleton instance with lower_case_nfd_strip_accents = false.
  static const FastBertNormalizerFactory& GetInstanceNoLowerCase() {
    static const FastBertNormalizerFactory* const kInstance =
        new FastBertNormalizerFactory(false);
    return *kInstance;
  }

  // Returns a singleton instance with lower_case_nfd_strip_accents = true.
  static const FastBertNormalizerFactory& GetInstanceLowerCase() {
    static const FastBertNormalizerFactory* const kInstance =
        new FastBertNormalizerFactory(true);
    return *kInstance;
  }

  // Returns the data to build a FastBertNormalizer.
  static absl::Status BuildFastBertNormalizer(
      bool lower_case_nfd_strip_accents, std::vector<uint32_t>& trie_data,
      int& data_for_codepoint_zero, std::string& mapped_value_string_pool);

  std::vector<uint32_t> trie_data_;
  int data_for_codepoint_zero_ = 0;
  std::string mapped_value_pool_ = "";
  std::unique_ptr<FastBertNormalizer> char_set_normalizer_ = nullptr;
};

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_BERT_NORMALIZER_MODEL_BUILDER_H_
