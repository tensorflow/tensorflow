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

#include "tensorflow_text/core/kernels/phrase_tokenizer_model_builder.h"

#include <stdint.h>

#include <memory>
#include <queue>
#include <stack>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow_text/core/kernels/phrase_tokenizer_model_generated.h"
#include "tensorflow_text/core/kernels/sentencepiece/double_array_trie_builder.h"
#include "tensorflow_text/core/kernels/string_vocab.h"
#include "tensorflow_text/core/kernels/whitespace_tokenizer_config_builder.h"

namespace tensorflow {
namespace text {
namespace {

// Builds the PhraseTokenizer model.
class PhraseBuilder {
 public:
  absl::Status BuildModel(const std::vector<std::string>& vocab,
                          const std::string& unk_token,
                          bool support_detokenization, int prob,
                          bool split_end_punctuation);

  absl::StatusOr<std::string> ExportToFlatBuffer() const;

 private:
  absl::optional<StringVocab> vocab_;
  std::vector<uint32_t> trie_data_;
  std::string unk_token_;
  int unk_token_id_;
  // Whether the tokenizer supports the detokenization function.
  bool support_detokenization_;
  int prob_;
  bool split_end_punctuation_;
};

absl::Status PhraseBuilder::BuildModel(const std::vector<std::string>& vocab,
                                       const std::string& unk_token,
                                       bool support_detokenization, int prob,
                                       bool split_end_punctuation) {
  unk_token_ = std::string(unk_token);
  support_detokenization_ = support_detokenization;
  prob_ = prob;
  split_end_punctuation_ = split_end_punctuation;

  vocab_.emplace(vocab);
  if (vocab_->Size() != vocab.size()) {
    return absl::FailedPreconditionError(
        "Tokens in the vocabulary must be unique.");
  }

  // Determine `unk_token_id_`.
  const absl::optional<int> unk_token_id = vocab_->LookupId(unk_token_);
  if (!unk_token_id.has_value()) {
    return absl::FailedPreconditionError("Cannot find unk_token in the vocab!");
  }
  unk_token_id_ = *unk_token_id;

  // build trie.
  trie_data_ = sentencepiece::BuildTrie(vocab);

  return absl::OkStatus();
}

absl::StatusOr<std::string> PhraseBuilder::ExportToFlatBuffer() const {
  flatbuffers::FlatBufferBuilder builder;

  const auto unk_token = builder.CreateString(unk_token_);

  std::vector<flatbuffers::Offset<flatbuffers::String>> vocab_fbs_vector;

  if (support_detokenization_) {
    vocab_fbs_vector.reserve(vocab_->Size());
    for (int i = 0; i < vocab_->Size(); ++i) {
      const absl::optional<absl::string_view> word = vocab_->LookupWord(i);
      if (!word.has_value()) {
        return absl::FailedPreconditionError(
            "Impossible. `token_id` is definitely within the range of vocab "
            "token ids; hence LookupWord() should always succeed.");
      }
      absl::string_view token = word.value();
      vocab_fbs_vector.emplace_back(builder.CreateString(token));
    }
  }

  auto vocab_array = builder.CreateVector(vocab_fbs_vector);

  std::string ws_config = BuildWhitespaceTokenizerConfig();
  auto whitespace_config = builder.CreateString(ws_config);
  auto trie_data = builder.CreateVector(trie_data_);

  TrieBuilder trie_builder(builder);
  trie_builder.add_nodes(trie_data);
  const auto trie_fbs = trie_builder.Finish();

  PhraseTokenizerConfigBuilder wtcb(builder);
  wtcb.add_unk_token(unk_token);
  wtcb.add_unk_token_id(unk_token_id_);
  wtcb.add_support_detokenization(support_detokenization_);
  wtcb.add_vocab_array(vocab_array);
  wtcb.add_whitespace_config(whitespace_config);
  wtcb.add_vocab_trie(trie_fbs);
  wtcb.add_prob(prob_);
  wtcb.add_split_end_punctuation(split_end_punctuation_);
  FinishPhraseTokenizerConfigBuffer(builder, wtcb.Finish());
  return std::string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                     builder.GetSize());
}
}  // namespace

absl::StatusOr<std::string> BuildPhraseModelAndExportToFlatBuffer(
    const std::vector<std::string>& vocab, const std::string& unk_token,
    bool support_detokenization, int prob, bool split_end_punctuation) {
  PhraseBuilder builder;
  SH_RETURN_IF_ERROR(builder.BuildModel(
      vocab, unk_token, support_detokenization, prob, split_end_punctuation));
  SH_ASSIGN_OR_RETURN(std::string flatbuffer, builder.ExportToFlatBuffer());
  return flatbuffer;
}

}  // namespace text
}  // namespace tensorflow
