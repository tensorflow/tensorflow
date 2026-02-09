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

#include "tensorflow_text/core/kernels/fast_wordpiece_tokenizer_model_builder.h"

#include <stdint.h>

#include <memory>
#include <queue>
#include <stack>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include /* cppitertools */ "imap.hpp"
#include "icu4c/source/common/unicode/umachine.h"
#include "icu4c/source/common/unicode/utf8.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow_text/core/kernels/darts_clone_trie_builder.h"
#include "tensorflow_text/core/kernels/darts_clone_trie_wrapper.h"
#include "tensorflow_text/core/kernels/fast_wordpiece_tokenizer_model_generated.h"
#include "tensorflow_text/core/kernels/fast_wordpiece_tokenizer_utils.h"
#include "tensorflow_text/core/kernels/sentence_fragmenter_v2.h"
#include "tensorflow_text/core/kernels/string_vocab.h"

namespace tensorflow {
namespace text {
namespace {

// A Unicode control char that never appears in the input as it is filtered
// during text normalization. It is used to build dummy nodes in the trie.
static constexpr char kInvalidControlChar = 0x11;

// A wrapper of vocab tokens that will be used to build the trie.
class TrieVocabToken {
 public:
  TrieVocabToken(absl::string_view token, int token_id,
                 absl::string_view suffix_indicator)
      : token_(std::string(token)), token_id_(token_id) {
    if (!suffix_indicator.empty() && token_ != suffix_indicator &&
        absl::StartsWith(token_, suffix_indicator)) {
      is_suffix_token_ = true;
      actual_token_start_offset_ = suffix_indicator.size();
    }
    // Iterate over the Unicode chars from the token, to initialize
    // contains_punctuation_ and actual_token_unicode_len_.
    int token_len = token.size();
    int cur_pos = actual_token_start_offset_;
    UChar32 c;
    while (cur_pos < token_len) {
      U8_NEXT(token, cur_pos, token_len, c);
      if (!contains_punctuation_ &&
          fast_wordpiece_tokenizer_utils::IsPunctuationOrChineseChar(c)) {
        contains_punctuation_ = true;
      }
      ++actual_token_unicode_len_;
    }
  }

  absl::string_view Token() const { return token_; }

  int TokenId() const { return token_id_; }

  bool IsSuffixToken() const { return is_suffix_token_; }

  bool ContainsPunctuation() const { return contains_punctuation_; }

  int TokenUnicodeLengthWithoutSuffixIndicator() const {
    return actual_token_unicode_len_;
  }

  int TokenLengthWithoutSuffixIndicator() const {
    return token_.size() - actual_token_start_offset_;
  }

 private:
  std::string token_;

  int token_id_ = -1;

  // By design, `is_suffix_token_`=false for the suffix indicator (e.g., "##")
  // itself.
  bool is_suffix_token_ = false;

  // The starting offset of the token string in `token_` without the suffix
  // indicator. By design, `actual_token_start_offset_`=0 for the suffix
  // indicator (e.g., "##") itself.
  int actual_token_start_offset_ = 0;

  // Length of the actual token string in Unicode character.
  int actual_token_unicode_len_ = 0;

  // True when the actual token string contains punctuation, e.g. "test.x",
  // "##.", ".test", "...", "!", etc.
  bool contains_punctuation_ = false;
};

// The failure struct to store failure links and failure pops.
struct FailureStruct {
  // The failure link, denoted as f(v), of each node v.
  //
  // Null node is represented by fast_wordpiece_tokenizer_utils::kNullNode.
  uint32_t failure_link = fast_wordpiece_tokenizer_utils::kNullNode;

  // The failure pop list, denoted as F(v), of a node v.
  //
  // It is stored as a pair of offset and length that represents a continuous
  // vector in `failure_pops_pool_`. This pair is encoded using
  // EncodeFailurePopList() in fast_wordpiece_tokenizer_utils.h.
  uint32_t failure_pops_offset_length =
      fast_wordpiece_tokenizer_utils::kNullFailurePopsList;
};

// Builds the FastWordpieceTokenizer model.
class FastWordpieceBuilder {
 public:
  // When no_pretokenization is false, we split the input string by punctuation
  // chars (in addition to whitespaces) and then tokenize it to wordpieces.
  absl::Status BuildModel(const std::vector<std::string>& vocab,
                          int max_bytes_per_token,
                          absl::string_view suffix_indicator,
                          absl::string_view unk_token,
                          bool no_pretokenization,
                          bool support_detokenization);

  absl::StatusOr<std::string> ExportToFlatBuffer() const;

 private:
  absl::StatusOr<std::vector<TrieVocabToken>> PrepareVocabTokensToBuildTrie();

  absl::Status ConstructTrie(
      const std::vector<TrieVocabToken>& tokens_to_build_trie);

  absl::Status BuildFailureStructure(
      const std::vector<TrieVocabToken>& tokens_to_build_trie);

  // Builds the set of outgoing edge labels for each trie node and returns a
  // mapping (node_id -> set<char>). Used in BuildFailureStructure().
  absl::StatusOr<std::vector<absl::flat_hash_set<char>>>
  BuildOutgoingEdgeLabelsForTrie(
      const std::vector<TrieVocabToken>& tokens_to_build_trie);

  // Builds the set of outgoing edge labels for nodes along the trie path of
  // `vocab_token`. Used in BuildOutgoingEdgeLabelsForTrie().
  absl::Status BuildOutgoingEdgeLabelsAlongVocabToken(
      const TrieVocabToken& vocab_token,
      std::vector<absl::flat_hash_set<char>>& node_outgoing_edge_labels);

  // Assigns failure link f(cur_node) to `failure_link` and populates failure
  // pops F(cur_node) (based on `one_step_pops` and
  // `parent_failure_pops_offset_length`).
  absl::Status AssignFailureLinkAndPops(uint32_t cur_node,
                                        uint32_t failure_link,
                                        const std::vector<int>& one_step_pops,
                                        int parent_failure_pops_offset_length);

  // If `failure_pops_offset_length` encodes a valid failure pop list, appends
  // the failure pop list to the end of `out_failure_pops`. Otherwise, does
  // nothing.
  void GetFailurePopsAndAppendToOut(uint32_t failure_pops_offset_length,
                                    std::vector<int>& out_failure_pops);

  absl::Status PrecomputeResultForSuffixIndicator();

  inline void BreakTrieLinkFromParentToChild(uint32_t child_node_id) {
    // In trie, the least significant 8 bits encode the label of the trie link
    // from the parent to the node itself.
    //
    // Reference:
    // https://github.com/s-yata/darts-clone/blob/e40ce4627526985a7767444b6ed6893ab6ff8983/include/darts.h#L65-L70.
    //
    // For example, if there is a trie link `u` -> `v` with label (say) 'a'
    // (ASCII 97 or 0x61), then the least significant 8 bits of node `v` will be
    // 0x61. By erasing its least significant 8 bits to 0, it effectively
    // prevents the node from being reachable from its parent, i.e. breaking the
    // trie link from the parent to the node itself.
    trie_array_[child_node_id] &= 0xFFFFFF00;
  }

  inline void EraseValueOfNode(uint32_t node_id) {
    // In trie, the 9th least significant bit of a node's value marks whether
    // the node has a leaf node (i.e., having a value stored on the node).
    //
    // Reference:
    // https://github.com/s-yata/darts-clone/blob/e40ce4627526985a7767444b6ed6893ab6ff8983/include/darts.h#L54-L58
    //
    // By setting the 9th least significant bit to 0, it effectively erases any
    // value (i.e., token id in our case) associated with the node.
    trie_array_[node_id] &= 0xFFFFFEFF;
  }

  absl::optional<StringVocab> vocab_;

  int max_bytes_per_token_ = -1;

  std::string suffix_indicator_;

  std::string unk_token_;

  int unk_token_id_ = -1;

  // A wrapper to access the trie encoded by `trie_array_`.
  absl::optional<trie_utils::DartsCloneTrieWrapper> trie_;

  // The actual data of the trie.
  std::vector<uint32_t> trie_array_;

  // The "suffix_root" node on the trie whose trie path (from the root to the
  // node) is the suffix indicator string.
  uint32_t trie_suffix_root_ = fast_wordpiece_tokenizer_utils::kNullNode;

  // The dummy node to serve as the failure link of punctuation nodes.
  uint32_t trie_punct_failure_link_node_ =
      fast_wordpiece_tokenizer_utils::kNullNode;

  // Whether to build the end-to-end tokenizer that tokenizes general texts.
  // When set to false, it splits the input on punctuation/whitespace and treat
  // each punctuation as an independent word.
  bool no_pretokenization_;

  // Whether the tokenizer supports the detokenization function.
  bool support_detokenization_;

  std::vector<FailureStruct> failure_struct_array_;

  // Each element in the failure pops pool is an encoded vocab token.
  // See EncodeToken() in fast_wordpiece_tokenizer_utils.h.
  std::vector<int> failure_pops_pool_;

  // The precomputed result for the suffix indicator. Each element in the
  // failure pops pool is an encoded vocab token. See EncodeToken() in
  // fast_wordpiece_tokenizer_utils.h.
  std::vector<int> precomputed_result_for_suffix_indicator_;

  // The mapping from node id to whether the corresponding token is a
  // punctuation char.
  absl::flat_hash_map<uint32_t, bool> node_id_is_punc_map_;
};

absl::Status FastWordpieceBuilder::BuildModel(
    const std::vector<std::string>& vocab, int max_bytes_per_token,
    absl::string_view suffix_indicator, absl::string_view unk_token,
    bool no_pretokenization, bool support_detokenization) {
  unk_token_ = std::string(unk_token);
  suffix_indicator_ = std::string(suffix_indicator);
  max_bytes_per_token_ = max_bytes_per_token;
  no_pretokenization_ = no_pretokenization;
  support_detokenization_ = support_detokenization;

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

  // Construct the trie and the failure structure.
  SH_ASSIGN_OR_RETURN(auto tokens_to_build_trie,
                      PrepareVocabTokensToBuildTrie());
  SH_RETURN_IF_ERROR(ConstructTrie(tokens_to_build_trie));
  SH_RETURN_IF_ERROR(BuildFailureStructure(tokens_to_build_trie));

  // Precompute the result when the input is the suffix indicator string itself.
  SH_RETURN_IF_ERROR(PrecomputeResultForSuffixIndicator());

  return absl::OkStatus();
}

absl::StatusOr<std::vector<TrieVocabToken>>
FastWordpieceBuilder::PrepareVocabTokensToBuildTrie() {
  // To simplify the inference (fewer corner cases),
  //  * We ensure that `trie_suffix_root_` is always available on the trie.
  //  * We ensure that `trie_suffix_root_` does not have data (i.e., the suffix
  //    indicator string is not in the set of the keys of the trie).
  //  * We don't actually add the end-of-input symbol "$" but use an alternative
  //    logic. See FastWordpieceTokenizer::HandleTheRemainingStringOnTriePath().

  if (vocab_->Size() > fast_wordpiece_tokenizer_utils::kMaxSupportedVocabSize) {
    return absl::FailedPreconditionError(
        absl::StrCat("Vocab size exceeds the max supported (",
                     fast_wordpiece_tokenizer_utils::kMaxSupportedVocabSize,
                     "). Found vocab size: ", vocab_->Size(), "."));
  }

  // Collect a subset of tokens (and variations) to build the trie.
  std::vector<TrieVocabToken> tokens_to_build_trie;
  tokens_to_build_trie.reserve(vocab_->Size());
  for (int token_id = 0; token_id < vocab_->Size(); ++token_id) {
    const absl::optional<absl::string_view> word = vocab_->LookupWord(token_id);
    if (!word.has_value()) {
      return absl::FailedPreconditionError(
          "Impossible. `token_id` is definitely within the range of vocab "
          "token ids; hence LookupWord() should always succeed.");
    }
    if (word->empty()) {
      // It does not make sense to add the empty string "" to the vocabulary. In
      // addition, darts_clone does not allow an empty Trie key.
      //
      // We allow this only for compatibility with the original Wordpiece
      // algorithm.
      LOG(WARNING)
          << "The empty string is found in the vocabulary, which takes place "
             "in the token id space but will never be used in the result. "
             "Consider cleaning it from the vocabulary.";
      continue;
    }
    if (*word == suffix_indicator_) {
      // In real-life cases, no need to add the suffix indicator string (e.g.,
      // "##") to the vocabulary.
      //
      // We allow this only for compatibility with the original Wordpiece
      // algorithm.
      LOG(WARNING)
          << "The empty suffix token is found in the vocabulary, which takes "
             "place in token id space but will (almost) never be used in the "
             "result. Consider cleaning it from the vocabulary.";

      // The token id of the suffix indicator is used only when the input is
      // the suffix indicator itself. That case is handled elsewhere, in
      // PrecomputeResultForSuffixIndicator().
      //
      // Therefore, we don't insert the suffix indicator string as a key into
      // the trie. As a result, `trie_suffix_root_` node will never have data.

      continue;
    }
    TrieVocabToken vocab_token(*word, token_id, suffix_indicator_);
    if (vocab_token.TokenLengthWithoutSuffixIndicator() >
        fast_wordpiece_tokenizer_utils::kMaxVocabTokenLengthInUTF8Bytes) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Vocab token utf8 length (excluding suffix indicator) exceeds the "
          "max supported (",
          fast_wordpiece_tokenizer_utils::kMaxVocabTokenLengthInUTF8Bytes,
          "). The vocab token is: ", *word,
          " with utf8 length (excluding suffix indicator): ",
          vocab_token.TokenLengthWithoutSuffixIndicator(), "."));
    }
    // Skip word that contains punctuation but is not a punctuation itself.
    // <unk>, <pad>, ##. are skipped in this step.
    if (!no_pretokenization_ && vocab_token.ContainsPunctuation() &&
        (vocab_token.TokenUnicodeLengthWithoutSuffixIndicator() > 1 ||
         vocab_token.IsSuffixToken())) {
      continue;
    }
    tokens_to_build_trie.emplace_back(vocab_token);
  }

  if (tokens_to_build_trie.empty()) {
    return absl::FailedPreconditionError(
        "No valid vocab tokens were found to build the trie.");
  }
  if (!suffix_indicator_.empty()) {
    const bool suffix_token_exists = std::any_of(
        tokens_to_build_trie.begin(), tokens_to_build_trie.end(),
        [](const TrieVocabToken& token) { return token.IsSuffixToken(); });
    if (!suffix_token_exists) {
      // No suffix tokens in the vocab.  That would lead to no trie node for
      // the suffix indicator, which creates corner cases in the inference.
      // To prevent that, we add a dummy suffix token, e.g., "##" +
      // kInvalidControlChar (if the suffix indicator is "##"), which is never
      // matched during inference.
      tokens_to_build_trie.emplace_back(TrieVocabToken(
          absl::StrCat(suffix_indicator_, std::string(1, kInvalidControlChar)),
          unk_token_id_, suffix_indicator_));
    }
  }

  if (!no_pretokenization_) {
    // Special treatment for all Unicode punctuation chars that are not already
    // in the trie.
    // The maximum codepoint in Unicode is 0x0010FFFF.
    for (UChar32 cp = 1; cp <= 0x0010FFFF; ++cp) {
      if (!U_IS_UNICODE_CHAR(cp) ||
          !fast_wordpiece_tokenizer_utils::IsPunctuationOrChineseChar(cp)) {
        continue;
      }
      // Get the UTF8 encoding of the codepoint cp.
      char buf[4];
      int len = 0;
      U8_APPEND_UNSAFE(buf, len, cp);
      absl::string_view buf_view(buf, len);
      // Set the token id of punctuation chars that don't exist in the vocab as
      // unk_token_id_.
      if (!vocab_->LookupId(buf_view)) {
        TrieVocabToken vocab_token(buf_view, unk_token_id_, suffix_indicator_);
        tokens_to_build_trie.emplace_back(vocab_token);
      }
    }
    // Insert a dummy node to serve as the failure link targets for punctuation
    // nodes.
    tokens_to_build_trie.emplace_back(TrieVocabToken(
        std::string(1, kInvalidControlChar), unk_token_id_, suffix_indicator_));
  }
  return tokens_to_build_trie;
}

absl::Status FastWordpieceBuilder::ConstructTrie(
    const std::vector<TrieVocabToken>& tokens_to_build_trie) {
  std::vector<std::string> keys;
  std::vector<int> values;
  for (const TrieVocabToken& vocab_token : tokens_to_build_trie) {
    keys.emplace_back(vocab_token.Token());
    SH_ASSIGN_OR_RETURN(int encoded_value,
                        fast_wordpiece_tokenizer_utils::EncodeToken(
                            vocab_token.TokenId(),
                            vocab_token.TokenLengthWithoutSuffixIndicator(),
                            vocab_token.IsSuffixToken()));
    values.push_back(encoded_value);
  }
  SH_ASSIGN_OR_RETURN(trie_array_,
                      trie_utils::BuildDartsCloneTrie(keys, values));
  SH_ASSIGN_OR_RETURN(
      trie_utils::DartsCloneTrieWrapper trie,
      trie_utils::DartsCloneTrieWrapper::Create(trie_array_.data()));
  trie_.emplace(std::move(trie));

  if (trie_array_.size() >
      fast_wordpiece_tokenizer_utils::kMaxSupportedTrieSize) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Not supported since the constructed Darts trie size (",
        trie_array_.size(), ") is greater than the maximum supported size (",
        fast_wordpiece_tokenizer_utils::kMaxSupportedTrieSize, ")."));
  }

  // Locate the trie suffix root.
  auto node = trie_->CreateTraversalCursorPointToRoot();
  if (!trie_->TryTraverseSeveralSteps(node, suffix_indicator_)) {
    return absl::FailedPreconditionError(
        "Cannot locate trie_suffix_root_. This should never happen.");
  }
  trie_suffix_root_ = node.node_id;

  if (!no_pretokenization_) {
    // Locate the dummy node for the failure link for punctuation nodes.
    node = trie_->CreateTraversalCursorPointToRoot();
    if (!trie_->TryTraverseSeveralSteps(node,
                                        std::string(1, kInvalidControlChar))) {
      return absl::FailedPreconditionError(
          "Cannot locate the dummy node for the failure link for punctuation "
          "nodes. This should never happen.");
    }
    trie_punct_failure_link_node_ = node.node_id;

    // We make `trie_punct_failure_link_node_` a standalone dummy node.
    EraseValueOfNode(trie_punct_failure_link_node_);
    BreakTrieLinkFromParentToChild(trie_punct_failure_link_node_);
  }
  return absl::OkStatus();
}

absl::Status FastWordpieceBuilder::BuildOutgoingEdgeLabelsAlongVocabToken(
    const TrieVocabToken& vocab_token,
    std::vector<absl::flat_hash_set<char>>& node_outgoing_edge_labels) {
  const absl::string_view token = vocab_token.Token();
  trie_utils::DartsCloneTrieWrapper::TraversalCursor cur_node;
  int char_pos = 0;
  trie_->SetTraversalCursor(cur_node, trie_->kRootNodeId);
  while (char_pos < token.size()) {
    const char edge_label = token[char_pos];
    node_outgoing_edge_labels[cur_node.node_id].insert(edge_label);
    if (!trie_->TryTraverseOneStep(cur_node, edge_label)) {
      // Should never happen, since we built trie using all of `vocab_token`.
      return absl::FailedPreconditionError(absl::StrCat(
               "Cannot traverse from parent id ", cur_node.node_id,
               " to child following the edge with label value of ",
               static_cast<int>(edge_label),
               " when processing a vocabulary token with token ID ",
               vocab_token.TokenId(), " (0-based). This error happened at ",
               "position ", char_pos, " (0-based) of the token. Before that, ",
               "the prefix \"", token.substr(0, char_pos),
               "\" of the token had been processed. This should never happen. ",
               "This probably indicates that there are some unicode ",
               "issues (e.g., byte '\\x0' in the middle) for the above ",
               "mentioned token in the vocabulary file. All bytes of this ",
               "questionable token (ID ", vocab_token.TokenId(), ") are: [",
               absl::StrJoin(
                   iter::imap([](auto ch) { return static_cast<int>(ch); },
                              vocab_token.Token()),
                   ", "),
               "]."));
    }
    ++char_pos;
  }
  // Record whether the current node represents a punctuation char in the map.
  node_id_is_punc_map_[cur_node.node_id] =
      !vocab_token.IsSuffixToken() && vocab_token.ContainsPunctuation() &&
      vocab_token.TokenUnicodeLengthWithoutSuffixIndicator() == 1;
  return absl::OkStatus();
}

absl::StatusOr<std::vector<absl::flat_hash_set<char>>>
FastWordpieceBuilder::BuildOutgoingEdgeLabelsForTrie(
    const std::vector<TrieVocabToken>& tokens_to_build_trie) {
  std::vector<absl::flat_hash_set<char>> node_outgoing_edge_labels(
      trie_array_.size());
  const std::string dummy_token_for_trie_punct_failure_link_node =
      std::string(1, kInvalidControlChar);
  for (const TrieVocabToken& vocab_token : tokens_to_build_trie) {
    if (vocab_token.Token() == dummy_token_for_trie_punct_failure_link_node)
      continue;
    SH_RETURN_IF_ERROR(BuildOutgoingEdgeLabelsAlongVocabToken(
        vocab_token, node_outgoing_edge_labels));
  }
  return node_outgoing_edge_labels;
}

// Computes failure links and failure pops using BFS traversal.
absl::Status FastWordpieceBuilder::BuildFailureStructure(
    const std::vector<TrieVocabToken>& tokens_to_build_trie) {
  // Build the set of outgoing edge labels for each trie node (node_id ->
  // set<char>). This is needed by BFS because darts-clone does not provide an
  // API to enumerate the outgoing links for a node.
  SH_ASSIGN_OR_RETURN(
      std::vector<absl::flat_hash_set<char>> node_outgoing_edge_labels,
      BuildOutgoingEdgeLabelsForTrie(tokens_to_build_trie));

  failure_struct_array_.resize(trie_array_.size());
  // Initialize the BFS queue.
  std::queue<uint32_t> bfs_queue({trie_->kRootNodeId});
  if (trie_suffix_root_ != trie_->kRootNodeId) {
    // When `suffix_indicator_` is empty, `trie_suffix_root_` will collapse
    // with root. In this case, we don't visit it twice.
    //
    // In addition, we have ensured that `trie_suffix_root_` will never be null.
    // See PrepareVocabTokensToBuildTrie().
    bfs_queue.push(trie_suffix_root_);
  }

  // The BFS loop.
  while (!bfs_queue.empty()) {
    uint32_t parent_id = bfs_queue.front();
    bfs_queue.pop();

    // Explore the children of the parent node.
    //
    // Fix the iteration order of the outgoing edges to ensure that the model is
    // always built in the same way (i.e., visiting nodes in the same order).
    std::vector<char> outgoing_labels_sorted(
        node_outgoing_edge_labels[parent_id].begin(),
        node_outgoing_edge_labels[parent_id].end());
    std::sort(outgoing_labels_sorted.begin(), outgoing_labels_sorted.end());
    for (const char edge_label : outgoing_labels_sorted) {
      auto child_node = trie_->CreateTraversalCursor(parent_id);
      if (!trie_->TryTraverseOneStep(child_node, edge_label)) {
        // Should never happen, due to how we built `node_outgoing_edge_labels`;
        // see BuildOutgoingEdgeLabelsAlongVocabToken().
        return absl::FailedPreconditionError(absl::StrCat(
            "Failed to traverse to child following edge ",
            absl::string_view(&edge_label, 1), " at parent ", parent_id, "."));
      }
      if (child_node.node_id == trie_suffix_root_) {
        // Avoid visiting `trie_suffix_root_` twice.
        continue;
      }

      // For the child node v, compute failure link f(v) and failure pops F(v).
      //
      // In the comments below, str(v) is the string on the path from the trie
      // root to the node v, and V is the vocabulary used to build the trie.

      int child_data_value = -1;
      if (trie_->TryGetData(child_node, child_data_value)) {
        uint32_t failure_link = trie_suffix_root_;
        // Check whether the current node represents a punctuation char.
        // Since the current node has data and thus corresponds to some token,
        // it must be in the map `node_id_is_punc_map_`
        if (!node_id_is_punc_map_.contains(child_node.node_id)) {
          return absl::FailedPreconditionError(
              "Failed to find if an end node in the trie is a punctuation char "
              "in node_id_is_punc_map_. It should never happen.");
        }
        if (!no_pretokenization_ &&
            node_id_is_punc_map_.at(child_node.node_id)) {
          // For end-to-end tokenizer, we set the failure link node of every
          // punctuation char as a special node trie_punct_failure_link_node_
          // which is a dummy node (no parent, no descendants, failure link is
          // null). Hence, by detecting the landing node, we know we just
          // matched a punctuation char. We then split it as a single word.
          failure_link = trie_punct_failure_link_node_;
        }
        // Case 1 (easy): str(v) is in V. Assume that during tokenization of a
        // word, we reached node v, but can't continue further, because the
        // current char from the input word does not match any of the edges
        // outgoing from v. In that case, str(v) is already the max match, so
        // it's the only wordpiece we add to the list of wordpieces we committed
        // to. Hence, F(v) = [str(v)]. The next wordpiece from the current word
        // is a suffix, so we move to node f(v) = trie_suffix_root_, which
        // represents the suffix indicator (e.g., "##"), from where we continue
        // the match process. In summary, we have:
        //  * f(v) = trie_suffix_root_.
        //  * F(v) = [str(v)].
        SH_RETURN_IF_ERROR(AssignFailureLinkAndPops(
            /*cur_node=*/child_node.node_id, /*failure_link=*/failure_link,
            /*one_step_pops=*/{child_data_value},
            /*parent_failure_pops_offset_length=*/
            fast_wordpiece_tokenizer_utils::kNullFailurePopsList));
        bfs_queue.push(child_node.node_id);
        continue;
      }

      // Case 2 (complex): str(v) is not in V.
      //
      // Consider the same scenario as in Case 1, where we can't continue
      // further from v, but now, str(v) is not a valid wordpiece. Instead,
      // we need to consider the wordpieces that the MaxMatch algorithm would
      // generate for the beginning of str(v) (these wordpieces are stored in
      // F(v)). f(v) (the state we transit to) should correspond to the trie
      // node for the remaining suffix of str(v).
      //
      // We could compute F(v) and f(v) by running the original WordPiece
      // algorithm. Instead, we do it even faster, by using F(u) and f(u) (the
      // similar info for the parent node u). Intuitively F(v) consists of (1)
      // the tokens from F(u) and (2) the possible tokens that the MaxMatch
      // algorithm would generate for str(f(u)).c, where str(f(u)) is the suffix
      // of str(u) not covered by the concatenation of the tokens from F(u), "."
      // means concatenation, and c is the edge label character from u to v.
      //
      //
      // Let u be the parent node, and c be the edge label from u to v. To
      // compute f(v) and F(v), the loop below uses a node variable z (called
      // `itr_node`) and a list G (called `one_steps_pops`). Initially, z is set
      // to be f(u), and G is empty.
      //  1. If z is null, f(v) will be null, too (see Note 2 below for what
      //  this means). We're done.
      //  2. Check if there is a trie edge out of node z, for label c, leading
      //    to node goto(z, c). If so, set f(v) = goto(z,c) and F(v) = F(u) + G.
      //    We're done and break.
      //  3. Otherwise, collect the pop tokens (by G = G + F(z)) and
      //    follows the failure link (by z = f(z)).
      //  4. Goes to Step 1 and continue the loop.
      //
      // Note 1: processing node v depends on the info for nodes z that are
      // closer to the root than v. Due to our use of the BFS traversal, that
      // info is guaranteed to exist when we examine node v.
      //
      // Note 2: f(v) is null means that during the tokenization process of some
      // input word, if the trie matching cannot continue at node v, there are
      // no failure links that we can follow, and (it can be proved that in such
      // a case) the input word can't be tokenized with the current vocab.
      //
      // For formal discussions and proofs, please refer to the academic paper
      // https://arxiv.org/abs/2012.15524
      const FailureStruct& parent_fs = failure_struct_array_[parent_id];
      if (parent_fs.failure_link != fast_wordpiece_tokenizer_utils::kNullNode) {
        std::vector<int> one_step_pops;
        auto itr_node = trie_->CreateTraversalCursor(parent_fs.failure_link);
        while (true) {
          if (trie_->TryTraverseOneStep(itr_node, edge_label)) {
            // Set the failure link and failure pops for `child_node`.
            SH_RETURN_IF_ERROR(AssignFailureLinkAndPops(
                /*cur_node=*/child_node.node_id,
                /*failure_link=*/itr_node.node_id, one_step_pops,
                parent_fs.failure_pops_offset_length));
            break;
          }
          const FailureStruct& itr_node_fs =
              failure_struct_array_[itr_node.node_id];
          if (itr_node_fs.failure_link ==
              fast_wordpiece_tokenizer_utils::kNullNode) {
            // Cannot follow anymore: failure link of `child_node` will be null.
            break;
          }
          // Append the failure pops of `itr_node` to `one_step_pops`.
          GetFailurePopsAndAppendToOut(itr_node_fs.failure_pops_offset_length,
                                       one_step_pops);
          // Follow the failure link.
          trie_->SetTraversalCursor(itr_node, itr_node_fs.failure_link);
        }
      }

      bfs_queue.push(child_node.node_id);
    }
  }

  if (!no_pretokenization_ && !suffix_indicator_.empty()) {
    // Rewire trie links along suffix_indicator_.
    // If the suffix indicator contains a punctuation char, let `u`--(`c`)-->`v`
    // be the first trie edge along the suffix indicator such that the edge
    // label (i.e. `c`) is a punctuation char. Note that `u`, `v` are trie
    // nodes. `c` is the edge label. We make the following change:
    //
    // Case 1: if `u` is the root, we remove the trie edge from `v` to its child
    // along the suffix indicator.
    // Case 2: if `u` is not the root, we remove the trie edge from `u` to `v`.
    //
    // Example 1: if suffix_indicator_ is "##" (as in BERT), we remove the trie
    // link from "#" to "##".  The goal here is to make sure we match the
    // punctuation character "#" as a token by itself, without matching "##"
    // (as we split by punctuation, "##" is not a valid token).
    // Example 2: if suffix_indicator is "foo#", we remove the trie link from
    // "foo" to "foo#".
    int cur_pos = 0;
    int next_pos = 0;
    bool prev_node_id_is_root = false;
    auto node = trie_->CreateTraversalCursorPointToRoot();
    UChar32 c;
    int suffix_indicator_length = suffix_indicator_.size();
    while (cur_pos < suffix_indicator_length) {
      next_pos = cur_pos;
      U8_NEXT(suffix_indicator_, next_pos, suffix_indicator_length, c);
      prev_node_id_is_root = (node.node_id == trie_->kRootNodeId);
      absl::string_view cur_unicode_char(suffix_indicator_.data() + cur_pos,
                                         next_pos - cur_pos);
      if (!trie_->TryTraverseSeveralSteps(node, cur_unicode_char)) {
        return absl::FailedPreconditionError(
            "Cannot locate a character in suffix_indicator_. It should never "
            "happen.");
      }
      if (fast_wordpiece_tokenizer_utils::IsPunctuationOrChineseChar(c)) {
        // If the previous node is a root node, read the next char to break the
        //  link from the current punctuation char to its next child node.
        if (prev_node_id_is_root) {
          cur_pos = next_pos;
          U8_FWD_1(suffix_indicator_, next_pos, suffix_indicator_length);
          const absl::string_view next_unicode_char(
              suffix_indicator_.data() + cur_pos, next_pos - cur_pos);
          auto child_node = node;
          if (!trie_->TryTraverseSeveralSteps(child_node, next_unicode_char)) {
            return absl::FailedPreconditionError(
                "Cannot locate a character in suffix_indicator_. It should "
                "never happen.");
          }
          BreakTrieLinkFromParentToChild(child_node.node_id);
        } else {
          BreakTrieLinkFromParentToChild(node.node_id);
        }
        break;
      }
      cur_pos = next_pos;
    }
  }
  return absl::OkStatus();
}

absl::Status FastWordpieceBuilder::AssignFailureLinkAndPops(
    uint32_t cur_node, uint32_t failure_link,
    const std::vector<int>& one_step_pops,
    int parent_failure_pops_offset_length) {
  if (failure_link == fast_wordpiece_tokenizer_utils::kNullNode) {
    return absl::OkStatus();
  }
  FailureStruct& cur_node_fs = failure_struct_array_[cur_node];
  cur_node_fs.failure_link = failure_link;

  // Let v be `cur_node` and u be the parent node.
  if (one_step_pops.empty()) {
    // Case 1: F(v) = F(u). So we just share the same vector.
    cur_node_fs.failure_pops_offset_length = parent_failure_pops_offset_length;
  } else {
    // Case 2: F(v) = F(u) + `one_step_pops`. We need to create a new vector and
    // append to `failure_pops_pool_`.
    const int failure_pops_offset = failure_pops_pool_.size();
    if (failure_pops_offset >
        fast_wordpiece_tokenizer_utils::kMaxSupportedFailurePoolOffset) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Failure pops list offset is ", failure_pops_offset,
          ", which exceeds maximum supported offset ",
          fast_wordpiece_tokenizer_utils::kMaxSupportedFailurePoolOffset,
          ". The vocabulary seems to be too large to be supported."));
    }
    // First copy F(u).
    GetFailurePopsAndAppendToOut(parent_failure_pops_offset_length,
                                 failure_pops_pool_);
    // Then append `one_step_pops`.
    failure_pops_pool_.insert(failure_pops_pool_.end(), one_step_pops.begin(),
                              one_step_pops.end());
    const int failure_pops_length =
        failure_pops_pool_.size() - failure_pops_offset;
    if (failure_pops_length >
        fast_wordpiece_tokenizer_utils::kMaxFailurePopsListSize) {
      // This should not happen, because `kBitsToEncodeFailurePopsListSize` is
      // set to be less than or equal to `kBitsToEncodeVocabTokenLength` (see
      // fast_wordpiece_tokenizer_utils.h).
      return absl::FailedPreconditionError(absl::StrCat(
          "Failure pops list size is ", failure_pops_length,
          ", which exceeds maximum supported size ",
          fast_wordpiece_tokenizer_utils::kMaxFailurePopsListSize, "."));
    }

    cur_node_fs.failure_pops_offset_length =
        fast_wordpiece_tokenizer_utils::EncodeFailurePopList(
            failure_pops_offset, failure_pops_length);
  }
  return absl::OkStatus();
}

void FastWordpieceBuilder::GetFailurePopsAndAppendToOut(
    uint32_t failure_pops_offset_length, std::vector<int>& out_failure_pops) {
  if (failure_pops_offset_length ==
      fast_wordpiece_tokenizer_utils::kNullFailurePopsList) {
    return;
  }
  int failure_pops_offset, failure_pops_length;
  fast_wordpiece_tokenizer_utils::GetFailurePopsOffsetAndLength(
      failure_pops_offset_length, failure_pops_offset, failure_pops_length);
  out_failure_pops.insert(
      out_failure_pops.end(), failure_pops_pool_.begin() + failure_pops_offset,
      failure_pops_pool_.begin() + failure_pops_offset + failure_pops_length);
}

absl::Status FastWordpieceBuilder::PrecomputeResultForSuffixIndicator() {
  std::vector<std::string> subwords;
  std::vector<int> begin_offset;
  std::vector<int> end_offset;
  int num_word_pieces;
  // Use the original WordPiece implementation.
  LookupStatus status = WordpieceTokenize(
      suffix_indicator_, max_bytes_per_token_, /*max_chars_per_subtoken=*/-1,
      suffix_indicator_, /*use_unknown_token=*/true, unk_token_,
      /*split_unknown_characters=*/false, &vocab_.value(), &subwords,
      &begin_offset, &end_offset, &num_word_pieces);
  precomputed_result_for_suffix_indicator_.reserve(subwords.size());
  if (!status.success) {
    return absl::FailedPreconditionError(status.error_msg);
  }
  for (int i = 0; i < subwords.size(); ++i) {
    const absl::optional<int> subword_id = vocab_->LookupId(subwords[i]);
    if (!subword_id.has_value()) {
      return absl::FailedPreconditionError(
          "Impossible because `subwords[i]` must be in the vocabulary!");
    }
    TrieVocabToken token(subwords[i], *subword_id, suffix_indicator_);
    SH_ASSIGN_OR_RETURN(
        int encoded_value,
        fast_wordpiece_tokenizer_utils::EncodeToken(
            token.TokenId(), token.TokenLengthWithoutSuffixIndicator(),
            token.IsSuffixToken()));
    precomputed_result_for_suffix_indicator_.push_back(encoded_value);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> FastWordpieceBuilder::ExportToFlatBuffer() const {
  flatbuffers::FlatBufferBuilder builder;

  const auto trie_array = builder.CreateVector(trie_array_);
  std::vector<tensorflow::text::FailureStruct> failure_struct_fbs_vector;
  failure_struct_fbs_vector.reserve(failure_struct_array_.size());
  for (const auto& item : failure_struct_array_) {
    failure_struct_fbs_vector.emplace_back(item.failure_link,
                                           item.failure_pops_offset_length);
  }
  const auto failure_structure_array =
      builder.CreateVectorOfStructs(failure_struct_fbs_vector);
  const auto failure_pops_pool = builder.CreateVector(failure_pops_pool_);
  const auto precomputed_result_for_suffix_indicator =
      builder.CreateVector(precomputed_result_for_suffix_indicator_);
  const auto suffix_indicator = builder.CreateString(suffix_indicator_);
  const auto unk_token = builder.CreateString(unk_token_);

  std::vector<flatbuffers::Offset<flatbuffers::String>> vocab_fbs_vector;
  std::vector<bool> vocab_is_suffix_fbs_vector;

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
      bool is_suffix_token = false;
      if (!suffix_indicator_.empty() && token != suffix_indicator_ &&
          absl::StartsWith(token, suffix_indicator_)) {
        is_suffix_token = true;
        // For suffix tokens, we remove the suffix indicator to save spac and
        // for ease of use in detokenization (where the suffix indicator will be
        // stripped anyway).
        token = token.substr(suffix_indicator_.size());
      }
      vocab_fbs_vector.emplace_back(builder.CreateString(token));
      vocab_is_suffix_fbs_vector.emplace_back(is_suffix_token);
    }
  }

  auto vocab_array = builder.CreateVector(vocab_fbs_vector);
  auto vocab_is_suffix_array = builder.CreateVector(vocab_is_suffix_fbs_vector);

  FastWordpieceTokenizerConfigBuilder wtcb(builder);
  wtcb.add_trie_array(trie_array);
  wtcb.add_failure_struct_array(failure_structure_array);
  wtcb.add_failure_pops_pool(failure_pops_pool);
  wtcb.add_trie_suffix_root(trie_suffix_root_);
  wtcb.add_trie_punct_failure_link_node(trie_punct_failure_link_node_);

  wtcb.add_max_bytes_per_token(max_bytes_per_token_);
  wtcb.add_suffix_indicator(suffix_indicator);
  wtcb.add_unk_token(unk_token);
  wtcb.add_unk_token_id(unk_token_id_);
  wtcb.add_precomputed_result_for_suffix_indicator(
      precomputed_result_for_suffix_indicator);
  wtcb.add_end_to_end(!no_pretokenization_);
  wtcb.add_support_detokenization(support_detokenization_);
  wtcb.add_vocab_array(vocab_array);
  wtcb.add_vocab_is_suffix_array(vocab_is_suffix_array);
  FinishFastWordpieceTokenizerConfigBuffer(builder, wtcb.Finish());
  return std::string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                     builder.GetSize());
}
}  // namespace

absl::StatusOr<std::string> BuildModelAndExportToFlatBuffer(
    const std::vector<std::string>& vocab, int max_bytes_per_token,
    absl::string_view suffix_indicator, absl::string_view unk_token,
    bool no_pretokenization, bool support_detokenization) {
  FastWordpieceBuilder builder;
  SH_RETURN_IF_ERROR(builder.BuildModel(vocab, max_bytes_per_token,
                                        suffix_indicator, unk_token,
                                        no_pretokenization,
                                        support_detokenization));
  SH_ASSIGN_OR_RETURN(std::string flatbuffer, builder.ExportToFlatBuffer());
  return flatbuffer;
}

}  // namespace text
}  // namespace tensorflow
