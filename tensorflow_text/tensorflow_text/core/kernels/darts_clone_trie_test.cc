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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow_text/core/kernels/darts_clone_trie_builder.h"
#include "tensorflow_text/core/kernels/darts_clone_trie_wrapper.h"

namespace tensorflow {
namespace text {
namespace trie_utils {

using ::testing::status::StatusIs;

TEST(DartsCloneTrieTest, CreateCursorPointToRootAndTryTraverseOneStep) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc"};

  // Create the trie instance.
  ASSERT_OK_AND_ASSIGN(std::vector<uint32_t> trie_array,
                       BuildDartsCloneTrie(vocab_tokens));
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(trie_array.data()));

  DartsCloneTrieWrapper::TraversalCursor cursor;
  int data;

  cursor = trie.CreateTraversalCursorPointToRoot();  // Create a cursor to point
                                                     // to the root.
  EXPECT_TRUE(trie.TryTraverseOneStep(cursor, 'A'));
  EXPECT_FALSE(trie.TryGetData(cursor, data));
  EXPECT_TRUE(trie.TryTraverseOneStep(cursor, 'b'));
  EXPECT_FALSE(trie.TryGetData(cursor, data));
  EXPECT_TRUE(trie.TryTraverseOneStep(cursor, 'c'));
  EXPECT_TRUE(trie.TryGetData(cursor, data));
  EXPECT_THAT(data, 2);
  EXPECT_FALSE(trie.TryTraverseOneStep(cursor, 'c'));
}

TEST(DartsCloneTrieTest, CreateCursorAndTryTraverseSeveralSteps) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc"};

  // Create the trie instance.
  ASSERT_OK_AND_ASSIGN(std::vector<uint32_t> trie_array,
                       BuildDartsCloneTrie(vocab_tokens));
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(trie_array.data()));

  DartsCloneTrieWrapper::TraversalCursor cursor;
  int data;

  cursor = trie.CreateTraversalCursor(trie.kRootNodeId);  // Create a cursor to
                                                          // point to the root.
  EXPECT_TRUE(trie.TryTraverseSeveralSteps(cursor, "def"));
  EXPECT_TRUE(trie.TryGetData(cursor, data));
  EXPECT_THAT(data, 0);
}

TEST(DartsCloneTrieTest, TraversePathNotExisted) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc"};

  // Create the trie instance.
  ASSERT_OK_AND_ASSIGN(std::vector<uint32_t> trie_array,
                       BuildDartsCloneTrie(vocab_tokens));
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(trie_array.data()));

  DartsCloneTrieWrapper::TraversalCursor cursor;

  trie.SetTraversalCursor(
      cursor,
      trie.kRootNodeId);  // Use SetTraversalCursor() to point to the root.
  EXPECT_FALSE(trie.TryTraverseSeveralSteps(cursor, "dez"));
}

TEST(DartsCloneTrieTest, TraverseOnUtf8Path) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc"};

  // Create the trie instance.
  ASSERT_OK_AND_ASSIGN(std::vector<uint32_t> trie_array,
                       BuildDartsCloneTrie(vocab_tokens));
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(trie_array.data()));

  DartsCloneTrieWrapper::TraversalCursor cursor;
  int data;

  trie.SetTraversalCursor(
      cursor,
      trie.kRootNodeId);  // Use SetTraversalCursor() to point to the root.
  EXPECT_TRUE(trie.TryTraverseSeveralSteps(cursor, "\xe1\xb8\x8aZZ"));
  EXPECT_TRUE(trie.TryGetData(cursor, data));
  EXPECT_THAT(data, 1);
}

TEST(DartsCloneTrieTest, TraverseOnPartialUtf8Path) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc"};

  // Create the trie instance.
  ASSERT_OK_AND_ASSIGN(std::vector<uint32_t> trie_array,
                       BuildDartsCloneTrie(vocab_tokens));
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(trie_array.data()));

  DartsCloneTrieWrapper::TraversalCursor cursor;
  int data;

  trie.SetTraversalCursor(
      cursor,
      trie.kRootNodeId);  // Use SetTraversalCursor() to point to the root.
  EXPECT_TRUE(trie.TryTraverseSeveralSteps(cursor, "\xe1\xb8"));
  EXPECT_FALSE(trie.TryGetData(cursor, data));
}

TEST(DartsCloneTrieTest, TraverseOnUtf8PathNotExisted) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc"};

  // Create the trie instance.
  ASSERT_OK_AND_ASSIGN(std::vector<uint32_t> trie_array,
                       BuildDartsCloneTrie(vocab_tokens));
  ASSERT_OK_AND_ASSIGN(DartsCloneTrieWrapper trie,
                       DartsCloneTrieWrapper::Create(trie_array.data()));

  DartsCloneTrieWrapper::TraversalCursor cursor;

  trie.SetTraversalCursor(
      cursor,
      trie.kRootNodeId);  // Use SetTraversalCursor() to point to the root.
  EXPECT_FALSE(trie.TryTraverseSeveralSteps(cursor, "\xe1\xb8\x84"));
}

TEST(DartsCloneTrieBuildError, KeysValuesSizeDifferent) {
  // The test vocabulary.
  std::vector<std::string> keys{"def", "\xe1\xb8\x8aZZ", "Abc"};
  std::vector<int> values{1, 2, 3, 4};

  // Create the trie instance.
  ASSERT_THAT(BuildDartsCloneTrie(keys, values),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST(DartsCloneTrieBuildError, DuplicatedKeys) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc", "def"};

  // Create the trie instance.
  ASSERT_THAT(BuildDartsCloneTrie(vocab_tokens),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST(DartsCloneTrieBuildError, EmptyStringsInKeys) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc", ""};

  // Create the trie instance.
  ASSERT_THAT(BuildDartsCloneTrie(vocab_tokens),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST(DartsCloneTrieBuildError, NegativeValues) {
  // The test vocabulary.
  std::vector<std::string> vocab_tokens{"def", "\xe1\xb8\x8aZZ", "Abc"};
  std::vector<int> vocab_values{0, -1, 1};

  // Create the trie instance.
  ASSERT_THAT(BuildDartsCloneTrie(vocab_tokens, vocab_values),
              StatusIs(util::error::INVALID_ARGUMENT));
}

}  // namespace trie_utils
}  // namespace text
}  // namespace tensorflow
