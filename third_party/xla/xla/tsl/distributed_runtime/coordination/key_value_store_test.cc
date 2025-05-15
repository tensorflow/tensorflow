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

#include "xla/tsl/distributed_runtime/coordination/key_value_store.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test.h"

namespace tsl {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Optional;
using ::testing::Pair;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

// Converts a list of KeyValueEntries into a list of pairs.
std::vector<std::pair<std::string, std::string>> AsPairs(
    absl::Span<const tensorflow::KeyValueEntry> entries) {
  std::vector<std::pair<std::string, std::string>> pairs;
  for (const tensorflow::KeyValueEntry& entry : entries) {
    pairs.push_back({entry.key(), entry.value()});
  }
  return pairs;
}

TEST(KeyValueStore, OverwritingInsertSucceeds) {
  KeyValueStore store;
  ASSERT_OK(store.Put("foo", "bar", /*allow_overwrite=*/true));
  EXPECT_THAT(store.Get("foo"), Optional(Eq("bar")));
  ASSERT_OK(store.Put("foo", "moo", /*allow_overwrite=*/true));
  EXPECT_THAT(store.Get("foo"), Optional(Eq("moo")));
}

TEST(KeyValueStore, OverwritingInsertFails) {
  KeyValueStore store;
  ASSERT_OK(store.Put("foo", "bar", /*allow_overwrite=*/false));
  EXPECT_THAT(store.Put("foo", "bar", /*allow_overwrite=*/false),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(KeyValueStore, GetExistingKey) {
  KeyValueStore store;
  ASSERT_OK(store.Put("foo", "bar", /*allow_overwrite=*/false));
  EXPECT_THAT(store.Get("foo"), Optional(Eq("bar")));
}

TEST(KeyValueStore, GetMissingKey) {
  KeyValueStore store;
  EXPECT_EQ(store.Get("foo"), std::nullopt);
}

TEST(KeyValueStore, GetPrefixNoMatch) {
  KeyValueStore store;
  ASSERT_OK(store.Put("a", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("c", "", /*allow_overwrite=*/true));
  EXPECT_THAT(AsPairs(store.GetPrefix("b")), IsEmpty());
}

TEST(KeyValueStore, GetPrefixExactMatch) {
  KeyValueStore store;
  ASSERT_OK(store.Put("a", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("b", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("bb", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("bbb", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("c", "", /*allow_overwrite=*/true));
  EXPECT_THAT(AsPairs(store.GetPrefix("b")),
              ElementsAre(Pair("b", ""), Pair("bb", ""), Pair("bbb", "")));
}

TEST(KeyValueStore, GetPrefixNoExactMatch) {
  KeyValueStore store;
  ASSERT_OK(store.Put("a", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("bb", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("bbb", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("c", "", /*allow_overwrite=*/true));
  EXPECT_THAT(AsPairs(store.GetPrefix("b")),
              ElementsAre(Pair("bb", ""), Pair("bbb", "")));
}

TEST(KeyValueStore, GetPrefixWithDirectoryFormat) {
  KeyValueStore store;
  ASSERT_OK(store.Put("a/1", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("b/1", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("b/2", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("b/3/4", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("b/3/5", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("c/1", "", /*allow_overwrite=*/true));
  EXPECT_THAT(AsPairs(store.GetPrefix("b/")),
              ElementsAre(Pair("b/1", ""), Pair("b/2", ""), Pair("b/3/4", ""),
                          Pair("b/3/5", "")));
}

TEST(KeyValueStore, AddCallbackAndThenKey) {
  KeyValueStore store;
  bool callback_called = false;
  store.AddCallbackForKey("foo",
                          [&](const absl::StatusOr<absl::string_view>& s) {
                            ASSERT_OK(s);
                            ASSERT_THAT(s, IsOkAndHolds("bar"));
                            callback_called = true;
                          });
  ASSERT_FALSE(callback_called);
  ASSERT_OK(store.Put("foo", "bar", /*allow_overwrite=*/true));
  EXPECT_TRUE(callback_called);
}

TEST(KeyValueStore, AddKeyThenCallback) {
  KeyValueStore store;
  ASSERT_OK(store.Put("foo", "bar", /*allow_overwrite=*/true));
  bool callback_called = false;
  store.AddCallbackForKey("foo",
                          [&](const absl::StatusOr<absl::string_view>& s) {
                            ASSERT_OK(s);
                            ASSERT_THAT(s, IsOkAndHolds("bar"));
                            callback_called = true;
                          });
  EXPECT_TRUE(callback_called);
}

TEST(KeyValueStore, DeleteKey) {
  KeyValueStore store;
  ASSERT_OK(store.Put("a", "", /*allow_overwrite=*/true));
  store.Delete("a");
  EXPECT_EQ(store.Get("a"), std::nullopt);
}

TEST(KeyValueStore, DeleteMissingKey) {
  KeyValueStore store;
  ASSERT_OK(store.Put("a", "", /*allow_overwrite=*/true));
  store.Delete("b");
}

TEST(KeyValueStore, DeletePrefixNoMatch) {
  KeyValueStore store;
  ASSERT_OK(store.Put("a", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("c", "", /*allow_overwrite=*/true));
  store.DeletePrefix("b");
  EXPECT_THAT(AsPairs(store.GetPrefix("")),
              ElementsAre(Pair("a", ""), Pair("c", "")));
}

TEST(KeyValueStore, DeletePrefixExactMatch) {
  KeyValueStore store;
  ASSERT_OK(store.Put("a", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("b", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("bb", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("bbb", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("c", "", /*allow_overwrite=*/true));
  store.DeletePrefix("b");
  EXPECT_THAT(AsPairs(store.GetPrefix("")),
              ElementsAre(Pair("a", ""), Pair("c", "")));
}

TEST(KeyValueStore, DeletePrefixNoExactMatch) {
  KeyValueStore store;
  ASSERT_OK(store.Put("a", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("bb", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("bbb", "", /*allow_overwrite=*/true));
  ASSERT_OK(store.Put("c", "", /*allow_overwrite=*/true));
  store.DeletePrefix("b");
  EXPECT_THAT(AsPairs(store.GetPrefix("")),
              ElementsAre(Pair("a", ""), Pair("c", "")));
}

TEST(KeyValueStore, CallbacksCalledOnDestruction) {
  bool callback_called = false;
  {
    KeyValueStore store;
    store.AddCallbackForKey(
        "foo", [&](const absl::StatusOr<absl::string_view>& s) {
          ASSERT_THAT(s, StatusIs(absl::StatusCode::kCancelled));
          callback_called = true;
        });
  }
  EXPECT_TRUE(callback_called);
}

}  // namespace
}  // namespace tsl
