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

#include "xla/tsl/platform/sha256.h"

#include <array>
#include <string>

#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/test.h"

namespace tsl {
namespace {

TEST(SHA256Test, EmptyString) {
  auto digest = SHA256::Hash("");
  std::string hex = absl::BytesToHexString(absl::string_view(
      reinterpret_cast<const char*>(digest.data()), digest.size()));
  EXPECT_EQ(hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST(SHA256Test, KnownVectorAbc) {
  auto digest = SHA256::Hash("abc");
  std::string hex = absl::BytesToHexString(absl::string_view(
      reinterpret_cast<const char*>(digest.data()), digest.size()));
  EXPECT_EQ(hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(SHA256Test, StreamingMatchesOneShot) {
  SHA256 hasher;
  hasher.Update("a");
  hasher.Update("b");
  hasher.Update("c");
  auto stream_digest = hasher.Digest();
  auto oneshot_digest = SHA256::Hash("abc");
  EXPECT_EQ(stream_digest, oneshot_digest);
}

TEST(SHA256Test, ResetWorkAsExpected) {
  SHA256 hasher;
  hasher.Update("abc");
  hasher.Reset();
  hasher.Update("");
  auto digest = hasher.Digest();
  EXPECT_EQ(digest, SHA256::Hash(""));
}

TEST(SHA256Test, DigestIsIdempotentAndNonDestructive) {
  SHA256 hasher;
  hasher.Update("a");

  // Calling Digest multiple times should return the exact same result.
  auto digest_a1 = hasher.Digest();
  auto digest_a2 = hasher.Digest();
  EXPECT_EQ(digest_a1, SHA256::Hash("a"));
  EXPECT_EQ(digest_a1, digest_a2);

  // Calling DigestString should also match and not destroy context.
  std::string digest_str1 = hasher.DigestString();
  std::string digest_str2 = hasher.DigestString();
  EXPECT_EQ(digest_str1, digest_str2);
  EXPECT_EQ(digest_str1,
            std::string(reinterpret_cast<const char*>(digest_a1.data()),
                        digest_a1.size()));

  // Continuing to update after Digest should correctly hash all accumulated
  // data.
  hasher.Update("b");
  auto digest_ab = hasher.Digest();
  EXPECT_EQ(digest_ab, SHA256::Hash("ab"));

  hasher.Update("c");
  auto digest_abc = hasher.Digest();
  EXPECT_EQ(digest_abc, SHA256::Hash("abc"));
}

}  // namespace
}  // namespace tsl
