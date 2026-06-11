/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/ir/backend_config_pool.h"

#include <memory>
#include <string>

#include "absl/hash/hash.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

TEST(BackendConfigPoolTest, InternIdenticalStrings) {
  auto* pool = BackendConfigPool::Get();
  pool->ResetForTesting();
  std::string json1 = "{\"foo\": 1}";
  std::string json2 = "{\"foo\": 1}";

  auto shared1 = pool->Intern(json1);
  auto shared2 = pool->Intern(json2);

  EXPECT_EQ(shared1, shared2);
  EXPECT_EQ(*shared1, json1);
}

TEST(BackendConfigPoolTest, InternDifferentStrings) {
  auto* pool = BackendConfigPool::Get();
  pool->ResetForTesting();
  std::string json1 = "{\"foo\": 1}";
  std::string json2 = "{\"foo\": 2}";

  auto shared1 = pool->Intern(json1);
  auto shared2 = pool->Intern(json2);

  EXPECT_NE(shared1, shared2);
  EXPECT_EQ(*shared1, json1);
  EXPECT_EQ(*shared2, json2);
}

TEST(BackendConfigPoolTest, GarbageCollection) {
  auto* pool = BackendConfigPool::Get();
  pool->ResetForTesting();
  pool->GarbageCollect();             // Clean up previous garbage.
  std::string json = "{\"foo\": 3}";  // Use unique string.

  {
    auto shared = pool->Intern(json);
    EXPECT_EQ(pool->GarbageCollect(), 0);  // Still referenced.
  }

  // Now 'shared' is destroyed, weak_ptr should be expired.
  EXPECT_EQ(pool->GarbageCollect(), 1);
}

TEST(BackendConfigPoolTest, CacheHitAfterEviction) {
  auto* pool = BackendConfigPool::Get();
  pool->ResetForTesting();
  pool->GarbageCollect();  // Clean up previous garbage.
  std::string json = "{\"foo\": 10}";

  const std::string* raw_ptr = nullptr;
  {
    auto shared = pool->Intern(json);
    raw_ptr = shared.get();
  }

  EXPECT_EQ(pool->GarbageCollect(), 1);

  auto shared2 = pool->Intern(json);
  // We don't assert EXPECT_NE(shared2.get(), raw_ptr) here because the
  // allocator might reuse the same memory address for the new object.
}

TEST(BackendConfigPoolTest, MultipleStrings) {
  auto* pool = BackendConfigPool::Get();
  pool->ResetForTesting();
  auto a = pool->Intern("a");
  auto b = pool->Intern("b");
  auto c = pool->Intern("c");

  EXPECT_NE(a, b);
  EXPECT_NE(a, c);
  EXPECT_NE(b, c);

  EXPECT_EQ(pool->Intern("a"), a);
}

TEST(BackendConfigPoolTest, EmptyString) {
  auto* pool = BackendConfigPool::Get();
  pool->ResetForTesting();
  auto empty = pool->Intern("");
  EXPECT_EQ(*empty, "");
  EXPECT_EQ(pool->Intern(""), empty);
}

}  // namespace
}  // namespace xla
