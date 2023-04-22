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

#include "tensorflow/c/experimental/filesystem/plugins/gcs/expiring_lru_cache.h"

#include <memory>

#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/cloud/now_seconds_env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(ExpiringLRUCacheTest, MaxAge) {
  const string key = "a";
  std::unique_ptr<NowSecondsEnv> env(new NowSecondsEnv);
  tf_gcs_filesystem::ExpiringLRUCache<int> cache(
      1, 0, [&env]() { return env->NowSeconds(); });
  env->SetNowSeconds(1);
  // Verify that replacement of an existing element works, and updates the
  // timestamp of the entry.
  cache.Insert(key, 41);
  env->SetNowSeconds(2);
  cache.Insert(key, 42);
  // 1 second after the most recent insertion, the entry is still valid.
  env->SetNowSeconds(3);
  int value = 0;
  EXPECT_TRUE(cache.Lookup(key, &value));
  EXPECT_EQ(value, 42);
  // 2 seconds after the most recent insertion, the entry is no longer valid.
  env->SetNowSeconds(4);
  EXPECT_FALSE(cache.Lookup(key, &value));
  // Re-insert the entry.
  cache.Insert(key, 43);
  EXPECT_TRUE(cache.Lookup(key, &value));
  EXPECT_EQ(value, 43);
  // The entry is valid 1 second after the insertion...
  env->SetNowSeconds(5);
  value = 0;
  EXPECT_TRUE(cache.Lookup(key, &value));
  EXPECT_EQ(value, 43);
  // ...but is no longer valid 2 seconds after the insertion.
  env->SetNowSeconds(6);
  EXPECT_FALSE(cache.Lookup(key, &value));
}

TEST(ExpiringLRUCacheTest, MaxEntries) {
  // max_age of 0 means nothing will be cached.
  tf_gcs_filesystem::ExpiringLRUCache<int> cache1(0, 4);
  cache1.Insert("a", 1);
  int value = 0;
  EXPECT_FALSE(cache1.Lookup("a", &value));
  // Now set max_age = 1 and verify the LRU eviction logic.
  tf_gcs_filesystem::ExpiringLRUCache<int> cache2(1, 4);
  cache2.Insert("a", 1);
  cache2.Insert("b", 2);
  cache2.Insert("c", 3);
  cache2.Insert("d", 4);
  EXPECT_TRUE(cache2.Lookup("a", &value));
  EXPECT_EQ(value, 1);
  EXPECT_TRUE(cache2.Lookup("b", &value));
  EXPECT_EQ(value, 2);
  EXPECT_TRUE(cache2.Lookup("c", &value));
  EXPECT_EQ(value, 3);
  EXPECT_TRUE(cache2.Lookup("d", &value));
  EXPECT_EQ(value, 4);
  // Insertion of "e" causes "a" to be evicted, but the other entries are still
  // there.
  cache2.Insert("e", 5);
  EXPECT_FALSE(cache2.Lookup("a", &value));
  EXPECT_TRUE(cache2.Lookup("b", &value));
  EXPECT_EQ(value, 2);
  EXPECT_TRUE(cache2.Lookup("c", &value));
  EXPECT_EQ(value, 3);
  EXPECT_TRUE(cache2.Lookup("d", &value));
  EXPECT_EQ(value, 4);
  EXPECT_TRUE(cache2.Lookup("e", &value));
  EXPECT_EQ(value, 5);
}

TEST(ExpiringLRUCacheTest, LookupOrCompute) {
  // max_age of 0 means we should always compute.
  uint64 num_compute_calls = 0;
  tf_gcs_filesystem::ExpiringLRUCache<int>::ComputeFunc compute_func =
      [&num_compute_calls](const string& key, int* value, TF_Status* status) {
        *value = num_compute_calls;
        num_compute_calls++;
        return TF_SetStatus(status, TF_OK, "");
      };
  tf_gcs_filesystem::ExpiringLRUCache<int> cache1(0, 4);

  int value = -1;
  TF_Status status;
  cache1.LookupOrCompute("a", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 0);
  EXPECT_EQ(num_compute_calls, 1);
  // re-read the same value, expect another lookup
  cache1.LookupOrCompute("a", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 1);
  EXPECT_EQ(num_compute_calls, 2);

  // Define a new cache with max_age > 0 and verify correct behavior.
  tf_gcs_filesystem::ExpiringLRUCache<int> cache2(2, 4);
  num_compute_calls = 0;
  value = -1;

  // Read our first value
  cache2.LookupOrCompute("a", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 0);
  EXPECT_EQ(num_compute_calls, 1);
  // Re-read, exepct no additional function compute_func calls.
  cache2.LookupOrCompute("a", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 0);
  EXPECT_EQ(num_compute_calls, 1);

  // Read a sequence of additional values, eventually evicting "a".
  cache2.LookupOrCompute("b", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 1);
  EXPECT_EQ(num_compute_calls, 2);
  cache2.LookupOrCompute("c", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 2);
  EXPECT_EQ(num_compute_calls, 3);
  cache2.LookupOrCompute("d", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 3);
  EXPECT_EQ(num_compute_calls, 4);
  cache2.LookupOrCompute("e", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 4);
  EXPECT_EQ(num_compute_calls, 5);
  // Verify the other values remain in the cache.
  cache2.LookupOrCompute("b", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 1);
  EXPECT_EQ(num_compute_calls, 5);
  cache2.LookupOrCompute("c", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 2);
  EXPECT_EQ(num_compute_calls, 5);
  cache2.LookupOrCompute("d", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 3);
  EXPECT_EQ(num_compute_calls, 5);

  // Re-read "a", ensure it is re-computed.
  cache2.LookupOrCompute("a", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 5);
  EXPECT_EQ(num_compute_calls, 6);
}

TEST(ExpiringLRUCacheTest, Clear) {
  tf_gcs_filesystem::ExpiringLRUCache<int> cache(1, 4);
  cache.Insert("a", 1);
  cache.Insert("b", 2);
  cache.Insert("c", 3);
  cache.Insert("d", 4);
  int value = 0;
  EXPECT_TRUE(cache.Lookup("a", &value));
  EXPECT_EQ(value, 1);
  EXPECT_TRUE(cache.Lookup("b", &value));
  EXPECT_EQ(value, 2);
  EXPECT_TRUE(cache.Lookup("c", &value));
  EXPECT_EQ(value, 3);
  EXPECT_TRUE(cache.Lookup("d", &value));
  EXPECT_EQ(value, 4);
  cache.Clear();
  EXPECT_FALSE(cache.Lookup("a", &value));
  EXPECT_FALSE(cache.Lookup("b", &value));
  EXPECT_FALSE(cache.Lookup("c", &value));
  EXPECT_FALSE(cache.Lookup("d", &value));
}

TEST(ExpiringLRUCacheTest, Delete) {
  // Insert an entry.
  tf_gcs_filesystem::ExpiringLRUCache<int> cache(1, 4);
  cache.Insert("a", 1);
  int value = 0;
  EXPECT_TRUE(cache.Lookup("a", &value));
  EXPECT_EQ(value, 1);

  // Delete the entry.
  EXPECT_TRUE(cache.Delete("a"));
  EXPECT_FALSE(cache.Lookup("a", &value));

  // Try deleting the entry again.
  EXPECT_FALSE(cache.Delete("a"));
  EXPECT_FALSE(cache.Lookup("a", &value));
}

}  // namespace
}  // namespace tensorflow
