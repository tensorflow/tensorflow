/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tsl/distributed_runtime/persistent_cache_repository/fake_cache_repository.h"

#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/test.h"

namespace tsl {
namespace {

using Options = PersistentCacheRepository::Options;
using testing::IsOkAndHolds;
using testing::StatusIs;

TEST(FakeCacheRepositoryTest, PutThenGet) {
  FakeCacheRepository f(Options{});
  TF_EXPECT_OK(f.Put("k", "e"));
  EXPECT_THAT(f.Get("k"), IsOkAndHolds("e"));
}

TEST(FakeCacheRepositoryTest, MultiplePuts) {
  FakeCacheRepository f(Options{});
  TF_EXPECT_OK(f.Put("k", "e"));
  TF_EXPECT_OK(f.Put("k", "e"));  // idempotent
  EXPECT_THAT(f.Put("k", "e1"), StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(FakeCacheRepositoryTest, NotFound) {
  FakeCacheRepository f(Options{});
  EXPECT_THAT(f.Get("k"), StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace tsl
