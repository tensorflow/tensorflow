/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/service/gpu/kernel_reuse_cache.h"

#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using KernelReuseTest = ::testing::Test;

TEST_F(KernelReuseTest, ExportAndLoadWork) {
  KernelReuseCache cache;
  EXPECT_TRUE(cache.IsEmpty());
  auto [result, was_cached] = cache.GetWithStatus(
      "fingerprint", []() { return KernelReuseCache::Entry{}; });
  TF_EXPECT_OK(result);
  EXPECT_NE(result.value(), nullptr);
  EXPECT_FALSE(was_cached);
  EXPECT_FALSE(cache.IsEmpty());
  const CompilationCacheProto proto = cache.Export();
  cache.Clear();
  EXPECT_TRUE(cache.IsEmpty());
  TF_EXPECT_OK(cache.Load(proto));
  EXPECT_FALSE(cache.IsEmpty());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
