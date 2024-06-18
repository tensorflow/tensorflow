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

#include <gtest/gtest.h>
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"

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

TEST_F(KernelReuseTest, UpdatingDiskKernelCacheWorks) {
  std::string cache_file_path;
  CHECK(tsl::Env::Default()->LocalTempFilename(&cache_file_path));
  {
    const CompilationCacheProto proto = [](std::string kernel_name) {
      KernelReuseCache cache;
      auto [result, was_cached] = cache.GetWithStatus("fingerprint", [&]() {
        return KernelReuseCache::Entry{.kernel_name = kernel_name};
      });
      return cache.Export();
    }("k1");
    TF_EXPECT_OK(UpdateDiskKernelCache(cache_file_path, /*do_append=*/false,
                                       proto,
                                       {{.name = "k1", .binary = {5, 6}}}));
  }
  {
    const CompilationCacheProto proto = [](std::string kernel_name) {
      KernelReuseCache cache;
      auto [result, was_cached] = cache.GetWithStatus("fingerprint", [&]() {
        return KernelReuseCache::Entry{.kernel_name = kernel_name};
      });
      return cache.Export();
    }("k2");
    TF_EXPECT_OK(UpdateDiskKernelCache(cache_file_path, /*do_append=*/true,
                                       proto,
                                       {{.name = "k2", .binary = {7, 8}}}));
  }
  std::string serialized;
  TF_EXPECT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), cache_file_path, &serialized));
  CompilationCacheProto proto;
  EXPECT_TRUE(proto.ParseFromString(std::string(serialized)));
  EXPECT_EQ(proto.entries_size(), 2);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
