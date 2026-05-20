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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xla/service/gpu/kernel_reuse_cache.pb.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using KernelReuseTest = ::testing::Test;

TEST_F(KernelReuseTest, ExportAndLoadWork) {
  KernelReuseCache cache;
  EXPECT_TRUE(cache.IsEmpty());

  const std::string fingerprint = "fingerprint1";

  auto [promise, returned] = tsl::MakePromise<KernelReuseCache::Entry>();
  auto [future, was_cached] = cache.GetWithStatus(
      fingerprint, [returned = &returned]() { return *returned; });
  EXPECT_FALSE(was_cached);

  const std::string kernel_name = "kernel_name1";
  EXPECT_FALSE(future.IsReady());
  promise.Set(KernelReuseCache::Entry{kernel_name});

  TF_ASSERT_OK_AND_ASSIGN(const KernelReuseCache::Entry* result,
                          future.Await());
  EXPECT_THAT(result, testing::NotNull());
  EXPECT_EQ(result->kernel_name, kernel_name);
  EXPECT_FALSE(cache.IsEmpty());

  const CompilationCacheProto proto = cache.Export();
  cache.Clear();

  EXPECT_TRUE(cache.IsEmpty());

  EXPECT_THAT(proto, tsl::proto_testing::EquivToProto(R"pb(
                entries {
                  key: "kernel_name1"
                  value {
                    fingerprint: "fingerprint1"
                    launch_dimensions { num_blocks: 1 num_threads_per_block: 1 }
                    link_binary: true
                  }
                }
                compatibility_version: 1
              )pb"));

  TF_EXPECT_OK(cache.Load(proto));
  EXPECT_FALSE(cache.IsEmpty());

  {
    auto [future, was_cached] = cache.GetWithStatus(fingerprint, []() {
      return absl::UnimplementedError("Should be cached");
    });
    EXPECT_TRUE(was_cached);
    TF_ASSERT_OK_AND_ASSIGN(const KernelReuseCache::Entry* result,
                            future.Await());
    EXPECT_THAT(result, testing::NotNull());
    EXPECT_EQ(result->kernel_name, kernel_name);
  }
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
  EXPECT_TRUE(proto.ParseFromString(serialized));
  EXPECT_EQ(proto.entries_size(), 2);
}

}  // namespace
}  // namespace xla::gpu
