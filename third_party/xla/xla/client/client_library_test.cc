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

#include "xla/client/client_library.h"

#include <optional>
#include <set>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/client/compile_only_client.h"
#include "xla/client/local_client.h"
#include "xla/service/backend.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;

int IntraOpThreads(LocalClient* client) {
  return client->backend().eigen_intra_op_thread_pool()->NumThreads();
}

class ClientLibraryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(platform_, PlatformUtil::GetPlatform("Host"));
    ClientLibrary::DestroyLocalInstances();
  }
  void TearDown() override { ClientLibrary::DestroyLocalInstances(); }

  LocalClientOptions Options(
      int intra_op_threads,
      const std::optional<std::set<int>>& allowed_devices = std::nullopt) {
    return LocalClientOptions(platform_, /*number_of_replicas=*/1,
                              intra_op_threads, allowed_devices);
  }

  se::Platform* platform_ = nullptr;
};

TEST_F(ClientLibraryTest, DestroyLocalInstanceAllowsRecreationWithNewOptions) {
  ASSERT_OK_AND_ASSIGN(LocalClient * client,
                       ClientLibrary::GetOrCreateLocalClient(Options(2)));
  EXPECT_EQ(IntraOpThreads(client), 2);

  // Without destroying, the cached instance is returned and the new options
  // are ignored.
  ASSERT_OK_AND_ASSIGN(LocalClient * cached,
                       ClientLibrary::GetOrCreateLocalClient(Options(3)));
  EXPECT_EQ(cached, client);
  EXPECT_EQ(IntraOpThreads(cached), 2);

  ClientLibrary::DestroyLocalInstance(platform_);

  ASSERT_OK_AND_ASSIGN(LocalClient * recreated,
                       ClientLibrary::GetOrCreateLocalClient(Options(3)));
  EXPECT_EQ(IntraOpThreads(recreated), 3);
}

TEST_F(ClientLibraryTest, CachesSeparatelyByAllowedDevices) {
  ASSERT_OK_AND_ASSIGN(
      LocalClient * all_devices_client,
      ClientLibrary::GetOrCreateLocalClient(Options(2, std::nullopt)));
  ASSERT_OK_AND_ASSIGN(
      LocalClient * device_0_client,
      ClientLibrary::GetOrCreateLocalClient(Options(3, std::set<int>{0})));
  EXPECT_NE(device_0_client, all_devices_client);
  EXPECT_EQ(IntraOpThreads(all_devices_client), 2);
  EXPECT_EQ(IntraOpThreads(device_0_client), 3);

  ASSERT_OK_AND_ASSIGN(
      LocalClient * cached_all_devices,
      ClientLibrary::GetOrCreateLocalClient(Options(4, std::nullopt)));
  ASSERT_OK_AND_ASSIGN(
      LocalClient * cached_device_0,
      ClientLibrary::GetOrCreateLocalClient(Options(4, std::set<int>{0})));
  EXPECT_EQ(cached_all_devices, all_devices_client);
  EXPECT_EQ(cached_device_0, device_0_client);
  EXPECT_EQ(ClientLibrary::GetXlaService(platform_),
            all_devices_client->local_service());
  EXPECT_EQ(ClientLibrary::GetXlaService(platform_, std::set<int>{0}),
            device_0_client->local_service());

  ClientLibrary::DestroyLocalInstance(platform_);

  ASSERT_OK_AND_ASSIGN(
      LocalClient * recreated_device_0,
      ClientLibrary::GetOrCreateLocalClient(Options(4, std::set<int>{0})));
  ASSERT_OK_AND_ASSIGN(
      LocalClient * device_0_and_1_client,
      ClientLibrary::GetOrCreateLocalClient(Options(5, std::set<int>{0, 1})));
  EXPECT_EQ(IntraOpThreads(recreated_device_0), 4);
  EXPECT_EQ(ClientLibrary::GetXlaService(platform_),
            recreated_device_0->local_service());
  EXPECT_EQ(ClientLibrary::GetXlaService(platform_, std::set<int>{0, 1}),
            device_0_and_1_client->local_service());
}

TEST_F(ClientLibraryTest, DestroyLocalInstanceKeepsCompileOnlyInstance) {
  ASSERT_OK_AND_ASSIGN(CompileOnlyClient * compile_only_client,
                       ClientLibrary::GetOrCreateCompileOnlyClient(platform_));
  ASSERT_THAT(ClientLibrary::GetOrCreateLocalClient(Options(2)), IsOk());

  ClientLibrary::DestroyLocalInstance(platform_);

  ASSERT_OK_AND_ASSIGN(CompileOnlyClient * compile_only_client_after,
                       ClientLibrary::GetOrCreateCompileOnlyClient(platform_));
  EXPECT_EQ(compile_only_client_after, compile_only_client);
}

TEST_F(ClientLibraryTest, DestroyLocalInstanceIsNoOpWithoutInstance) {
  ClientLibrary::DestroyLocalInstance(nullptr);
  ClientLibrary::DestroyLocalInstance(platform_);

  ASSERT_OK_AND_ASSIGN(LocalClient * client,
                       ClientLibrary::GetOrCreateLocalClient(Options(2)));
  EXPECT_EQ(IntraOpThreads(client), 2);
}

}  // namespace
}  // namespace xla
