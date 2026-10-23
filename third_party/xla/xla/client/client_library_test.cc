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

  LocalClientOptions Options(int intra_op_threads) {
    return LocalClientOptions(platform_, /*number_of_replicas=*/1,
                              intra_op_threads);
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
