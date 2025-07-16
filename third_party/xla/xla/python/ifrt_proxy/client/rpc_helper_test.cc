// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/client/rpc_helper.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt_proxy/client/client_session.h"
#include "xla/python/ifrt_proxy/client/mock_client_session.h"
#include "xla/python/ifrt_proxy/client/version.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/test_utils.h"
#include "xla/python/ifrt_proxy/common/types.h"
#include "xla/python/ifrt_proxy/common/types.pb.h"
#include "tsl/platform/test.h"

using ::testing::_;
using ::testing::UnorderedElementsAre;

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

constexpr absl::Duration kMaxFlushTimeout = absl::Seconds(10);

void PausePeriodicFlushes() {
  // We want to (a) return 'paused=true' whenever the flusher thread tries to
  // find out whether flushing has been paused, and (b) wait for any ongoing
  // background flushes to complete. To achieve (b), we wait until the flusher
  // thread asks for the value of `paused` at least once.
  struct AtomicBool {
    absl::Mutex mu;
    bool b = false;
  };

  auto called_at_least_once = std::make_shared<AtomicBool>();
  auto periodic_flusher_pause_hook = [called_at_least_once](bool* paused) {
    *paused = true;
    absl::MutexLock l(&called_at_least_once->mu);
    called_at_least_once->b = true;
  };
  TestHookSet(TestHookName::kRpcBatcherPausePeriodicFlush,
              std::move(periodic_flusher_pause_hook));

  absl::MutexLock l(&called_at_least_once->mu);
  CHECK(called_at_least_once->mu.AwaitWithTimeout(
      absl::Condition(&called_at_least_once->b), kMaxFlushTimeout));
}

void ResumePeriodicFlushes() {
  TestHookClear(TestHookName::kRpcBatcherPausePeriodicFlush);
}

class RpcHelperTest : public ::testing::Test {
 public:
  RpcHelperTest() : requests_(kMaxFlushTimeout) {
    session_ = std::make_shared<MockClientSession>();
    IfrtProxyVersion version;
    version.set_protocol_version(kClientMaxVersion);
    version.set_ifrt_serdes_version_number(
        SerDesVersion::current().version_number().value());
    rpc_helper_ = std::make_shared<RpcHelper>(version, session_);
    EXPECT_CALL(*session_, Finish(_)).Times(1);
    ON_CALL(*session_, Enqueue)
        .WillByDefault([this](std::unique_ptr<IfrtRequest> req) {
          requests_.Push(std::move(req));
          return Future<ClientSession::Response>(
              absl::InternalError("Fake error response"));
        });
  }

  std::shared_ptr<MockClientSession> session_;
  std::shared_ptr<RpcHelper> rpc_helper_;
  TestQueue<std::unique_ptr<IfrtRequest>> requests_;
};

TEST_F(RpcHelperTest, BatchedPeriodicFlush) {
  PausePeriodicFlushes();
  rpc_helper_->Batch(RpcHelper::kDestructArray, ArrayHandle{1});
  rpc_helper_->Batch(RpcHelper::kDeleteArray, ArrayHandle{2});
  rpc_helper_->Batch(RpcHelper::kDestructArray, ArrayHandle{3});
  rpc_helper_->Batch(RpcHelper::kDeleteArray, ArrayHandle{4});
  rpc_helper_->Batch(RpcHelper::kDestructArray, ArrayHandle{9});
  rpc_helper_->Batch(RpcHelper::kDeleteArray, ArrayHandle{8});
  rpc_helper_->Batch(RpcHelper::kDestructArray, ArrayHandle{7});
  rpc_helper_->Batch(RpcHelper::kDeleteArray, ArrayHandle{6});
  ResumePeriodicFlushes();

  auto delete_req = requests_.Pop();
  auto destruct_req = requests_.Pop();

  if (destruct_req->has_delete_array_request()) {
    destruct_req.swap(delete_req);
  }

  EXPECT_THAT(destruct_req->destruct_array_request().array_handle(),
              UnorderedElementsAre(1, 3, 9, 7));
  EXPECT_THAT(delete_req->delete_array_request().array_handle(),
              UnorderedElementsAre(2, 4, 8, 6));
}

TEST_F(RpcHelperTest, BatchedNoPeriodicFlush) {
  PausePeriodicFlushes();
  rpc_helper_->Batch(RpcHelper::kDestructArray, ArrayHandle{1});
  rpc_helper_->Batch(RpcHelper::kDeleteArray, ArrayHandle{2});
  rpc_helper_->Batch(RpcHelper::kDestructArray, ArrayHandle{3});
  rpc_helper_->Batch(RpcHelper::kDeleteArray, ArrayHandle{4});
  rpc_helper_->Batch(RpcHelper::kDestructArray, ArrayHandle{9});
  rpc_helper_->Batch(RpcHelper::kDeleteArray, ArrayHandle{8});
  rpc_helper_->Batch(RpcHelper::kDestructArray, ArrayHandle{7});
  rpc_helper_->Batch(RpcHelper::kDeleteArray, ArrayHandle{6});

  // Send some non-batched request, which should flush all the batched requests.
  {
    auto dummy_request = std::make_unique<CheckFutureRequest>();
    dummy_request->set_future_handle(1);
    rpc_helper_->CheckFuture(std::move(dummy_request));
    requests_.AllowNonEmptyDestruction(/*allow=*/true);
  }

  auto delete_req = requests_.Pop();
  auto destruct_req = requests_.Pop();

  if (destruct_req->has_delete_array_request()) {
    destruct_req.swap(delete_req);
  }

  EXPECT_THAT(destruct_req->destruct_array_request().array_handle(),
              UnorderedElementsAre(1, 3, 9, 7));
  EXPECT_THAT(delete_req->delete_array_request().array_handle(),
              UnorderedElementsAre(2, 4, 8, 6));
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
