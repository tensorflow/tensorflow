/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/distributed_runtime/preemption/preemption_sync_manager.h"

#include <memory>
#include <string>
#include <utility>

#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/support/channel_arguments.h"
#include "absl/memory/memory.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_agent.h"
#include "tensorflow/core/distributed_runtime/preemption/preemption_notifier.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.h"
#include "tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_service_impl.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/coordination_config.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {
namespace {

constexpr char kJobName[] = "test_worker";

// Send fake preemption notices at any time for testing.
class FakePreemptionNotifier : public PreemptionNotifier {
 public:
  ~FakePreemptionNotifier() override {
    mutex_lock l(mu_);
    NotifyRegisteredListeners(
        errors::Cancelled("~FakePreemptionNotifier() was called."));
  }

  void AnnounceDeath(absl::Time death_time) {
    LOG(WARNING) << "Received preemption notice with death time: "
                 << death_time;
    {
      mutex_lock l(mu_);
      death_time_ = death_time;
      NotifyRegisteredListeners(death_time_);
    }
  }

  void Reset() override {}
};

class PreemptionSyncManagerTest : public ::testing::Test {
 protected:
  PreemptionSyncManagerTest() {
    // Setup coordination service.
    StartCoordinationService();
    InitializeAndConnectCoordinationAgents();

    // Create preempt sync manager for task 1.
    auto preempt_notifier = std::make_unique<FakePreemptionNotifier>();
    preempt_notifier_ = preempt_notifier.get();
    TF_CHECK_OK(preempt_sync_mgr_->Initialize(
        Env::Default(), coord_agent_.get(), std::move(preempt_notifier)));

    // Create preempt sync manager for task 2.
    auto preempt_notifier2 = std::make_unique<FakePreemptionNotifier>();
    preempt_notifier2_ = preempt_notifier2.get();
    TF_CHECK_OK(preempt_sync_mgr2_->Initialize(
        Env::Default(), coord_agent2_.get(), std::move(preempt_notifier2)));
  }

  // `to_task1` toggles which of the two tasks receives preemption notice.
  void SendPreemptionNotice(absl::Time death_time = absl::Now(),
                            bool to_task1 = true) {
    if (to_task1) {
      preempt_notifier_->AnnounceDeath(death_time);
    } else {
      preempt_notifier2_->AnnounceDeath(death_time);
    }
    // Block main thread for a short while to allow preemption sync manager to
    // process the notice.
    Env::Default()->SleepForMicroseconds(
        absl::ToInt64Microseconds(absl::Milliseconds(1)));
  }

  // Report to coordiation service that task two is unhealthy.
  void SimulateUnhealthyTaskTwo() {
    CoordinatedTask task2;
    task2.set_job_name(kJobName);
    task2.set_task_id(2);
    TF_CHECK_OK(
        coord_service_->ReportTaskError(task2, errors::Internal("test_error")));
  }

  // Allow access to objects under test.
  std::unique_ptr<PreemptionSyncManager> preempt_sync_mgr_ =
      CreatePreemptionSyncManager();
  std::unique_ptr<PreemptionSyncManager> preempt_sync_mgr2_ =
      CreatePreemptionSyncManager();

 private:
  // Utility methods to set up coordination service and agents.
  void StartCoordinationService() {
    ::grpc::ServerBuilder builder;
    coord_service_ = EnableCoordinationService();
    coord_compute_pool_ = std::make_unique<thread::ThreadPool>(
        Env::Default(), "CoordinationServiceRpcHandler",
        /*num_threads=*/1);
    coord_rpc_service_ = std::make_unique<GrpcCoordinationServiceImpl>(
        coord_compute_pool_.get(), &builder);
    grpc_server_ = builder.BuildAndStart();
    coord_rpc_thread_ = absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/"CoordinationServiceHandleRPCsLoop",
        [service = coord_rpc_service_.get()]() { service->HandleRPCsLoop(); }));
  }
  std::unique_ptr<CoordinationServiceInterface> EnableCoordinationService() {
    ServerDef server_def;
    server_def.set_protocol("grpc");
    server_def.set_job_name(kJobName);
    server_def.set_task_index(0);
    auto job_def = server_def.mutable_cluster()->add_job();
    job_def->set_name(kJobName);
    job_def->mutable_tasks()->insert({1, "TEST_ADDRESS_1"});
    job_def->mutable_tasks()->insert({2, "TEST_ADDRESS_2"});
    auto coordination_config = server_def.mutable_default_session_config()
                                   ->mutable_experimental()
                                   ->mutable_coordination_config();
    coordination_config->set_service_type("standalone");
    return CoordinationServiceInterface::EnableCoordinationService(
        "standalone", Env::Default(), server_def, /*cache=*/nullptr);
  }
  void InitializeAndConnectCoordinationAgents() {
    std::unique_ptr<CoordinationClient> coord_client =
        absl::WrapUnique(NewGrpcCoordinationClient(
            grpc_server_->InProcessChannel(::grpc::ChannelArguments())));
    std::unique_ptr<CoordinationClient> coord_client2 =
        absl::WrapUnique(NewGrpcCoordinationClient(
            grpc_server_->InProcessChannel(::grpc::ChannelArguments())));
    auto error_fn = [](const Status& status) {
      LOG(ERROR) << "Coordination service agent in error status: " << status;
    };
    TF_CHECK_OK(
        coord_agent_->Initialize(Env::Default(), kJobName, /*task_id=*/1,
                                 CoordinationServiceConfig::default_instance(),
                                 std::move(coord_client), error_fn));
    TF_CHECK_OK(
        coord_agent2_->Initialize(Env::Default(), kJobName, /*task_id=*/2,
                                  CoordinationServiceConfig::default_instance(),
                                  std::move(coord_client2), error_fn));
    TF_CHECK_OK(coord_agent_->Connect());
    TF_CHECK_OK(coord_agent2_->Connect());
  }

  // Coordination service.
  std::unique_ptr<::grpc::Server> grpc_server_;
  std::unique_ptr<CoordinationServiceInterface> coord_service_;
  std::unique_ptr<thread::ThreadPool> coord_compute_pool_;
  std::unique_ptr<AsyncServiceInterface> coord_rpc_service_;
  std::unique_ptr<Thread> coord_rpc_thread_;
  // Owned by task 1.
  std::unique_ptr<CoordinationServiceAgent> coord_agent_ =
      CreateCoordinationServiceAgent();
  FakePreemptionNotifier* preempt_notifier_;
  // Owned by task 2.
  std::unique_ptr<CoordinationServiceAgent> coord_agent2_ =
      CreateCoordinationServiceAgent();
  FakePreemptionNotifier* preempt_notifier2_;
};

/* Single task tests */
// TODO(b/230630494): Enable tests once the library is implemented.
TEST_F(PreemptionSyncManagerTest, DISABLED_NoPreemption_NoSyncPoint) {
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
}

// TODO(b/230630494): Enable tests once the library is implemented.
TEST_F(PreemptionSyncManagerTest, DISABLED_Preemption_SingleSyncPoint) {
  // Simulate task doing work and making progress.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
  SendPreemptionNotice();

  // Since this is the only task, sync point must be reached.
  EXPECT_TRUE(preempt_sync_mgr_->ReachedSyncPoint());

  // Now, we moved past the sync point.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
}

// TODO(b/230630494): Enable tests once the library is implemented.
TEST_F(PreemptionSyncManagerTest, DISABLED_DelayedPreemption_NoSyncPointYet) {
  // Simulate task doing work and making progress.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
  // Send notice about scheduled preemption in an hour.
  SendPreemptionNotice(absl::Now() + absl::Hours(1));

  // Protocol didn't trigger yet, so there should be no sync point.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
}
// TODO(b/230630494): Enable tests once the library is implemented.
TEST_F(PreemptionSyncManagerTest, DISABLED_UnhealthyTask_NoSyncPoint) {
  // Simulate task doing work and making progress.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
  SimulateUnhealthyTaskTwo();
  SendPreemptionNotice();

  // No sync point is created since one of the tasks is unhealthy.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
}

/* Two task tests */
// TODO(b/230630494): Enable tests once the library is implemented.
TEST_F(PreemptionSyncManagerTest, DISABLED_PreemptSlowTask) {
  // Simulate slow task 1 that is only at call #1.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
  // Simulate fast task 3 that is already at call #3.
  EXPECT_FALSE(preempt_sync_mgr2_->ReachedSyncPoint());
  EXPECT_FALSE(preempt_sync_mgr2_->ReachedSyncPoint());
  EXPECT_FALSE(preempt_sync_mgr2_->ReachedSyncPoint());
  SendPreemptionNotice();

  // Sync point should be set at call #4.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
  EXPECT_TRUE(preempt_sync_mgr_->ReachedSyncPoint());

  // Task 2 was already at call #3, so the next call should be the sync point.
  EXPECT_TRUE(preempt_sync_mgr2_->ReachedSyncPoint());
}

// Same as PreemptSlowTask, but we send the preemption notice to the faster
// task 2.
// TODO(b/230630494): Enable tests once the library is implemented.
TEST_F(PreemptionSyncManagerTest, DISABLED_PreemptFastTask) {
  // Simulate slow task 1 that is only at call #1.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
  // Simulate fast task 3 that is already at call #3.
  EXPECT_FALSE(preempt_sync_mgr2_->ReachedSyncPoint());
  EXPECT_FALSE(preempt_sync_mgr2_->ReachedSyncPoint());
  EXPECT_FALSE(preempt_sync_mgr2_->ReachedSyncPoint());
  SendPreemptionNotice(absl::Now(), /*=to_task1=*/false);

  // Sync point should be set at call #4.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint());
  EXPECT_TRUE(preempt_sync_mgr_->ReachedSyncPoint());

  // Task 2 was already at call #3, so the next call should be the sync point.
  EXPECT_TRUE(preempt_sync_mgr2_->ReachedSyncPoint());
}

}  // namespace
}  // namespace tensorflow
