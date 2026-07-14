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
#include "xla/tsl/distributed_runtime/preemption/preemption_sync_manager.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/support/channel_arguments.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_client.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/distributed_runtime/preemption/preemption_notifier.h"
#include "xla/tsl/distributed_runtime/rpc/async_service_interface.h"
#include "xla/tsl/distributed_runtime/rpc/coordination/grpc_coordination_client.h"
#include "xla/tsl/distributed_runtime/rpc/coordination/grpc_coordination_service_impl.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/protobuf/coordination_config.pb.h"

namespace tsl {
namespace {
using tensorflow::CoordinatedJob;
using tensorflow::CoordinatedTask;
using tensorflow::CoordinationServiceConfig;

constexpr char kJobName[] = "test_worker";

// Send fake preemption notices at any time for testing.
class FakePreemptionNotifier : public PreemptionNotifier {
 public:
  FakePreemptionNotifier() : PreemptionNotifier(/*env=*/nullptr) {}

  ~FakePreemptionNotifier() override {
    NotifyRegisteredListeners(
        absl::CancelledError("~FakePreemptionNotifier() was called."));
  }

  void AnnounceDeath(absl::Time death_time) {
    LOG(WARNING) << "Received preemption notice with death time: "
                 << death_time;
    NotifyRegisteredListeners(death_time);
  }
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
    CHECK_OK(preempt_sync_mgr_->Initialize(coord_agent_.get(),
                                           std::move(preempt_notifier)));

    // Create preempt sync manager for task 2.
    auto preempt_notifier2 = std::make_unique<FakePreemptionNotifier>();
    preempt_notifier2_ = preempt_notifier2.get();
    CHECK_OK(preempt_sync_mgr2_->Initialize(coord_agent2_.get(),
                                            std::move(preempt_notifier2)));
  }
  ~PreemptionSyncManagerTest() override {
    // Tear down coordination service objects in order.
    preempt_sync_mgr_ = nullptr;
    preempt_sync_mgr2_ = nullptr;
    coord_agent_ = nullptr;
    coord_agent2_ = nullptr;
    coord_service_ = nullptr;
    static_cast<tsl::GrpcCoordinationServiceImpl*>(coord_rpc_service_.get())
        ->SetCoordinationServiceInstance(nullptr);
    grpc_server_->Shutdown();
    coord_rpc_service_->Shutdown();
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
        absl::ToInt64Microseconds(absl::Seconds(1)));
  }

  // Report to coordination service that task two is unhealthy.
  void SimulateUnhealthyTaskTwo() {
    CoordinatedTask task2;
    task2.set_job_name(kJobName);
    task2.set_task_id(1);
    CHECK_OK(coord_service_->ReportTaskError(
        task2, absl::InternalError("test_error")));
  }

  // Allow access to objects under test.
  std::unique_ptr<PreemptionSyncManager> preempt_sync_mgr_ =
      CreatePreemptionSyncManager();
  std::unique_ptr<PreemptionSyncManager> preempt_sync_mgr2_ =
      CreatePreemptionSyncManager();

 protected:
  // Utility methods to set up coordination service and agents.
  void StartCoordinationService() {
    ::grpc::ServerBuilder builder;
    coord_service_ = EnableCoordinationService();
    coord_compute_pool_ = std::make_unique<thread::ThreadPool>(
        Env::Default(), "CoordinationServiceRpcHandler",
        /*num_threads=*/1);
    coord_rpc_service_ = std::make_unique<GrpcCoordinationServiceImpl>(
        coord_compute_pool_.get(), &builder);
    auto* grpc_coord_service =
        static_cast<GrpcCoordinationServiceImpl*>(coord_rpc_service_.get());
    grpc_coord_service->SetCoordinationServiceInstance(coord_service_.get());
    grpc_server_ = builder.BuildAndStart();
    coord_rpc_thread_ = absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/"CoordinationServiceHandleRPCsLoop",
        [service = coord_rpc_service_.get()]() { service->HandleRPCsLoop(); }));
  }
  std::unique_ptr<CoordinationService> EnableCoordinationService() {
    CoordinationServiceConfig config;
    config.set_service_type("standalone");
    CoordinatedJob* job = config.mutable_coordinated_job_list()->Add();
    job->set_name(kJobName);
    job->set_num_tasks(2);
    return CoordinationService::Create(Env::Default(), config,
                                       /*cache=*/nullptr);
  }
  void InitializeAndConnectCoordinationAgents() {
    std::unique_ptr<CoordinationClient> coord_client =
        absl::WrapUnique(NewGrpcCoordinationClient(
            grpc_server_->InProcessChannel(::grpc::ChannelArguments())));
    std::unique_ptr<CoordinationClient> coord_client2 =
        absl::WrapUnique(NewGrpcCoordinationClient(
            grpc_server_->InProcessChannel(::grpc::ChannelArguments())));
    auto error_fn = [](const absl::Status& status) {
      LOG(ERROR) << "Coordination service agent in error status: " << status;
    };
    CoordinationServiceConfig coord_config;
    coord_config.set_service_leader("test_leader");
    CHECK_OK(coord_agent_->Initialize(Env::Default(), kJobName,
                                      /*task_id=*/0, coord_config,
                                      std::move(coord_client), error_fn));
    CHECK_OK(coord_agent2_->Initialize(Env::Default(), kJobName,
                                       /*task_id=*/1, coord_config,
                                       std::move(coord_client2), error_fn));
    CHECK_OK(coord_agent_->Connect());
    CHECK_OK(coord_agent2_->Connect());
  }

  // Coordination service.
  std::unique_ptr<CoordinationService> coord_service_;
  std::unique_ptr<::grpc::Server> grpc_server_;
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
TEST_F(PreemptionSyncManagerTest, NoPreemption_NoSyncPoint) {
  int step_counter = 0;
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter++));
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter++));
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter++));
}

TEST_F(PreemptionSyncManagerTest, Preemption_SingleSyncPoint) {
  // Simulate task doing work and making progress.
  int step_counter = 0;
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter++));
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter++));
  SendPreemptionNotice();

  // Since this is the only task, sync point must be reached.
  EXPECT_TRUE(preempt_sync_mgr_->ReachedSyncPoint(step_counter++));

  // Now, we moved past the sync point.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter++));
}

TEST_F(PreemptionSyncManagerTest, DelayedPreemption_NoSyncPointYet) {
  int step_counter = 0;
  // Simulate task doing work and making progress.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter++));
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter++));
  // Send notice about scheduled preemption in an hour.
  SendPreemptionNotice(absl::Now() + absl::Hours(1));

  // Protocol didn't trigger yet, so there should be no sync point.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter++));
}

TEST_F(PreemptionSyncManagerTest, UnhealthyTask_NoSyncPoint) {
  int step_counter = 0;
  // Simulate task doing work and making progress.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter++));
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter++));
  SimulateUnhealthyTaskTwo();
  SendPreemptionNotice();

  // No sync point is created since one of the tasks is unhealthy.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter++));
}

TEST_F(PreemptionSyncManagerTest, ShutdownTasksWithoutPreemption) {
  int step_counter = 0;
  // Simulate task doing work and making progress.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter++));
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter++));

  // Shutdown coordination service agents.
  CHECK_OK(coord_agent_->Shutdown());
  CHECK_OK(coord_agent2_->Shutdown());
  // Protocol is not triggered, so there should be no sync point.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter++));
}

// Explicitly shut down without preemption.
TEST_F(PreemptionSyncManagerTest, ShutdownWithoutPreemption) {
  preempt_sync_mgr_->Shutdown();
}

// Explicitly shut down without initialization.
TEST_F(PreemptionSyncManagerTest, ShutdownWithoutInitialization) {
  std::unique_ptr<PreemptionSyncManager> m = CreatePreemptionSyncManager();
  m->Shutdown();
}

// Explicitly shut down with preemption.
TEST_F(PreemptionSyncManagerTest, ShutdownWithPreemption) {
  SendPreemptionNotice(absl::Now());
  preempt_sync_mgr_->Shutdown();
}

/* Two task tests */
TEST_F(PreemptionSyncManagerTest, PreemptSlowTask) {
  int step_counter0 = 0;
  int step_counter2 = 0;
  // Simulate slow task 1 that is only at call #1.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter0++));
  // Simulate fast task 3 that is already at call #3.
  EXPECT_FALSE(preempt_sync_mgr2_->ReachedSyncPoint(step_counter2++));
  EXPECT_FALSE(preempt_sync_mgr2_->ReachedSyncPoint(step_counter2++));
  EXPECT_FALSE(preempt_sync_mgr2_->ReachedSyncPoint(step_counter2++));
  SendPreemptionNotice();

  // Sync point should be set at call #4.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter0++));
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter0++));
  EXPECT_TRUE(preempt_sync_mgr_->ReachedSyncPoint(step_counter0++));

  // Task 2 was already at call #3, so the next call should be the sync point.
  EXPECT_TRUE(preempt_sync_mgr2_->ReachedSyncPoint(step_counter2++));
}

// Same as PreemptSlowTask, but we send the preemption notice to the faster
// task 2.
TEST_F(PreemptionSyncManagerTest, PreemptFastTask) {
  int step_counter0 = 0;
  int step_counter2 = 0;
  // Simulate slow task 1 that is only at call #1.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter0++));
  // Simulate fast task 3 that is already at call #3.
  EXPECT_FALSE(preempt_sync_mgr2_->ReachedSyncPoint(step_counter2++));
  EXPECT_FALSE(preempt_sync_mgr2_->ReachedSyncPoint(step_counter2++));
  EXPECT_FALSE(preempt_sync_mgr2_->ReachedSyncPoint(step_counter2++));
  SendPreemptionNotice(absl::Now(), /*=to_task1=*/false);

  // Sync point should be set at call #4.
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter0++));
  EXPECT_FALSE(preempt_sync_mgr_->ReachedSyncPoint(step_counter0++));
  EXPECT_TRUE(preempt_sync_mgr_->ReachedSyncPoint(step_counter0++));

  // Task 2 was already at call #3, so the next call should be the sync point.
  EXPECT_TRUE(preempt_sync_mgr2_->ReachedSyncPoint(step_counter2++));
}

}  // namespace
}  // namespace tsl
