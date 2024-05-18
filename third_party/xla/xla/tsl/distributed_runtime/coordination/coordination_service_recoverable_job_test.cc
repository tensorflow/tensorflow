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
#include <memory>
#include <string>
#include <utility>

#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/support/channel_arguments.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_client.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/distributed_runtime/rpc/async_service_interface.h"
#include "xla/tsl/distributed_runtime/rpc/coordination/grpc_coordination_client.h"
#include "xla/tsl/distributed_runtime/rpc/coordination/grpc_coordination_service_impl.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"
#include "tsl/protobuf/coordination_config.pb.h"

namespace tsl {
namespace {
using tensorflow::CoordinatedJob;
using tensorflow::CoordinationServiceConfig;

constexpr char kParameterServerJobName[] = "parameter_server";
constexpr char kWorkerJobName[] = "worker";
constexpr char kCoordinationServiceType[] = "standalone";
constexpr char kServiceLeader[] = "/job:parameter_server/replica:0/task:0";

class TestCoordinationClientCache : public CoordinationClientCache {
 public:
  void AddTask(const std::string& target, CoordinationClient* client) {
    absl::MutexLock l(&clients_mu_);
    clients_.emplace(target, client);
  }

  CoordinationClient* GetClient(const std::string& target) override {
    absl::MutexLock l(&clients_mu_);
    if (auto it = clients_.find(target); it != clients_.end()) {
      return it->second;
    }
    return nullptr;
  }

  std::unique_ptr<CoordinationClient> GetOwnedClient(
      const std::string& target) override {
    LOG(ERROR) << "GetOwnedClient is not supported.";
    return nullptr;
  }

 private:
  absl::Mutex clients_mu_;
  absl::flat_hash_map<std::string, CoordinationClient*> clients_
      ABSL_GUARDED_BY(clients_mu_);
};

class TestCoordinationServiceTaskState {
 public:
  TestCoordinationServiceTaskState() = default;

  ~TestCoordinationServiceTaskState() = default;

  void Shutdown() {
    coord_client_.reset();
    coord_agent_.reset();
    coord_compute_pool_.reset();
    static_cast<tsl::GrpcCoordinationServiceImpl*>(coord_rpc_service_.get())
        ->SetCoordinationServiceInstance(nullptr);
    grpc_server_->Shutdown();
    coord_rpc_service_->Shutdown();
  }

  void StartGrpcServer() {
    ::grpc::ServerBuilder builder;
    coord_compute_pool_ = std::make_unique<thread::ThreadPool>(
        Env::Default(), /*name=*/"CoordinationServiceRpcHandler",
        /*num_threads=*/5);
    coord_rpc_service_ = std::make_unique<GrpcCoordinationServiceImpl>(
        coord_compute_pool_.get(), &builder);
    auto* grpc_coord_service =
        static_cast<GrpcCoordinationServiceImpl*>(coord_rpc_service_.get());
    grpc_coord_service->SetCoordinationServiceAgentInstance(coord_agent_.get());
    grpc_server_ = builder.BuildAndStart();
    coord_client_ = absl::WrapUnique(NewGrpcCoordinationClient(
        grpc_server_->InProcessChannel(::grpc::ChannelArguments())));
    coord_rpc_thread_ = absl::WrapUnique(Env::Default()->StartThread(
        /*thread_options=*/{}, /*name=*/"CoordinationServiceHandleRPCsLoop",
        [service = coord_rpc_service_.get()]() { service->HandleRPCsLoop(); }));
  }

  void SetCoordinationService(CoordinationServiceInterface* service) {
    auto* grpc_coord_service =
        static_cast<GrpcCoordinationServiceImpl*>(coord_rpc_service_.get());
    grpc_coord_service->SetCoordinationServiceInstance(service);
  }

  void InitializeAndConnectCoordinationAgents(
      const std::string& job_name, int task_id,
      const CoordinationServiceConfig& coordination_config) {
    auto error_fn = [this, job_name](const absl::Status& status) {
      this->status_ = status;
      LOG(ERROR) << "Coordination service agent of " << job_name
                 << " is in error status: " << status;
    };

    TF_CHECK_OK(coord_agent_->Initialize(Env::Default(), job_name, task_id,
                                         coordination_config,
                                         std::move(coord_client_), error_fn));
    TF_CHECK_OK(coord_agent_->Connect());
    TF_CHECK_OK(status_);
  }

  CoordinationClient* GetCoordinationClient() { return coord_client_.get(); }

  absl::Status ReportError(const absl::Status& status) {
    return coord_agent_->ReportError(status);
  }

  absl::Status GetStatus() const { return status_; }

 private:
  std::unique_ptr<::grpc::Server> grpc_server_;
  std::unique_ptr<thread::ThreadPool> coord_compute_pool_;
  std::unique_ptr<AsyncServiceInterface> coord_rpc_service_;
  std::unique_ptr<Thread> coord_rpc_thread_;
  std::unique_ptr<CoordinationServiceAgent> coord_agent_ =
      CreateCoordinationServiceAgent();
  std::unique_ptr<CoordinationClient> coord_client_;
  absl::Status status_;
};

class CoordinationServiceRecoverableJobTest : public ::testing::Test {
 public:
  void SetUp() override {
    state_ps_0_.StartGrpcServer();
    state_ps_1_.StartGrpcServer();
    state_worker_0_.StartGrpcServer();
    state_worker_1_.StartGrpcServer();
  }

  void TearDown() override {
    state_ps_0_.Shutdown();
    state_ps_1_.Shutdown();
    state_worker_0_.Shutdown();
    state_worker_1_.Shutdown();
    coord_service_.reset();
  }

  void Initialize() {
    ConfigureCoordinationService();
    auto client_cache = std::make_unique<TestCoordinationClientCache>();
    client_cache->AddTask(
        /*target=*/kServiceLeader, state_ps_0_.GetCoordinationClient());
    client_cache->AddTask(
        /*target=*/"/job:parameter_server/replica:0/task:1",
        state_ps_1_.GetCoordinationClient());
    client_cache->AddTask(
        /*target=*/"/job:worker/replica:0/task:0",
        state_worker_0_.GetCoordinationClient());
    client_cache->AddTask(
        /*target=*/"/job:worker/replica:0/task:1",
        state_worker_1_.GetCoordinationClient());
    coord_service_ = CoordinationServiceInterface::EnableCoordinationService(
        Env::Default(), coordination_config_, std::move(client_cache));
    // Set the service pointer for all the tasks since it is needed for handling
    // error propagations. In reality, every task has its own service pointer.
    // To mimic that, we need multi-process tests.
    state_ps_0_.SetCoordinationService(coord_service_.get());
    state_ps_1_.SetCoordinationService(coord_service_.get());
    state_worker_0_.SetCoordinationService(coord_service_.get());
    state_worker_1_.SetCoordinationService(coord_service_.get());
    state_ps_0_.InitializeAndConnectCoordinationAgents(kParameterServerJobName,
                                                       /*task_id=*/0,
                                                       coordination_config_);
    state_ps_1_.InitializeAndConnectCoordinationAgents(kParameterServerJobName,
                                                       /*task_id=*/1,
                                                       coordination_config_);
    state_worker_0_.InitializeAndConnectCoordinationAgents(
        kWorkerJobName,
        /*task_id=*/0, coordination_config_);
    state_worker_1_.InitializeAndConnectCoordinationAgents(
        kWorkerJobName,
        /*task_id=*/1, coordination_config_);
  }

  void ConfigureCoordinationService() {
    // Assume the coordination service is deployed in the parameter server.
    coordination_config_.set_service_type(kCoordinationServiceType);
    coordination_config_.set_service_leader(kServiceLeader);
    CoordinatedJob* ps =
        coordination_config_.mutable_coordinated_job_list()->Add();
    ps->set_name(kParameterServerJobName);
    ps->set_num_tasks(2);
    CoordinatedJob* worker =
        coordination_config_.mutable_coordinated_job_list()->Add();
    worker->set_name(kWorkerJobName);
    worker->set_num_tasks(2);
  }

  void AddJobToRecoverableJobs(const std::string& job_name) {
    coordination_config_.add_recoverable_jobs(job_name);
  }

 protected:
  CoordinationServiceConfig coordination_config_;
  std::unique_ptr<CoordinationServiceInterface> coord_service_;
  TestCoordinationServiceTaskState state_ps_0_;
  TestCoordinationServiceTaskState state_ps_1_;
  TestCoordinationServiceTaskState state_worker_0_;
  TestCoordinationServiceTaskState state_worker_1_;
};

TEST_F(CoordinationServiceRecoverableJobTest,
       UnrecoverableWorkerFailurePropagated) {
  Initialize();
  TF_ASSERT_OK(state_worker_0_.ReportError(absl::InternalError("Test Error.")));

  // For unrecoverable task, error propagates to all connected tasks.
  EXPECT_TRUE(absl::IsInternal(state_ps_0_.GetStatus()));
  EXPECT_TRUE(absl::IsInternal(state_ps_1_.GetStatus()));
  EXPECT_TRUE(absl::IsInternal(state_worker_0_.GetStatus()));
  EXPECT_TRUE(absl::IsInternal(state_worker_1_.GetStatus()));
}

TEST_F(CoordinationServiceRecoverableJobTest,
       UnrecoverablePSFailurePropagated) {
  Initialize();
  TF_ASSERT_OK(state_ps_0_.ReportError(absl::InternalError("Test Error.")));

  // For unrecoverable task, error propagates to all connected tasks.
  EXPECT_TRUE(absl::IsInternal(state_ps_0_.GetStatus()));
  EXPECT_TRUE(absl::IsInternal(state_ps_1_.GetStatus()));
  EXPECT_TRUE(absl::IsInternal(state_worker_0_.GetStatus()));
  EXPECT_TRUE(absl::IsInternal(state_worker_1_.GetStatus()));
}

TEST_F(CoordinationServiceRecoverableJobTest,
       RecoverableWorkerFailureNotPropagated) {
  AddJobToRecoverableJobs(kWorkerJobName);
  Initialize();
  TF_ASSERT_OK(state_worker_0_.ReportError(absl::InternalError("Test Error.")));

  // For recoverable task, error does not propagate.
  EXPECT_TRUE(state_ps_0_.GetStatus().ok());
  EXPECT_TRUE(state_ps_1_.GetStatus().ok());
  EXPECT_TRUE(absl::IsInternal(state_worker_0_.GetStatus()));
  EXPECT_TRUE(state_worker_1_.GetStatus().ok());
}

}  // namespace
}  // namespace tsl
