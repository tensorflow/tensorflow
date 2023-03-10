/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/worker_client.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "absl/types/optional.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/test_cluster.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/service/worker_impl.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::data::testing::RangeSquareDataset;
using ::tensorflow::testing::StatusIs;
using ::testing::MatchesRegex;

constexpr const char kProtocol[] = "grpc";

class WorkerClientTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_cluster_ = std::make_unique<TestCluster>(/*num_workers=*/1);
    TF_ASSERT_OK(test_cluster_->Initialize());
    dispatcher_client_ = std::make_unique<DataServiceDispatcherClient>(
        test_cluster_->DispatcherAddress(), kProtocol);
  }

  // Creates a dataset and returns the dataset ID.
  StatusOr<std::string> RegisterDataset(const int64_t range) {
    const auto dataset_def = RangeSquareDataset(range);
    std::string dataset_id;
    TF_RETURN_IF_ERROR(dispatcher_client_->RegisterDataset(
        dataset_def, DataServiceMetadata(),
        /*requested_dataset_id=*/std::nullopt, dataset_id));
    return dataset_id;
  }

  // Creates a iteration and returns the iteration client ID.
  StatusOr<int64_t> CreateIteration(const std::string& dataset_id) {
    ProcessingModeDef processing_mode;
    processing_mode.set_sharding_policy(ProcessingModeDef::OFF);
    int64_t job_id = 0;
    TF_RETURN_IF_ERROR(dispatcher_client_->GetOrCreateJob(
        dataset_id, processing_mode, /*job_name=*/std::nullopt,
        /*num_consumers=*/std::nullopt, /*use_cross_trainer_cache=*/false,
        TARGET_WORKERS_AUTO, job_id));
    int64_t iteration_client_id = 0;
    TF_RETURN_IF_ERROR(dispatcher_client_->GetOrCreateIteration(
        job_id, /*repetition=*/0, iteration_client_id));
    return iteration_client_id;
  }

  // Gets the task for iteration `iteration_client_id`.
  StatusOr<int64_t> GetTaskToRead(const int64_t iteration_client_id) {
    ClientHeartbeatRequest request;
    ClientHeartbeatResponse response;
    request.set_iteration_client_id(iteration_client_id);
    TF_RETURN_IF_ERROR(dispatcher_client_->ClientHeartbeat(request, response));
    if (response.task_info().empty()) {
      return errors::NotFound(absl::Substitute(
          "No task found for iteration $0.", iteration_client_id));
    }
    return response.task_info(0).task_id();
  }

  StatusOr<std::unique_ptr<DataServiceWorkerClient>> GetWorkerClient(
      const std::string& data_transfer_protocol) {
    return CreateDataServiceWorkerClient(
        GetWorkerAddress(), /*protocol=*/kProtocol, data_transfer_protocol);
  }

  StatusOr<GetElementResult> GetElement(DataServiceWorkerClient& client,
                                        const int64_t task_id) {
    GetElementRequest request;
    GetElementResult result;
    request.set_task_id(task_id);
    TF_RETURN_IF_ERROR(client.GetElement(request, result));
    return result;
  }

  std::string GetDispatcherAddress() const {
    return test_cluster_->DispatcherAddress();
  }

  std::string GetWorkerAddress() const {
    return test_cluster_->WorkerAddress(0);
  }

  std::unique_ptr<TestCluster> test_cluster_;
  std::unique_ptr<DataServiceDispatcherClient> dispatcher_client_;
};

TEST_F(WorkerClientTest, LocalRead) {
  const int64_t range = 5;
  TF_ASSERT_OK_AND_ASSIGN(const std::string dataset_id, RegisterDataset(range));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t iteration_client_id,
                          CreateIteration(dataset_id));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t task_id,
                          GetTaskToRead(iteration_client_id));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<DataServiceWorkerClient> client,
                          GetWorkerClient(kLocalTransferProtocol));
  for (int64_t i = 0; i < range; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(GetElementResult result,
                            GetElement(*client, task_id));
    test::ExpectEqual(result.components[0], Tensor(int64_t{i * i}));
    EXPECT_FALSE(result.end_of_sequence);
  }

  // Remove the local worker from `LocalWorkers`. Since the client reads from a
  // local server, this should cause the request to fail.
  LocalWorkers::Remove(GetWorkerAddress());
  EXPECT_THAT(GetElement(*client, task_id),
              StatusIs(error::CANCELLED,
                       MatchesRegex("Local worker.*is no longer available.*")));
}

TEST_F(WorkerClientTest, LocalReadEmptyDataset) {
  TF_ASSERT_OK_AND_ASSIGN(const std::string dataset_id,
                          RegisterDataset(/*range=*/0));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t iteration_client_id,
                          CreateIteration(dataset_id));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t task_id,
                          GetTaskToRead(iteration_client_id));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<DataServiceWorkerClient> client,
                          GetWorkerClient(kLocalTransferProtocol));
  TF_ASSERT_OK_AND_ASSIGN(GetElementResult result,
                          GetElement(*client, task_id));
  EXPECT_TRUE(result.end_of_sequence);

  // Remove the local worker from `LocalWorkers`. Since the client reads from a
  // local server, this should cause the request to fail.
  LocalWorkers::Remove(GetWorkerAddress());
  EXPECT_THAT(GetElement(*client, task_id),
              StatusIs(error::CANCELLED,
                       MatchesRegex("Local worker.*is no longer available.*")));
}

TEST_F(WorkerClientTest, GrpcRead) {
  const int64_t range = 5;
  TF_ASSERT_OK_AND_ASSIGN(const std::string dataset_id, RegisterDataset(range));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t iteration_client_id,
                          CreateIteration(dataset_id));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t task_id,
                          GetTaskToRead(iteration_client_id));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<DataServiceWorkerClient> client,
                          GetWorkerClient(kGrpcTransferProtocol));
  for (int64_t i = 0; i < range; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(GetElementResult result,
                            GetElement(*client, task_id));
    test::ExpectEqual(result.components[0], Tensor(int64_t{i * i}));
    EXPECT_FALSE(result.end_of_sequence);
  }

  // Remove the local worker from `LocalWorkers`. Since the client reads from a
  // local server, this should cause the request to fail.
  LocalWorkers::Remove(GetWorkerAddress());
  EXPECT_THAT(GetElement(*client, task_id),
              StatusIs(error::CANCELLED,
                       MatchesRegex("Local worker.*is no longer available.*")));
}

TEST_F(WorkerClientTest, LocalServerShutsDown) {
  TF_ASSERT_OK_AND_ASSIGN(const std::string dataset_id,
                          RegisterDataset(/*range=*/5));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t iteration_client_id,
                          CreateIteration(dataset_id));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t task_id,
                          GetTaskToRead(iteration_client_id));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<DataServiceWorkerClient> client,
                          GetWorkerClient(kLocalTransferProtocol));

  // Stopping a worker causes local reads to return Cancelled status.
  test_cluster_->StopWorkers();
  EXPECT_THAT(GetElement(*client, task_id),
              StatusIs(error::CANCELLED,
                       MatchesRegex("Local worker.*is no longer available.*")));
}

TEST_F(WorkerClientTest, CancelClient) {
  TF_ASSERT_OK_AND_ASSIGN(const std::string dataset_id,
                          RegisterDataset(/*range=*/5));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t iteration_client_id,
                          CreateIteration(dataset_id));
  TF_ASSERT_OK_AND_ASSIGN(const int64_t task_id,
                          GetTaskToRead(iteration_client_id));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<DataServiceWorkerClient> client,
                          GetWorkerClient(kLocalTransferProtocol));

  client->TryCancel();
  EXPECT_THAT(GetElement(*client, task_id),
              StatusIs(error::CANCELLED,
                       MatchesRegex("Client for worker.*has been cancelled.")));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
