/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/data/service/compression_utils.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/master.grpc.pb.h"
#include "tensorflow/core/data/service/master.pb.h"
#include "tensorflow/core/data/service/server_lib.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {

namespace {
const char kProtocol[] = "grpc+local";

// Parse the address from a string in the form "<protocol>://<address>".
Status AddressFromTarget(const std::string& target, std::string* address) {
  std::vector<std::string> parts = absl::StrSplit(target, "://");
  if (parts.size() != 2) {
    return errors::InvalidArgument("target ", target, " split into ",
                                   parts.size(), " parts, not 2");
  }
  *address = parts[1];
  return Status::OK();
}

class TestCluster {
 public:
  explicit TestCluster(int num_workers) : num_workers_(num_workers) {}

  Status Initialize() {
    TF_RETURN_IF_ERROR(NewMasterServer(/*port=*/0, kProtocol, &master_));
    TF_RETURN_IF_ERROR(master_->Start());
    TF_RETURN_IF_ERROR(AddressFromTarget(master_->Target(), &master_address_));
    workers_.reserve(num_workers_);
    worker_addresses_.reserve(num_workers_);
    for (int i = 0; i < num_workers_; ++i) {
      TF_RETURN_IF_ERROR(AddWorker());
    }
    return Status::OK();
  }

  Status AddWorker() {
    workers_.emplace_back();
    TF_RETURN_IF_ERROR(NewWorkerServer(/*port=*/0, kProtocol, master_address_,
                                       &workers_.back()));
    TF_RETURN_IF_ERROR(workers_.back()->Start());
    worker_addresses_.emplace_back();
    TF_RETURN_IF_ERROR(AddressFromTarget(workers_.back()->Target(),
                                         &worker_addresses_.back()));
    return Status::OK();
  }

  std::string MasterAddress() { return master_address_; }

  std::string WorkerAddress(int index) { return worker_addresses_[index]; }

 private:
  int num_workers_;
  std::unique_ptr<GrpcDataServer> master_;
  std::string master_address_;
  std::vector<std::unique_ptr<GrpcDataServer>> workers_;
  std::vector<std::string> worker_addresses_;
};

Status RegisterDataset(MasterService::Stub* master_stub,
                       const GraphDef& dataset_graph, int64* dataset_id) {
  grpc_impl::ClientContext ctx;
  GetOrRegisterDatasetRequest req;
  *req.mutable_dataset()->mutable_graph() = dataset_graph;
  GetOrRegisterDatasetResponse resp;
  grpc::Status s = master_stub->GetOrRegisterDataset(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to register dataset", s);
  }
  *dataset_id = resp.dataset_id();
  return Status::OK();
}

Status BeginEpoch(MasterService::Stub* master_stub, int64 dataset_id,
                  int64* epoch_id) {
  grpc_impl::ClientContext ctx;
  BeginEpochRequest req;
  req.set_dataset_id(dataset_id);
  BeginEpochResponse resp;
  grpc::Status s = master_stub->BeginEpoch(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to begin epoch", s);
  }
  *epoch_id = resp.epoch_id();
  return Status::OK();
}

Status GetTasks(MasterService::Stub* master_stub, int64 epoch_id,
                std::vector<TaskInfo>* tasks) {
  grpc_impl::ClientContext ctx;
  GetTasksRequest req;
  req.set_epoch_id(epoch_id);
  GetTasksResponse resp;
  grpc::Status s = master_stub->GetTasks(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get tasks", s);
  }
  tasks->clear();
  for (auto& task : resp.task_info()) {
    tasks->push_back(task);
  }
  return Status::OK();
}

Status GetElement(WorkerService::Stub* worker_stub, int64 task_id,
                  std::vector<Tensor>* element, bool* end_of_sequence) {
  grpc_impl::ClientContext ctx;
  GetElementRequest req;
  req.set_task_id(task_id);
  GetElementResponse resp;
  grpc::Status s = worker_stub->GetElement(&ctx, req, &resp);
  if (!s.ok()) {
    return grpc_util::WrapError("Failed to get element", s);
  }
  *end_of_sequence = resp.end_of_sequence();
  if (!*end_of_sequence) {
    const CompressedElement& compressed = resp.compressed_element();
    TF_RETURN_IF_ERROR(service_util::Uncompress(compressed, element));
  }
  return Status::OK();
}

Status CheckWorkerOutput(const std::string& worker_address, int64 task_id,
                         std::vector<std::vector<Tensor>> expected_output) {
  auto worker_channel = grpc::CreateChannel(
      worker_address, grpc::experimental::LocalCredentials(LOCAL_TCP));
  std::unique_ptr<WorkerService::Stub> worker_stub =
      WorkerService::NewStub(worker_channel);
  for (std::vector<Tensor>& expected : expected_output) {
    bool end_of_sequence;
    std::vector<Tensor> element;
    TF_RETURN_IF_ERROR(
        GetElement(worker_stub.get(), task_id, &element, &end_of_sequence));
    if (end_of_sequence) {
      return errors::Internal("Reached end of sequence too early.");
    }
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::ExpectEqual(element, expected,
                                                       /*compare_order=*/true));
  }
  // Call GetElement a couple more times to verify tha end_of_sequence keeps
  // returning true.
  bool end_of_sequence;
  std::vector<Tensor> element;
  TF_RETURN_IF_ERROR(
      GetElement(worker_stub.get(), task_id, &element, &end_of_sequence));
  if (!end_of_sequence) {
    return errors::Internal("Expected end_of_sequence to be true");
  }
  TF_RETURN_IF_ERROR(
      GetElement(worker_stub.get(), task_id, &element, &end_of_sequence));
  if (!end_of_sequence) {
    return errors::Internal("Expected end_of_sequence to be true");
  }
  return Status::OK();
}

}  // namespace

TEST(DataService, IterateDatasetOneWorker) {
  TestCluster cluster(1);
  TF_ASSERT_OK(cluster.Initialize());
  test_util::GraphDefTestCase test_case;
  TF_ASSERT_OK(test_util::map_test_case(&test_case));
  auto master_channel = grpc::CreateChannel(
      cluster.MasterAddress(), grpc::experimental::LocalCredentials(LOCAL_TCP));
  std::unique_ptr<MasterService::Stub> master_stub =
      MasterService::NewStub(master_channel);

  int64 dataset_id;
  TF_ASSERT_OK(
      RegisterDataset(master_stub.get(), test_case.graph_def, &dataset_id));
  int64 epoch_id;
  TF_ASSERT_OK(BeginEpoch(master_stub.get(), dataset_id, &epoch_id));
  std::vector<TaskInfo> tasks;
  TF_ASSERT_OK(GetTasks(master_stub.get(), epoch_id, &tasks));
  ASSERT_EQ(tasks.size(), 1);
  ASSERT_EQ(tasks[0].worker_address(), cluster.WorkerAddress(0));

  TF_ASSERT_OK(CheckWorkerOutput(tasks[0].worker_address(), tasks[0].id(),
                                 test_case.output));
}

TEST(DataService, IterateDatasetTwoWorkers) {
  TestCluster cluster(2);
  TF_ASSERT_OK(cluster.Initialize());
  test_util::GraphDefTestCase test_case;
  TF_ASSERT_OK(test_util::map_test_case(&test_case));
  auto master_channel = grpc::CreateChannel(
      cluster.MasterAddress(), grpc::experimental::LocalCredentials(LOCAL_TCP));
  std::unique_ptr<MasterService::Stub> master_stub =
      MasterService::NewStub(master_channel);

  int64 dataset_id;
  TF_ASSERT_OK(
      RegisterDataset(master_stub.get(), test_case.graph_def, &dataset_id));
  int64 epoch_id;
  TF_ASSERT_OK(BeginEpoch(master_stub.get(), dataset_id, &epoch_id));
  std::vector<TaskInfo> tasks;
  TF_ASSERT_OK(GetTasks(master_stub.get(), epoch_id, &tasks));
  ASSERT_EQ(tasks.size(), 2);

  // Each worker produces the full dataset.
  for (TaskInfo task : tasks) {
    TF_ASSERT_OK(
        CheckWorkerOutput(task.worker_address(), task.id(), test_case.output));
  }
}

TEST(DataService, AddWorkerMidEpoch) {
  TestCluster cluster(1);
  TF_ASSERT_OK(cluster.Initialize());
  test_util::GraphDefTestCase test_case;
  TF_ASSERT_OK(test_util::map_test_case(&test_case));
  auto master_channel = grpc::CreateChannel(
      cluster.MasterAddress(), grpc::experimental::LocalCredentials(LOCAL_TCP));
  std::unique_ptr<MasterService::Stub> master_stub =
      MasterService::NewStub(master_channel);

  int64 dataset_id;
  TF_ASSERT_OK(
      RegisterDataset(master_stub.get(), test_case.graph_def, &dataset_id));
  int64 epoch_id;
  TF_ASSERT_OK(BeginEpoch(master_stub.get(), dataset_id, &epoch_id));
  std::vector<TaskInfo> tasks;
  TF_ASSERT_OK(GetTasks(master_stub.get(), epoch_id, &tasks));
  ASSERT_EQ(tasks.size(), 1);
  TF_ASSERT_OK(cluster.AddWorker());
  TF_ASSERT_OK(GetTasks(master_stub.get(), epoch_id, &tasks));
  ASSERT_EQ(tasks.size(), 2);

  // Each worker produces the full dataset.
  for (TaskInfo task : tasks) {
    TF_ASSERT_OK(
        CheckWorkerOutput(task.worker_address(), task.id(), test_case.output));
  }
}

}  // namespace data
}  // namespace tensorflow
