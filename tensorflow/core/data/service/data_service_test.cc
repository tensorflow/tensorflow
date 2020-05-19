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

#include "tensorflow/core/data/service/data_service.h"

#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/data/compression_utils.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/master.grpc.pb.h"
#include "tensorflow/core/data/service/master.pb.h"
#include "tensorflow/core/data/service/server_lib.h"
#include "tensorflow/core/data/service/test_cluster.h"
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
constexpr const char kProtocol[] = "grpc+local";

TEST(DataService, ParseParallelEpochsProcessingMode) {
  ProcessingMode mode;
  TF_ASSERT_OK(ParseProcessingMode("parallel_epochs", &mode));
  EXPECT_EQ(mode, ProcessingMode::PARALLEL_EPOCHS);
}

TEST(DataService, ParseOneEpochProcessingMode) {
  ProcessingMode mode;
  TF_ASSERT_OK(ParseProcessingMode("one_epoch", &mode));
  EXPECT_EQ(mode, ProcessingMode::ONE_EPOCH);
}

TEST(DataService, ParseInvalidProcessingMode) {
  ProcessingMode mode;
  Status s = ParseProcessingMode("invalid", &mode);
  EXPECT_EQ(s.code(), error::Code::INVALID_ARGUMENT);
}

TEST(DataService, ProcessingModeToString) {
  EXPECT_EQ("parallel_epochs",
            ProcessingModeToString(ProcessingMode::PARALLEL_EPOCHS));
  EXPECT_EQ("one_epoch", ProcessingModeToString(ProcessingMode::ONE_EPOCH));
}

Status CheckWorkerOutput(const std::string& worker_address, int64 task_id,
                         std::vector<std::vector<Tensor>> expected_output) {
  DataServiceWorkerClient worker(worker_address, kProtocol);
  for (std::vector<Tensor>& expected : expected_output) {
    bool end_of_sequence;
    CompressedElement compressed;
    TF_RETURN_IF_ERROR(
        worker.GetElement(task_id, &compressed, &end_of_sequence));
    if (end_of_sequence) {
      return errors::Internal("Reached end of sequence too early.");
    }
    std::vector<Tensor> element;
    TF_RETURN_IF_ERROR(UncompressElement(compressed, &element));
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::ExpectEqual(element, expected,
                                                       /*compare_order=*/true));
  }
  // Call GetElement a couple more times to verify tha end_of_sequence keeps
  // returning true.
  bool end_of_sequence;
  CompressedElement compressed;
  TF_RETURN_IF_ERROR(worker.GetElement(task_id, &compressed, &end_of_sequence));
  if (!end_of_sequence) {
    return errors::Internal("Expected end_of_sequence to be true");
  }
  TF_RETURN_IF_ERROR(worker.GetElement(task_id, &compressed, &end_of_sequence));
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
  DataServiceMasterClient master(cluster.MasterAddress(), kProtocol);

  int64 dataset_id;
  TF_ASSERT_OK(master.RegisterDataset(test_case.graph_def, &dataset_id));
  int64 job_id;
  TF_ASSERT_OK(
      master.CreateJob(dataset_id, ProcessingMode::PARALLEL_EPOCHS, &job_id));
  std::vector<TaskInfo> tasks;
  bool job_finished;
  TF_ASSERT_OK(master.GetTasks(job_id, &tasks, &job_finished));
  ASSERT_EQ(tasks.size(), 1);
  EXPECT_EQ(tasks[0].worker_address(), cluster.WorkerAddress(0));
  EXPECT_FALSE(job_finished);

  TF_EXPECT_OK(CheckWorkerOutput(tasks[0].worker_address(), tasks[0].id(),
                                 test_case.output));
}

TEST(DataService, IterateDatasetTwoWorkers) {
  TestCluster cluster(2);
  TF_ASSERT_OK(cluster.Initialize());
  test_util::GraphDefTestCase test_case;
  TF_ASSERT_OK(test_util::map_test_case(&test_case));
  DataServiceMasterClient master(cluster.MasterAddress(), kProtocol);

  int64 dataset_id;
  TF_ASSERT_OK(master.RegisterDataset(test_case.graph_def, &dataset_id));
  int64 job_id;
  TF_ASSERT_OK(
      master.CreateJob(dataset_id, ProcessingMode::PARALLEL_EPOCHS, &job_id));
  std::vector<TaskInfo> tasks;
  bool job_finished;
  TF_EXPECT_OK(master.GetTasks(job_id, &tasks, &job_finished));
  EXPECT_EQ(tasks.size(), 2);
  EXPECT_FALSE(job_finished);

  // Each worker produces the full dataset.
  for (TaskInfo task : tasks) {
    TF_EXPECT_OK(
        CheckWorkerOutput(task.worker_address(), task.id(), test_case.output));
  }
}

TEST(DataService, AddWorkerMidEpoch) {
  TestCluster cluster(1);
  TF_ASSERT_OK(cluster.Initialize());
  test_util::GraphDefTestCase test_case;
  TF_ASSERT_OK(test_util::map_test_case(&test_case));
  DataServiceMasterClient master(cluster.MasterAddress(), kProtocol);

  int64 dataset_id;
  TF_ASSERT_OK(master.RegisterDataset(test_case.graph_def, &dataset_id));
  int64 job_id;
  TF_ASSERT_OK(
      master.CreateJob(dataset_id, ProcessingMode::PARALLEL_EPOCHS, &job_id));
  std::vector<TaskInfo> tasks;
  bool job_finished;
  TF_ASSERT_OK(master.GetTasks(job_id, &tasks, &job_finished));
  EXPECT_EQ(tasks.size(), 1);
  EXPECT_FALSE(job_finished);
  TF_ASSERT_OK(cluster.AddWorker());
  TF_EXPECT_OK(master.GetTasks(job_id, &tasks, &job_finished));
  EXPECT_EQ(tasks.size(), 2);
  EXPECT_FALSE(job_finished);

  // Each worker produces the full dataset.
  for (TaskInfo task : tasks) {
    TF_EXPECT_OK(
        CheckWorkerOutput(task.worker_address(), task.id(), test_case.output));
  }
}

}  // namespace data
}  // namespace tensorflow
