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
#include "tensorflow/core/data/service/dispatcher_state.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/journal.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {

namespace {
using Dataset = DispatcherState::Dataset;
using Worker = DispatcherState::Worker;
using IterationKey = DispatcherState::IterationKey;
using Job = DispatcherState::Job;
using Iteration = DispatcherState::Iteration;
using Task = DispatcherState::Task;
using ::tensorflow::testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

Status RegisterDataset(const std::string& dataset_id, uint64 fingerprint,
                       DispatcherState& state) {
  Update update;
  RegisterDatasetUpdate* register_dataset = update.mutable_register_dataset();
  register_dataset->set_dataset_id(dataset_id);
  register_dataset->set_fingerprint(fingerprint);
  register_dataset->set_dedupe_by_dataset_id(false);
  return state.Apply(update);
}

Status RegisterDataset(const std::string& dataset_id, DispatcherState& state) {
  return RegisterDataset(dataset_id, /*fingerprint=*/1, state);
}

Status RegisterDataset(const std::string& dataset_id, uint64_t fingerprint,
                       bool dedupe_by_dataset_id, DispatcherState& state) {
  Update update;
  RegisterDatasetUpdate* register_dataset = update.mutable_register_dataset();
  register_dataset->set_dataset_id(dataset_id);
  register_dataset->set_fingerprint(fingerprint);
  register_dataset->set_dedupe_by_dataset_id(dedupe_by_dataset_id);
  return state.Apply(update);
}

Status RegisterWorker(std::string worker_address, DispatcherState& state) {
  Update update;
  update.mutable_register_worker()->set_worker_address(worker_address);
  return state.Apply(update);
}

Status CreateJob(int64_t job_id, const std::string& dataset_id,
                 const std::string& job_name, DispatcherState& state) {
  Update update;
  CreateJobUpdate* create_job = update.mutable_create_job();
  create_job->set_job_id(job_id);
  create_job->set_dataset_id(dataset_id);
  create_job->set_job_name(job_name);
  return state.Apply(update);
}

Status CreateIteration(int64_t iteration_id, const std::string& dataset_id,
                       const IterationKey& named_iteration_key,
                       DispatcherState& state) {
  int64_t job_id = state.NextAvailableJobId();
  TF_RETURN_IF_ERROR(
      CreateJob(job_id, dataset_id, named_iteration_key.name, state));
  Update update;
  CreateIterationUpdate* create_iteration = update.mutable_create_iteration();
  create_iteration->set_job_id(job_id);
  create_iteration->set_iteration_id(iteration_id);
  create_iteration->set_repetition(named_iteration_key.repetition);
  return state.Apply(update);
}

Status CreateIteration(int64_t iteration_id, const std::string& dataset_id,
                       DispatcherState& state) {
  IterationKey key(/*name=*/absl::StrCat(random::New64()), /*repetition=*/0);
  return CreateIteration(iteration_id, dataset_id, key, state);
}

Status AcquireIterationClientId(int64_t iteration_id,
                                int64_t iteration_client_id,
                                DispatcherState& state) {
  Update update;
  AcquireIterationClientUpdate* acquire_iteration_client =
      update.mutable_acquire_iteration_client();
  acquire_iteration_client->set_iteration_id(iteration_id);
  acquire_iteration_client->set_iteration_client_id(iteration_client_id);
  return state.Apply(update);
}

Status ReleaseIterationClientId(int64_t iteration_client_id,
                                int64_t release_time, DispatcherState& state) {
  Update update;
  ReleaseIterationClientUpdate* release_iteration_client =
      update.mutable_release_iteration_client();
  release_iteration_client->set_iteration_client_id(iteration_client_id);
  release_iteration_client->set_time_micros(release_time);
  return state.Apply(update);
}

Status CreateTask(int64_t task_id, int64_t iteration_id,
                  const std::string& worker_address, DispatcherState& state) {
  Update update;
  CreateTaskUpdate* create_task = update.mutable_create_task();
  create_task->set_task_id(task_id);
  create_task->set_iteration_id(iteration_id);
  create_task->set_worker_address(worker_address);
  return state.Apply(update);
}

Status FinishTask(int64_t task_id, DispatcherState& state) {
  Update update;
  FinishTaskUpdate* finish_task = update.mutable_finish_task();
  finish_task->set_task_id(task_id);
  return state.Apply(update);
}

}  // namespace

TEST(DispatcherState, RegisterDataset) {
  uint64 fingerprint = 20;
  DispatcherState state;
  std::string dataset_id = state.NextAvailableDatasetId();
  int64_t dataset_id_int;
  ASSERT_TRUE(absl::SimpleAtoi(dataset_id, &dataset_id_int));
  TF_EXPECT_OK(RegisterDataset(dataset_id, fingerprint, state));
  EXPECT_EQ(state.NextAvailableDatasetId(), absl::StrCat(dataset_id_int + 1));

  {
    std::shared_ptr<const Dataset> dataset;
    TF_EXPECT_OK(state.DatasetFromFingerprint(fingerprint, dataset));
    EXPECT_EQ(dataset->dataset_id, dataset_id);
    EXPECT_TRUE(dataset->metadata.element_spec().empty());
    EXPECT_EQ(dataset->metadata.compression(),
              DataServiceMetadata::COMPRESSION_UNSPECIFIED);
  }
  {
    std::shared_ptr<const Dataset> dataset;
    TF_EXPECT_OK(state.DatasetFromId(dataset_id, dataset));
    EXPECT_EQ(dataset->fingerprint, fingerprint);
    EXPECT_TRUE(dataset->metadata.element_spec().empty());
    EXPECT_EQ(dataset->metadata.compression(),
              DataServiceMetadata::COMPRESSION_UNSPECIFIED);
  }
}

TEST(DispatcherState, RegisterDatasetWithExplicitID) {
  const uint64_t fingerprint = 20;
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset("dataset_id", fingerprint,
                               /*dedupe_by_dataset_id=*/true, state));
  std::shared_ptr<const Dataset> dataset;
  TF_EXPECT_OK(state.DatasetFromId("dataset_id", dataset));
  EXPECT_EQ(dataset->dataset_id, "dataset_id");
  // The fingerprint is not registered if the user requests an explicit ID.
  EXPECT_THAT(state.DatasetFromFingerprint(fingerprint, dataset),
              StatusIs(error::NOT_FOUND));
}

TEST(DispatcherState, RegisterDatasetsWithDifferentIDs) {
  const uint64_t fingerprint = 20;
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset("dataset_id1", fingerprint,
                               /*dedupe_by_dataset_id=*/true, state));
  TF_EXPECT_OK(RegisterDataset("dataset_id2", fingerprint,
                               /*dedupe_by_dataset_id=*/true, state));
  std::shared_ptr<const Dataset> dataset;
  TF_EXPECT_OK(state.DatasetFromId("dataset_id1", dataset));
  EXPECT_EQ(dataset->dataset_id, "dataset_id1");
  TF_EXPECT_OK(state.DatasetFromId("dataset_id2", dataset));
  EXPECT_EQ(dataset->dataset_id, "dataset_id2");
}

TEST(DispatcherState, RegisterDatasetsWithExplicitAndAnonymousIDs) {
  const uint64_t fingerprint = 20;
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset("dataset_id1", fingerprint,
                               /*dedupe_by_dataset_id=*/true, state));
  TF_EXPECT_OK(RegisterDataset("dataset_id2", fingerprint,
                               /*dedupe_by_dataset_id=*/false, state));
  std::shared_ptr<const Dataset> dataset;
  TF_EXPECT_OK(state.DatasetFromId("dataset_id1", dataset));
  EXPECT_EQ(dataset->dataset_id, "dataset_id1");
  TF_EXPECT_OK(state.DatasetFromId("dataset_id2", dataset));
  EXPECT_EQ(dataset->dataset_id, "dataset_id2");
  // The fingerprint is not registered if the user requests an explicit ID. So
  // the following query returns "dataset_id2".
  TF_EXPECT_OK(state.DatasetFromFingerprint(fingerprint, dataset));
  EXPECT_EQ(dataset->dataset_id, "dataset_id2");
}

TEST(DispatcherState, RegisterDatasetCompression) {
  DispatcherState state;
  const std::string dataset_id = state.NextAvailableDatasetId();
  Update update;
  RegisterDatasetUpdate* register_dataset = update.mutable_register_dataset();
  register_dataset->set_dataset_id(dataset_id);
  register_dataset->mutable_metadata()->set_compression(
      DataServiceMetadata::COMPRESSION_SNAPPY);
  TF_ASSERT_OK(state.Apply(update));
  {
    std::shared_ptr<const Dataset> dataset;
    TF_EXPECT_OK(state.DatasetFromId(dataset_id, dataset));
    EXPECT_EQ(dataset->metadata.compression(),
              DataServiceMetadata::COMPRESSION_SNAPPY);
  }
}

TEST(DispatcherState, RegisterDatasetElementSpec) {
  DispatcherState state;
  const std::string dataset_id = state.NextAvailableDatasetId();
  Update update;
  RegisterDatasetUpdate* register_dataset = update.mutable_register_dataset();
  register_dataset->set_dataset_id(dataset_id);
  register_dataset->set_fingerprint(20);
  register_dataset->mutable_metadata()->set_element_spec(
      "encoded_element_spec");
  TF_ASSERT_OK(state.Apply(update));
  {
    std::shared_ptr<const Dataset> dataset;
    TF_EXPECT_OK(state.DatasetFromId(dataset_id, dataset));
    EXPECT_EQ(dataset->metadata.element_spec(), "encoded_element_spec");
  }
}

TEST(DispatcherState, MissingDatasetId) {
  DispatcherState state;
  std::shared_ptr<const Dataset> dataset;
  Status s = state.DatasetFromId("missing_dataset_id", dataset);
  EXPECT_EQ(s.code(), error::NOT_FOUND);
}

TEST(DispatcherState, MissingDatasetFingerprint) {
  DispatcherState state;
  std::shared_ptr<const Dataset> dataset;
  Status s = state.DatasetFromFingerprint(0, dataset);
  EXPECT_EQ(s.code(), error::NOT_FOUND);
}

TEST(DispatcherState, NextAvailableDatasetId) {
  DispatcherState state;
  std::string dataset_id = state.NextAvailableDatasetId();
  int64_t dataset_id_int;
  ASSERT_TRUE(absl::SimpleAtoi(dataset_id, &dataset_id_int));

  uint64 fingerprint = 20;
  TF_EXPECT_OK(RegisterDataset(dataset_id, fingerprint, state));
  EXPECT_NE(state.NextAvailableDatasetId(), dataset_id);
  EXPECT_EQ(state.NextAvailableDatasetId(), absl::StrCat(dataset_id_int + 1));
  EXPECT_EQ(state.NextAvailableDatasetId(), state.NextAvailableDatasetId());
}

TEST(DispatcherState, RegisterWorker) {
  DispatcherState state;
  std::string address = "test_worker_address";
  TF_EXPECT_OK(RegisterWorker(address, state));
  std::shared_ptr<const Worker> worker;
  TF_EXPECT_OK(state.WorkerFromAddress(address, worker));
  EXPECT_EQ(worker->address, address);
}

TEST(DispatcherState, RegisterWorkerInFixedWorkerSet) {
  experimental::DispatcherConfig config;
  config.add_worker_addresses("/worker/task/0");
  config.add_worker_addresses("/worker/task/1");
  config.add_worker_addresses("/worker/task/2");

  DispatcherState state(config);
  TF_EXPECT_OK(state.ValidateWorker("/worker/task/0:20000"));
  TF_EXPECT_OK(state.ValidateWorker("/worker/task/1:20000"));
  TF_EXPECT_OK(state.ValidateWorker("/worker/task/2:20000"));
  TF_EXPECT_OK(RegisterWorker("/worker/task/0:20000", state));
  TF_EXPECT_OK(RegisterWorker("/worker/task/1:20000", state));
  TF_EXPECT_OK(RegisterWorker("/worker/task/2:20000", state));

  std::shared_ptr<const Worker> worker;
  TF_EXPECT_OK(state.WorkerFromAddress("/worker/task/0:20000", worker));
  EXPECT_EQ(worker->address, "/worker/task/0:20000");
}

TEST(DispatcherState, RegisterInvalidWorkerInFixedWorkerSet) {
  experimental::DispatcherConfig config;
  config.add_worker_addresses("/worker/task/0");
  config.add_worker_addresses("/worker/task/1");
  config.add_worker_addresses("/worker/task/2");

  DispatcherState state(config);
  EXPECT_THAT(state.ValidateWorker("localhost:20000"),
              StatusIs(error::FAILED_PRECONDITION,
                       HasSubstr("The worker's address is not configured")));

  // Tests that `RegisterWorker` always returns OK, and ignores errors. This is
  // because the journal records are supposed to be valid. If there is an error,
  // it should be caught by `ValidateWorker` and not written to the journal.
  TF_EXPECT_OK(RegisterWorker("localhost:20000", state));
  std::shared_ptr<const Worker> worker;
  EXPECT_THAT(state.WorkerFromAddress("/worker/task/0:20000", worker),
              StatusIs(error::NOT_FOUND,
                       "Worker with address /worker/task/0:20000 not found."));
}

TEST(DispatcherState, ListWorkers) {
  DispatcherState state;
  std::string address_1 = "address_1";
  std::string address_2 = "address_2";
  {
    std::vector<std::shared_ptr<const Worker>> workers = state.ListWorkers();
    EXPECT_THAT(workers, IsEmpty());
  }
  TF_EXPECT_OK(RegisterWorker(address_1, state));
  {
    std::vector<std::shared_ptr<const Worker>> workers = state.ListWorkers();
    EXPECT_THAT(workers, SizeIs(1));
  }
  TF_EXPECT_OK(RegisterWorker(address_2, state));
  {
    std::vector<std::shared_ptr<const Worker>> workers = state.ListWorkers();
    EXPECT_THAT(workers, SizeIs(2));
  }
}

TEST(DispatcherState, MissingWorker) {
  DispatcherState state;
  std::shared_ptr<const Worker> worker;
  Status s = state.WorkerFromAddress("test_worker_address", worker);
  EXPECT_EQ(s.code(), error::NOT_FOUND);
}

TEST(DispatcherState, UnknownUpdate) {
  DispatcherState state;
  Update update;
  Status s = state.Apply(update);
  EXPECT_EQ(s.code(), error::INTERNAL);
}

TEST(DispatcherState, JobName) {
  DispatcherState state;
  std::string dataset_id = state.NextAvailableDatasetId();
  int64_t job_id = state.NextAvailableJobId();
  std::string job_name = "test_name";
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateJob(job_id, dataset_id, job_name, state));
  std::shared_ptr<const Job> job;
  TF_EXPECT_OK(state.JobByName(job_name, job));
  EXPECT_EQ(state.NextAvailableJobId(), job_id + 1);
  EXPECT_EQ(job->dataset_id, dataset_id);
  EXPECT_FALSE(job->use_cross_trainer_cache);
}

TEST(DispatcherState, JobData) {
  DispatcherState state;
  std::string dataset_id = state.NextAvailableDatasetId();
  int64_t job_id = state.NextAvailableJobId();
  int64_t num_consumers = 8;
  bool use_cross_trainer_cache = true;
  TF_ASSERT_OK(RegisterDataset(dataset_id, state));
  Update update;
  CreateJobUpdate* create_job = update.mutable_create_job();
  create_job->set_job_id(job_id);
  create_job->set_dataset_id(dataset_id);
  create_job->set_num_consumers(num_consumers);
  create_job->set_use_cross_trainer_cache(use_cross_trainer_cache);
  TF_ASSERT_OK(state.Apply(update));
  std::shared_ptr<const Job> job;
  TF_ASSERT_OK(state.JobFromId(job_id, job));
  EXPECT_EQ(job->num_consumers, num_consumers);
  EXPECT_EQ(job->use_cross_trainer_cache, use_cross_trainer_cache);
}

TEST(DispatcherState, CrossTrainerCacheTask) {
  DispatcherState state;
  std::string dataset_id = state.NextAvailableDatasetId();
  std::string worker_address = "test_worker_address";
  TF_ASSERT_OK(RegisterDataset(dataset_id, state));

  int64_t job_id = state.NextAvailableJobId();
  Update job_update;
  CreateJobUpdate* create_job = job_update.mutable_create_job();
  create_job->set_job_id(job_id);
  create_job->set_dataset_id(dataset_id);
  create_job->set_use_cross_trainer_cache(true);
  TF_ASSERT_OK(state.Apply(job_update));

  int64_t iteration_id = state.NextAvailableIterationId();
  Update iteration_update;
  CreateIterationUpdate* create_iteration =
      iteration_update.mutable_create_iteration();
  create_iteration->set_job_id(job_id);
  create_iteration->set_iteration_id(iteration_id);
  TF_ASSERT_OK(state.Apply(iteration_update));

  int64_t task_id = state.NextAvailableTaskId();
  TF_EXPECT_OK(CreateTask(task_id, iteration_id, worker_address, state));
  std::shared_ptr<const Task> task;
  TF_EXPECT_OK(state.TaskFromId(task_id, task));
  EXPECT_EQ(task->iteration->iteration_id, iteration_id);
  EXPECT_EQ(task->task_id, task_id);
  EXPECT_EQ(task->worker_address, worker_address);
  EXPECT_TRUE(task->iteration->job->use_cross_trainer_cache);
}

TEST(DispatcherState, CreateTask) {
  std::string dataset_id = "dataset_id";
  int64_t iteration_id = 3;
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  int64_t task_id = state.NextAvailableTaskId();
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateIteration(iteration_id, dataset_id, state));
  TF_EXPECT_OK(CreateTask(task_id, iteration_id, worker_address, state));
  EXPECT_EQ(state.NextAvailableTaskId(), task_id + 1);
  {
    std::shared_ptr<const Task> task;
    TF_EXPECT_OK(state.TaskFromId(task_id, task));
    EXPECT_EQ(task->iteration->iteration_id, iteration_id);
    EXPECT_EQ(task->task_id, task_id);
    EXPECT_EQ(task->worker_address, worker_address);
    EXPECT_FALSE(task->iteration->job->use_cross_trainer_cache);
  }
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForIteration(iteration_id, tasks));
    EXPECT_THAT(tasks, SizeIs(1));
  }
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForWorker(worker_address, tasks));
    EXPECT_EQ(1, tasks.size());
  }
}

TEST(DispatcherState, CreateTasksForSameIteration) {
  std::string dataset_id = "dataset_id";
  int64_t iteration_id = 3;
  int64_t task_id_1 = 8;
  int64_t task_id_2 = 9;
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateIteration(iteration_id, dataset_id, state));
  TF_EXPECT_OK(CreateTask(task_id_1, iteration_id, worker_address, state));
  TF_EXPECT_OK(CreateTask(task_id_2, iteration_id, worker_address, state));
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForIteration(iteration_id, tasks));
    EXPECT_THAT(tasks, SizeIs(2));
  }
}

TEST(DispatcherState, CreateTasksForDifferentIterations) {
  std::string dataset_id = "dataset_id";
  int64_t iteration_id_1 = 3;
  int64_t iteration_id_2 = 4;
  int64_t task_id_1 = 8;
  int64_t task_id_2 = 9;
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateIteration(iteration_id_1, dataset_id, state));
  TF_EXPECT_OK(CreateIteration(iteration_id_2, dataset_id, state));
  TF_EXPECT_OK(CreateTask(task_id_1, iteration_id_1, worker_address, state));
  TF_EXPECT_OK(CreateTask(task_id_2, iteration_id_2, worker_address, state));
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForIteration(iteration_id_1, tasks));
    EXPECT_THAT(tasks, SizeIs(1));
  }
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForIteration(iteration_id_2, tasks));
    EXPECT_THAT(tasks, SizeIs(1));
  }
}

TEST(DispatcherState, CreateTasksForSameWorker) {
  std::string dataset_id = "dataset_id";
  int64_t iteration_id = 3;
  int64_t task_id_1 = 8;
  int64_t task_id_2 = 9;
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateIteration(iteration_id, dataset_id, state));
  TF_EXPECT_OK(CreateTask(task_id_1, iteration_id, worker_address, state));
  TF_EXPECT_OK(CreateTask(task_id_2, iteration_id, worker_address, state));
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForWorker(worker_address, tasks));
    EXPECT_EQ(2, tasks.size());
  }
}

TEST(DispatcherState, CreateTasksForDifferentWorkers) {
  std::string dataset_id = "dataset_id";
  int64_t iteration_id = 3;
  int64_t task_id_1 = 8;
  int64_t task_id_2 = 9;
  std::string worker_address_1 = "test_worker_address_1";
  std::string worker_address_2 = "test_worker_address_2";
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateIteration(iteration_id, dataset_id, state));
  TF_EXPECT_OK(CreateTask(task_id_1, iteration_id, worker_address_1, state));
  TF_EXPECT_OK(CreateTask(task_id_2, iteration_id, worker_address_2, state));
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForWorker(worker_address_1, tasks));
    EXPECT_EQ(1, tasks.size());
  }
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForWorker(worker_address_2, tasks));
    EXPECT_EQ(1, tasks.size());
  }
}

TEST(DispatcherState, GetTasksForWorkerEmpty) {
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  TF_EXPECT_OK(RegisterWorker(worker_address, state));
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForWorker(worker_address, tasks));
    EXPECT_EQ(0, tasks.size());
  }
}

TEST(DispatcherState, FinishTask) {
  std::string dataset_id = "dataset_id";
  int64_t iteration_id = 3;
  int64_t task_id = 4;
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateIteration(iteration_id, dataset_id, state));
  TF_EXPECT_OK(CreateTask(task_id, iteration_id, worker_address, state));
  TF_EXPECT_OK(FinishTask(task_id, state));
  std::shared_ptr<const Task> task;
  TF_EXPECT_OK(state.TaskFromId(task_id, task));
  EXPECT_TRUE(task->finished);
  std::shared_ptr<const Iteration> iteration;
  TF_EXPECT_OK(state.IterationFromId(iteration_id, iteration));
  EXPECT_TRUE(iteration->finished);
}

TEST(DispatcherState, FinishMultiTaskIteration) {
  std::string dataset_id = "dataset_id";
  int64_t iteration_id = 3;
  int64_t task_id_1 = 4;
  int64_t task_id_2 = 5;
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateIteration(iteration_id, dataset_id, state));
  TF_EXPECT_OK(CreateTask(task_id_1, iteration_id, worker_address, state));
  TF_EXPECT_OK(CreateTask(task_id_2, iteration_id, worker_address, state));

  TF_EXPECT_OK(FinishTask(task_id_1, state));
  {
    std::shared_ptr<const Iteration> iteration;
    TF_EXPECT_OK(state.IterationFromId(iteration_id, iteration));
    EXPECT_FALSE(iteration->finished);
  }

  TF_EXPECT_OK(FinishTask(task_id_2, state));
  {
    std::shared_ptr<const Iteration> iteration;
    TF_EXPECT_OK(state.IterationFromId(iteration_id, iteration));
    EXPECT_TRUE(iteration->finished);
  }
}

TEST(DispatcherState, AcquireIterationClientId) {
  std::string dataset_id = "dataset_id";
  int64_t iteration_id = 3;
  int64_t iteration_client_id_1 = 1;
  int64_t iteration_client_id_2 = 2;
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateIteration(iteration_id, dataset_id, state));
  TF_EXPECT_OK(
      AcquireIterationClientId(iteration_id, iteration_client_id_1, state));
  {
    std::shared_ptr<const Iteration> iteration;
    TF_EXPECT_OK(state.IterationFromId(iteration_id, iteration));
    EXPECT_EQ(iteration->num_clients, 1);
    TF_EXPECT_OK(
        AcquireIterationClientId(iteration_id, iteration_client_id_2, state));
    EXPECT_EQ(iteration->num_clients, 2);
  }
  {
    std::shared_ptr<const Iteration> iteration;
    TF_EXPECT_OK(
        state.IterationForIterationClientId(iteration_client_id_1, iteration));
    EXPECT_EQ(iteration->iteration_id, iteration_id);
  }
  {
    std::shared_ptr<const Iteration> iteration;
    TF_EXPECT_OK(
        state.IterationForIterationClientId(iteration_client_id_2, iteration));
    EXPECT_EQ(iteration->iteration_id, iteration_id);
  }
}

TEST(DispatcherState, ReleaseIterationClientId) {
  std::string dataset_id = "dataset_id";
  int64_t iteration_id = 3;
  int64_t iteration_client_id = 6;
  int64_t release_time = 100;
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateIteration(iteration_id, dataset_id, state));
  TF_EXPECT_OK(
      AcquireIterationClientId(iteration_id, iteration_client_id, state));
  TF_EXPECT_OK(
      ReleaseIterationClientId(iteration_client_id, release_time, state));
  std::shared_ptr<const Iteration> iteration;
  TF_EXPECT_OK(state.IterationFromId(iteration_id, iteration));
  EXPECT_EQ(iteration->num_clients, 0);
  Status s =
      state.IterationForIterationClientId(iteration_client_id, iteration);
  EXPECT_EQ(s.code(), error::NOT_FOUND);
}

TEST(DispatcherState, ListActiveClientsEmpty) {
  std::string dataset_id = "dataset_id";
  int64_t iteration_id = 3;
  int64_t iteration_client_id = 6;
  int64_t release_time = 100;
  DispatcherState state;
  EXPECT_THAT(state.ListActiveClientIds(), IsEmpty());
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateIteration(iteration_id, dataset_id, state));
  TF_EXPECT_OK(
      AcquireIterationClientId(iteration_id, iteration_client_id, state));
  TF_EXPECT_OK(
      ReleaseIterationClientId(iteration_client_id, release_time, state));
  EXPECT_THAT(state.ListActiveClientIds(), IsEmpty());
}

TEST(DispatcherState, ListActiveClients) {
  std::string dataset_id = "dataset_id";
  int64_t iteration_id = 3;
  int64_t iteration_client_id_1 = 6;
  int64_t iteration_client_id_2 = 7;
  int64_t iteration_client_id_3 = 8;
  int64_t release_time = 100;
  DispatcherState state;
  TF_EXPECT_OK(RegisterDataset(dataset_id, state));
  TF_EXPECT_OK(CreateIteration(iteration_id, dataset_id, state));
  TF_EXPECT_OK(
      AcquireIterationClientId(iteration_id, iteration_client_id_1, state));
  TF_EXPECT_OK(
      AcquireIterationClientId(iteration_id, iteration_client_id_2, state));
  TF_EXPECT_OK(
      ReleaseIterationClientId(iteration_client_id_2, release_time, state));
  TF_EXPECT_OK(
      AcquireIterationClientId(iteration_id, iteration_client_id_3, state));
  EXPECT_THAT(state.ListActiveClientIds(), UnorderedElementsAre(6, 8));
}

}  // namespace data
}  // namespace tensorflow
