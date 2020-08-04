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

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/journal.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {

namespace {
using Dataset = DispatcherState::Dataset;
using NamedJobKey = DispatcherState::NamedJobKey;
using Job = DispatcherState::Job;
using Task = DispatcherState::Task;
using ::testing::SizeIs;

Status RegisterDatasetWithIdAndFingerprint(int64 id, uint64 fingerprint,
                                           DispatcherState* state) {
  Update update;
  RegisterDatasetUpdate* register_dataset = update.mutable_register_dataset();
  register_dataset->set_dataset_id(id);
  register_dataset->set_fingerprint(fingerprint);
  TF_RETURN_IF_ERROR(state->Apply(update));
  return Status::OK();
}

Status CreateAnonymousJob(int64 job_id, int64 dataset_id,
                          DispatcherState* state) {
  Update update;
  CreateJobUpdate* create_job = update.mutable_create_job();
  create_job->set_job_id(job_id);
  create_job->set_dataset_id(dataset_id);
  create_job->set_processing_mode(ProcessingModeDef::PARALLEL_EPOCHS);
  TF_RETURN_IF_ERROR(state->Apply(update));
  return Status::OK();
}

Status CreateNamedJob(int64 job_id, int64 dataset_id, NamedJobKey named_job_key,
                      DispatcherState* state) {
  Update update;
  CreateJobUpdate* create_job = update.mutable_create_job();
  create_job->set_job_id(job_id);
  create_job->set_dataset_id(dataset_id);
  create_job->set_processing_mode(ProcessingModeDef::PARALLEL_EPOCHS);
  NamedJobKeyDef* key = create_job->mutable_named_job_key();
  key->set_name(named_job_key.name);
  key->set_index(named_job_key.index);
  TF_RETURN_IF_ERROR(state->Apply(update));
  return Status::OK();
}

Status CreateTask(int64 task_id, int64 job_id, int64 dataset_id,
                  StringPiece worker_address, DispatcherState* state) {
  Update update;
  CreateTaskUpdate* create_task = update.mutable_create_task();
  create_task->set_task_id(task_id);
  create_task->set_job_id(job_id);
  create_task->set_dataset_id(dataset_id);
  create_task->set_worker_address(worker_address);
  TF_RETURN_IF_ERROR(state->Apply(update));
  return Status::OK();
}

Status FinishTask(int64 task_id, DispatcherState* state) {
  Update update;
  FinishTaskUpdate* finish_task = update.mutable_finish_task();
  finish_task->set_task_id(task_id);
  TF_RETURN_IF_ERROR(state->Apply(update));
  return Status::OK();
}
}  // namespace

TEST(DispatcherState, RegisterDataset) {
  int64 id = 10;
  uint64 fingerprint = 20;
  DispatcherState state;
  TF_EXPECT_OK(RegisterDatasetWithIdAndFingerprint(id, fingerprint, &state));
  EXPECT_EQ(state.NextAvailableDatasetId(), id + 1);

  {
    std::shared_ptr<const Dataset> dataset;
    TF_EXPECT_OK(state.DatasetFromFingerprint(fingerprint, &dataset));
    EXPECT_EQ(id, dataset->dataset_id);
  }
  {
    std::shared_ptr<const Dataset> dataset;
    TF_EXPECT_OK(state.DatasetFromId(id, &dataset));
    EXPECT_EQ(fingerprint, dataset->fingerprint);
  }
}

TEST(DispatcherState, MissingDatasetId) {
  DispatcherState state;
  std::shared_ptr<const Dataset> dataset;
  Status s = state.DatasetFromId(0, &dataset);
  EXPECT_EQ(s.code(), error::NOT_FOUND);
}

TEST(DispatcherState, MissingDatasetFingerprint) {
  DispatcherState state;
  std::shared_ptr<const Dataset> dataset;
  Status s = state.DatasetFromFingerprint(0, &dataset);
  EXPECT_EQ(s.code(), error::NOT_FOUND);
}

TEST(DispatcherState, NextAvailableDatasetId) {
  DispatcherState state;
  int64 id = state.NextAvailableDatasetId();
  uint64 fingerprint = 20;
  TF_EXPECT_OK(RegisterDatasetWithIdAndFingerprint(id, fingerprint, &state));
  EXPECT_NE(id, state.NextAvailableDatasetId());
  EXPECT_EQ(state.NextAvailableDatasetId(), state.NextAvailableDatasetId());
}

TEST(DispatcherState, UnknownUpdate) {
  DispatcherState state;
  Update update;
  Status s = state.Apply(update);
  EXPECT_EQ(s.code(), error::INTERNAL);
}

TEST(DispatcherState, AnonymousJob) {
  int64 job_id = 3;
  int64 dataset_id = 10;
  DispatcherState state;
  TF_EXPECT_OK(RegisterDatasetWithIdAndFingerprint(dataset_id, 1, &state));
  TF_EXPECT_OK(CreateAnonymousJob(job_id, dataset_id, &state));
  std::shared_ptr<const Job> job;
  TF_EXPECT_OK(state.JobFromId(job_id, &job));
  EXPECT_EQ(state.NextAvailableJobId(), job_id + 1);
  EXPECT_EQ(dataset_id, job->dataset_id);
  EXPECT_EQ(job_id, job->job_id);
  EXPECT_FALSE(job->finished);
}

TEST(DispatcherState, NamedJob) {
  int64 job_id = 3;
  int64 dataset_id = 10;
  DispatcherState state;
  TF_EXPECT_OK(RegisterDatasetWithIdAndFingerprint(dataset_id, 1, &state));
  NamedJobKey named_job_key("test", 1);
  TF_EXPECT_OK(CreateNamedJob(job_id, dataset_id, named_job_key, &state));
  std::shared_ptr<const Job> job;
  TF_EXPECT_OK(state.NamedJobByKey(named_job_key, &job));
  EXPECT_EQ(state.NextAvailableJobId(), job_id + 1);
  EXPECT_EQ(dataset_id, job->dataset_id);
  EXPECT_EQ(job_id, job->job_id);
  EXPECT_FALSE(job->finished);
}

TEST(DispatcherState, CreateTask) {
  int64 job_id = 3;
  int64 dataset_id = 10;
  int64 task_id = 8;
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  TF_EXPECT_OK(RegisterDatasetWithIdAndFingerprint(dataset_id, 1, &state));
  TF_EXPECT_OK(CreateAnonymousJob(job_id, dataset_id, &state));
  TF_EXPECT_OK(CreateTask(task_id, job_id, dataset_id, worker_address, &state));
  EXPECT_EQ(state.NextAvailableTaskId(), task_id + 1);
  {
    std::shared_ptr<const Task> task;
    TF_EXPECT_OK(state.TaskFromId(task_id, &task));
    EXPECT_EQ(task_id, task->task_id);
    EXPECT_EQ(job_id, task->job_id);
    EXPECT_EQ(dataset_id, task->dataset_id);
    EXPECT_EQ(worker_address, task->worker_address);
  }
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForJob(job_id, &tasks));
    EXPECT_THAT(tasks, SizeIs(1));
  }
}

TEST(DispatcherState, CreateTasksForSameJob) {
  int64 job_id = 3;
  int64 dataset_id = 10;
  int64 task_id_1 = 8;
  int64 task_id_2 = 9;
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  TF_EXPECT_OK(RegisterDatasetWithIdAndFingerprint(dataset_id, 1, &state));
  TF_EXPECT_OK(CreateAnonymousJob(job_id, dataset_id, &state));
  TF_EXPECT_OK(
      CreateTask(task_id_1, job_id, dataset_id, worker_address, &state));
  TF_EXPECT_OK(
      CreateTask(task_id_2, job_id, dataset_id, worker_address, &state));
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForJob(job_id, &tasks));
    EXPECT_EQ(2, tasks.size());
  }
}

TEST(DispatcherState, CreateTasksForDifferentJobs) {
  int64 job_id_1 = 3;
  int64 job_id_2 = 4;
  int64 dataset_id = 10;
  int64 task_id_1 = 8;
  int64 task_id_2 = 9;
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  TF_EXPECT_OK(RegisterDatasetWithIdAndFingerprint(dataset_id, 1, &state));
  TF_EXPECT_OK(CreateAnonymousJob(job_id_1, dataset_id, &state));
  TF_EXPECT_OK(CreateAnonymousJob(job_id_2, dataset_id, &state));
  TF_EXPECT_OK(
      CreateTask(task_id_1, job_id_1, dataset_id, worker_address, &state));
  TF_EXPECT_OK(
      CreateTask(task_id_2, job_id_2, dataset_id, worker_address, &state));
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForJob(job_id_1, &tasks));
    EXPECT_EQ(1, tasks.size());
  }
  {
    std::vector<std::shared_ptr<const Task>> tasks;
    TF_EXPECT_OK(state.TasksForJob(job_id_2, &tasks));
    EXPECT_EQ(1, tasks.size());
  }
}

TEST(DispatcherState, FinishTask) {
  int64 job_id = 3;
  int64 dataset_id = 10;
  int64 task_id = 4;
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  TF_EXPECT_OK(RegisterDatasetWithIdAndFingerprint(dataset_id, 1, &state));
  TF_EXPECT_OK(CreateAnonymousJob(job_id, dataset_id, &state));
  TF_EXPECT_OK(CreateTask(task_id, job_id, dataset_id, worker_address, &state));
  TF_EXPECT_OK(FinishTask(task_id, &state));
  std::shared_ptr<const Task> task;
  TF_EXPECT_OK(state.TaskFromId(task_id, &task));
  EXPECT_TRUE(task->finished);
  std::shared_ptr<const Job> job;
  TF_EXPECT_OK(state.JobFromId(job_id, &job));
  EXPECT_TRUE(job->finished);
}

TEST(DispatcherState, FinishMultiTaskJob) {
  int64 job_id = 3;
  int64 dataset_id = 10;
  int64 task_id_1 = 4;
  int64 task_id_2 = 5;
  std::string worker_address = "test_worker_address";
  DispatcherState state;
  TF_EXPECT_OK(RegisterDatasetWithIdAndFingerprint(dataset_id, 1, &state));
  TF_EXPECT_OK(CreateAnonymousJob(job_id, dataset_id, &state));
  TF_EXPECT_OK(
      CreateTask(task_id_1, job_id, dataset_id, worker_address, &state));
  TF_EXPECT_OK(
      CreateTask(task_id_2, job_id, dataset_id, worker_address, &state));

  TF_EXPECT_OK(FinishTask(task_id_1, &state));
  {
    std::shared_ptr<const Job> job;
    TF_EXPECT_OK(state.JobFromId(job_id, &job));
    EXPECT_FALSE(job->finished);
  }

  TF_EXPECT_OK(FinishTask(task_id_2, &state));
  {
    std::shared_ptr<const Job> job;
    TF_EXPECT_OK(state.JobFromId(job_id, &job));
    EXPECT_TRUE(job->finished);
  }
}

}  // namespace data
}  // namespace tensorflow
