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
Status RegisterDatasetWithIdAndFingerprint(int64 id, uint64 fingerprint,
                                           DispatcherState* state) {
  NoopJournalWriter journal_writer;
  Update update;
  RegisterDatasetUpdate* register_dataset = update.mutable_register_dataset();
  register_dataset->set_dataset_id(id);
  register_dataset->set_fingerprint(fingerprint);
  TF_RETURN_IF_ERROR(state->Apply(update, &journal_writer));
  return Status::OK();
}

Status CreateAnonymousJob(int64 job_id, int64 dataset_id,
                          DispatcherState* state) {
  NoopJournalWriter journal_writer;
  Update update;
  CreateJobUpdate* create_job = update.mutable_create_job();
  create_job->set_job_id(job_id);
  create_job->set_dataset_id(dataset_id);
  create_job->set_processing_mode(ProcessingModeDef::PARALLEL_EPOCHS);
  TF_RETURN_IF_ERROR(state->Apply(update, &journal_writer));
  return Status::OK();
}

Status CreateNamedJob(int64 job_id, int64 dataset_id,
                      DispatcherState::NamedJobKey named_job_key,
                      DispatcherState* state) {
  NoopJournalWriter journal_writer;
  Update update;
  CreateJobUpdate* create_job = update.mutable_create_job();
  create_job->set_job_id(job_id);
  create_job->set_dataset_id(dataset_id);
  create_job->set_processing_mode(ProcessingModeDef::PARALLEL_EPOCHS);
  NamedJobKeyDef* key = create_job->mutable_named_job_key();
  key->set_name(named_job_key.name);
  key->set_index(named_job_key.index);
  TF_RETURN_IF_ERROR(state->Apply(update, &journal_writer));
  return Status::OK();
}

Status FinishJob(int64 job_id, DispatcherState* state) {
  NoopJournalWriter journal_writer;
  Update update;
  FinishJobUpdate* finish_job = update.mutable_finish_job();
  finish_job->set_job_id(job_id);
  TF_RETURN_IF_ERROR(state->Apply(update, &journal_writer));
  return Status::OK();
}
}  // namespace

TEST(DispatcherState, RegisterDataset) {
  int64 id = 10;
  uint64 fingerprint = 20;
  DispatcherState state;
  TF_EXPECT_OK(RegisterDatasetWithIdAndFingerprint(id, fingerprint, &state));

  {
    std::shared_ptr<const DispatcherState::Dataset> dataset;
    TF_EXPECT_OK(state.DatasetFromFingerprint(fingerprint, &dataset));
    EXPECT_EQ(id, dataset->dataset_id);
  }
  {
    std::shared_ptr<const DispatcherState::Dataset> dataset;
    TF_EXPECT_OK(state.DatasetFromId(id, &dataset));
    EXPECT_EQ(fingerprint, dataset->fingerprint);
  }
}

TEST(DispatcherState, MissingDatasetId) {
  DispatcherState state;
  std::shared_ptr<const DispatcherState::Dataset> dataset;
  Status s = state.DatasetFromId(0, &dataset);
  EXPECT_EQ(s.code(), error::NOT_FOUND);
}

TEST(DispatcherState, MissingDatasetFingerprint) {
  DispatcherState state;
  std::shared_ptr<const DispatcherState::Dataset> dataset;
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
  NoopJournalWriter journal_writer;
  DispatcherState state;
  Update update;
  Status s = state.Apply(update, &journal_writer);
  EXPECT_EQ(s.code(), error::INTERNAL);
}

TEST(DispatcherState, AnonymousJob) {
  int64 job_id = 3;
  int64 dataset_id = 10;
  DispatcherState state;
  Update update;
  TF_EXPECT_OK(RegisterDatasetWithIdAndFingerprint(dataset_id, 1, &state));
  TF_EXPECT_OK(CreateAnonymousJob(job_id, dataset_id, &state));
  std::shared_ptr<const DispatcherState::Job> job;
  TF_EXPECT_OK(state.JobFromId(job_id, &job));
  EXPECT_EQ(dataset_id, job->dataset_id);
  EXPECT_EQ(job_id, job->job_id);
  EXPECT_FALSE(job->finished);
}

TEST(DispatcherState, NamedJob) {
  int64 job_id = 3;
  int64 dataset_id = 10;
  DispatcherState state;
  Update update;
  TF_EXPECT_OK(RegisterDatasetWithIdAndFingerprint(dataset_id, 1, &state));
  DispatcherState::NamedJobKey named_job_key("test", 1);
  TF_EXPECT_OK(CreateNamedJob(job_id, dataset_id, named_job_key, &state));
  std::shared_ptr<const DispatcherState::Job> job;
  TF_EXPECT_OK(state.NamedJobByKey(named_job_key, &job));
  EXPECT_EQ(dataset_id, job->dataset_id);
  EXPECT_EQ(job_id, job->job_id);
  EXPECT_FALSE(job->finished);
}

TEST(DispatcherState, FinishJob) {
  int64 job_id = 3;
  int64 dataset_id = 10;
  DispatcherState state;
  Update update;
  TF_EXPECT_OK(RegisterDatasetWithIdAndFingerprint(dataset_id, 1, &state));
  TF_EXPECT_OK(CreateAnonymousJob(job_id, dataset_id, &state));
  TF_EXPECT_OK(FinishJob(job_id, &state));
  std::shared_ptr<const DispatcherState::Job> job;
  TF_EXPECT_OK(state.JobFromId(job_id, &job));
  EXPECT_TRUE(job->finished);
}

}  // namespace data
}  // namespace tensorflow
