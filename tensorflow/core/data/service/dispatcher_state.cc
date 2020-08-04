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

#include <memory>

#include "tensorflow/core/data/service/journal.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {

DispatcherState::DispatcherState() {}

Status DispatcherState::Apply(Update update) {
  switch (update.update_type_case()) {
    case Update::kRegisterDataset:
      RegisterDataset(update.register_dataset());
      break;
    case Update::kCreateJob:
      CreateJob(update.create_job());
      break;
    case Update::kCreateTask:
      CreateTask(update.create_task());
      break;
    case Update::kFinishTask:
      FinishTask(update.finish_task());
      break;
    case Update::UPDATE_TYPE_NOT_SET:
      return errors::Internal("Update type not set.");
  }

  return Status::OK();
}

void DispatcherState::RegisterDataset(
    const RegisterDatasetUpdate& register_dataset) {
  int64 id = register_dataset.dataset_id();
  int64 fingerprint = register_dataset.fingerprint();
  auto dataset = std::make_shared<Dataset>(id, fingerprint,
                                           register_dataset.dataset_def());
  DCHECK(!datasets_by_id_.contains(id));
  datasets_by_id_[id] = dataset;
  DCHECK(!datasets_by_fingerprint_.contains(fingerprint));
  datasets_by_fingerprint_[fingerprint] = dataset;
  next_available_dataset_id_ = std::max(next_available_dataset_id_, id + 1);
}

void DispatcherState::CreateJob(const CreateJobUpdate& create_job) {
  int64 job_id = create_job.job_id();
  absl::optional<NamedJobKey> named_job_key;
  if (create_job.has_named_job_key()) {
    named_job_key.emplace(create_job.named_job_key().name(),
                          create_job.named_job_key().index());
  }
  auto job = std::make_shared<Job>(job_id, create_job.dataset_id(),
                                   ProcessingMode(create_job.processing_mode()),
                                   named_job_key);
  DCHECK(!jobs_.contains(job_id));
  jobs_[job_id] = job;
  if (named_job_key.has_value()) {
    DCHECK(!named_jobs_.contains(named_job_key.value()));
    named_jobs_[named_job_key.value()] = job;
  }
  next_available_job_id_ = std::max(next_available_job_id_, job_id + 1);
}

void DispatcherState::CreateTask(const CreateTaskUpdate& create_task) {
  int64 task_id = create_task.task_id();
  auto& task = tasks_[task_id];
  DCHECK_EQ(task, nullptr);
  task = std::make_shared<Task>(task_id, create_task.job_id(),
                                create_task.dataset_id(),
                                create_task.worker_address());
  tasks_by_job_[create_task.job_id()].push_back(task);
  next_available_task_id_ = std::max(next_available_task_id_, task_id + 1);
}

void DispatcherState::FinishTask(const FinishTaskUpdate& finish_task) {
  VLOG(2) << "Marking task " << finish_task.task_id() << " as finished";
  int64 task_id = finish_task.task_id();
  auto& task = tasks_[task_id];
  DCHECK(task != nullptr);
  task->finished = true;
  bool all_finished = true;
  for (const auto& task_for_job : tasks_by_job_[task->job_id]) {
    if (!task_for_job->finished) {
      all_finished = false;
    }
  }
  VLOG(3) << "Job " << task->job_id << " finished: " << all_finished;
  jobs_[task->job_id]->finished = all_finished;
}

int64 DispatcherState::NextAvailableDatasetId() const {
  return next_available_dataset_id_;
}

Status DispatcherState::DatasetFromId(
    int64 id, std::shared_ptr<const Dataset>* dataset) const {
  auto it = datasets_by_id_.find(id);
  if (it == datasets_by_id_.end()) {
    return errors::NotFound("Dataset id ", id, " not found");
  }
  *dataset = it->second;
  return Status::OK();
}

Status DispatcherState::DatasetFromFingerprint(
    uint64 fingerprint, std::shared_ptr<const Dataset>* dataset) const {
  auto it = datasets_by_fingerprint_.find(fingerprint);
  if (it == datasets_by_fingerprint_.end()) {
    return errors::NotFound("Dataset fingerprint ", fingerprint, " not found");
  }
  *dataset = it->second;
  return Status::OK();
}

std::vector<std::shared_ptr<const DispatcherState::Job>>
DispatcherState::ListJobs() {
  std::vector<std::shared_ptr<const DispatcherState::Job>> jobs;
  jobs.reserve(jobs_.size());
  for (const auto& it : jobs_) {
    jobs.push_back(it.second);
  }
  return jobs;
}

Status DispatcherState::JobFromId(int64 id,
                                  std::shared_ptr<const Job>* job) const {
  auto it = jobs_.find(id);
  if (it == jobs_.end()) {
    return errors::NotFound("Job id ", id, " not found");
  }
  *job = it->second;
  return Status::OK();
}

Status DispatcherState::NamedJobByKey(NamedJobKey named_job_key,
                                      std::shared_ptr<const Job>* job) const {
  auto it = named_jobs_.find(named_job_key);
  if (it == named_jobs_.end()) {
    return errors::NotFound("Named job key (", named_job_key.name, ", ",
                            named_job_key.index, ") not found");
  }
  *job = it->second;
  return Status::OK();
}

int64 DispatcherState::NextAvailableJobId() const {
  return next_available_job_id_;
}

Status DispatcherState::TaskFromId(int64 id,
                                   std::shared_ptr<const Task>* task) const {
  auto it = tasks_.find(id);
  if (it == tasks_.end()) {
    return errors::NotFound("Task ", id, " not found");
  }
  *task = it->second;
  return Status::OK();
}

Status DispatcherState::TasksForJob(
    int64 job_id, std::vector<std::shared_ptr<const Task>>* tasks) const {
  auto it = tasks_by_job_.find(job_id);
  if (it == tasks_by_job_.end()) {
    return errors::NotFound("Job ", job_id, " not found");
  }
  tasks->clear();
  tasks->reserve(it->second.size());
  for (const auto& task : it->second) {
    tasks->push_back(task);
  }
  return Status::OK();
}

int64 DispatcherState::NextAvailableTaskId() const {
  return next_available_task_id_;
}

}  // namespace data
}  // namespace tensorflow
