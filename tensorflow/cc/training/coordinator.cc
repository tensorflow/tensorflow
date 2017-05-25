/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/training/coordinator.h"

namespace tensorflow {

Coordinator::Coordinator() : Coordinator(std::vector<error::Code>()) {}

Coordinator::Coordinator(const std::vector<error::Code>& clean_stop_errors)
    : should_stop_(false) {
  if (clean_stop_errors.empty()) {
    clean_stop_errors_.insert(error::OUT_OF_RANGE);
  } else {
    for (const auto& code : clean_stop_errors) {
      clean_stop_errors_.insert(static_cast<int>(code));
    }
  }
}

Coordinator::~Coordinator() {
  RequestStop().IgnoreError();
  Join().IgnoreError();
}

Status Coordinator::RegisterRunner(std::unique_ptr<RunnerInterface> runner) {
  {
    mutex_lock l(mu_);
    if (should_stop_) {
      return Status(error::FAILED_PRECONDITION,
                    "The coordinator has been stopped.");
    }
  }
  mutex_lock l(runners_lock_);
  runners_.push_back(std::move(runner));
  return Status::OK();
}

bool Coordinator::AllRunnersStopped() {
  mutex_lock l(runners_lock_);
  for (const auto& runner : runners_) {
    if (runner->IsRunning()) {
      return false;
    }
  }
  return true;
}

Status Coordinator::RequestStop() {
  mutex_lock l(mu_);
  if (should_stop_) {
    return Status(error::FAILED_PRECONDITION,
                  "The Coordinator is not running.");
  }
  should_stop_ = true;
  wait_for_stop_.notify_all();
  return Status::OK();
}

bool Coordinator::ShouldStop() {
  mutex_lock l(mu_);
  return should_stop_;
}

Status Coordinator::Join() {
  // TODO(yuefengz): deal with stragglers.
  {
    mutex_lock l(mu_);
    if (!should_stop_) {
      return Status(error::FAILED_PRECONDITION,
                    "Joining coordinator without requesting to stop.");
    }
  }

  {
    mutex_lock l(runners_lock_);
    for (const auto& t : runners_) {
      ReportStatus(t->Join());
    }
    runners_.clear();
  }
  return GetStatus();
}

void Coordinator::ReportStatus(const Status& status) {
  mutex_lock l(status_lock_);
  if (status.ok() || !status_.ok() ||
      clean_stop_errors_.count(static_cast<int>(status.code())) > 0) {
    return;
  }
  status_ = status;
}

Status Coordinator::GetStatus() {
  mutex_lock l(status_lock_);
  return status_;
}

void Coordinator::WaitForStop() {
  mutex_lock l(mu_);
  while (!should_stop_) {
    wait_for_stop_.wait(l);
  }
}

Status Coordinator::ExportCostGraph(CostGraphDef* cost_graph) const {
  mutex_lock l(runners_lock_);
  for (auto& t : runners_) {
    Status s = t->ExportCostGraph(cost_graph);
    if (!s.ok()) {
      return s;
    }
  }
  return Status::OK();
}

}  // namespace
