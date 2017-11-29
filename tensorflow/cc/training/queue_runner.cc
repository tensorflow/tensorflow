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

#include "tensorflow/cc/training/queue_runner.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

Status QueueRunner::New(const QueueRunnerDef& queue_runner_def,
                        std::unique_ptr<QueueRunner>* result) {
  result->reset(new QueueRunner());
  return (*result)->Init(queue_runner_def);
}

Status QueueRunner::New(const QueueRunnerDef& queue_runner_def,
                        Coordinator* coord,
                        std::unique_ptr<QueueRunner>* result) {
  result->reset(new QueueRunner());
  (*result)->coord_ = coord;
  return (*result)->Init(queue_runner_def);
}

void QueueRunner::AddErrorCallback(const std::function<void(Status)>& cb) {
  mutex_lock l(cb_mu_);
  callbacks_.push_back(cb);
}

void QueueRunner::ClearErrorCallbacks() {
  mutex_lock l(cb_mu_);
  callbacks_.clear();
}

Status QueueRunner::Init(const QueueRunnerDef& queue_runner_def) {
  queue_name_ = queue_runner_def.queue_name();
  enqueue_op_names_.clear();
  enqueue_op_names_.insert(enqueue_op_names_.end(),
                           queue_runner_def.enqueue_op_name().begin(),
                           queue_runner_def.enqueue_op_name().end());
  size_t op_names_size = enqueue_op_names_.size();
  if (op_names_size > kint32max) {
    return Status(error::INVALID_ARGUMENT,
                  "Enqueue ops to run cannot exceed kint32max");
  }
  runs_ = static_cast<int>(op_names_size);
  if (runs_ == 0) {
    return Status(error::INVALID_ARGUMENT, "Empty enqueue ops to run.");
  }
  close_op_name_ = queue_runner_def.close_op_name();
  cancel_op_name_ = queue_runner_def.cancel_op_name();
  if (queue_runner_def.queue_closed_exception_types_size() == 0) {
    queue_closed_exception_types_.insert(error::OUT_OF_RANGE);
  } else {
    for (const auto& code : queue_runner_def.queue_closed_exception_types()) {
      queue_closed_exception_types_.insert(static_cast<int>(code));
    }
  }

  int nthreads = runs_;
  if (coord_) {
    // One more thread to call Stop()
    nthreads++;
  }
  thread_pool_.reset(new thread::ThreadPool(
      Env::Default(), SanitizeThreadSuffix(queue_name_), nthreads));

  return Status::OK();
}

QueueRunner::~QueueRunner() {
  // Cannot run Stop() here because the session might already be closed or
  // destroyed.
  Join().IgnoreError();
}

Status QueueRunner::Start(Session* sess) { return Start(sess, 0); }

Status QueueRunner::StartAndCollectCostGraph(Session* sess,
                                             const RunOptions* run_options) {
  SetRunArgumentsAndCostGraph(run_options);
  return Start(sess, 0);
}

Status QueueRunner::Start(Session* sess, int wait_for) {
  counter_.reset(new BlockingCounter(runs_));
  for (const string& enqueue_op : enqueue_op_names_) {
    thread_pool_->Schedule(
        std::bind(&QueueRunner::Run, this, sess, enqueue_op));
  }
  if (coord_) {
    thread_pool_->Schedule(std::bind(&QueueRunner::Stop, this, sess));
  }
  // Wait for up to 'wait_for' milliseconds.
  if (wait_for > 0) {
    if (!counter_->WaitFor(std::chrono::milliseconds(wait_for))) {
      return Status(error::DEADLINE_EXCEEDED,
                    "Queues not fed before the timeout");
    }
    // Check the status of the queue runner as well as the result of the enqueue
    // operations.
    mutex_lock l(mu_);
    if (!enqueue_status_.ok()) {
      return enqueue_status_;
    } else {
      return status_;
    }
  }
  return Status::OK();
}

Status QueueRunner::StartAndCollectCostGraph(Session* session, int wait_for_ms,
                                             const RunOptions* run_options) {
  SetRunArgumentsAndCostGraph(run_options);
  return Start(session, wait_for_ms);
}

void QueueRunner::Stop(Session* sess) {
  if (coord_ != nullptr) {
    coord_->WaitForStop();
  }
  if (!cancel_op_name_.empty()) {
    UpdateStatus(RealRun(sess, cancel_op_name_, false));
  }
  stopped_ = true;
}

Status QueueRunner::Join() {
  thread_pool_.reset();
  mutex_lock l(mu_);
  return status_;
}

void QueueRunner::UpdateStatus(const Status& status) {
  {
    mutex_lock l(mu_);
    if (!status_.ok() || status.ok() || IsQueueClosed(status)) {
      return;
    }
    status_ = status;
  }
  if (coord_) {
    coord_->ReportStatus(status);
  }
  mutex_lock l(cb_mu_);
  for (auto& cb : callbacks_) {
    cb(status);
  }
}

void QueueRunner::Run(Session* sess, const string& enqueue_op) {
  bool first_iteration = true;
  Status status;
  while (status.ok()) {
    if (coord_ && coord_->ShouldStop()) {
      break;
    }
    status = RealRun(sess, enqueue_op, true);
    if (first_iteration) {
      if (!status.ok()) {
        mutex_lock l(mu_);
        enqueue_status_ = status;
      }
      counter_->DecrementCount();
      first_iteration = false;
    }
  }
  bool last_run = false;
  {
    mutex_lock l(mu_);
    runs_--;
    last_run = (runs_ == 0);
  }

  // Close the queue unless the coordinator is shutting down since the cancel op
  // will be run anway in this case.
  if (IsQueueClosed(status) && (!coord_ || !coord_->ShouldStop())) {
    if (last_run && !close_op_name_.empty()) {
      UpdateStatus(RealRun(sess, close_op_name_, false));
    }
  } else if (!status.ok()) {
    LOG(ERROR) << "Queue runner thread got a failure status: "
               << status.ToString();
    UpdateStatus(status);
    if (coord_) {
      coord_->RequestStop().IgnoreError();
    }
  }
}

Status QueueRunner::GetStatus() {
  mutex_lock l(mu_);
  return status_;
}

Status QueueRunner::ExportCostGraph(CostGraphDef* cost_graph) const {
  if (!cg_mu_) {
    return Status(error::FAILED_PRECONDITION,
                  "This QueueRunner doesn't collect a cost graph.");
  }
  mutex_lock l(*cg_mu_);
  cost_graph->MergeFrom(*cost_graph_);
  return Status::OK();
}

void QueueRunner::SetRunArgumentsAndCostGraph(const RunOptions* run_options) {
  cg_mu_.reset(new mutex());
  {
    mutex_lock l(*cg_mu_);
    cost_graph_.reset(new CostGraphDef());
  }
  if (run_options) {
    run_options_ = *run_options;
  }
}

Status QueueRunner::RealRun(Session* sess, const string& op,
                            bool update_costs) {
  Status s;
  if (update_costs && cg_mu_) {
    RunMetadata metadata;
    s = sess->Run(run_options_, {}, {}, {op}, nullptr, &metadata);
    mutex_lock l(*cg_mu_);
    cost_graph_->Swap(metadata.mutable_cost_graph());
  } else {
    s = sess->Run({}, {}, {op}, nullptr);
  }
  return s;
}

}  // namespace tensorflow
