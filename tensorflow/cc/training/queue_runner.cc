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
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

QueueRunner::QueueRunner() : started_(false) {}

QueueRunner::QueueRunner(const QueueRunnerDef& queue_runner_def)
    : started_(false) {
  TF_CHECK_OK(Init(queue_runner_def));
}

Status QueueRunner::Init(const QueueRunnerDef& queue_runner_def) {
  if (started_.load()) {
    return Status(error::ALREADY_EXISTS, "QueueRunner is already running.");
  }
  queue_name_ = queue_runner_def.queue_name();
  enqueue_op_names_.insert(enqueue_op_names_.end(),
                           queue_runner_def.enqueue_op_name().begin(),
                           queue_runner_def.enqueue_op_name().end());
  runs_ = enqueue_op_names_.size();
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

  thread_pool_.reset(
      new thread::ThreadPool(Env::Default(), queue_name_, runs_));
  should_stop_ = false;
  return Status::OK();
}

QueueRunner::~QueueRunner() {
  // Cannot run Stop() here because the session might already be closed or
  // destroyed.
  Join();
}

Status QueueRunner::Start(Session* sess) {
  if (runs_ == 0) {
    return Status(
        error::INVALID_ARGUMENT,
        "No enqueue ops to run. You may want to Init the QueueRunner first.");
  }
  started_ = true;
  for (const string& enqueue_op : enqueue_op_names_) {
    thread_pool_->Schedule(
        std::bind(&QueueRunner::Run, this, sess, enqueue_op));
  }
  return Status::OK();
}

Status QueueRunner::Stop(Session* sess) {
  should_stop_ = true;
  if (cancel_op_name_.empty()) {
    return Status::OK();
  } else {
    return sess->Run({}, {}, {cancel_op_name_}, nullptr);
  }
}

Status QueueRunner::Join() {
  thread_pool_.reset();
  started_ = false;
  return status_;
}

void QueueRunner::Run(Session* sess, const string& enqueue_op) {
  bool decremented = false;
  while (!should_stop_.load()) {
    auto status = sess->Run({}, {}, {enqueue_op}, nullptr);
    if (status.ok()) {
      continue;
    } else if (queue_closed_exception_types_.count(
                   static_cast<int>(status.code())) > 0) {
      mutex_lock l(mu_);
      runs_--;
      decremented = true;
      should_stop_ = true;

      // If all enqueue ops have finished, run the close op.
      if (runs_ == 0 && !close_op_name_.empty()) {
        auto s = sess->Run({}, {}, {close_op_name_}, nullptr);
        if (!s.ok() && status_.ok() &&
            queue_closed_exception_types_.count(static_cast<int>(s.code())) ==
                0) {
          status_ = s;
        }
      }
    } else {
      {
        mutex_lock l(mu_);
        should_stop_ = true;
        // Only record the first failure status.
        if (status_.ok()) {
          status_ = status;
        }
      }
      // Stop the queue runner immediately to propagate the error to
      // subsequent queues.
      Stop(sess);
    }
  }

  if (!decremented) {
    mutex_lock l(mu_);
    runs_--;
  }
}

}  // namespace tensorflow
