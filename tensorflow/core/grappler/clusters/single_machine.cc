/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/cc/training/queue_runner.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace grappler {

SingleMachine::SingleMachine(int timeout_s, int num_cpu_cores, int num_gpus)
    : Cluster(timeout_s),
      num_gpus_(num_gpus),
      running_(false),
      closing_(false) {
  thread_pool_.reset(new thread::ThreadPool(
      Env::Default(), SanitizeThreadSuffix("single_machine"), 2));

  (*options_.config.mutable_device_count())["CPU"] = 1;
  if (num_gpus > 0) {
    (*options_.config.mutable_device_count())["GPU"] = num_gpus;
  }
  CHECK_GE(num_cpu_cores, 1);
  options_.config.set_intra_op_parallelism_threads(num_cpu_cores);
  options_.config.set_inter_op_parallelism_threads(num_cpu_cores);
}

SingleMachine::~SingleMachine() {
  CloseSession(false /*use_timeout*/).IgnoreError();

  // Prevent the destructor from deleting mu_ until CloseSession() is done.
  mutex_lock l(mu_);
}

Status SingleMachine::Provision() {
  Status status = ResetSession();
  if (!status.ok()) {
    return status;
  }

  DeviceAttributes attr;
  attr.set_name("/job:localhost/replica:0/task:0/cpu:0");
  attr.set_device_type("CPU");
  devices_.push_back(attr);

  for (int i = 0; i < num_gpus_; ++i) {
    DeviceAttributes attr;
    attr.set_name(strings::StrCat("/job:localhost/replica:0/task:0/gpu:", i));
    attr.set_device_type("GPU");
    devices_.push_back(attr);
  }
  return Status::OK();
}

Status SingleMachine::Initialize(const GrapplerItem& item) {
  if (last_graph_ != &item.graph) {
    init_ops_ = item.init_ops;
    last_graph_ = nullptr;
    queue_runner_defs_ = item.queue_runners;
  }
  return Status::OK();
}

Status SingleMachine::Run(const GraphDef& graph_def,
                          const std::vector<std::pair<string, Tensor>>& feed,
                          const std::vector<string>& fetch,
                          RunMetadata* metadata) {
  if (last_graph_ != &graph_def) {
    Status status = ResetSession();
    if (status.ok()) {
      status = session_->Create(graph_def);
    }
    if (!init_ops_.empty() && status.ok()) {
      status = RunWithTimeout({}, init_ops_, nullptr);
    }
    for (int i = 0; i < queue_runner_defs_.size() && status.ok(); ++i) {
      std::unique_ptr<QueueRunner> queue_runner;
      TF_RETURN_IF_ERROR(QueueRunner::New(queue_runner_defs_[i],
                                          coordinator_.get(), &queue_runner));
      TF_RETURN_IF_ERROR(queue_runner->Start(session_.get()));
      TF_RETURN_IF_ERROR(coordinator_->RegisterRunner(std::move(queue_runner)));
      status = coordinator_->GetStatus();
    }

    if (status.ok()) {
      last_graph_ = &graph_def;
    } else {
      return status;
    }

    // Warmup TensorFlow if needed
    for (int i = 0;
         i < options_.config.graph_options().build_cost_model_after(); ++i) {
      status = RunWithTimeout(feed, fetch, nullptr);
      if (!status.ok()) {
        return status;
      }
    }
  }

  return RunWithTimeout(feed, fetch, metadata);
}

Status SingleMachine::RunWithTimeout(
    const std::vector<std::pair<string, Tensor>>& feed,
    const std::vector<string>& fetch, RunMetadata* run_metadata) {
  mutex_lock l(mu_);
  // We shouldn't be running or closing the session at this point.
  CHECK(!running_);
  CHECK(!closing_);

  running_ = true;
  metadata_ = RunMetadata();

  thread_pool_->Schedule([this, feed, fetch] {
    Status status =
        session_->Run(run_options_, feed, {}, fetch, nullptr, &this->metadata_);
    mutex_lock l(mu_);
    status_ = status;
    running_ = false;
    done_running_.notify_all();
  });

  while (running_) {
    std::cv_status timeout =
        done_running_.wait_for(l, std::chrono::milliseconds(timeout_s_ * 1000));
    if (timeout != std::cv_status::no_timeout) {
      last_graph_ = nullptr;
      return Status(error::DEADLINE_EXCEEDED,
                    strings::StrCat("Failed to run the graph after ",
                                    timeout_s_, " seconds, aborting"));
    }
  }
  if (run_metadata && status_.ok()) {
    *run_metadata = metadata_;
  }
  return status_;
}

Status SingleMachine::CloseSession(bool use_timeout) {
  if (!session_) {
    return Status::OK();
  }

  mutex_lock l(close_mu_);

  if (!closing_) {
    closing_ = true;

    thread_pool_->Schedule([this] {
      if (this->coordinator_) {
        this->coordinator_->RequestStop().IgnoreError();
        // Wait for all the runners to have closed their queues.
        while (!this->coordinator_->AllRunnersStopped()) {
          sleep(1);
        }
        // Now we can close the session. This should cancel any pending I/O
        // operation.
        this->session_->Close().IgnoreError();
        // Last but not least, we can delete the coordinator.
        this->coordinator_.reset();
      } else {
        this->session_->Close().IgnoreError();
      }

      // Wait for any previous run to finish.
      mutex_lock l(mu_);
      while (running_) {
        done_running_.wait(l);
      }

      mutex_lock l2(close_mu_);
      closing_ = false;
      done_closing_.notify_all();
    });
  }

  while (closing_) {
    if (!use_timeout) {
      done_closing_.wait(l);
    } else {
      std::cv_status timeout = done_closing_.wait_for(
          l, std::chrono::milliseconds(timeout_s_ * 1000));
      if (timeout != std::cv_status::no_timeout) {
        // Let the caller know that we can't shutdown the session, and therefore
        // can't process any further.
        return Status(
            error::UNAVAILABLE,
            strings::StrCat("Failed to close the previous session after ",
                            timeout_s_, " seconds, aborting"));
      }
    }
  }

  return Status::OK();
}

Status SingleMachine::ResetSession() {
  if (session_) {
    LOG(INFO) << "Cleaning up previous session";

    // Make sure the session is properly closed
    Status status = CloseSession(true /*use_timeout*/);
    if (!status.ok()) {
      return status;
    }

    // Flush all the pending closures (if any).
    thread_pool_.reset(new thread::ThreadPool(
        Env::Default(), SanitizeThreadSuffix("single_machine"), 2));

    // We need to Reset the session to ensure that all the variables are
    // deleted. But first we need to delete the session since Reset()
    // deletes some of the containers referenced by the session.
    session_.reset();
    status = Reset(options_, {});
    if (!status.ok()) {
      return status;
    }
  }

  LOG(INFO) << "Starting new session";

  session_.reset(NewSession(options_));
  CHECK(session_ != nullptr);

  coordinator_.reset(new Coordinator());

  return Status::OK();
}

}  // namespace grappler
}  // namespace tensorflow
