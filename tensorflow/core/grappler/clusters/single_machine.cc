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

#include <memory>

#include "tensorflow/cc/training/queue_runner.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace grappler {

SingleMachine::SingleMachine(int timeout_s, int num_cpu_cores, int num_gpus)
    : Cluster(timeout_s),
      num_gpus_(num_gpus),
      expected_init_time_s_(0),
      closing_(false) {
  thread_pool_.reset(new thread::ThreadPool(
      Env::Default(), SanitizeThreadSuffix("single_machine"), 2));

  (*options_.config.mutable_device_count())["CPU"] = 1;
  if (num_gpus > 0) {
    (*options_.config.mutable_device_count())["GPU"] = num_gpus;
  }
  CHECK_GE(num_cpu_cores, 1);
  options_.config.set_intra_op_parallelism_threads(num_cpu_cores);
  // Create a session specific thread pool to ensure the threads are reset when
  // the session is reset.
  options_.config.add_session_inter_op_thread_pool()->set_num_threads(
      num_cpu_cores);
  if (timeout_s > 0) {
    options_.config.set_operation_timeout_in_ms(timeout_s * 1000);
  }
}

SingleMachine::~SingleMachine() {
  CloseSession(false /*use_timeout*/).IgnoreError();

  // Reset the thread-pool so that there are no outstanding Session::Run(...)s
  // when we delete the session.
  thread_pool_.reset();
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
  mutex_lock l(this->last_graph_mu_);
  if (last_graph_ != &item.graph || last_graph_id_ != item.id) {
    init_ops_ = item.init_ops;
    expected_init_time_s_ = item.expected_init_time;
    last_graph_ = nullptr;
    queue_runner_defs_ = item.queue_runners;
    last_graph_id_ = item.id;
  }
  return Status::OK();
}

Status SingleMachine::Run(const GraphDef& graph_def,
                          const std::vector<std::pair<string, Tensor>>& feed,
                          const std::vector<string>& fetch,
                          RunMetadata* metadata) {
  {
    mutex_lock l(this->last_graph_mu_);
    if (last_graph_ != &graph_def) {
      TF_RETURN_IF_ERROR(ResetSession());
      TF_RETURN_IF_ERROR(session_->Create(graph_def));
      if (!init_ops_.empty()) {
        init_metadata_ = RunMetadata();
        int64 timeout_s = timeout_s_ + expected_init_time_s_;
        TF_RETURN_IF_ERROR(
            RunWithTimeout({}, init_ops_, &init_metadata_, timeout_s));
        // The compute cost for init ops is likely to be pessimistic since init
        // ops are run only once before warmup. Therefore we only keep their
        // memory costs.
        for (auto node : *init_metadata_.mutable_cost_graph()->mutable_node()) {
          node.clear_compute_cost();
        }
      }
      for (int i = 0; i < queue_runner_defs_.size(); ++i) {
        std::unique_ptr<QueueRunner> queue_runner;
        TF_RETURN_IF_ERROR(QueueRunner::New(queue_runner_defs_[i],
                                            coordinator_.get(), &queue_runner));
        TF_RETURN_IF_ERROR(queue_runner->StartAndCollectCostGraph(
            session_.get(), &run_options_));
        TF_RETURN_IF_ERROR(
            coordinator_->RegisterRunner(std::move(queue_runner)));
        TF_RETURN_IF_ERROR(coordinator_->GetStatus());
      }

      // Warmup TensorFlow if needed
      for (int i = 0;
           i < options_.config.graph_options().build_cost_model_after(); ++i) {
        TF_RETURN_IF_ERROR(RunWithTimeout(feed, fetch, nullptr));
      }

      last_graph_ = &graph_def;
    }
  }

  TF_RETURN_IF_ERROR(RunWithTimeout(feed, fetch, metadata));

  if (metadata) {
    // Add the costs of initialization and the queue runners.
    metadata->MergeFrom(init_metadata_);
    return coordinator_->ExportCostGraph(metadata->mutable_cost_graph());
  } else {
    return Status::OK();
  }
}

Status SingleMachine::RunWithTimeout(
    const std::vector<std::pair<string, Tensor>>& feed,
    const std::vector<string>& fetch, RunMetadata* run_metadata) {
  return RunWithTimeout(feed, fetch, run_metadata, timeout_s_);
}

Status SingleMachine::RunWithTimeout(
    const std::vector<std::pair<string, Tensor>>& feed,
    const std::vector<string>& fetch, RunMetadata* run_metadata,
    int64 timeout_s) {
  // We shouldn't be running or closing the session at this point.
  {
    mutex_lock l(close_mu_);
    CHECK(!closing_);
  }
  auto status = std::make_shared<Status>();
  auto local_metadata = std::make_shared<RunMetadata>();
  const bool executed_in_time = ExecuteWithTimeout(
      [this, status, local_metadata, &feed, &fetch]() {
        *status = session_->Run(run_options_, feed, {}, fetch, nullptr,
                                local_metadata.get());
      },
      timeout_s * 1000, thread_pool_.get());
  if (!executed_in_time) {
    return errors::DeadlineExceeded("Failed to run the graph after ", timeout_s,
                                    " seconds, aborting");
  } else if (run_metadata && status->ok()) {
    *run_metadata = *local_metadata;
  }
  return *status;
}

Status SingleMachine::CloseSession(bool use_timeout) {
  if (!session_) {
    return Status::OK();
  }

  {
    mutex_lock l(close_mu_);

    if (!closing_) {
      closing_ = true;
    }
  }

  const bool executed_in_time = ExecuteWithTimeout(
      [&]() {
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

        mutex_lock l2(close_mu_);
        closing_ = false;
      },
      use_timeout ? timeout_s_ * 1000 : -1, thread_pool_.get());

  if (!executed_in_time) {
    // Let the caller know that we can't shutdown the session, and therefore
    // can't process any further.
    return errors::Unavailable("Failed to close the previous session after ",
                               timeout_s_, " seconds, aborting");
  }

  return Status::OK();
}

Status SingleMachine::ResetSession() {
  if (session_) {
    LOG(INFO) << "Cleaning up previous session";

    // Make sure the session is properly closed
    TF_RETURN_IF_ERROR(CloseSession(true /*use_timeout*/));

    // Flush all the pending closures (if any).
    thread_pool_.reset(new thread::ThreadPool(
        Env::Default(), SanitizeThreadSuffix("single_machine"), 2));

    // We need to Reset the session to ensure that all the variables are
    // deleted. But first we need to delete the session since Reset()
    // deletes some of the containers referenced by the session.
    session_.reset();
    TF_RETURN_IF_ERROR(Reset(options_, {}));
  }

  LOG(INFO) << "Starting new session";

  session_.reset(NewSession(options_));
  CHECK(session_ != nullptr);

  coordinator_.reset(new Coordinator());

  return Status::OK();
}

}  // namespace grappler
}  // namespace tensorflow
