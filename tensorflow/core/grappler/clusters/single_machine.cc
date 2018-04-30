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

#include <atomic>
#include <memory>

#include "tensorflow/cc/training/queue_runner.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/grappler/clusters/utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace grappler {

static std::atomic<bool> already_provisioned(false);

SingleMachine::SingleMachine(int timeout_s, int num_cpu_cores, int num_gpus)
    : Cluster(timeout_s), expected_init_time_s_(0), closing_(false) {
  VLOG(1) << "Number of CPU cores: " << num_cpu_cores
          << " Number of GPUs: " << num_gpus;
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
  // This is really ugly: to avoid leaking variables, we need to reset the tf
  // session every time we're done processing a grappler item. However,
  // variables are global, and therefore we can't have more than 1 session alive
  // at a time. This check detects when more that one cluster is provisioned.
  if (already_provisioned) {
    return errors::Unavailable(
        "Can't provision more than one single cluster at a time");
  }

  TF_RETURN_IF_ERROR(ResetSession());

  std::vector<DeviceAttributes> devices;
  TF_RETURN_IF_ERROR(session_->ListDevices(&devices));
  for (const auto& dev : devices) {
    DeviceProperties attr;
    if (dev.device_type() == "CPU") {
      attr = GetLocalCPUInfo();
    } else if (dev.device_type() == "GPU") {
      DeviceNameUtils::ParsedName parsed;
      if (!DeviceNameUtils::ParseFullName(dev.name(), &parsed)) {
        return errors::InvalidArgument(
            strings::StrCat("Not able to parse GPU device name: ", dev.name()));
      }
      TfGpuId tf_gpu_id(parsed.id);
      CudaGpuId cuda_gpu_id;
      Status s = GpuIdManager::TfToCudaGpuId(tf_gpu_id, &cuda_gpu_id);
      if (!s.ok()) {
        return errors::Unavailable("Unknown TF GPU device with id ",
                                   tf_gpu_id.value(), ": ", s.ToString());
      }
      attr = GetLocalGPUInfo(cuda_gpu_id);
    } else if (dev.device_type().find("XLA") == string::npos) {
      // Filter out the fake XLA devices to avoid double counting the actual
      // hardware resources that are available.
      attr.set_type(dev.device_type());
    }
    // Overwrite the memory size since users might have requested to use only a
    // fraction of the available device memory.
    attr.set_memory_size(dev.memory_limit());
    devices_[dev.name()] = attr;
  }
  already_provisioned = true;

  // Clear highmark stats of all local allocators.
  if (cpu_allocator_stats_enabled_) {
    TF_RETURN_IF_ERROR(ClearAllocatorStats());
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

Status SingleMachine::Shutdown() {
  TF_RETURN_IF_ERROR(ShutdownSession());

  mutex_lock l(this->last_graph_mu_);
  last_graph_ = nullptr;
  already_provisioned = false;

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
        // Also clear the timeline to save memory
        init_metadata_.clear_step_stats();
      }
      // We can have at most one hardware trace. Use it for the main graph, and
      // downgrade tracing of the queue runners to a software trace.
      RunOptions queue_options = run_options_;
      if (queue_options.trace_level() >= RunOptions::HARDWARE_TRACE) {
        queue_options.set_trace_level(RunOptions::SOFTWARE_TRACE);
      }
      for (size_t i = 0; i < queue_runner_defs_.size(); ++i) {
        std::unique_ptr<QueueRunner> queue_runner;
        TF_RETURN_IF_ERROR(QueueRunner::New(queue_runner_defs_[i],
                                            coordinator_.get(), &queue_runner));

        TF_RETURN_IF_ERROR(queue_runner->StartAndCollectCostGraph(
            session_.get(), queue_options));
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

  if (metadata) {
    TF_RETURN_IF_ERROR(RunWithTimeout(feed, fetch, metadata));
    // Merge the costs of the initialization and the queue runners.
    CostGraphDef queue_costs;
    TF_RETURN_IF_ERROR(coordinator_->ExportCostGraph(&queue_costs));
    MergeCosts(metadata->mutable_cost_graph(), init_metadata_.cost_graph(),
               queue_costs);
  } else {
    return RunWithTimeout(feed, fetch, nullptr);
  }
  return Status::OK();
}

Status SingleMachine::EnablePeakMemoryStats(bool enable) {
  EnableCPUAllocatorStats(enable);
  cpu_allocator_stats_enabled_ = enable;
  // No need to enable GPU allocator stats since its stats are always collected.
  return Status::OK();
}

Status SingleMachine::GetPeakMemoryUsage(
    std::unordered_map<string, uint64>* device_peak_memory) const {
  // Cpu_allocator->TracksAllocationSizes() returns true doesn't always mean the
  // the AllocatorStats would be collected.
  if (!cpu_allocator_stats_enabled_) {
    return Status(error::INVALID_ARGUMENT,
                  "Tracking allocation for CPU is not enabled.");
  }

  const DeviceMgr* device_mgr;
  TF_RETURN_IF_ERROR(session_->LocalDeviceManager(&device_mgr));
  std::vector<Device*> devices = device_mgr->ListDevices();

  device_peak_memory->clear();
  for (Device* device : devices) {
    AllocatorStats stats;
    auto* allocator = device->GetAllocator(AllocatorAttributes());
    if (!allocator->TracksAllocationSizes()) {
      return Status(error::INVALID_ARGUMENT,
                    "Tracking allocation is not enabled.");
    }
    allocator->GetStats(&stats);
    (*device_peak_memory)[device->name()] = stats.max_bytes_in_use;
  }

  return Status::OK();
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
      [this, status, local_metadata, feed, fetch]() {
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
  if (!session_ || !thread_pool_) {
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

Status SingleMachine::ShutdownSession() {
  TF_RETURN_IF_ERROR(CloseSession(true /*use_timeout*/));

  // Delete the threadpool: this ensures that all the pending closures complete
  // before we return. Note that if TF deadlocked on us, the closures will
  // never complete, and the call to thread_pool_.reset() will never return:
  // therefore we need to delete the threadpool with the background thread.
  // That thread itself will also never complete, so the user should
  // abort the process to avoid leaking too many resources.
  auto n = std::make_shared<Notification>();
  Env::Default()->SchedClosure([this, n]() {
    thread_pool_.reset();
    n->Notify();
  });
  int64 timeout_us = 1000000ll * timeout_s_;
  const bool notified = WaitForNotificationWithTimeout(n.get(), timeout_us);
  if (!notified) {
    // Let the caller know that we can't shutdown the session properly since
    // there are calls to Session::Run() still running.
    return errors::Unavailable("The session is still running graphs after ",
                               timeout_s_, " seconds");
  }

  return Status::OK();
}

Status SingleMachine::ResetSession() {
  if (session_) {
    LOG(INFO) << "Cleaning up previous session";

    // Make sure the session is properly closed
    TF_RETURN_IF_ERROR(ShutdownSession());

    // Destroying the object deletes all its variables as well. This is only
    // true for DirectSession.
    session_.reset();
  }

  LOG(INFO) << "Starting new session";

  // Create a new threadpool
  thread_pool_.reset(new thread::ThreadPool(
      Env::Default(), SanitizeThreadSuffix("single_machine"), 2));

  session_.reset(NewSession(options_));
  if (!session_) {
    return errors::Unknown("Failed to create session");
  }
  coordinator_.reset(new Coordinator());

  return Status::OK();
}

void SingleMachine::MergeCosts(CostGraphDef* graph_costs,
                               const CostGraphDef& init_costs,
                               const CostGraphDef& queue_costs) {
  graph_costs->mutable_node()->Reserve(graph_costs->node_size() +
                                       init_costs.node_size() +
                                       queue_costs.node_size());
  std::unordered_set<string> nodes_seen;
  int queue_costs_id_offset = graph_costs->node_size();
  for (const auto& node : graph_costs->node()) {
    nodes_seen.insert(node.name());
    if (node.id() >= queue_costs_id_offset) {
      queue_costs_id_offset = node.id() + 1;
    }
  }

  int init_costs_id_offset = queue_costs_id_offset + queue_costs.node_size();
  // The costs obtained by running the main graph could be more stable than
  // the one we get from the queue runners since the queue runners run
  // asynchronously.
  for (const auto& node : queue_costs.node()) {
    if (nodes_seen.find(node.name()) != nodes_seen.end()) {
      continue;
    }

    auto* new_node = graph_costs->add_node();
    new_node->MergeFrom(node);

    new_node->set_id(node.id() + queue_costs_id_offset);
    if (new_node->id() >= init_costs_id_offset) {
      init_costs_id_offset = new_node->id() + 1;
    }

    for (auto& input_info : *new_node->mutable_input_info()) {
      input_info.set_preceding_node(input_info.preceding_node() +
                                    queue_costs_id_offset);
    }
    for (auto& control_input : *new_node->mutable_control_input()) {
      control_input += queue_costs_id_offset;
    }
  }

  // Don't overwrite the costs with that generated during initialization since
  // these are possibly outdated.
  for (const auto& node : init_costs.node()) {
    if (nodes_seen.find(node.name()) != nodes_seen.end()) {
      continue;
    }

    auto* new_node = graph_costs->add_node();
    new_node->MergeFrom(node);

    new_node->set_id(node.id() + init_costs_id_offset);
    for (auto& input_info : *new_node->mutable_input_info()) {
      input_info.set_preceding_node(input_info.preceding_node() +
                                    init_costs_id_offset);
    }
    for (auto& control_input : *new_node->mutable_control_input()) {
      control_input += init_costs_id_offset;
    }
  }
}

Status SingleMachine::ClearAllocatorStats() const {
  // Cpu_allocator->TracksAllocationSizes() returns true doesn't always mean the
  // the AllocatorStats would be collected.
  if (!cpu_allocator_stats_enabled_) {
    return Status(error::INVALID_ARGUMENT,
                  "Tracking allocation for CPU is not enabled.");
  }

  const DeviceMgr* device_mgr;
  TF_RETURN_IF_ERROR(session_->LocalDeviceManager(&device_mgr));
  std::vector<Device*> devices = device_mgr->ListDevices();

  for (Device* device : devices) {
    AllocatorStats stats;
    auto* allocator = device->GetAllocator(AllocatorAttributes());
    if (!allocator->TracksAllocationSizes()) {
      return Status(error::INVALID_ARGUMENT,
                    "Tracking allocation is not enabled.");
    }
    allocator->ClearStats();
  }
  return Status::OK();
}

}  // namespace grappler
}  // namespace tensorflow
