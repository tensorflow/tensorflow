/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/graph_mgr.h"

#include <vector>

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

GraphMgr::GraphMgr(const WorkerEnv* worker_env)
    : worker_env_(worker_env), table_(5) {}

GraphMgr::~GraphMgr() {
  for (auto p : table_) p.second->Unref();
}

GraphMgr::Item::~Item() {
  for (const auto& unit : this->units) {
    CHECK_NOTNULL(unit.device);
    delete unit.root;
    delete unit.lib;
    unit.device->op_segment()->RemoveHold(this->session);
  }
  delete this->lib_def;
}

// NOTE: node->device_name() is not set by GraphConstructor.  We
// expects that NodeDef in GraphDef given to workers fully specifies
// device names.
static string SplitByDevice(const Node* node) {
  return node->assigned_device_name();
}

// Validates "gdef" device specifications.
static Status ValidateGraphDefForDevices(const GraphDef& gdef) {
  DeviceNameUtils::ParsedName parsed;
  for (const auto& ndef : gdef.node()) {
    if (!DeviceNameUtils::ParseFullName(ndef.device(), &parsed)) {
      return errors::InvalidArgument("Missing device name in: ",
                                     SummarizeNodeDef(ndef));
    }
  }
  return Status::OK();
}

// Creates executors given a graph definition "gdef" of a "session".
// If a node in "gdef" is shared by other graphs in "session", the
// same op kernel is reused. E.g., typically a params node is shared
// by multiple graphs in a session.
//
// If "gdef" is assigned to multiple devices, extra nodes (e.g.,
// send/recv nodes) maybe added. The extra nodes' name are generated
// by calling "new_name(old_name)".
//
// "executors" are filled with one executor per device if success and
// the caller takes the ownership of returned executors.
Status GraphMgr::InitItem(const string& session, const GraphDef& gdef,
                          const GraphOptions& graph_options, Item* item) {
  item->session = session;
  item->lib_def = new FunctionLibraryDefinition(gdef.library());

  TF_RETURN_IF_ERROR(ValidateGraphDefForDevices(gdef));

  if (gdef.versions().producer() >= 5) {
    // Validate the graph: we assume that merging two valid graphs
    // should maintain graph validity.
    TF_RETURN_IF_ERROR(graph::ValidateGraphDef(gdef, *item->lib_def));
  }

  // Constructs the graph out of "gdef".
  Graph graph(item->lib_def);
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = true;
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, gdef, &graph));

  // Splits "graph" into multiple subgraphs by device names.
  std::unordered_map<string, GraphDef> partitions;
  PartitionOptions popts;
  popts.node_to_loc = SplitByDevice;
  popts.new_name = [this](const string& prefix) {
    mutex_lock l(mu_);
    return strings::StrCat(prefix, "_G", next_id_++);
  };
  popts.get_incarnation = [this](const string& name) {
    Device* device = nullptr;
    Status s = worker_env_->device_mgr->LookupDevice(name, &device);
    if (s.ok()) {
      return device->attributes().incarnation();
    } else {
      return PartitionOptions::kIllegalIncarnation;
    }
  };
  popts.control_flow_added = true;
  popts.scheduling_for_recvs = graph_options.enable_recv_scheduling();
  TF_RETURN_IF_ERROR(Partition(popts, &graph, &partitions));
  if (popts.scheduling_for_recvs) {
    TF_RETURN_IF_ERROR(AddControlEdges(popts, &partitions));
  }

  thread::ThreadPool* pool = worker_env_->compute_pool;
  auto runner = [pool](std::function<void()> fn) { pool->Schedule(fn); };

  LocalExecutorParams params;

  Status s;
  item->units.reserve(partitions.size());
  const auto& optimizer_opts = graph_options.optimizer_options();
  GraphOptimizer optimizer(optimizer_opts);
  for (auto&& p : partitions) {
    const string& device_name = p.first;
    GraphDef* def = &p.second;
    item->units.resize(item->units.size() + 1);
    ExecutionUnit* unit = &(item->units.back());

    // Find the device.
    s = worker_env_->device_mgr->LookupDevice(device_name, &unit->device);
    if (!s.ok()) break;

    // Construct the subgraph.
    Graph* subgraph = new Graph(item->lib_def);
    // Give the device an opportunity to rewrite its subgraph.
    unit->device->MaybeRewriteGraph(gdef.library(), def);
    s = ConvertGraphDefToGraph(opts, *def, subgraph);
    if (!s.ok()) {
      delete subgraph;
      break;
    }
    // Top-level nodes in the graph uses the op segment to cache
    // kernels. Therefore, as long as the executor is alive, we need
    // to ensure the kernels cached for the session are alive.
    auto opseg = unit->device->op_segment();
    opseg->AddHold(session);

    // Function library runtime.
    unit->lib =
        NewFunctionLibraryRuntime(worker_env_->device_mgr, unit->device, runner,
                                  def->versions().producer(), item->lib_def,
                                  graph_options.optimizer_options());

    // Construct the root executor for the subgraph.
    params.device = unit->device;
    auto lib = unit->lib;
    params.function_library = lib;
    params.create_kernel = [session, lib, opseg](const NodeDef& ndef,
                                                 OpKernel** kernel) {
      // Caches the kernel only if the node is stateful.
      if (!lib->IsStateful(ndef.op())) {
        return lib->CreateKernel(ndef, kernel);
      }
      auto create_fn = [lib, &ndef](OpKernel** kernel) {
        return lib->CreateKernel(ndef, kernel);
      };
      // Kernels created for subgraph nodes need to be cached.  On
      // cache miss, create_fn() is invoked to create a kernel based
      // on the function library here + global op registry.
      return opseg->FindOrCreate(session, ndef.name(), kernel, create_fn);
    };
    params.delete_kernel = [lib](OpKernel* kernel) {
      // If the node is stateful, opseg owns it. Otherwise, delete it.
      if (kernel && !lib->IsStateful(kernel->type_string())) {
        delete kernel;
      }
    };

    optimizer.Optimize(lib, params.device, &subgraph);
    s = EnsureMemoryTypes(DeviceType(unit->device->device_type()),
                          unit->device->name(), subgraph);
    if (!s.ok()) {
      delete subgraph;
      break;
    }
    s = NewLocalExecutor(params, subgraph, &unit->root);
    if (!s.ok()) {
      break;
    }
  }
  return s;
}

Status GraphMgr::Register(const string& session, const GraphDef& gdef,
                          const GraphOptions& graph_options, string* handle) {
  Item* item = new Item;
  Status s = InitItem(session, gdef, graph_options, item);
  if (!s.ok()) {
    item->Unref();
    return s;
  }

  // Inserts one item into table_.
  {
    mutex_lock l(mu_);
    *handle = strings::Printf("%016llx", ++next_id_);
    item->handle = *handle;
    CHECK(table_.insert({*handle, item}).second);
  }
  return Status::OK();
}

Status GraphMgr::Deregister(const string& handle) {
  Item* item = nullptr;
  // Removes one item from table_.
  {
    mutex_lock l(mu_);
    auto iter = table_.find(handle);
    if (iter == table_.end()) {
      return errors::Aborted("Graph handle is not found: ", handle,
                             ". Possibly, this worker just restarted.");
    }
    item = iter->second;
    table_.erase(iter);
  }
  item->Unref();
  return Status::OK();
}

Status GraphMgr::DeregisterAll() {
  std::vector<Item*> items;
  // Removes all items from table_.
  {
    mutex_lock l(mu_);
    for (const auto& entry : table_) {
      items.push_back(entry.second);
    }
    table_.clear();
  }
  for (auto item : items) {
    item->Unref();
  }
  return Status::OK();
}

Status GraphMgr::Execute(const string& handle, const int64 step_id,
                         const ExecutorOpts& opts,
                         StepStatsCollector* collector,
                         CancellationManager* cancellation_manager,
                         const NamedTensors& in, NamedTensors* out) {
  Notification n;
  Status status;
  ExecuteAsync(handle, step_id, opts, collector, cancellation_manager, in, out,
               [&n, &status](const Status& s) {
                 status = s;
                 n.Notify();
               });
  n.WaitForNotification();
  return status;
}

void GraphMgr::ExecuteAsync(const string& handle, const int64 step_id,
                            const ExecutorOpts& opts,
                            StepStatsCollector* collector,
                            CancellationManager* cancellation_manager,
                            const NamedTensors& in, NamedTensors* out,
                            StatusCallback done) {
  // Lookup an item. Holds one ref while executing.
  Item* item = nullptr;
  {
    mutex_lock l(mu_);
    auto iter = table_.find(handle);
    if (iter != table_.end()) {
      item = iter->second;
      item->Ref();
    }
  }

  if (item == nullptr) {
    done(errors::Aborted("Graph handle is not found: ", handle));
    return;
  }

  const int num_units = item->units.size();
  CHECK_GE(num_units, 1);

  Rendezvous* rendezvous = worker_env_->rendezvous_mgr->Find(step_id);

  // Sends values specified by the caller.
  for (const auto& p : in) {
    const string& key = p.first;
    const Tensor& val = p.second;
    const Status s = rendezvous->Send(key, Rendezvous::Args(), val, false);
    if (!s.ok()) {
      done(s);
      item->Unref();
      rendezvous->Unref();
      return;
    }
  }

  // Starts parallel Executors.
  //
  // NOTE: Transfer one ref of rendezvous and one ref of item to
  // RunAllDone.
  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_units, rendezvous, std::bind(&ME::RunAllDone, this, item, rendezvous,
                                       out, done, std::placeholders::_1));
  Executor::Args args;
  {
    mutex_lock l(mu_);
    args.step_id = ++next_id_;
  }
  args.rendezvous = rendezvous;
  args.cancellation_manager = cancellation_manager;
  args.stats_collector = collector;
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(args.step_id, handle);
  }
  thread::ThreadPool* pool = worker_env_->compute_pool;
  args.runner = [pool](std::function<void()> fn) { pool->Schedule(fn); };
  for (const auto& unit : item->units) {
    unit.root->RunAsync(args, barrier->Get());
  }
}

void GraphMgr::RunAllDone(Item* item, Rendezvous* rendezvous, NamedTensors* out,
                          StatusCallback done, Status s) {
  if (s.ok()) {
    // Receives values requested by the caller.
    for (auto& p : *out) {
      const string& key = p.first;
      Tensor* val = &p.second;
      bool is_dead = false;
      s = rendezvous->Recv(key, Rendezvous::Args(), val, &is_dead);
      if (is_dead) {
        s = errors::InvalidArgument("The tensor returned for ", key,
                                    " was not valid.");
      }
      if (!s.ok()) break;
    }
  }
  done(s);
  rendezvous->Unref();
  item->Unref();
}

}  // end namespace tensorflow
