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

#include "tensorflow/core/distributed_runtime/graph_mgr.h"

#include <vector>

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

GraphMgr::GraphMgr(const WorkerEnv* worker_env,
                   RendezvousMgrInterface* rendezvous_mgr)
    : worker_env_(worker_env), rendezvous_mgr_(rendezvous_mgr), table_(5) {
  CHECK(rendezvous_mgr) << "Rendezvous mgr was null";
  // The default value of sync_on_finish will be flipped soon and this
  // environment variable will be removed as well.
  Status status =
      ReadBoolFromEnvVar("TF_SYNC_ON_FINISH", true, &sync_on_finish_);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
  }
}

GraphMgr::~GraphMgr() {
  for (auto p : table_) p.second->Unref();
}

GraphMgr::Item::~Item() {
  for (const auto& unit : this->units) {
    CHECK_NOTNULL(unit.device);
    if (!graph_mgr->skip_cost_models_) {
      graph_mgr->cost_model_manager_.RemoveCostModelForGraph(unit.graph);
    }
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
  item->lib_def =
      new FunctionLibraryDefinition(OpRegistry::Global(), gdef.library());

  TF_RETURN_IF_ERROR(ValidateGraphDefForDevices(gdef));

  if (gdef.versions().producer() >= 5) {
    // Validate the graph: we assume that merging two valid graphs
    // should maintain graph validity.
    TF_RETURN_IF_ERROR(graph::ValidateGraphDef(gdef, *item->lib_def));
  }

  // Constructs the graph out of "gdef".
  Graph graph(OpRegistry::Global());
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
  popts.get_incarnation = [this](const string& name) -> int64 {
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

  std::unordered_map<string, std::unique_ptr<Graph>> partition_graphs;
  for (const auto& partition : partitions) {
    std::unique_ptr<Graph> device_graph(new Graph(OpRegistry::Global()));
    GraphConstructorOptions device_opts;
    // There are internal operations (e.g., send/recv) that we now allow.
    device_opts.allow_internal_ops = true;
    device_opts.expect_device_spec = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(device_opts, partition.second,
                                              device_graph.get()));
    partition_graphs.emplace(partition.first, std::move(device_graph));
  }

  GraphOptimizationPassOptions optimization_options;
  optimization_options.flib_def = item->lib_def;
  optimization_options.partition_graphs = &partition_graphs;
  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_PARTITIONING, optimization_options));

  LocalExecutorParams params;

  item->units.reserve(partitions.size());
  item->graph_mgr = this;
  const auto& optimizer_opts = graph_options.optimizer_options();
  GraphOptimizer optimizer(optimizer_opts);
  for (auto& p : partition_graphs) {
    const string& device_name = p.first;
    std::unique_ptr<Graph>& subgraph = p.second;
    item->units.resize(item->units.size() + 1);
    ExecutionUnit* unit = &(item->units.back());

    // Find the device.
    Status s =
        worker_env_->device_mgr->LookupDevice(device_name, &unit->device);
    if (!s.ok()) {
      // Remove the empty unit from the item as the item destructor wants all
      // units to have valid devices.
      item->units.pop_back();
      return s;
    }

    // Give the device an opportunity to rewrite its subgraph.
    TF_RETURN_IF_ERROR(
        unit->device->MaybeRewriteGraph(gdef.library(), &subgraph));

    // Top-level nodes in the graph uses the op segment to cache
    // kernels. Therefore, as long as the executor is alive, we need
    // to ensure the kernels cached for the session are alive.
    auto opseg = unit->device->op_segment();
    opseg->AddHold(session);

    // Function library runtime.
    unit->lib = NewFunctionLibraryRuntime(
        worker_env_->device_mgr, worker_env_->env, unit->device,
        subgraph->versions().producer(), item->lib_def,
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

    optimizer.Optimize(lib, worker_env_->env, params.device, &subgraph);
    TF_RETURN_IF_ERROR(
        EnsureMemoryTypes(DeviceType(unit->device->device_type()),
                          unit->device->name(), subgraph.get()));
    unit->graph = subgraph.get();
    unit->build_cost_model = graph_options.build_cost_model();
    if (unit->build_cost_model > 0) {
      skip_cost_models_ = false;
    }
    TF_RETURN_IF_ERROR(
        NewLocalExecutor(params, subgraph.release(), &unit->root));
  }
  return Status::OK();
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

Status GraphMgr::SendInputsToRendezvous(Rendezvous* rendezvous,
                                        const NamedTensors& in) {
  Rendezvous::ParsedKey parsed;
  for (const auto& p : in) {
    const string& key = p.first;
    const Tensor& val = p.second;

    Status s = Rendezvous::ParseKey(key, &parsed);
    if (s.ok()) {
      s = rendezvous->Send(parsed, Rendezvous::Args(), val, false);
    }
    if (!s.ok()) {
      return s;
    }
  }
  return Status::OK();
}

Status GraphMgr::RecvOutputsFromRendezvous(Rendezvous* rendezvous,
                                           NamedTensors* out) {
  // Receives values requested by the caller.
  Rendezvous::ParsedKey parsed;
  for (auto& p : *out) {
    const string& key = p.first;
    Tensor* val = &p.second;
    bool is_dead = false;
    Status s = Rendezvous::ParseKey(key, &parsed);
    if (s.ok()) {
      s = rendezvous->Recv(parsed, Rendezvous::Args(), val, &is_dead);
    }
    if (is_dead) {
      s = errors::InvalidArgument("The tensor returned for ", key,
                                  " was not valid.");
    }
    if (!s.ok()) return s;
  }
  return Status::OK();
}

void GraphMgr::RecvOutputsFromRendezvousAsync(Rendezvous* rendezvous,
                                              NamedTensors* out,
                                              const StatusCallback& done) {
  if (out->empty()) {
    done(Status::OK());
    return;
  }
  // We compute the args before calling RecvAsync because we need to ensure that
  // out isn't being iterated over after done is called, since done deletes out.
  std::vector<std::tuple<string, Tensor*, Rendezvous::ParsedKey>> args;
  for (auto& p : *out) {
    Rendezvous::ParsedKey parsed;
    Status s = Rendezvous::ParseKey(p.first, &parsed);
    if (!s.ok()) {
      done(s);
      return;
    }
    args.push_back(std::make_tuple(p.first, &p.second, parsed));
  }

  typedef struct {
    mutex mu;
    int done_counter;
    Status shared_status = Status::OK();
  } CallState;
  CallState* call_state = new CallState;
  call_state->done_counter = out->size();
  for (auto& p : args) {
    const string& key = std::get<0>(p);
    Tensor* val = std::get<1>(p);
    Rendezvous::ParsedKey parsed = std::get<2>(p);
    rendezvous->RecvAsync(
        parsed, Rendezvous::Args(),
        [val, done, key, call_state](const Status& s,
                                     const Rendezvous::Args& send_args,
                                     const Rendezvous::Args& recv_args,
                                     const Tensor& v, const bool is_dead) {
          Status status = s;
          if (status.ok()) {
            *val = v;
            if (is_dead) {
              status = errors::InvalidArgument("The tensor returned for ", key,
                                               " was not valid.");
            }
          }
          call_state->mu.lock();
          call_state->shared_status.Update(status);
          call_state->done_counter--;
          // If we are the last async call to return, call the done callback.
          if (call_state->done_counter == 0) {
            const Status& final_status = call_state->shared_status;
            call_state->mu.unlock();
            done(final_status);
            delete call_state;
            return;
          }
          call_state->mu.unlock();
        });
  }
}

Status GraphMgr::SendInputs(const int64 step_id, const NamedTensors& in) {
  Rendezvous* rendezvous = rendezvous_mgr_->Find(step_id);
  Status s = SendInputsToRendezvous(rendezvous, in);
  rendezvous->Unref();
  return s;
}

Status GraphMgr::RecvOutputs(const int64 step_id, NamedTensors* out) {
  Rendezvous* rendezvous = rendezvous_mgr_->Find(step_id);
  Status s = RecvOutputsFromRendezvous(rendezvous, out);
  rendezvous->Unref();
  return s;
}

void GraphMgr::RecvOutputsAsync(const int64 step_id, NamedTensors* out,
                                StatusCallback done) {
  Rendezvous* rendezvous = rendezvous_mgr_->Find(step_id);
  RecvOutputsFromRendezvousAsync(rendezvous, out,
                                 [done, rendezvous](const Status s) {
                                   rendezvous->Unref();
                                   done(s);
                                 });
}

void GraphMgr::ExecuteAsync(const string& handle, const int64 step_id,
                            const ExecutorOpts& opts,
                            StepStatsCollector* collector,
                            CostGraphDef* cost_graph,
                            CancellationManager* cancellation_manager,
                            const NamedTensors& in, StatusCallback done) {
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

  Rendezvous* rendezvous = rendezvous_mgr_->Find(step_id);

  // Sends values specified by the caller.
  Status s = SendInputsToRendezvous(rendezvous, in);
  if (!s.ok()) {
    done(s);
    item->Unref();
    rendezvous->Unref();
    return;
  }

  StartParallelExecutors(handle, step_id, item, rendezvous, collector,
                         cost_graph, cancellation_manager,
                         [this, item, rendezvous, done](const Status& s) {
                           done(s);
                           rendezvous->Unref();
                           item->Unref();
                         });
}

void GraphMgr::StartParallelExecutors(const string& handle, int64 step_id,
                                      Item* item, Rendezvous* rendezvous,
                                      StepStatsCollector* collector,
                                      CostGraphDef* cost_graph,
                                      CancellationManager* cancellation_manager,
                                      StatusCallback done) {
  const int num_units = item->units.size();
  CHECK_GE(num_units, 1);
  ScopedStepContainer* step_container =
      new ScopedStepContainer(step_id, [this](const string& name) {
        worker_env_->device_mgr->ClearContainers({name});
      });
  // NOTE: Transfer one ref of rendezvous and item.
  ExecutorBarrier* barrier =
      new ExecutorBarrier(num_units, rendezvous,
                          [this, item, collector, cost_graph, step_container,
                           done](const Status& s) {
                            BuildCostModel(item, collector, cost_graph);
                            done(s);
                            delete step_container;
                          });
  Executor::Args args;
  {
    mutex_lock l(mu_);
    args.step_id = ++next_id_;
  }
  args.rendezvous = rendezvous;
  args.cancellation_manager = cancellation_manager;
  args.stats_collector = collector;
  args.step_container = step_container;
  args.sync_on_finish = sync_on_finish_;
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(args.step_id, handle);
  }
  thread::ThreadPool* pool = worker_env_->compute_pool;
  using std::placeholders::_1;
  // Line below is equivalent to this code, but does one less indirect call:
  //  args.runner = [pool](std::function<void()> fn) { pool->Schedule(fn); };
  args.runner = std::bind(&thread::ThreadPool::Schedule, pool, _1);
  for (const auto& unit : item->units) {
    unit.root->RunAsync(args, barrier->Get());
  }
}

void GraphMgr::BuildCostModel(Item* item, StepStatsCollector* collector,
                              CostGraphDef* cost_graph) {
  if (collector && !skip_cost_models_) {
    // Build the cost model
    std::unordered_map<string, const Graph*> device_to_graph;
    for (const auto& unit : item->units) {
      if (unit.build_cost_model > 0) {
        device_to_graph[unit.device->name()] = unit.graph;
      }
    }
    collector->BuildCostModel(&cost_model_manager_, device_to_graph);

    if (cost_graph != nullptr) {
      for (const auto& unit : item->units) {
        cost_model_manager_.AddToCostGraphDef(unit.graph, cost_graph)
            .IgnoreError();
      }
    }
  }
}

}  // end namespace tensorflow
