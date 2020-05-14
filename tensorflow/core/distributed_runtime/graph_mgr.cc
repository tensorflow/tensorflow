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

#include <chrono>  // NOLINT(build/c++11)
#include <vector>

#include "tensorflow/core/common_runtime/build_graph_options.h"
#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/debugger_state_interface.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/rendezvous_util.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

GraphMgr::GraphMgr(const WorkerEnv* worker_env, DeviceMgr* device_mgr)
    : worker_env_(worker_env), device_mgr_(device_mgr), table_(5) {
  // The default value of sync_on_finish will be flipped soon and this
  // environment variable will be removed as well.
  Status status =
      ReadBoolFromEnvVar("TF_SYNC_ON_FINISH", true, &sync_on_finish_);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
  }
}

GraphMgr::~GraphMgr() {
  for (const auto& p : table_) p.second->Unref();
}

GraphMgr::Item::~Item() {
  for (const auto& unit : this->units) {
    CHECK_NOTNULL(unit.device);
    if (!graph_mgr->skip_cost_models_) {
      graph_mgr->cost_model_manager_.RemoveCostModelForGraph(unit.graph.get());
    }
    delete unit.root;
    unit.device->op_segment()->RemoveHold(this->session);
  }
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
                                     FormatNodeDefForError(ndef));
    }
  }
  return Status::OK();
}

Status GraphMgr::DecorateAndPublishGraphForDebug(
    const DebugOptions& debug_options, Graph* graph, Device* device) {
  std::unique_ptr<DebugGraphDecoratorInterface> decorator;
  TF_RETURN_IF_ERROR(
      DebugGraphDecoratorRegistry::CreateDecorator(debug_options, &decorator));
  TF_RETURN_IF_ERROR(decorator->DecorateGraph(graph, device));
  TF_RETURN_IF_ERROR(decorator->PublishGraph(*graph, device->name()));
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
Status GraphMgr::InitItem(
    const string& handle, const GraphDef& gdef, WorkerSession* session,
    const GraphOptions& graph_options, const DebugOptions& debug_options,
    const ConfigProto& config_proto, int64 collective_graph_key,
    DistributedFunctionLibraryRuntime* cluster_flr, Item* item) {
  item->session = handle;
  item->collective_graph_key = collective_graph_key;
  item->lib_def.reset(
      new FunctionLibraryDefinition(OpRegistry::Global(), gdef.library()));

  TF_RETURN_IF_ERROR(ValidateGraphDefForDevices(gdef));

  // We don't explicitly Validate the graph def because ConvertGraphDefToGraph
  // does that below.

  item->proc_flr.reset(new ProcessFunctionLibraryRuntime(
      device_mgr_, worker_env_->env, /*config=*/&config_proto,
      gdef.versions().producer(), item->lib_def.get(),
      graph_options.optimizer_options(), worker_env_->compute_pool, cluster_flr,
      /*custom_kernel_creator=*/nullptr, /*session_metadata=*/nullptr,
      Rendezvous::Factory{
          [this, session](const int64 step_id, const DeviceMgr*,
                          Rendezvous** r) -> Status {
            auto* remote_r = this->worker_env_->rendezvous_mgr->Find(step_id);
            TF_RETURN_IF_ERROR(remote_r->Initialize(session));
            *r = remote_r;
            return Status::OK();
          },
          [this](const int64 step_id) {
            this->worker_env_->rendezvous_mgr->Cleanup(step_id);
            return Status::OK();
          }}));

  // Constructs the graph out of "gdef".
  Graph graph(OpRegistry::Global());
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = true;
  opts.validate_nodes = true;
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
    Status s = device_mgr_->LookupDevice(name, &device);
    if (s.ok()) {
      return device->attributes().incarnation();
    } else {
      return PartitionOptions::kIllegalIncarnation;
    }
  };
  popts.flib_def = &graph.flib_def();
  popts.control_flow_added = true;
  popts.scheduling_for_recvs = graph_options.enable_recv_scheduling();
  TF_RETURN_IF_ERROR(Partition(popts, &graph, &partitions));
  if (popts.scheduling_for_recvs) {
    TF_RETURN_IF_ERROR(AddControlEdges(popts, &partitions));
  }

  std::unordered_map<string, std::unique_ptr<Graph>> partition_graphs;
  for (auto& partition : partitions) {
    std::unique_ptr<Graph> device_graph(new Graph(OpRegistry::Global()));
    GraphConstructorOptions device_opts;
    // There are internal operations (e.g., send/recv) that we now allow.
    device_opts.allow_internal_ops = true;
    device_opts.expect_device_spec = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
        device_opts, std::move(partition.second), device_graph.get()));
    partition_graphs.emplace(partition.first, std::move(device_graph));
  }

  GraphOptimizationPassOptions optimization_options;
  optimization_options.flib_def = item->lib_def.get();
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
    Status s = device_mgr_->LookupDevice(device_name, &unit->device);
    if (!s.ok()) {
      // Remove the empty unit from the item as the item destructor wants all
      // units to have valid devices.
      item->units.pop_back();
      return s;
    }

    // Give the device an opportunity to rewrite its subgraph.
    TF_RETURN_IF_ERROR(unit->device->MaybeRewriteGraph(&subgraph));

    // Top-level nodes in the graph uses the op segment to cache
    // kernels. Therefore, as long as the executor is alive, we need
    // to ensure the kernels cached for the session are alive.
    auto opseg = unit->device->op_segment();
    opseg->AddHold(handle);

    // Function library runtime.
    FunctionLibraryRuntime* lib = item->proc_flr->GetFLR(unit->device->name());
    if (lib == nullptr) {
      return errors::InvalidArgument("Cannot find FLR for device: ",
                                     unit->device->name());
    }

    // Construct the root executor for the subgraph.
    params.device = unit->device;
    params.function_library = lib;
    params.create_kernel =
        [handle, lib, opseg](const std::shared_ptr<const NodeProperties>& props,
                             OpKernel** kernel) {
          // NOTE(mrry): We must not share function kernels (implemented
          // using `CallOp`) between subgraphs, because `CallOp::handle_`
          // is tied to a particular subgraph. Even if the function itself
          // is stateful, the `CallOp` that invokes it is not.
          if (!OpSegment::ShouldOwnKernel(lib, props->node_def.op())) {
            return lib->CreateKernel(props, kernel);
          }
          auto create_fn = [lib, &props](OpKernel** kernel) {
            return lib->CreateKernel(props, kernel);
          };
          // Kernels created for subgraph nodes need to be cached.  On
          // cache miss, create_fn() is invoked to create a kernel based
          // on the function library here + global op registry.
          return opseg->FindOrCreate(handle, props->node_def.name(), kernel,
                                     create_fn);
        };
    params.delete_kernel = [lib](OpKernel* kernel) {
      if (kernel && !OpSegment::ShouldOwnKernel(lib, kernel->type_string())) {
        delete kernel;
      }
    };

    optimizer.Optimize(lib, worker_env_->env, params.device, &subgraph,
                       /*shape_map=*/nullptr);

    // TensorFlow Debugger (tfdbg) inserts debug nodes in the graph.
    if (!debug_options.debug_tensor_watch_opts().empty()) {
      TF_RETURN_IF_ERROR(DecorateAndPublishGraphForDebug(
          debug_options, subgraph.get(), params.device));
    }

    TF_RETURN_IF_ERROR(
        EnsureMemoryTypes(DeviceType(unit->device->device_type()),
                          unit->device->name(), subgraph.get()));
    unit->graph = std::move(subgraph);
    unit->build_cost_model = graph_options.build_cost_model();
    if (unit->build_cost_model > 0) {
      skip_cost_models_ = false;
    }
    TF_RETURN_IF_ERROR(NewLocalExecutor(params, *unit->graph, &unit->root));
  }
  return Status::OK();
}

Status GraphMgr::Register(
    const string& handle, const GraphDef& gdef, WorkerSession* session,
    const GraphOptions& graph_options, const DebugOptions& debug_options,
    const ConfigProto& config_proto, int64 collective_graph_key,
    DistributedFunctionLibraryRuntime* cluster_flr, string* graph_handle) {
  Item* item = new Item;
  Status s = InitItem(handle, gdef, session, graph_options, debug_options,
                      config_proto, collective_graph_key, cluster_flr, item);
  if (!s.ok()) {
    item->Unref();
    return s;
  }

  // Inserts one item into table_.
  {
    mutex_lock l(mu_);
    *graph_handle =
        strings::Printf("%016llx", static_cast<long long>(++next_id_));
    item->handle = *graph_handle;
    CHECK(table_.insert({*graph_handle, item}).second);
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

Status GraphMgr::SendInputs(const int64 step_id, const NamedTensors& in) {
  Rendezvous* rendezvous = worker_env_->rendezvous_mgr->Find(step_id);
  std::vector<string> keys;
  std::vector<Tensor> tensors_to_send;
  keys.reserve(in.size());
  tensors_to_send.reserve(in.size());
  size_t input_size = 0;
  for (const auto& p : in) {
    keys.push_back(p.first);
    tensors_to_send.push_back(p.second);
    input_size += p.second.AllocatedBytes();
  }
  metrics::RecordGraphInputTensors(input_size);
  Status s =
      SendTensorsToRendezvous(rendezvous, nullptr, {}, keys, tensors_to_send);
  rendezvous->Unref();
  return s;
}

Status GraphMgr::RecvOutputs(const int64 step_id, NamedTensors* out) {
  Rendezvous* rendezvous = worker_env_->rendezvous_mgr->Find(step_id);
  Status s = RecvOutputsFromRendezvous(rendezvous, out, Rendezvous::Args());
  rendezvous->Unref();
  if (!s.ok()) {
    // Failing to fetch the outputs should not be possible, so rewrite the error
    // status to an INTERNAL error.
    s = errors::Internal("Failed to fetch outputs for step ", step_id,
                         ". (Original error message: ", s.ToString(), ")");
  }
  size_t output_size = 0;
  for (auto& p : *out) {
    output_size += p.second.AllocatedBytes();
  }
  metrics::RecordGraphOutputTensors(output_size);
  return s;
}

void GraphMgr::RecvOutputsAsync(const int64 step_id, NamedTensors* out,
                                StatusCallback done) {
  Rendezvous* rendezvous = worker_env_->rendezvous_mgr->Find(step_id);
  std::vector<string> keys;
  std::vector<Tensor>* received_keys = new std::vector<Tensor>;
  keys.reserve(out->size());
  received_keys->reserve(out->size());
  for (const auto& p : *out) {
    keys.push_back(p.first);
    received_keys->push_back(p.second);
  }
  RecvOutputsFromRendezvousAsync(
      rendezvous, nullptr, {}, keys, received_keys,
      [done, rendezvous, received_keys, out, keys](const Status s) {
        rendezvous->Unref();
        size_t output_size = 0;
        for (int i = 0; i < keys.size(); ++i) {
          (*out)[keys[i]] = (*received_keys)[i];
          output_size += (*out)[keys[i]].AllocatedBytes();
        }
        metrics::RecordGraphOutputTensors(output_size);
        delete received_keys;
        done(s);
      });
}

void GraphMgr::ExecuteAsync(const string& handle, const int64 step_id,
                            WorkerSession* session, const ExecutorOpts& opts,
                            StepStatsCollector* collector,
                            MutableRunGraphResponseWrapper* response,
                            CancellationManager* cancellation_manager,
                            const NamedTensors& in, StatusCallback done) {
  const uint64 start_time_usecs = Env::Default()->NowMicros();
  profiler::TraceMe activity(
      [step_id] { return absl::StrCat("RunGraph#id=", step_id, "#"); },
      profiler::TraceMeLevel::kInfo);
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

  CostGraphDef* cost_graph = nullptr;
  if (response != nullptr) {
    cost_graph = response->mutable_cost_graph();
    if (opts.record_partition_graphs()) {
      for (const ExecutionUnit& unit : item->units) {
        GraphDef graph_def;
        unit.graph->ToGraphDef(&graph_def);
        response->AddPartitionGraph(graph_def);
      }
    }
  }

  RemoteRendezvous* rendezvous = worker_env_->rendezvous_mgr->Find(step_id);
  Status s = rendezvous->Initialize(session);
  CollectiveExecutor::Handle* ce_handle =
      item->collective_graph_key != BuildGraphOptions::kNoCollectiveGraphKey
          ? new CollectiveExecutor::Handle(
                worker_env_->collective_executor_mgr->FindOrCreate(step_id),
                true)
          : nullptr;
  // Sends values specified by the caller.
  size_t input_size = 0;
  if (s.ok()) {
    std::vector<string> keys;
    std::vector<Tensor> tensors_to_send;
    keys.reserve(in.size());
    tensors_to_send.reserve(in.size());
    for (auto& p : in) {
      keys.push_back(p.first);
      tensors_to_send.push_back(p.second);
      input_size += p.second.AllocatedBytes();
    }
    s = SendTensorsToRendezvous(rendezvous, nullptr, {}, keys, tensors_to_send);
  }

  if (!s.ok()) {
    done(s);
    delete ce_handle;
    item->Unref();
    rendezvous->Unref();
    return;
  }

  StartParallelExecutors(
      handle, step_id, item, rendezvous, ce_handle, collector, cost_graph,
      cancellation_manager, session,
      [item, rendezvous, ce_handle, done, start_time_usecs, input_size,
       step_id](const Status& s) {
        profiler::TraceMe activity(
            [step_id] {
              return absl::StrCat("RunGraphDone#id=", step_id, "#");
            },
            profiler::TraceMeLevel::kInfo);
        done(s);
        metrics::RecordGraphInputTensors(input_size);
        metrics::UpdateGraphExecTime(Env::Default()->NowMicros() -
                                     start_time_usecs);
        rendezvous->Unref();
        item->Unref();
        delete ce_handle;
      });
}

void GraphMgr::StartParallelExecutors(
    const string& handle, int64 step_id, Item* item, Rendezvous* rendezvous,
    CollectiveExecutor::Handle* ce_handle, StepStatsCollector* collector,
    CostGraphDef* cost_graph, CancellationManager* cancellation_manager,
    WorkerSession* session, StatusCallback done) {
  const int num_units = item->units.size();
  CHECK_GE(num_units, 1);
  ScopedStepContainer* step_container = new ScopedStepContainer(
      step_id,
      [this](const string& name) { device_mgr_->ClearContainers({name}); });
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
  args.step_id = step_id;
  args.rendezvous = rendezvous;
  args.collective_executor = ce_handle ? ce_handle->get() : nullptr;
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
  auto default_runner = std::bind(&thread::ThreadPool::Schedule, pool, _1);
  for (const auto& unit : item->units) {
    // TODO(zhengxq): if the device picks its own threadpool, we need to assign
    //     less threads to the main compute pool by default.
    thread::ThreadPool* device_thread_pool =
        unit.device->tensorflow_device_thread_pool();
    if (!device_thread_pool) {
      args.runner = default_runner;
    } else {
      args.runner =
          std::bind(&thread::ThreadPool::Schedule, device_thread_pool, _1);
    }
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
        device_to_graph[unit.device->name()] = unit.graph.get();
      }
    }
    collector->BuildCostModel(&cost_model_manager_, device_to_graph);

    if (cost_graph != nullptr) {
      for (const auto& unit : item->units) {
        cost_model_manager_.AddToCostGraphDef(unit.graph.get(), cost_graph)
            .IgnoreError();
      }
    }
  }
}

}  // end namespace tensorflow
