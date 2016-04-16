/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/direct_session.h"

#include <atomic>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/gpu/gpu_tracer.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/common_runtime/simple_placer.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

namespace {

thread::ThreadPool* NewThreadPool(const SessionOptions& options) {
  int32 inter_op_parallelism_threads =
      options.config.inter_op_parallelism_threads();
  if (inter_op_parallelism_threads == 0) {
    // Default to using the number of cores available in the process.
    inter_op_parallelism_threads = port::NumSchedulableCPUs();
  }
  VLOG(1) << "Direct session inter op parallelism threads: "
          << inter_op_parallelism_threads;
  return new thread::ThreadPool(options.env, "Compute",
                                inter_op_parallelism_threads);
}

thread::ThreadPool* GlobalThreadPool(const SessionOptions& options) {
  static thread::ThreadPool* const thread_pool = NewThreadPool(options);
  return thread_pool;
}

// TODO(vrv): Figure out how to unify the many different functions
// that generate RendezvousKey, since many of them have to be
// consistent with each other.
string GetRendezvousKey(const string& tensor_name,
                        const DeviceAttributes& device_info,
                        const FrameAndIter& frame_iter) {
  return strings::StrCat(device_info.name(), ";",
                         strings::FpToString(device_info.incarnation()), ";",
                         device_info.name(), ";", tensor_name, ";",
                         frame_iter.frame_id, ":", frame_iter.iter_id);
}

}  // namespace

std::atomic_int_fast64_t DirectSession::step_id_counter_(1);

// NOTE: On Android with a single device, there is never
// a risk of an OpKernel blocking indefinitely:
//
// 1) No operations do I/O that depends on other simultaneous kernels,
//
// 2) Recv nodes always complete immediately: The inputs are sent into
//    the local rendezvous before we start the executor, so the
//    corresponding recvs will not block.
//
// Based on these assumptions, we can use the same thread pool for
// both "non-blocking" and "blocking" OpKernels on Android.
//
// This may change down the road when we add support for multiple
// devices that run concurrently, in which case we will need to
// revisit this decision.
void DirectSession::SchedClosure(std::function<void()> c) {
// TODO(sanjay): Get rid of __ANDROID__ path
#ifdef __ANDROID__
  // On Android, there is no implementation of ThreadPool that takes
  // std::function, only Closure, which we cannot easily convert.
  //
  // Instead, we just run the function in-line, which is currently
  // safe given the reasoning above.
  c();
#else
  thread_pool_->Schedule(c);
#endif  // __ANDROID__
}

DirectSession::DirectSession(const SessionOptions& options,
                             const DeviceMgr* device_mgr)
    : options_(options),
      device_mgr_(device_mgr),
      cancellation_manager_(new CancellationManager()),
      operation_timeout_in_ms_(options_.config.operation_timeout_in_ms()) {
  if (options_.config.use_per_session_threads()) {
    thread_pool_ = NewThreadPool(options_);
  } else {
    thread_pool_ = GlobalThreadPool(options);
  }
  // NOTE(mrry): We do not need to use a unique string for the session
  // handle, because DirectSession owns its devices. This may change
  // in future versions.
  session_handle_ = "direct";
  int devices_added = 0;
  if (options.config.log_device_placement()) {
    const string mapping_str = device_mgr_->DeviceMappingString();
    if (mapping_str.empty()) {
      printf("Device mapping: no known devices.\n");
    } else {
      printf("Device mapping:\n%s", mapping_str.c_str());
    }
    LOG(INFO) << "Device mapping:\n" << mapping_str;
  }
  for (auto d : device_mgr_->ListDevices()) {
    devices_.push_back(d);
    device_set_.AddDevice(d);
    d->op_segment()->AddHold(session_handle_);

    // The first device added is special: it is the 'client device' (a
    // CPU device) from which we feed and fetch Tensors.
    if (devices_added == 0) {
      device_set_.set_client_device(d);
    }
    ++devices_added;
  }
}

DirectSession::~DirectSession() {
  for (auto& it : partial_runs_) {
    delete it.second;
  }
  for (auto it : executors_) {
    delete it.second;
  }
  for (auto d : device_mgr_->ListDevices()) {
    d->op_segment()->RemoveHold(session_handle_);
  }
  delete cancellation_manager_;
  if (options_.config.use_per_session_threads()) {
    delete thread_pool_;
  }
  for (auto it : cost_models_) {
    delete it.second;
  }
}

Status DirectSession::Create(const GraphDef& graph) {
  mutex_lock l(graph_def_lock_);
  if (graph_created_) {
    return errors::AlreadyExists(
        "A Graph has already been created for this session.");
  }
  return ExtendLocked(graph);
}

Status DirectSession::Extend(const GraphDef& graph) {
  mutex_lock l(graph_def_lock_);
  return ExtendLocked(graph);
}

Status DirectSession::ExtendLocked(const GraphDef& graph) {
  // Merge versions
  if (graph_def_.has_versions()) {
    if (graph_def_.versions().producer() != graph.versions().producer()) {
      return errors::InvalidArgument(
          "Can't extend GraphDef at version ", graph_def_.versions().producer(),
          " with graph at version ", graph.versions().producer());
    }
    VersionDef* versions = graph_def_.mutable_versions();
    versions->set_min_consumer(
        std::max(versions->min_consumer(), graph.versions().min_consumer()));
    if (graph.versions().bad_consumers_size()) {
      // Add new bad_consumers that aren't already marked bad.
      //
      // Note: This implementation is quadratic time if there are many calls to
      // ExtendLocked with many bad consumers.  Since this is unlikely, and
      // fixing it would require data structures outside of this routine,
      // quadratic time it is.
      auto* bad_consumers = versions->mutable_bad_consumers();
      const std::unordered_set<int> existing(bad_consumers->begin(),
                                             bad_consumers->end());
      for (const int v : graph.versions().bad_consumers()) {
        if (existing.find(v) == existing.end()) {
          bad_consumers->Add(v);
        }
      }
    }
  } else {
    graph_def_.mutable_versions()->CopyFrom(graph.versions());
  }

  const int node_size_before_merge = graph_def_.node_size();
  graph_def_.MergeFrom(graph);

  FunctionLibraryDefinition fdefs(graph_def_.library());
  // Add default attributes to all new nodes in the graph.
  Status s =
      AddDefaultAttrsToGraphDef(&graph_def_, fdefs, node_size_before_merge);
  if (!s.ok()) {
    // One of the nodes was invalid, return the state of graph_def_
    // to what it was before this function.
    const int nodes_added = graph_def_.node_size() - node_size_before_merge;
    graph_def_.mutable_node()->DeleteSubrange(node_size_before_merge,
                                              nodes_added);
    return s;
  }

  if (graph_def_.versions().producer() >= 5) {
    // Validate the graph: we assume that merging two valid graphs
    // should maintain graph validity.
    TF_RETURN_IF_ERROR(graph::ValidateGraphDef(graph_def_, fdefs));
  }

  graph_created_ = true;  // In case this is first call
  return Status::OK();
}

// TODO(yuanbyu): Simplify by treating Run() as "PRunSetup(); PRun()".
Status DirectSession::Run(const NamedTensorList& inputs,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          std::vector<Tensor>* outputs) {
  RunMetadata run_metadata;
  return Run(RunOptions(), inputs, output_names, target_nodes, outputs,
             &run_metadata);
}

Status DirectSession::Run(const RunOptions& run_options,
                          const NamedTensorList& inputs,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          std::vector<Tensor>* outputs,
                          RunMetadata* run_metadata) {
  {
    mutex_lock l(graph_def_lock_);
    if (!graph_created_) {
      return errors::InvalidArgument(
          "Session was not created with a graph before Run()!");
    }
  }

  // Extract the inputs names for this run of the session.
  std::vector<string> input_tensor_names;
  input_tensor_names.reserve(inputs.size());
  for (const auto& it : inputs) {
    input_tensor_names.push_back(it.first);
  }

  // Check if we already have an executor for these arguments.
  ExecutorsAndKeys* executors_and_keys;
  RunStateArgs run_state_args;
  TF_RETURN_IF_ERROR(GetOrCreateExecutors(input_tensor_names, output_names,
                                          target_nodes, &executors_and_keys,
                                          &run_state_args));

  // Create a run state and start execution.
  RunState run_state(input_tensor_names, output_names);
  run_state.rendez = new IntraProcessRendezvous(device_mgr_.get());

  // Send inputs.
  TF_RETURN_IF_ERROR(SendInputs(inputs, executors_and_keys, run_state.rendez));

  // Start parallel Executors.
  const int num_executors = executors_and_keys->items.size();
  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_executors, run_state.rendez, [&run_state](const Status& ret) {
        {
          mutex_lock l(run_state.mu_);
          run_state.status.Update(ret);
        }
        run_state.executors_done.Notify();
      });

  Executor::Args args;
  args.step_id = step_id_counter_.fetch_add(1);
  args.rendezvous = run_state.rendez;
  args.cancellation_manager = cancellation_manager_;
  args.runner = [this](Executor::Args::Closure c) { SchedClosure(c); };
  args.session_state = &session_state_;
  args.tensor_store = &run_state.tensor_store;
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(args.step_id, run_state_args.handle);
  }

  std::unique_ptr<GPUTracer> tracer;
  if (run_options.trace_level() == RunOptions::FULL_TRACE ||
      options_.config.graph_options().build_cost_model()) {
    args.stats_collector = new StepStatsCollector(
        run_metadata->mutable_step_stats(), &cost_models_);
    run_state.collector = args.stats_collector;
    if (tracer && run_options.trace_level() == RunOptions::FULL_TRACE) {
      tracer.reset(CreateGPUTracer());
      tracer->Start();
    }
  }

  for (const auto& item : executors_and_keys->items) {
    item.executor->RunAsync(args, barrier->Get());
  }

  WaitForNotification(&run_state, run_options.timeout_in_ms() > 0
                                      ? run_options.timeout_in_ms()
                                      : operation_timeout_in_ms_);
  if (tracer) {
    tracer->Stop();
    tracer->Collect(args.stats_collector);
  }

  {
    mutex_lock l(run_state.mu_);
    TF_RETURN_IF_ERROR(run_state.status);
  }

  // Receive outputs.
  TF_RETURN_IF_ERROR(
      RecvOutputs(output_names, executors_and_keys, &run_state, outputs));

  // Save the output tensors of this run we choose to keep.
  TF_RETURN_IF_ERROR(
      run_state.tensor_store.SaveTensors(output_names, &session_state_));

  return Status::OK();
}

Status DirectSession::PRunSetup(const std::vector<string>& input_names,
                                const std::vector<string>& output_names,
                                const std::vector<string>& target_nodes,
                                string* handle) {
  {
    mutex_lock l(graph_def_lock_);
    if (!graph_created_) {
      return errors::InvalidArgument(
          "Session was not created with a graph before PRunSetup()!");
    }
  }

  // Check if we already have an executor for these arguments.
  ExecutorsAndKeys* executors_and_keys;
  RunStateArgs run_state_args;
  run_state_args.is_partial_run = true;
  Status s = GetOrCreateExecutors(input_names, output_names, target_nodes,
                                  &executors_and_keys, &run_state_args);
  TF_RETURN_IF_ERROR(s);

  // Create the run state and save it for future PRun calls.
  RunState* run_state = new RunState(input_names, output_names);
  run_state->rendez = new IntraProcessRendezvous(device_mgr_.get());
  {
    mutex_lock l(executor_lock_);
    if (!partial_runs_.insert({run_state_args.handle, run_state}).second) {
      return errors::Internal("The handle '", run_state_args.handle,
                              "' created for this partial run is not unique.");
    }
  }

  // Start parallel Executors.
  Notification& executors_done = run_state->executors_done;
  Status* run_status = &run_state->status;
  const int num_executors = executors_and_keys->items.size();
  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_executors, run_state->rendez,
      [&executors_done, run_status, this](const Status& ret) {
        if (!ret.ok()) {
          mutex_lock l(executor_lock_);
          *run_status = ret;
        }
        executors_done.Notify();
      });

  Executor::Args args;
  args.step_id = step_id_counter_.fetch_add(1);
  args.rendezvous = run_state->rendez;
  args.cancellation_manager = cancellation_manager_;
  args.runner = [this](Executor::Args::Closure c) { SchedClosure(c); };
  args.session_state = &session_state_;
  args.tensor_store = &run_state->tensor_store;
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(args.step_id, run_state_args.handle);
  }

  if (options_.config.graph_options().build_cost_model()) {
    run_state->collector = new StepStatsCollector(nullptr, &cost_models_);
    args.stats_collector = run_state->collector;
  }

  for (auto& item : executors_and_keys->items) {
    Executor* exec = item.executor;
    exec->RunAsync(args, barrier->Get());
  }

  *handle = run_state_args.handle;
  return Status::OK();
}

Status DirectSession::PRun(const string& handle, const NamedTensorList& inputs,
                           const std::vector<string>& output_names,
                           std::vector<Tensor>* outputs) {
  std::vector<string> parts = str_util::Split(handle, ';');
  const string& key = parts[0];
  // Get the executors for this partial run.
  ExecutorsAndKeys* executors_and_keys;
  RunState* run_state;
  {
    mutex_lock l(executor_lock_);  // could use reader lock
    auto exc_it = executors_.find(key);
    if (exc_it == executors_.end()) {
      return errors::InvalidArgument(
          "Must run 'setup' before performing partial runs!");
    }
    executors_and_keys = exc_it->second;

    auto prun_it = partial_runs_.find(handle);
    if (prun_it == partial_runs_.end()) {
      return errors::InvalidArgument(
          "Must run 'setup' before performing partial runs!");
    }
    run_state = prun_it->second;

    // Make sure that this is a new set of feeds that are still pending.
    for (const auto& input : inputs) {
      auto it = run_state->pending_inputs.find(input.first);
      if (it == run_state->pending_inputs.end()) {
        return errors::InvalidArgument("The feed ", input.first,
                                       " had already been fed.");
      }
    }
    // Check that this is a new set of fetches that are still pending.
    for (const auto& output : output_names) {
      auto it = run_state->pending_outputs.find(output);
      if (it == run_state->pending_outputs.end()) {
        return errors::InvalidArgument("The fetch ", output,
                                       " had already been fetched.");
      }
    }
  }

  // Check that this new set of fetches can be computed from all the
  // feeds we have supplied.
  TF_RETURN_IF_ERROR(
      CheckFetch(inputs, output_names, executors_and_keys, run_state));

  // Send inputs.
  Status s = SendInputs(inputs, executors_and_keys, run_state->rendez);

  // Receive outputs.
  if (s.ok()) {
    s = RecvOutputs(output_names, executors_and_keys, run_state, outputs);
  }

  // Save the output tensors of this run we choose to keep.
  if (s.ok()) {
    s = run_state->tensor_store.SaveTensors(output_names, &session_state_);
  }

  {
    mutex_lock l(executor_lock_);
    // Delete the run state if there is an error or all fetches are done.
    bool done = true;
    if (s.ok()) {
      {
        mutex_lock l(run_state->mu_);
        if (!run_state->status.ok()) {
          LOG(WARNING) << "An error unrelated to this prun has been detected. "
                       << run_state->status;
        }
      }
      for (const auto& it : inputs) {
        run_state->pending_inputs.erase(it.first);
      }
      for (const auto& name : output_names) {
        run_state->pending_outputs.erase(name);
      }
      done = (run_state->pending_inputs.size() == 0 &&
              run_state->pending_outputs.size() == 0);
    }
    if (done) {
      WaitForNotification(run_state, operation_timeout_in_ms_);
      partial_runs_.erase(handle);
      delete run_state;
    }
  }
  return s;
}

Status DirectSession::SendInputs(const NamedTensorList& inputs,
                                 const ExecutorsAndKeys* executors_and_keys,
                                 IntraProcessRendezvous* rendez) {
  Status s;
  // Insert the input tensors into the local rendezvous by their
  // rendezvous key.
  for (const auto& input : inputs) {
    auto it = executors_and_keys->input_keys.find(input.first);
    if (it == executors_and_keys->input_keys.end()) {
      return errors::InvalidArgument("'", input.first,
                                     "' is not a pre-defined feed!");
    }
    const string& input_key = it->second;
    s = rendez->Send(input_key, Rendezvous::Args(), input.second, false);
    if (!s.ok()) {
      rendez->StartAbort(s);
      return s;
    }
  }
  return Status::OK();
}

Status DirectSession::RecvOutputs(const std::vector<string>& output_names,
                                  const ExecutorsAndKeys* executors_and_keys,
                                  RunState* run_state,
                                  std::vector<Tensor>* outputs) {
  Status s;
  if (!output_names.empty()) {
    outputs->resize(output_names.size());
  }

  // Get the outputs from the rendezvous
  for (size_t output_offset = 0; output_offset < output_names.size();
       ++output_offset) {
    const string& output_name = output_names[output_offset];
    auto it = executors_and_keys->output_keys.find(output_name);
    if (it == executors_and_keys->output_keys.end()) {
      return errors::InvalidArgument("'", output_name,
                                     "' was not defined as a fetch"
                                     " target in PRunSetup.");
    }
    const string& output_key = it->second;
    Tensor output_tensor;
    bool is_dead;

    // Fetch data from the Rendezvous.
    IntraProcessRendezvous* rendez = run_state->rendez;
    s = rendez->Recv(output_key, Rendezvous::Args(), &output_tensor, &is_dead);
    if (is_dead && s.ok()) {
      s = errors::InvalidArgument("The tensor returned for ",
                                  output_names[output_offset],
                                  " was not valid.");
    }
    if (!s.ok()) {
      rendez->StartAbort(s);
      outputs->clear();
      return s;
    }

    (*outputs)[output_offset] = output_tensor;
  }
  return Status::OK();
}

Status DirectSession::CheckFetch(const NamedTensorList& feeds,
                                 const std::vector<string>& fetches,
                                 const ExecutorsAndKeys* executors_and_keys,
                                 const RunState* run_state) {
  const Graph* graph = executors_and_keys->graph;
  const NameNodeMap* name_to_node = executors_and_keys->name_to_node;

  // Build the set of pending feeds that we haven't seen.
  std::unordered_set<TensorId, TensorId::Hasher> pending_feeds;
  {
    mutex_lock l(executor_lock_);
    for (const string& feed : run_state->pending_inputs) {
      TensorId id(ParseTensorName(feed));
      auto it = name_to_node->find(id.first);
      if (it == name_to_node->end()) {
        return errors::NotFound("Feed ", feed, ": not found");
      }
      pending_feeds.insert(id);
    }
  }
  for (const auto& it : feeds) {
    TensorId id(ParseTensorName(it.first));
    pending_feeds.erase(id);
  }

  // Initialize the stack with the fetch nodes.
  std::vector<const Node*> stack;
  for (const string& fetch : fetches) {
    TensorId id(ParseTensorName(fetch));
    auto it = name_to_node->find(id.first);
    if (it == name_to_node->end()) {
      return errors::NotFound("Fetch ", fetch, ": not found");
    }
    stack.push_back(it->second);
  }

  // Any tensor needed for fetches can't be in pending_feeds.
  std::vector<bool> visited(graph->num_node_ids(), false);
  while (!stack.empty()) {
    const Node* n = stack.back();
    stack.pop_back();

    for (const Edge* in_edge : n->in_edges()) {
      const Node* in_node = in_edge->src();
      if (pending_feeds.count({in_node->name(), in_edge->src_output()}) > 0) {
        return errors::InvalidArgument("Fetch ", in_node->name(), ":",
                                       in_edge->src_output(),
                                       " can't be computed from the feeds"
                                       " that have been fed so far.");
      }
      if (!visited[in_node->id()]) {
        visited[in_node->id()] = true;
        stack.push_back(in_node);
      }
    }
  }
  return Status::OK();
}

Status DirectSession::GetOrCreateExecutors(
    gtl::ArraySlice<string> inputs, gtl::ArraySlice<string> outputs,
    gtl::ArraySlice<string> target_nodes, ExecutorsAndKeys** executors_and_keys,
    RunStateArgs* run_state_args) {
  // Sort the inputs and outputs, so we don't create separate
  // executors when a user passes in the same inputs/outputs in
  // different orders.
  //
  // We could consider some other signature instead of sorting that
  // preserves the same property to avoid the sort in the future.
  std::vector<string> inputs_sorted(inputs.begin(), inputs.end());
  std::vector<string> outputs_sorted(outputs.begin(), outputs.end());
  std::vector<string> tn_sorted(target_nodes.begin(), target_nodes.end());
  std::sort(inputs_sorted.begin(), inputs_sorted.end());
  std::sort(outputs_sorted.begin(), outputs_sorted.end());
  std::sort(tn_sorted.begin(), tn_sorted.end());

  const string key = strings::StrCat(str_util::Join(inputs_sorted, ","), "->",
                                     str_util::Join(outputs_sorted, ","), "/",
                                     str_util::Join(tn_sorted, ","));

  // Set the handle.
  {
    mutex_lock l(mu_);
    run_state_args->handle = strings::StrCat(key, ";", name_counter_++);
  }

  // See if we already have the executors for this run.
  {
    mutex_lock l(executor_lock_);  // could use reader lock
    auto it = executors_.find(key);
    if (it != executors_.end()) {
      *executors_and_keys = it->second;
      return Status::OK();
    }
  }

  // The executor_lock_ is intentionally released while executor is
  // being created.
  FunctionLibraryDefinition* fdefs;
  std::unordered_map<string, Graph*> graphs;
  Status s = CreateGraphs(inputs, outputs, target_nodes, &fdefs, &graphs,
                          run_state_args);
  TF_RETURN_IF_ERROR(s);

  std::unique_ptr<ExecutorsAndKeys> ek(new ExecutorsAndKeys);
  ek->func_defs = fdefs;
  if (run_state_args->is_partial_run) {
    ek->graph = run_state_args->graph;
    ek->name_to_node = new NameNodeMap;
    std::unordered_set<StringPiece, StringPiece::Hasher> names;
    for (const string& input : inputs) {
      TensorId id(ParseTensorName(input));
      names.emplace(id.first);
    }
    for (const string& output : outputs) {
      TensorId id(ParseTensorName(output));
      names.emplace(id.first);
    }
    for (Node* n : run_state_args->graph->nodes()) {
      if (names.count(n->name()) > 0) {
        ek->name_to_node->insert({n->name(), n});
      }
    }
  }
  ek->items.reserve(graphs.size());
  auto runner = [this](Executor::Args::Closure c) { SchedClosure(c); };
  const auto& optimizer_opts =
      options_.config.graph_options().optimizer_options();
  GraphOptimizer optimizer(optimizer_opts);
  for (auto iter = graphs.begin(); iter != graphs.end(); ++iter) {
    const string& partition_name = iter->first;
    Graph* partition_graph = iter->second;
    const int graph_def_version = partition_graph->versions().producer();

    Device* device;
    s = device_mgr_->LookupDevice(partition_name, &device);
    if (!s.ok()) break;

    ek->items.resize(ek->items.size() + 1);
    auto* item = &(ek->items.back());
    item->flib = NewFunctionLibraryRuntime(device, runner, graph_def_version,
                                           fdefs, optimizer_opts);

    LocalExecutorParams params;
    params.device = device;
    params.function_library = item->flib;
    auto lib = item->flib;
    auto opseg = device->op_segment();
    params.create_kernel = [this, lib, opseg](const NodeDef& ndef,
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
      return opseg->FindOrCreate(session_handle_, ndef.name(), kernel,
                                 create_fn);
    };
    params.delete_kernel = [lib](OpKernel* kernel) {
      // If the node is stateful, opseg owns it. Otherwise, delete it.
      if (kernel && !lib->IsStateful(kernel->type_string())) {
        delete kernel;
      }
    };

    optimizer.Optimize(lib, device, &partition_graph);
    s = ValidateMemoryTypes(DeviceType(device->device_type()), partition_graph);
    if (!s.ok()) {
      break;
    }
    // NewLocalExecutor takes ownership of *partition_graph.
    iter->second = nullptr;
    item->executor = nullptr;
    s = NewLocalExecutor(params, partition_graph, &item->executor);
    if (!s.ok()) {
      break;
    }
  }
  if (!s.ok()) {
    gtl::STLDeleteValues(&graphs);
    return s;
  }

  // Compute the rendezvous keys to avoid recomputing them every time.
  //
  // We always use the first device as the device name portion of the
  // key, even if we're feeding another graph.
  for (const string& input : inputs) {
    ek->input_keys[input] = GetRendezvousKey(
        input, device_set_.client_device()->attributes(), FrameAndIter(0, 0));
  }
  for (const string& output : outputs) {
    ek->output_keys[output] = GetRendezvousKey(
        output, device_set_.client_device()->attributes(), FrameAndIter(0, 0));
  }

  // Reacquire the lock, try to insert into the map.
  mutex_lock l(executor_lock_);
  const bool inserted = executors_.insert(std::make_pair(key, ek.get())).second;
  if (!inserted) {
    // Another thread created the entry before us, so delete the
    // one we created and return the already created one.
    auto it = executors_.find(key);
    *executors_and_keys = it->second;
  } else {
    *executors_and_keys = ek.release();
  }

  return Status::OK();
}

void DirectSession::SaveStatefulNodes(Graph* graph) {
  for (Node* n : graph->nodes()) {
    if (n->op_def().is_stateful()) {
      VLOG(2) << "Saving " << n->DebugString();
      stateful_placements_[n->name()] = n->assigned_device_name();
    }
  }
}

void DirectSession::RestoreStatefulNodes(Graph* graph) {
  for (Node* n : graph->nodes()) {
    if (n->op_def().is_stateful()) {
      auto iter = stateful_placements_.find(n->name());
      if (iter != stateful_placements_.end()) {
        n->set_assigned_device_name(iter->second);
        VLOG(2) << "Restored " << n->DebugString();
      }
    }
  }
}

Status DirectSession::CreateGraphs(gtl::ArraySlice<string> feeds,
                                   gtl::ArraySlice<string> fetches,
                                   gtl::ArraySlice<string> target_nodes,
                                   FunctionLibraryDefinition** func_defs,
                                   std::unordered_map<string, Graph*>* outputs,
                                   RunStateArgs* run_state_args) {
  std::unique_ptr<FunctionLibraryDefinition> fdefs;
  std::unique_ptr<Graph> graph;
  {
    mutex_lock l(graph_def_lock_);
    fdefs.reset(new FunctionLibraryDefinition(graph_def_.library()));
    graph.reset(new Graph(fdefs.get()));
    GraphConstructorOptions opts;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, graph_def_, graph.get()));
  }

  // Remember the graph in run state if this is a partial run.
  if (run_state_args->is_partial_run) {
    run_state_args->graph = new Graph(fdefs.get());
    CopyGraph(*graph.get(), run_state_args->graph);
  }

  TF_RETURN_IF_ERROR(subgraph::RewriteGraphForExecution(
      graph.get(), feeds, fetches, target_nodes,
      device_set_.client_device()->attributes()));

  // Run the simple placer after rewriting the graph.
  std::unordered_map<string, int32> node_name_to_cost_map;
  for (Node* n : graph->nodes()) {
    node_name_to_cost_map[n->name()] = n->cost_id();
  }
  SimplePlacer placer(graph.get(), &device_set_, &node_name_to_cost_map,
                      &options_);

  {
    mutex_lock l(mu_);
    // Restore stateful nodes.
    RestoreStatefulNodes(graph.get());
    TF_RETURN_IF_ERROR(placer.Run());
    // Save stateful nodes.
    SaveStatefulNodes(graph.get());
  }

  // Partition the graph across devices.
  std::unordered_map<string, GraphDef> partitions;
  PartitionOptions popts;
  popts.node_to_loc = [](const Node* node) {
    return node->assigned_device_name();
  };
  popts.new_name = [this](const string& prefix) {
    mutex_lock l(mu_);
    return strings::StrCat(prefix, "/_", name_counter_++);
  };
  popts.get_incarnation = [](const string& name) {
    // The direct session does not have changing incarnation numbers.
    // Just return '1'.
    return 1;
  };
  popts.control_flow_added = false;
  TF_RETURN_IF_ERROR(Partition(popts, graph.get(), &partitions));

  std::vector<string> device_names;
  for (auto device : devices_) {
    // Extract the LocalName from the device.
    device_names.push_back(DeviceNameUtils::LocalName(device->name()));
  }

  // Check for valid partitions.
  for (const auto& partition : partitions) {
    const string& local_partition_name =
        DeviceNameUtils::LocalName(partition.first);
    if (std::count(device_names.begin(), device_names.end(),
                   local_partition_name) == 0) {
      return errors::InvalidArgument(
          "Creating a partition for ", local_partition_name,
          " which doesn't exist in the list of available devices. Available "
          "devices: ",
          str_util::Join(device_names, ","));
    }
  }

  Status s;
  for (auto&& partition : partitions) {
    const string& partition_name = partition.first;

    GraphDef* graph_def = &partition.second;
    VLOG(2) << "Created " << graph_def->DebugString() << " for "
            << partition_name;

    // Give the device an opportunity to rewrite its subgraph.
    Device* d;
    s = device_mgr_->LookupDevice(partition_name, &d);
    if (!s.ok()) break;
    {
      mutex_lock l(graph_def_lock_);
      // TODO(pbar) The library is currently shared and immutable. There
      // may be possible use cases where a device may want to modify
      // function definitions - in which case the library would need to be
      // replicated per device.
      s = d->MaybeRewriteGraph(graph_def_.library(), graph_def);
      if (!s.ok()) {
        break;
      }
    }
    Graph* device_graph = new Graph(fdefs.get());
    GraphConstructorOptions device_opts;
    // There are internal operations (e.g., send/recv) that we now
    // allow.
    device_opts.allow_internal_ops = true;
    device_opts.expect_device_spec = true;
    s = ConvertGraphDefToGraph(device_opts, *graph_def, device_graph);
    if (!s.ok()) {
      delete device_graph;
      break;
    }
    outputs->insert(std::make_pair(partition_name, device_graph));
  }
  if (!s.ok()) {
    // Also delete other graphs created during the loop.
    gtl::STLDeleteValues(outputs);
    return s;
  }
  *func_defs = fdefs.release();
  return Status::OK();
}

::tensorflow::Status DirectSession::Close() {
  cancellation_manager_->StartCancel();
  return ::tensorflow::Status::OK();
}

DirectSession::RunState::~RunState() {
  if (rendez != nullptr) {
    if (!executors_done.HasBeenNotified()) {
      rendez->StartAbort(errors::Cancelled("PRun cancellation"));
      executors_done.WaitForNotification();
    }
    rendez->Unref();
  }
  if (collector != nullptr) {
    delete collector;
  }
}

void DirectSession::WaitForNotification(RunState* run_state,
                                        int64 timeout_in_ms) {
  if (timeout_in_ms > 0) {
    bool timed_out =
        run_state->executors_done.WaitForNotificationWithTimeout(timeout_in_ms);
    if (timed_out) {
      {
        mutex_lock l(run_state->mu_);
        run_state->status.Update(Status(error::DEADLINE_EXCEEDED,
                                        "Timed out waiting for notification"));
      }
      // TODO(sherrym): This cancels all steps in the session, even ones that
      // have not exceeded their deadline. An alternative would be to use a
      // two-level cancellation manager with a Session-global one containing
      // several step-local ones. Probably the RunState should have its own
      // CancellationManager.
      cancellation_manager_->StartCancel();
    }
  } else {
    run_state->executors_done.WaitForNotification();
  }
}

class DirectSessionFactory : public SessionFactory {
 public:
  DirectSessionFactory() {}

  bool AcceptsOptions(const SessionOptions& options) override {
    return options.target.empty();
  }

  Session* NewSession(const SessionOptions& options) override {
    std::vector<Device*> devices;
    DeviceFactory::AddDevices(options, "/job:localhost/replica:0/task:0",
                              &devices);
    return new DirectSession(options, new DeviceMgr(devices));
  }
};

class DirectSessionRegistrar {
 public:
  DirectSessionRegistrar() {
    SessionFactory::Register("DIRECT_SESSION", new DirectSessionFactory());
  }
};
static DirectSessionRegistrar registrar;

}  // namespace tensorflow
