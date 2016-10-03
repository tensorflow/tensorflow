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

#include "tensorflow/core/distributed_runtime/master_session.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/profile_handler.h"
#include "tensorflow/core/common_runtime/simple_graph_execution_state.h"
#include "tensorflow/core/common_runtime/stats_publisher_interface.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/master_session_interface.h"
#include "tensorflow/core/distributed_runtime/scheduler.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/master.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {
// A little bit of per-step state.
struct PerStepState {
  bool collect_costs = false;
  bool collect_timeline = false;
  bool collect_rpcs = false;
  Microseconds start_micros = Microseconds(0);
  Microseconds end_micros = Microseconds(0);
  std::vector<StepStats> step_stats;  // per partition
  StepStats rpc_stats;                // for RPC layer
};

// A session encapsulates a graph computation (resource allocation,
// placement, execution, etc.).
class MasterSession : public MasterSessionInterface {
 public:
  // This session encapsulates the graph computation for a graph.
  //
  // The session places nodes on devices in "remote_devs" and executes
  // operations on these devices.
  //
  // The caller takes ownership of all remote devices.
  MasterSession(const SessionOptions& options, const MasterEnv* env,
                std::vector<Device*>* remote_devs,
                StatsPublisherFactory stats_publisher_factory);

  // Initialize the MasterSession for "def".  Must be called before Extend(),
  // Run(), or Close().
  //
  // The callee may clear "def".
  Status Create(GraphDef* def) override;

  // Returns the session handle.
  const string& handle() const override { return handle_; }

  // Returns the last access time (the number of micro-seconds since
  // some fixed point in time) of this session.
  uint64 last_access_time_usec() const override {
    return last_access_time_usec_.load();
  }

  // Attempt to extend the graph according to the given "req".
  // (See master.proto for details of valid extensions.)
  //
  // PRECONDITION: The current version of this session's graph
  //   is "req->current_graph_version".
  //
  // POSTCONDITION: The current version of this session's graph
  //   is "resp->new_graph_version".
  //
  // Extend() may block the caller thread for a long time.
  Status Extend(const ExtendSessionRequest* req,
                ExtendSessionResponse* resp) override;

  // Run one step.
  Status Run(CallOptions* opts, const RunStepRequest* req,
             RunStepResponse* resp) override;

  // Close this session and delete "*this". Returns OK if all known
  // states are cleanup successfully.
  //
  // Close() may block the caller thread for a long time.
  Status Close() override;

 private:
  SessionOptions session_opts_;

  // Not owned.
  const MasterEnv* env_;

  // The opaque session handle.
  const string handle_;

  // Owned.
  std::vector<Device*> remote_devs_;

  // The device set used by this session.
  DeviceSet devices_;

  StatsPublisherFactory stats_publisher_factory_;

  std::atomic_ulong last_access_time_usec_;

  mutex mu_;
  std::unique_ptr<SimpleGraphExecutionState> execution_state_;
  int64 graph_version_;

  // We keep a map from a signature of a run request to the
  // ReffedClientGraph the can execute it.  We keep up to one old copy
  // of each ReffedClientGraph around because if it gets deallocated
  // before a new substitute has been created, Variables can go out of
  // scope and lose their state.
  class ReffedClientGraph;
  typedef std::unordered_map<uint64, ReffedClientGraph*> RCGMap;
  RCGMap runs_ GUARDED_BY(mu_);
  RCGMap obsolete_ GUARDED_BY(mu_);

  // Active RunStep calls.
  condition_variable num_running_is_zero_;
  int32 num_running_ GUARDED_BY(mu_) = 0;

  std::unordered_map<uint64, int64> subgraph_execution_counts_ GUARDED_BY(mu_);

  // We need to ensure that certain nodes added (e.g., send and recv
  // nodes) are unique across all sub-graphs within this session.
  int64 next_node_id_ GUARDED_BY(mu_) = 0;

  // Used to cancel running steps on Close().
  CancellationManager* cancellation_manager_;

  // Private dtor. The client must call Close().
  virtual ~MasterSession();

  Status StartStep(const RunStepRequest& req, BuildGraphOptions* opts,
                   int64* count, ReffedClientGraph** graph);
  void ClearRunsTable(std::vector<ReffedClientGraph*>* to_unref,
                      RCGMap* rcg_map) EXCLUSIVE_LOCKS_REQUIRED(mu_);
  Status DoRunWithLocalExecution(CallOptions* opts, const RunStepRequest* req,
                                 RunStepResponse* resp);
  void UpdateLastAccessTime();

  TF_DISALLOW_COPY_AND_ASSIGN(MasterSession);
};

// MasterSession wraps SimpleClientGraph in a reference counted object.
// This way, MasterSession can clear up the cache mapping Run requests to
// compiled graphs while the compiled graph is still being used.
//
// TODO(zhifengc): Cleanup this class. It's becoming messy.
class MasterSession::ReffedClientGraph : public core::RefCounted {
 public:
  ReffedClientGraph(const string& handle, const BuildGraphOptions& bopts,
                    std::unique_ptr<SimpleClientGraph> cg,
                    const SessionOptions& session_opts,
                    StatsPublisherFactory stats_publisher_factory)
      : session_handle_(handle),
        client_graph_(std::move(cg)),
        bopts_(bopts),
        session_opts_(session_opts) {
    VLOG(1) << "Created ReffedClientGraph for node with "
            << client_graph_->graph.num_node_ids();

    stats_publisher_ = stats_publisher_factory(handle, bopts, session_opts);
  }

  ~ReffedClientGraph() override { DeregisterPartitions(); }

  const SimpleClientGraph* client_graph() { return client_graph_.get(); }

  std::unique_ptr<ProfileHandler> GetProfileHandler(uint64 step,
                                                    int64 execution_count,
                                                    const RunOptions& ropts) {
    return stats_publisher_->GetProfileHandler(step, execution_count, ropts);
  }

  // Turn RPC logging on or off, both at the WorkerCache used by this
  // master process, and at each remote worker in use for the current
  // partitions.
  void SetRPCLogging(const MasterEnv* env, bool active) {
    env->worker_cache->SetLogging(active);
    // Logging is a best-effort activity, so we make async calls to turn
    // it on/off and don't make use of the responses.
    for (auto& p : partitions_) {
      LoggingRequest* req = new LoggingRequest;
      req->set_rpc_logging(active);
      LoggingResponse* resp = new LoggingResponse;
      Ref();
      p.worker->LoggingAsync(req, resp, [this, req, resp](const Status& s) {
        delete req;
        delete resp;
        // ReffedClientGraph owns p.worker so we need to hold a ref to
        // ensure that the method doesn't attempt to access p.worker after
        // ReffedClient graph has deleted it.
        // TODO(suharshs): Simplify this ownership model.
        Unref();
      });
    }
  }

  // Retrieve all RPC logs data accumulated for the current step, both
  // from the local WorkerCache in use by this master process and from
  // all the remote workers executing the remote partitions.
  void RetrieveLogs(const MasterEnv* env, int64 step_id, StepStats* ss) {
    // Get the local data first, because it sets *ss without merging.
    env->worker_cache->RetrieveLogs(step_id, ss);

    // Then merge in data from all the remote workers.
    LoggingRequest req;
    req.add_fetch_step_id(step_id);
    int waiting_for = partitions_.size();
    if (waiting_for > 0) {
      mutex scoped_mu;
      // TODO(suharshs): Use BlockingCounter instead?
      Notification all_done;
      for (auto& p : partitions_) {
        LoggingResponse* resp = new LoggingResponse;
        p.worker->LoggingAsync(
            &req, resp, [step_id, ss, resp, &scoped_mu, &waiting_for,
                         &all_done](const Status& s) {
              bool notify_all_done = false;
              {
                mutex_lock l(scoped_mu);
                --waiting_for;
                if (waiting_for == 0) {
                  notify_all_done = true;
                }
                if (s.ok()) {
                  for (auto& lss : resp->step()) {
                    if (step_id != lss.step_id()) {
                      LOG(ERROR) << "Wrong step_id in LoggingResponse";
                      continue;
                    }
                    ss->MergeFrom(lss.step_stats());
                  }
                }
                delete resp;
              }
              // Must not call all_done.Notify() until out of critical
              // section where *ss is updated.
              if (notify_all_done) {
                all_done.Notify();
              }
            });
      }
      all_done.WaitForNotification();
    }
  }

  // Local execution methods.

  // Partitions the graph into subgraphs and registers them on
  // workers.
  Status RegisterPartitions(const MasterEnv* env, const PartitionOptions& popts,
                            const FunctionDefLibrary& func_def_lib);

  // Runs one step of all partitions.
  Status RunPartitions(const MasterEnv* env, int64 step_id,
                       int64 execution_count,
                       SimpleGraphExecutionState* execution_state,
                       PerStepState* pss, CallOptions* opts,
                       const RunStepRequest& req, RunStepResponse* resp,
                       CancellationManager* cm);

  // Calls workers to cleanup states for the step "step_id".  Calls
  // `done` when all cleanup RPCs have completed.
  void CleanupPartitionsAsync(int64 step_id, StatusCallback done);

  // Post-processing of any runtime statistics gathered during execution.
  void ProcessStats(const MasterEnv* env, int64 step_id, PerStepState* pss,
                    SimpleGraphExecutionState* execution_state,
                    ProfileHandler* ph, RunStepResponse* resp);
  void ProcessDeviceStats(ProfileHandler* ph,
                          SimpleGraphExecutionState* execution_state,
                          const DeviceStepStats& ds, bool is_rpc);

  string DetailText(const NodeDef& def, const NodeExecStats& ns) {
    int64 tot = 0;
    for (auto& no : ns.output()) {
      tot += no.tensor_description().allocation_description().requested_bytes();
    }
    string bytes;
    if (tot >= 0.1 * 1048576.0) {
      bytes = strings::Printf("[%.1fMB] ", tot / 1048576.0);
    }
    return strings::StrCat(
        bytes, def.name(), " = ", def.op(), "(",
        str_util::Join(
            std::vector<StringPiece>(def.input().begin(), def.input().end()),
            ", "),
        ")");
  }

 private:
  const string session_handle_;
  const std::unique_ptr<SimpleClientGraph> client_graph_;
  std::unordered_set<const Node*> nodes_needing_input_mapping_;
  BuildGraphOptions bopts_;
  const SessionOptions session_opts_;

  // Graph partitioned into per-location subgraphs.
  struct Part {
    // Worker name.
    string name;

    // Graph definition.
    GraphDef gdef;

    // Maps feed names to rendezvous keys. Empty most of the time.
    std::unordered_map<string, string> feed_key;

    // Maps rendezvous keys to fetch names. Empty most of the time.
    std::unordered_map<string, string> key_fetch;

    // The interface to the worker. Owned.
    WorkerInterface* worker = nullptr;

    // After registeration with the worker, graph_handle identifies
    // this partition on the worker.
    string graph_handle;

    Part() : feed_key(3), key_fetch(3) {}
  };

  // partitions_ is immutable after RegisterPartitions() call
  // finishes.  RunPartitions() can access partitions_ safely without
  // acquiring locks.
  std::vector<Part> partitions_;

  mutable mutex mu_;

  // Partition initialization and registration only needs to happen
  // once. init_started_ && !init_done_ indicates the initialization
  // is on going.
  bool init_started_ GUARDED_BY(mu_) = false;
  Notification init_done_;

  // init_result_ remembers the initialization error if any.
  Status init_result_ GUARDED_BY(mu_);

  std::unique_ptr<StatsPublisherInterface> stats_publisher_;

  // Send/Recv nodes that are the result of client-added
  // feeds and fetches must be tracked so that the tensors
  // can be added to the local rendezvous.
  static void TrackFeedsAndFetches(Part* part, const PartitionOptions& popts);

  // The actual graph partitioning and registration implementation.
  Status DoRegisterPartitions(const MasterEnv* env,
                              const PartitionOptions& popts,
                              const FunctionDefLibrary& func_def_lib);

  // Deregisters the partitions on the workers.  Called in the
  // destructor and does not wait for the rpc completion.
  void DeregisterPartitions();

  TF_DISALLOW_COPY_AND_ASSIGN(ReffedClientGraph);
};

Status MasterSession::ReffedClientGraph::RegisterPartitions(
    const MasterEnv* env, const PartitionOptions& popts,
    const FunctionDefLibrary& func_def_lib) {
  {  // Ensure register once.
    mu_.lock();
    if (!init_started_) {
      init_started_ = true;
      mu_.unlock();
      Status s = DoRegisterPartitions(env, popts, func_def_lib);
      std::vector<const GraphDef*> graph_defs;
      graph_defs.reserve(partitions_.size());
      for (const Part& part : partitions_) {
        graph_defs.push_back(&part.gdef);
      }
      stats_publisher_->PublishGraphProto(graph_defs);
      mu_.lock();
      init_result_ = s;
      init_done_.Notify();
    } else {
      mu_.unlock();
      init_done_.WaitForNotification();
      mu_.lock();
    }
    Status result = init_result_;
    mu_.unlock();
    return result;
  }
}

static string SplitByWorker(const Node* node) {
  string task;
  string device;
  CHECK(DeviceNameUtils::SplitDeviceName(node->assigned_device_name(), &task,
                                         &device))
      << "node: " << node->name() << " dev: " << node->assigned_device_name();
  return task;
}

void MasterSession::ReffedClientGraph::TrackFeedsAndFetches(
    Part* part, const PartitionOptions& popts) {
  for (int i = 0; i < part->gdef.node_size(); ++i) {
    NodeDef* ndef = part->gdef.mutable_node(i);
    const bool is_recv = ndef->op() == "_Recv";
    const bool is_send = ndef->op() == "_Send";

    if (is_recv || is_send) {
      string name;
      TF_CHECK_OK(GetNodeAttr(*ndef, "tensor_name", &name));
      string send_device;
      TF_CHECK_OK(GetNodeAttr(*ndef, "send_device", &send_device));
      string recv_device;
      TF_CHECK_OK(GetNodeAttr(*ndef, "recv_device", &recv_device));
      uint64 send_device_incarnation;
      TF_CHECK_OK(
          GetNodeAttr(*ndef, "send_device_incarnation",
                      reinterpret_cast<int64*>(&send_device_incarnation)));
      const string& key =
          Rendezvous::CreateKey(send_device, send_device_incarnation,
                                recv_device, name, FrameAndIter(0, 0));

      // Only send/recv nodes that were added as feeds and fetches
      // (client-terminated) should be tracked.  Other send/recv nodes
      // are for transferring data between partitions / memory spaces.
      bool client_terminated;
      TF_CHECK_OK(GetNodeAttr(*ndef, "client_terminated", &client_terminated));
      if (client_terminated) {
        if (is_recv) {
          part->feed_key.insert({name, key});
        } else {
          part->key_fetch.insert({key, name});
        }
      }
    }
  }
}

Status MasterSession::ReffedClientGraph::DoRegisterPartitions(
    const MasterEnv* env, const PartitionOptions& popts_in,
    const FunctionDefLibrary& func_def_lib) {
  PartitionOptions popts = popts_in;
  if (popts.need_to_record_start_times) {
    CostModel cost_model(true);
    cost_model.InitFromGraph(client_graph()->graph);
    // TODO(yuanbyu): Use the real cost model.
    // execution_state_->MergeFromGlobal(&cost_model);
    SlackAnalysis sa(&client_graph()->graph, &cost_model);
    sa.ComputeAsap(&popts.start_times);
  }

  // Partition the graph.
  Status s;
  std::unordered_map<string, GraphDef> graph_partitions;
  s = Partition(popts, &client_graph_->graph, &graph_partitions);
  if (!s.ok()) return s;
  partitions_.reserve(graph_partitions.size());
  for (auto& name_def : graph_partitions) {
    partitions_.resize(partitions_.size() + 1);
    Part* part = &partitions_.back();
    part->name = name_def.first;
    part->gdef.Swap(&name_def.second);
    // For simplicity, we ship the library completely to every worker.
    *(part->gdef.mutable_library()) = func_def_lib;
    TrackFeedsAndFetches(part, popts);
    part->worker = env->worker_cache->CreateWorker(part->name);
    if (part->worker == nullptr) {
      s = errors::NotFound("worker ", part->name);
      break;
    }
  }
  if (!s.ok()) {
    for (Part& part : partitions_) {
      delete part.worker;
    }
    return s;
  }
  struct Call {
    RegisterGraphRequest req;
    RegisterGraphResponse resp;
    Status status;
    Notification done;
  };
  const int num = partitions_.size();
  gtl::InlinedVector<Call, 4> calls(num);
  for (int i = 0; i < num; ++i) {
    const Part& part = partitions_[i];
    Call* c = &calls[i];
    c->req.set_session_handle(session_handle_);
    *c->req.mutable_graph_def() = part.gdef;
    *c->req.mutable_graph_options() = session_opts_.config.graph_options();
    VLOG(2) << "Register " << part.gdef.DebugString();
    auto cb = [c](const Status& s) {
      c->status = s;
      c->done.Notify();
    };
    part.worker->RegisterGraphAsync(&c->req, &c->resp, cb);
  }
  for (int i = num - 1; i >= 0; --i) {
    Call* c = &calls[i];
    c->done.WaitForNotification();
    s.Update(c->status);
    partitions_[i].graph_handle = c->resp.graph_handle();
  }
  return s;
}

static bool CopyIfNeeded(TensorProto* in, TensorProto* out) {
  if (in->tensor_content().empty()) {
    // If the tensor is not encoded in tensor_content or contains 0
    // elements, we can return it to the client directly.
    out->Swap(in);
  } else {
    Tensor t(in->dtype());
    if (!t.FromProto(cpu_allocator(), *in)) return false;
    t.AsProtoField(out);
  }
  return true;
}

// Helper class to manage "num" parallel RunGraph calls.
class RunManyGraphs {
 public:
  explicit RunManyGraphs(int num) : calls_(num), num_pending_(num) {}

  ~RunManyGraphs() {}

  // Returns the index-th call.
  struct Call {
    CallOptions opts;
    RunGraphRequest req;
    RunGraphResponse resp;
  };
  Call* get(int index) { return &calls_[index]; }

  // When the index-th call is done, updates the overall status.
  void WhenDone(int index, const Status& s) {
    TRACEPRINTF("Partition %d %s", index, s.ToString().c_str());
    {
      mutex_lock l(mu_);
      if (!s.ok()) {
        UpdateStatusLocked(s);
      }
      --num_pending_;
      cv_pending_.notify_all();
    }
  }

  void StartCancel() {
    mutex_lock l(mu_);
    UpdateStatusLocked(errors::Cancelled("RunManyGraphs"));
  }

  void Wait() {
    mutex_lock l(mu_);
    while (num_pending_ > 0) {
      cv_pending_.wait(l);
    }
  }

  Status status() const {
    mutex_lock l(mu_);
    return status_;
  }

 private:
  gtl::InlinedVector<Call, 4> calls_;

  // TODO(jeff,sanjay): Replace bookkeeping state here with a
  // BlockingCounter abstraction that we define in
  // tensorflow/core/lib/core.
  mutable mutex mu_;
  condition_variable cv_pending_;
  int num_pending_;
  Status status_ GUARDED_BY(mu_);

  void UpdateStatusLocked(const Status& s) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (status_.ok()) {
      status_ = s;
      for (Call& call : calls_) {
        call.opts.StartCancel();
      }
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(RunManyGraphs);
};

int64 CostFrequency(int64 x) {
  if (x < 10) {
    return 1;  // 100%
  } else if (x < 100) {
    return 10;  // 10%
  } else if (x < 1000) {
    return 100;  // 1%
  } else {
    return 1000;  // 0.1%
  }
}

Status MasterSession::ReffedClientGraph::RunPartitions(
    const MasterEnv* env, int64 step_id, int64 execution_count,
    SimpleGraphExecutionState* execution_state, PerStepState* pss,
    CallOptions* call_opts, const RunStepRequest& req, RunStepResponse* resp,
    CancellationManager* cm) {
  VLOG(2) << "RunPartitions step_id " << step_id << " execution_count "
          << execution_count;
  // Builds an index for feeds provided by the client.
  std::unordered_map<StringPiece, const TensorProto*, StringPiece::Hasher>
      feeds(3);

  for (const auto& feed : req.feed()) {
    if (!feeds.insert({feed.name(), &feed.tensor()}).second) {
      return errors::InvalidArgument("Duplicated feeds: ", feed.name());
    }
  }

  // Prepares a number of calls to workers. One call per partition.

  // Collect execution cost stats on a smoothly decreasing frequency.
  ExecutorOpts exec_opts;
  if (pss->collect_costs) {
    exec_opts.set_record_costs(true);
  }
  if (pss->collect_timeline) {
    exec_opts.set_record_timeline(true);
  }
  if (pss->collect_rpcs) {
    SetRPCLogging(env, true);
  }
  if (pss->collect_costs || pss->collect_timeline) {
    pss->step_stats.resize(partitions_.size());
  }

  const int num = partitions_.size();
  RunManyGraphs calls(num);

  for (int i = 0; i < num; ++i) {
    const Part& part = partitions_[i];
    RunManyGraphs::Call* c = calls.get(i);
    c->req.set_graph_handle(part.graph_handle);
    c->req.set_step_id(step_id);
    *c->req.mutable_exec_opts() = exec_opts;
    // If any feeds are provided, send the feed values together
    // in the RunGraph request.
    for (const auto& feed_key : part.feed_key) {
      const string& feed = feed_key.first;
      const string& key = feed_key.second;
      const TensorProto* val = feeds[feed];
      if (val == nullptr) {
        return errors::InvalidArgument("No feed is provided for feed=", feed,
                                       ", key=", key);
      }
      auto* send = c->req.add_send();
      send->set_key(key);
      *(send->mutable_val()) = *val;  // TODO(mrry): make it faster if needed.
    }
    for (const auto& key_fetch : part.key_fetch) {
      const string& key = key_fetch.first;
      c->req.add_recv_key(key);
    }
  }

  // Issues RunGraph calls.
  for (int i = 0; i < num; ++i) {
    const Part& part = partitions_[i];
    RunManyGraphs::Call* call = calls.get(i);
    TRACEPRINTF("Partition %d %s", i, part.name.c_str());
    part.worker->RunGraphAsync(
        &call->opts, &call->req, &call->resp,
        std::bind(&RunManyGraphs::WhenDone, &calls, i, std::placeholders::_1));
  }

  // Waits for the RunGraph calls.
  call_opts->SetCancelCallback([&calls]() { calls.StartCancel(); });
  auto token = cm->get_cancellation_token();
  bool success =
      cm->RegisterCallback(token, [&calls]() { calls.StartCancel(); });
  if (!success) {
    calls.StartCancel();
  }
  calls.Wait();
  call_opts->ClearCancelCallback();
  if (success) {
    cm->DeregisterCallback(token);
  } else {
    return errors::Cancelled("Step was cancelled");
  }

  // Collects fetches.
  Status status = calls.status();
  if (status.ok()) {
    for (int i = 0; i < num; ++i) {
      const Part& part = partitions_[i];
      for (auto& recv : *(calls.get(i)->resp.mutable_recv())) {
        auto* ret = resp->add_tensor();
        auto iter = part.key_fetch.find(recv.key());
        if (iter == part.key_fetch.end()) {
          status.Update(errors::Internal("Unexpected fetch key: ", recv.key()));
          break;
        }
        const string& fetch = iter->second;
        ret->set_name(fetch);
        if (!CopyIfNeeded(recv.mutable_val(), ret->mutable_tensor())) {
          status.Update(
              errors::Internal("Unexpected unparseable tensor: ", recv.key()));
          break;
        }
      }
      if (pss->collect_timeline && calls.get(i)->resp.has_step_stats()) {
        pss->step_stats[i].Swap(calls.get(i)->resp.mutable_step_stats());
      }
    }
  }
  return status;
}

namespace {

class CleanupBroadcastHelper {
 public:
  CleanupBroadcastHelper(int64 step_id, int num_calls, StatusCallback done)
      : resps_(num_calls), num_pending_(num_calls), done_(std::move(done)) {
    req_.set_step_id(step_id);
  }

  // Returns a non-owned pointer to a request buffer for all calls.
  CleanupGraphRequest* request() { return &req_; }

  // Returns a non-owned pointer to a response buffer for the ith call.
  CleanupGraphResponse* response(int i) { return &resps_[i]; }

  // Called when the ith response is received.
  void call_done(int i, const Status& s) {
    bool run_callback = false;
    Status status_copy;
    {
      mutex_lock l(mu_);
      status_.Update(s);
      if (--num_pending_ == 0) {
        run_callback = true;
        status_copy = status_;
      }
    }
    if (run_callback) {
      done_(status_copy);
      // This is the last call, so delete the helper object.
      delete this;
    }
  }

 private:
  // A single request shared between all workers.
  CleanupGraphRequest req_;
  // One response buffer for each worker.
  gtl::InlinedVector<CleanupGraphResponse, 4> resps_;

  mutex mu_;
  // Number of requests remaining to be collected.
  int num_pending_ GUARDED_BY(mu_);
  // Aggregate status of the operation.
  Status status_ GUARDED_BY(mu_);
  // Callback to be called when all operations complete.
  StatusCallback done_;

  TF_DISALLOW_COPY_AND_ASSIGN(CleanupBroadcastHelper);
};

}  // namespace

void MasterSession::ReffedClientGraph::CleanupPartitionsAsync(
    int64 step_id, StatusCallback done) {
  const int num = partitions_.size();
  // Helper object will be deleted when the final call completes.
  CleanupBroadcastHelper* helper =
      new CleanupBroadcastHelper(step_id, num, std::move(done));
  for (int i = 0; i < num; ++i) {
    const Part& part = partitions_[i];
    part.worker->CleanupGraphAsync(
        helper->request(), helper->response(i),
        [helper, i](const Status& s) { helper->call_done(i, s); });
  }
}

void MasterSession::ReffedClientGraph::ProcessStats(
    const MasterEnv* env, int64 step_id, PerStepState* pss,
    SimpleGraphExecutionState* execution_state, ProfileHandler* ph,
    RunStepResponse* resp) {
  if (!pss->collect_costs && !pss->collect_timeline) return;

  // Out-of-band logging data is collected now, during post-processing.
  if (pss->collect_timeline) {
    SetRPCLogging(env, false);
    RetrieveLogs(env, step_id, &pss->rpc_stats);
  }
  for (size_t i = 0; i < partitions_.size(); ++i) {
    const StepStats& ss = pss->step_stats[i];
    if (pss->collect_costs) {
      execution_state->UpdateCostsFromStats(ss);
    }
    if (ph) {
      for (const auto& ds : ss.dev_stats()) {
        ProcessDeviceStats(ph, execution_state, ds, false /*is_rpc*/);
      }
    }
  }
  if (ph) {
    for (const auto& ds : pss->rpc_stats.dev_stats()) {
      ProcessDeviceStats(ph, execution_state, ds, true /*is_rpc*/);
    }
    ph->StepDone(pss->start_micros, pss->end_micros,
                 Microseconds(0) /*cleanup_time*/, 0 /*total_runops*/,
                 Status::OK());
  }
  // Assemble all stats for this timeline into a merged StepStats.
  StepStats step_stats_proto;
  if (pss->collect_timeline) {
    step_stats_proto = pss->rpc_stats;
    for (size_t i = 0; i < partitions_.size(); ++i) {
      const StepStats& ss = pss->step_stats[i];
      step_stats_proto.MergeFrom(ss);
    }
    stats_publisher_->PublishStatsProto(step_stats_proto);
    // Copy the stats back, but only for on-demand profiling to avoid slowing
    // down calls that trigger the automatic profiling.
    if (session_opts_.config.graph_options().timeline_step() <= 0) {
      resp->mutable_metadata()->mutable_step_stats()->Swap(&step_stats_proto);
    }
  }
}

void MasterSession::ReffedClientGraph::ProcessDeviceStats(
    ProfileHandler* ph, SimpleGraphExecutionState* execution_state,
    const DeviceStepStats& ds, bool is_rpc) {
  const string& dev_name = ds.device();
  VLOG(1) << "Device " << dev_name << " reports stats for "
          << ds.node_stats_size() << " nodes";
  for (const auto& ns : ds.node_stats()) {
    if (is_rpc) {
      // We don't have access to a good Node pointer, so we rely on
      // sufficient data being present in the NodeExecStats.
      ph->RecordOneOp(dev_name, ns, true /*is_copy*/, "", ns.node_name(),
                      ns.timeline_label());
    } else {
      NodeDef ndef;
      Status s = execution_state->GlobalNodeDefByName(ns.node_name(), &ndef);
      const bool found_node_in_graph = s.ok();
      if (!found_node_in_graph && ns.timeline_label().empty()) {
        // The counter incrementing is not thread-safe. But we don't really
        // care.
        // TODO(zhengxq): we should implement a LOG_FIRST_N and LOG_EVERY_N for
        // more general usage.
        static int log_counter = 0;
        if (log_counter < 10) {
          log_counter++;
          LOG(WARNING) << "Failed to find node " << ns.node_name()
                       << " for dev " << dev_name;
        }
        continue;
      }
      string optype = found_node_in_graph ? ndef.op() : ns.node_name();
      string details;
      if (!ns.timeline_label().empty()) {
        details = ns.timeline_label();
      } else if (found_node_in_graph) {
        details = DetailText(ndef, ns);
      } else {
        // Leave details string empty
      }
      ph->RecordOneOp(dev_name, ns, false /*is_copy*/, ns.node_name(), optype,
                      details);
    }
  }
}

// Asynchronously deregisters subgraphs on the workers, without waiting for the
// result.
void MasterSession::ReffedClientGraph::DeregisterPartitions() {
  struct Call {
    DeregisterGraphRequest req;
    DeregisterGraphResponse resp;
  };
  for (Part& part : partitions_) {
    Call* c = new Call;
    c->req.set_graph_handle(part.graph_handle);
    WorkerInterface* w = part.worker;
    auto cb = [c, w](const Status& s) {
      if (!s.ok()) {
        // This error is potentially benign, so we don't log at the
        // error level.
        LOG(INFO) << "DeregisterGraph error: " << s;
      }
      delete c;
      delete w;
    };
    w->DeregisterGraphAsync(&c->req, &c->resp, cb);
  }
}

void BuildBuildGraphOptions(const RunStepRequest& req,
                            BuildGraphOptions* opts) {
  for (const auto& feed : req.feed()) {
    opts->feed_endpoints.push_back(feed.name());
  }
  for (const auto& fetch : req.fetch()) {
    opts->fetch_endpoints.push_back(fetch);
  }
  for (const auto& target : req.target()) {
    opts->target_nodes.push_back(target);
  }

  std::sort(opts->feed_endpoints.begin(), opts->feed_endpoints.end());
  std::sort(opts->target_nodes.begin(), opts->target_nodes.end());
  std::sort(opts->fetch_endpoints.begin(), opts->fetch_endpoints.end());
}

uint64 HashBuildGraphOptions(const BuildGraphOptions& opts) {
  uint64 h = 0x2b992ddfa23249d6ull;
  for (const string& name : opts.feed_endpoints) {
    h = Hash64(name.c_str(), name.size(), h);
  }
  for (const string& name : opts.target_nodes) {
    h = Hash64(name.c_str(), name.size(), h);
  }
  for (const string& name : opts.fetch_endpoints) {
    h = Hash64(name.c_str(), name.size(), h);
  }
  return h;
}

string BuildGraphOptionsString(const BuildGraphOptions& opts) {
  string buf;
  for (const string& name : opts.feed_endpoints) {
    strings::StrAppend(&buf, " FdE: ", name);
  }
  strings::StrAppend(&buf, "\n");
  for (const string& name : opts.target_nodes) {
    strings::StrAppend(&buf, " TN: ", name);
  }
  strings::StrAppend(&buf, "\n");
  for (const string& name : opts.fetch_endpoints) {
    strings::StrAppend(&buf, " FeE: ", name);
  }
  strings::StrAppend(&buf, "\n");
  return buf;
}

MasterSession::MasterSession(const SessionOptions& opt, const MasterEnv* env,
                             std::vector<Device*>* remote_devs,
                             StatsPublisherFactory stats_publisher_factory)
    : session_opts_(opt),
      env_(env),
      handle_(strings::FpToString(random::New64())),
      stats_publisher_factory_(std::move(stats_publisher_factory)),
      graph_version_(0),
      runs_(5),
      cancellation_manager_(new CancellationManager) {
  UpdateLastAccessTime();

  swap(remote_devs_, *remote_devs);
  VLOG(1) << "Session " << handle_ << " #local " << env->local_devices.size()
          << " #remote " << remote_devs_.size();
  for (Device* d : remote_devs_) {
    devices_.AddDevice(d);
  }
  int num_local_devices = 0;
  for (Device* d : env->local_devices) {
    devices_.AddDevice(d);
    if (num_local_devices == 0) {
      // Uses the first local device as the client device.
      devices_.set_client_device(d);
    }
    num_local_devices++;
  }
  LOG(INFO) << "Start master session " << handle_
            << " with config: " << std::endl
            << session_opts_.config.DebugString();
}

MasterSession::~MasterSession() {
  delete cancellation_manager_;
  for (const auto& iter : runs_) iter.second->Unref();
  for (const auto& iter : obsolete_) iter.second->Unref();
  for (Device* dev : remote_devs_) delete dev;
}

void MasterSession::UpdateLastAccessTime() {
  last_access_time_usec_.store(Env::Default()->NowMicros());
}

Status MasterSession::Create(GraphDef* graph_def) {
  if (session_opts_.config.graph_options().place_pruned_graph()) {
    // TODO(b/29900832): Fix this or remove the option.
    LOG(WARNING) << "Distributed session does not support the "
                    "place_pruned_graph option.";
  }

  SimpleGraphExecutionStateOptions options;
  options.device_set = &devices_;
  options.session_options = &session_opts_;
  execution_state_.reset(
      new SimpleGraphExecutionState(graph_def->library(), options));
  TF_RETURN_IF_ERROR(execution_state_->Create(graph_def));

  return Status::OK();
}

Status MasterSession::Extend(const ExtendSessionRequest* req,
                             ExtendSessionResponse* resp) {
  UpdateLastAccessTime();
  std::unique_ptr<SimpleGraphExecutionState> extended_execution_state;
  {
    mutex_lock l(mu_);
    // TODO(mrry): Redesign the locking with reader/writer locks to prevent
    //   starvation due to concurrent steps being issued. This is not
    //   immediately important because we expect Extend to be used in
    //   development/interactive exploration, and not during high-throughput
    //   training.
    while (num_running_ != 0) {
      num_running_is_zero_.wait(l);
    }

    if (graph_version_ != req->current_graph_version()) {
      return errors::Aborted("Current version is ", graph_version_,
                             " but caller expected ",
                             req->current_graph_version(), ".");
    }

    CHECK(execution_state_);
    TF_RETURN_IF_ERROR(
        execution_state_->Extend(req->graph_def(), &extended_execution_state));

    CHECK(extended_execution_state);
    // The old execution state will be released outside the lock.
    execution_state_.swap(extended_execution_state);
    ++graph_version_;
    resp->set_new_graph_version(graph_version_);
  }
  return Status::OK();
}

Status MasterSession::StartStep(const RunStepRequest& req,
                                BuildGraphOptions* opts, int64* count,
                                ReffedClientGraph** rcg) {
  BuildBuildGraphOptions(req, opts);
  const uint64 hash = HashBuildGraphOptions(*opts);
  ReffedClientGraph* to_unref = nullptr;
  {
    mutex_lock l(mu_);
    // Keep track of how many times this subgraph has been executed in
    // this session.
    int64* c = &subgraph_execution_counts_[hash];
    *count = (*c)++;
    auto iter = runs_.find(hash);
    if (iter == runs_.end()) {
      // We have not seen this subgraph before. Build the subgraph and
      // cache it.
      VLOG(1) << "Unseen hash " << hash << " for "
              << BuildGraphOptionsString(*opts);
      std::unique_ptr<SimpleClientGraph> client_graph;
      TF_RETURN_IF_ERROR(execution_state_->BuildGraph(*opts, &client_graph));
      auto entry =
          new ReffedClientGraph(handle_, *opts, std::move(client_graph),
                                session_opts_, stats_publisher_factory_);
      iter = runs_.insert({hash, entry}).first;
      auto obs_iter = obsolete_.find(hash);
      if (obs_iter != obsolete_.end()) {
        to_unref = obs_iter->second;
        obsolete_.erase(obs_iter);
      }
      VLOG(1) << "Preparing to execute new graph";
    }
    *rcg = iter->second;
    (*rcg)->Ref();
  }
  if (to_unref) to_unref->Unref();
  return Status::OK();
}

void MasterSession::ClearRunsTable(std::vector<ReffedClientGraph*>* to_unref,
                                   RCGMap* rcg_map) {
  VLOG(1) << "Discarding all reffed graphs";
  for (auto p : *rcg_map) {
    ReffedClientGraph* rcg = p.second;
    if (to_unref) {
      to_unref->push_back(rcg);
    } else {
      rcg->Unref();
    }
  }
  rcg_map->clear();
}

Status MasterSession::Run(CallOptions* opts, const RunStepRequest* req,
                          RunStepResponse* resp) {
  UpdateLastAccessTime();
  {
    mutex_lock l(mu_);
    ++num_running_;
  }
  Status status = DoRunWithLocalExecution(opts, req, resp);
  {
    mutex_lock l(mu_);
    --num_running_;
    if (num_running_ == 0) {
      num_running_is_zero_.notify_all();
    }
  }
  return status;
}

Status MasterSession::DoRunWithLocalExecution(CallOptions* opts,
                                              const RunStepRequest* req,
                                              RunStepResponse* resp) {
  VLOG(2) << "DoRunWithLocalExecution "
          << "req: " << req->DebugString();
  PerStepState pss;
  pss.start_micros = Env::Default()->NowMicros();

  // Prepare.
  BuildGraphOptions bgopts;
  ReffedClientGraph* rcg = nullptr;
  int64 count = 0;
  TF_RETURN_IF_ERROR(StartStep(*req, &bgopts, &count, &rcg));

  // Unref "rcg" when out of scope.
  core::ScopedUnref unref(rcg);

  // Registers subgraphs if haven't done so.
  PartitionOptions popts;
  popts.node_to_loc = SplitByWorker;
  popts.new_name = [this](const string& prefix) {
    mutex_lock l(mu_);
    return strings::StrCat(prefix, "_S", next_node_id_++);
  };
  popts.get_incarnation = [this](const string& name) -> int64 {
    Device* d = devices_.FindDeviceByName(name);
    if (d == nullptr) {
      return PartitionOptions::kIllegalIncarnation;
    } else {
      return d->attributes().incarnation();
    }
  };
  popts.control_flow_added = false;
  const bool enable_bfloat16_sendrecv =
      session_opts_.config.graph_options().enable_bfloat16_sendrecv();
  popts.should_cast = [enable_bfloat16_sendrecv](const Edge* e) {
    if (e->IsControlEdge()) {
      return DT_FLOAT;
    }
    DataType dtype = BaseType(e->src()->output_type(e->src_output()));
    if (enable_bfloat16_sendrecv && dtype == DT_FLOAT) {
      return DT_BFLOAT16;
    } else {
      return dtype;
    }
  };
  if (session_opts_.config.graph_options().enable_recv_scheduling()) {
    popts.scheduling_for_recvs = true;
    popts.need_to_record_start_times = true;
  }

  TF_RETURN_IF_ERROR(rcg->RegisterPartitions(
      env_, popts, rcg->client_graph()->flib_def->ToProto()));

  // Keeps the highest 8 bits 0x01: we reserve some bits of the
  // step_id for future use.
  const uint64 step_id = (random::New64() & ((1uLL << 56) - 1)) | (1uLL << 56);
  TRACEPRINTF("stepid %llu", step_id);

  std::unique_ptr<ProfileHandler> ph;
  pss.collect_timeline = req->options().trace_level() == RunOptions::FULL_TRACE;
  pss.collect_costs = (0 == (count % CostFrequency(count)));
  ph = rcg->GetProfileHandler(step_id, count, req->options());
  if (ph) {
    pss.collect_timeline = true;
    pss.collect_rpcs = true;
  }

  TF_RETURN_IF_ERROR(rcg->RunPartitions(env_, step_id, count,
                                        execution_state_.get(), &pss, opts,
                                        *req, resp, cancellation_manager_));

  pss.end_micros = Env::Default()->NowMicros();

  // Schedule post-processing and cleanup to be done asynchronously.
  rcg->Ref();
  rcg->ProcessStats(env_, step_id, &pss, execution_state_.get(), ph.get(),
                    resp);
  rcg->CleanupPartitionsAsync(step_id, [rcg](const Status& s) {
    if (!s.ok()) {
      LOG(ERROR) << "Cleanup partition error: " << s;
    }
    rcg->Unref();
  });
  return Status::OK();
}

Status MasterSession::Close() {
  cancellation_manager_->StartCancel();
  std::vector<ReffedClientGraph*> to_unref;
  {
    mutex_lock l(mu_);
    while (num_running_ != 0) {
      num_running_is_zero_.wait(l);
    }
    ClearRunsTable(&to_unref, &runs_);
    ClearRunsTable(&to_unref, &obsolete_);
  }
  for (ReffedClientGraph* rcg : to_unref) rcg->Unref();
  delete this;
  return Status::OK();
}

}  // end namespace

namespace internal {

MasterSessionInterface* NewMasterSession(
    const SessionOptions& options, const MasterEnv* env,
    std::vector<Device*>* remote_devs,
    StatsPublisherFactory stats_publisher_factory) {
  return new MasterSession(options, env, remote_devs, stats_publisher_factory);
}

}  // end namespace internal
}  // end namespace tensorflow
