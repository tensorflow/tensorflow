#include "tensorflow/core/common_runtime/direct_session.h"

#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/common_runtime/simple_placer.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

namespace {

thread::ThreadPool* kernel_thread_pool_ = nullptr;
static bool InitModule(const SessionOptions& options) {
  int32 inter_op_parallelism_threads =
      options.config.inter_op_parallelism_threads();
  if (inter_op_parallelism_threads == 0) {
    // Default to using the number of cores available in the process.
    inter_op_parallelism_threads = port::NumSchedulableCPUs();
  }
  LOG(INFO) << "Direct session inter op parallelism threads: "
            << inter_op_parallelism_threads;
  kernel_thread_pool_ = new thread::ThreadPool(options.env, "Compute",
                                               inter_op_parallelism_threads);
  return true;
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

// NOTE: On Android with a single device, there is never
// a risk of an OpKernel blocking indefinitely:
//
// 1) No operations do I/O that depends on other simultaneous kernels,
//
// 2) Recv nodes always complete immediately: The inputs are sent into
//    the local rendezvous before we start the executor, so the
//    corresonding recvs will not block.
//
// Based on these assumptions, we can use the same thread pool for
// both "non-blocking" and "blocking" OpKernels on Android.
//
// This may change down the road when we add support for multiple
// devices that run concurrently, in which case we will need to
// revisit this decision.
void SchedClosure(std::function<void()> c) {
// TODO(sanjay): Get rid of __ANDROID__ path
#ifdef __ANDROID__
  // On Android, there is no implementation of ThreadPool that takes
  // std::function, only Closure, which we cannot easily convert.
  //
  // Instead, we just run the function in-line, which is currently
  // safe given the reasoning above.
  c();
#else
  kernel_thread_pool_->Schedule(c);
#endif  // __ANDROID__
}

}  // namespace

DirectSession::DirectSession(const SessionOptions& options,
                             const DeviceMgr* device_mgr)
    : options_(options),
      device_mgr_(device_mgr),
      cancellation_manager_(new CancellationManager()) {
  static bool init = InitModule(options);
  CHECK(init);  // Avoids compiler warning that init is unused.
  session_handle_ = strings::FpToString(random::New64());
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
  for (auto d : device_mgr_->ListDevices()) {
    d->op_segment()->RemoveHold(session_handle_);
  }
  for (auto it : executors_) {
    delete it.second;
  }
  delete cancellation_manager_;
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
  graph_created_ = true;  // In case this is first call
  graph_def_.MergeFrom(graph);
  return Status::OK();
}

Status DirectSession::Run(const std::vector<std::pair<string, Tensor>>& inputs,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          std::vector<Tensor>* outputs) {
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
  Status s = GetOrCreateExecutors(input_tensor_names, output_names,
                                  target_nodes, &executors_and_keys);
  if (!s.ok()) {
    return s;
  }

  IntraProcessRendezvous* rendez =
      new IntraProcessRendezvous(device_mgr_.get());
  core::ScopedUnref rendez_unref(rendez);

  // Insert the input tensors into the local rendezvous by their
  // rendezvous key.
  for (const auto& input : inputs) {
    const string& input_key = executors_and_keys->input_keys[input.first];
    s = rendez->Send(input_key, Rendezvous::Args(), input.second, false);
    if (!s.ok()) {
      rendez->StartAbort(s);
      return s;
    }
  }

  // Start parallel Executors.
  Notification executors_done;
  const int num_executors = executors_and_keys->device_executors.size();
  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_executors, rendez, [&executors_done, &s](const Status& ret) {
        s = ret;
        executors_done.Notify();
      });

  Executor::Args args;
  args.rendezvous = rendez;
  args.cancellation_manager = cancellation_manager_;
  args.runner = SchedClosure;

  for (auto device_executor : executors_and_keys->device_executors) {
    Executor* exec = device_executor.second;
    exec->RunAsync(args, barrier->Get());
  }

  executors_done.WaitForNotification();

  TF_RETURN_IF_ERROR(s);

  if (!output_names.empty()) {
    outputs->resize(output_names.size());
  }

  // Get the outputs from the rendezvous
  for (size_t output_offset = 0; output_offset < output_names.size();
       ++output_offset) {
    const string& output_key =
        executors_and_keys->output_keys[output_names[output_offset]];
    Tensor output_tensor;
    bool is_dead;

    // Fetch data from the Rendezvous.
    s = rendez->Recv(output_key, Rendezvous::Args(), &output_tensor, &is_dead);
    if (is_dead) {
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

  return s;
}

Status DirectSession::GetOrCreateExecutors(
    gtl::ArraySlice<string> inputs, gtl::ArraySlice<string> outputs,
    gtl::ArraySlice<string> target_nodes,
    ExecutorsAndKeys** executors_and_keys) {
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
  std::unordered_map<string, Graph*> graphs;
  Status s = CreateGraphs(inputs, outputs, target_nodes, &graphs);
  if (!s.ok()) {
    return s;
  }

  bool has_control_flow = false;
  for (const auto& graph : graphs) {
    for (const Node* n : graph.second->nodes()) {
      if (IsControlFlow(n)) {
        has_control_flow = true;
        break;
      }
    }
    if (has_control_flow) break;
  }

  std::unique_ptr<ExecutorsAndKeys> ek(new ExecutorsAndKeys);

  for (const auto& graph : graphs) {
    const string& partition_name = graph.first;
    Graph* partition_graph = graph.second;

    Device* d;
    s = device_mgr_->LookupDevice(partition_name, &d);
    if (!s.ok()) {
      return s;
    }

    LocalExecutorParams params;
    params.has_control_flow = has_control_flow;
    params.device = d;
    params.create_kernel = [this, d](const NodeDef& ndef, OpKernel** kernel) {
      return CreateCachedKernel(d, session_handle_, nullptr, ndef, kernel);
    };
    params.delete_kernel = [this, d](OpKernel* kernel) {
      DeleteCachedKernel(d, session_handle_, kernel);
    };

    Executor* tmp_exec;
    s = NewLocalExecutor(params, partition_graph, &tmp_exec);
    if (!s.ok()) {
      return s;
    }
    ek->device_executors.insert(std::make_pair(graph.first, tmp_exec));
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

Status DirectSession::CreateGraphs(
    gtl::ArraySlice<string> feeds, gtl::ArraySlice<string> fetches,
    gtl::ArraySlice<string> target_nodes,
    std::unordered_map<string, Graph*>* outputs) {
  Graph graph(OpRegistry::Global());
  GraphConstructorOptions opts;

  {
    mutex_lock l(graph_def_lock_);
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, graph_def_, &graph));
  }

  TF_RETURN_IF_ERROR(subgraph::RewriteGraphForExecution(
      &graph, feeds, fetches, target_nodes,
      device_set_.client_device()->attributes()));

  // Run the simple placer after rewriting the graph.
  std::unordered_map<string, int32> node_name_to_cost_map;
  for (Node* n : graph.nodes()) {
    node_name_to_cost_map[n->name()] = n->cost_id();
  }
  SimplePlacer placer(&graph, &device_set_, &node_name_to_cost_map, &options_);

  {
    mutex_lock l(mu_);
    // Restore stateful nodes.
    RestoreStatefulNodes(&graph);
    TF_RETURN_IF_ERROR(placer.Run());
    // Save stateful nodes.
    SaveStatefulNodes(&graph);
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
  TF_RETURN_IF_ERROR(Partition(popts, &graph, &partitions));

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

  for (const auto& partition : partitions) {
    const string& partition_name = partition.first;

    const GraphDef& graph_def = partition.second;
    VLOG(2) << "Created " << graph_def.DebugString() << " for "
            << partition_name;

    Graph* device_graph = new Graph(OpRegistry::Global());
    GraphConstructorOptions device_opts;
    // There are internal operations (e.g., send/recv) that we now
    // allow.
    device_opts.allow_internal_ops = true;
    device_opts.expect_device_spec = true;
    Status s = ConvertGraphDefToGraph(device_opts, graph_def, device_graph);
    if (!s.ok()) {
      delete device_graph;
      // Also delete other graphs created during the loop.
      gtl::STLDeleteValues(outputs);
      return s;
    }
    outputs->insert(std::make_pair(partition_name, device_graph));
  }

  return Status::OK();
}

::tensorflow::Status DirectSession::Close() {
  cancellation_manager_->StartCancel();
  return ::tensorflow::Status::OK();
}

class DirectSessionFactory : public SessionFactory {
 public:
  DirectSessionFactory() {}

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
