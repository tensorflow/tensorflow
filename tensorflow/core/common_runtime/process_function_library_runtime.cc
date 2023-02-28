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
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "tensorflow/core/common_runtime/build_graph_options.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/function_optimization_registry.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/optimize_function_graph_utils.h"
#include "tensorflow/core/common_runtime/partitioning_utils.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/common_runtime/rendezvous_util.h"
#include "tensorflow/core/common_runtime/replicate_per_replica_nodes.h"
#include "tensorflow/core/common_runtime/single_threaded_executor.h"
#include "tensorflow/core/common_runtime/stats_publisher_interface.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/reffed_status_callback.h"
#include "tensorflow/tsl/platform/statusor.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/protobuf/remote_tensor_handle.pb.h"
#endif  // IS_MOBILE_PLATFORM

namespace tensorflow {

const char ProcessFunctionLibraryRuntime::kDefaultFLRDevice[] = "null";

void ProcessFunctionLibraryRuntime::FunctionData::DistributedInit(
    DistributedFunctionLibraryRuntime* parent, const string& function_name,
    const FunctionLibraryDefinition& lib_def, AttrSlice attrs,
    const FunctionLibraryRuntime::InstantiateOptions& options,
    FunctionLibraryRuntime::DoneCallback done) {
  {
    mutex_lock l(mu_);
    is_cross_process_ = true;
    if (init_started_) {
      init_done_.WaitForNotification();
      done(init_result_);
      return;
    }
    init_started_ = true;
  }
  parent->Instantiate(function_name, lib_def, attrs, options, &local_handle_,
                      [this, done](const Status& s) {
                        init_done_.Notify();
                        done(s);
                      });
}

ProcessFunctionLibraryRuntime::ProcessFunctionLibraryRuntime(
    const DeviceMgr* device_mgr, Env* env, const ConfigProto* config,
    int graph_def_version, const FunctionLibraryDefinition* lib_def,
    const OptimizerOptions& optimizer_options,
    thread::ThreadPool* default_thread_pool,
    DistributedFunctionLibraryRuntime* parent,
    const SessionMetadata* session_metadata,
    Rendezvous::Factory rendezvous_factory,
    StatsPublisherFactory stats_publisher_factory, int32 stream_id)
    : parent_(parent),
      env_(env),
      config_(config ? std::make_optional(*config) : std::nullopt),
      device_mgr_(device_mgr),
      lib_def_(lib_def),
      default_thread_pool_(default_thread_pool),
      flr_map_(new std::unordered_map<Device*,
                                      std::unique_ptr<FunctionLibraryRuntime>>),
      next_handle_(0),
      session_metadata_(session_metadata),
      rendezvous_factory_(std::move(rendezvous_factory)),
      optimizer_options_(optimizer_options),
      graph_def_version_(graph_def_version),
      stats_publisher_factory_(std::move(stats_publisher_factory)) {
  if (device_mgr == nullptr) {
    (*flr_map_)[nullptr] = NewFunctionLibraryRuntime(
        nullptr, env, config_ ? &(*config_) : nullptr, nullptr,
        graph_def_version, lib_def_, default_thread_pool, optimizer_options,
        session_metadata_, this);
    return;
  }
  InitializeDeviceAndFlr(stream_id);
}

/* static */
Status ProcessFunctionLibraryRuntime::SendTensors(
    const string& source_device, const string& target_device,
    const string& key_prefix, int64_t src_incarnation,
    gtl::ArraySlice<Tensor> tensors_to_send, DeviceContext* device_context,
    const std::vector<AllocatorAttributes>& alloc_attrs,
    RendezvousInterface* rendezvous) {
  std::vector<string> keys;
  for (int i = 0; i < tensors_to_send.size(); ++i) {
    string name = strings::StrCat(key_prefix, i);
    string key = Rendezvous::CreateKey(source_device, src_incarnation,
                                       target_device, name, FrameAndIter(0, 0));
    keys.push_back(key);
  }
  TF_RETURN_IF_ERROR(SendTensorsToRendezvous(
      rendezvous, device_context, alloc_attrs, keys, tensors_to_send));
  return OkStatus();
}

/* static */
void ProcessFunctionLibraryRuntime::ReceiveTensorsAsync(
    const string& source_device, const string& target_device,
    const string& key_prefix, int64_t src_incarnation, int64_t num_tensors,
    DeviceContext* device_context,
    const std::vector<AllocatorAttributes>& alloc_attrs,
    RendezvousInterface* rendezvous, std::vector<Tensor>* received_tensors,
    StatusCallback done) {
  std::vector<string> keys;
  for (int64_t i = 0; i < num_tensors; ++i) {
    string name = strings::StrCat(key_prefix, i);
    string key = Rendezvous::CreateKey(source_device, src_incarnation,
                                       target_device, name, FrameAndIter(0, 0));
    keys.push_back(key);
  }
  RecvOutputsFromRendezvousAsync(rendezvous, device_context, alloc_attrs, keys,
                                 received_tensors, std::move(done));
}

Status ProcessFunctionLibraryRuntime::GetRetTypes(
    FunctionLibraryRuntime::Handle h, DataTypeVector* ret_types) {
  FunctionLibraryRuntime* flr = nullptr;
  {
    tf_shared_lock l(mu_);
    auto miter = mdevice_data_.find(h);
    if (miter != mdevice_data_.end()) {
      *ret_types = miter->second->ret_types_;
      return OkStatus();
    }
    auto fiter = function_data_.find(h);
    if (fiter != function_data_.end()) {
      flr = GetFLR(fiter->second->target_device());
    }
  }
  if (flr != nullptr) {
    return flr->GetRetTypes(h, ret_types);
  }
  return errors::InvalidArgument("Handle ", h, " not found.");
}

Status ProcessFunctionLibraryRuntime::GetDeviceIncarnation(
    const string& device_name, int64_t* incarnation) const {
  FunctionLibraryRuntime* flr = GetFLR(device_name);
  if (flr == nullptr) {
    return errors::InvalidArgument("Device name: ", device_name, " not found.");
  }
  *incarnation = flr->device()->attributes().incarnation();
  return OkStatus();
}

Status ProcessFunctionLibraryRuntime::GetDeviceContext(
    const string& device_name, DeviceContext** device_context) const {
  *device_context = nullptr;
  FunctionLibraryRuntime* flr = GetFLR(device_name);
  if (flr == nullptr) {
    return errors::InvalidArgument("Device name: ", device_name, " not found.");
  }
  Device* device = flr->device();
  string device_type = device->parsed_name().type;
  if (device_type == "CPU" || device_type == "TPU_SYSTEM") {
    // "TPU_SYSTEM" indicates that `device` is a CPU.
    return OkStatus();
  }

  if (device->IsRemoteCallAllowed()) {
    auto* dev_info = flr->device()->tensorflow_accelerator_device_info();
    if (dev_info) {
      *device_context = dev_info->default_context;
      return OkStatus();
    }
  }

  return errors::Internal("Device type: ", device_type,
                          " is currently unsupported for remote ",
                          "function executions");
}

void ProcessFunctionLibraryRuntime::InitializeDeviceAndFlr(int32 stream_id) {
  // Reset device_set_ by one of the two following scenarios:
  // 1) Both cluster-FLR and its remote_device_mgr is available: include local
  //    devices (if any) from the local device_mgr_ as Device type, and include
  //    remote devices from cluster's remote_device_mgr as RemoteDevice type.
  // 2) Include local devices from the local device_mgr_.
  // In both scenarios, no device is added more than one times.
  mutex_lock l(mu_);
  device_set_ = std::make_shared<DeviceSet>();
  if (parent_ != nullptr && parent_->remote_device_mgr() != nullptr) {
    for (auto d : parent_->remote_device_mgr()->ListDevices()) {
      Device* device = nullptr;
      if (device_mgr_->LookupDevice(d->name(), &device) == OkStatus()) {
        // If this device exists in device_mgr, i.e., a local device,
        // add this device from the instance included in device_mgr_
        device_set_->AddDevice(device);
      } else {
        device_set_->AddDevice(d);
      }
    }
  } else {
    for (auto d : device_mgr_->ListDevices()) {
      device_set_->AddDevice(d);
    }
  }

  // Update flr_map_ by adding new devices
  for (Device* d : device_mgr_->ListDevices()) {
    if ((*flr_map_)[d] == nullptr) {
      (*flr_map_)[d] = NewFunctionLibraryRuntime(
          device_mgr_, env_, config_ ? &(*config_) : nullptr,
          device_mgr_->LookupStream(d, stream_id), graph_def_version_, lib_def_,
          default_thread_pool_, optimizer_options_, session_metadata_, this);
    }
  }
}

FunctionLibraryRuntime* ProcessFunctionLibraryRuntime::GetFLR(
    const string& device_name) const {
  Device* device = nullptr;
  if (device_name != kDefaultFLRDevice) {
    if (!device_mgr_->LookupDevice(device_name, &device).ok()) {
      VLOG(4) << "Could not find device: " << device_name;
      return nullptr;
    }
  }
  const auto& iter = flr_map_->find(device);
  if (iter == flr_map_->end()) {
    VLOG(1) << "Could not find device: " << device_name
            << "in the local process.";
    return nullptr;
  }
  return iter->second.get();
}

FunctionLibraryRuntime::Handle ProcessFunctionLibraryRuntime::AddHandle(
    const string& function_key, const string& device_name,
    FunctionLibraryRuntime::LocalHandle local_handle) {
  mutex_lock l(mu_);
  return AddHandleLocked(function_key, device_name, local_handle);
}

FunctionLibraryRuntime::Handle ProcessFunctionLibraryRuntime::AddHandleLocked(
    const string& function_key, const string& device_name,
    FunctionLibraryRuntime::LocalHandle local_handle) {
  auto h = next_handle_;
  function_data_[h] =
      std::make_unique<FunctionData>(device_name, local_handle, function_key);
  table_[function_key] = h;
  next_handle_++;
  return h;
}

FunctionLibraryRuntime::Handle
ProcessFunctionLibraryRuntime::AddMultiDeviceHandle(
    std::unique_ptr<MultiDeviceFunctionData> data, const string& function_key) {
  mutex_lock l(mu_);
  auto h = next_handle_;
  mdevice_data_[h] = std::move(data);
  table_[function_key] = h;
  next_handle_++;
  return h;
}

bool ProcessFunctionLibraryRuntime::HasMultiDeviceHandle(
    FunctionLibraryRuntime::Handle handle) const {
  bool multi_device;
  {
    tf_shared_lock l(mu_);
    multi_device = mdevice_data_.find(handle) != mdevice_data_.end();
  }
  return multi_device;
}

FunctionLibraryRuntime::Handle ProcessFunctionLibraryRuntime::GetHandle(
    const string& function_key) const {
  tf_shared_lock l(mu_);
  return gtl::FindWithDefault(table_, function_key, kInvalidHandle);
}

FunctionLibraryRuntime::LocalHandle
ProcessFunctionLibraryRuntime::GetHandleOnDevice(
    const string& device_name, FunctionLibraryRuntime::Handle handle,
    bool include_multi_device) const {
  tf_shared_lock l(mu_);

  auto miter = mdevice_data_.find(handle);
  if (miter != mdevice_data_.end()) {
    if (!include_multi_device) return kInvalidLocalHandle;

    const MultiDeviceFunctionData& data = *miter->second;
    if (data.glue_.size() != 1) return kInvalidLocalHandle;

    const auto& pair = *data.glue_.begin();
    const string& func_device_name = pair.first;
    const ComponentFunctionData& component_data = pair.second;
    if (func_device_name != device_name) return kInvalidLocalHandle;

    // Replace the given handle with the handle for the single component
    // function.
    handle = component_data.handle;
  }

  auto iter = function_data_.find(handle);
  if (iter == function_data_.end()) {
    return kInvalidLocalHandle;
  }
  FunctionData* function_data = iter->second.get();
  if (function_data->target_device() != device_name) {
    return kInvalidLocalHandle;
  }
  return function_data->local_handle();
}

string ProcessFunctionLibraryRuntime::GetDeviceName(
    FunctionLibraryRuntime::Handle handle) const {
  tf_shared_lock l(mu_);
  auto iter = function_data_.find(handle);
  CHECK(iter != function_data_.end());
  FunctionData* function_data = iter->second.get();
  return function_data->target_device();
}

ProcessFunctionLibraryRuntime::MultiDeviceFunctionData*
ProcessFunctionLibraryRuntime::IsMultiDevice(
    FunctionLibraryRuntime::Handle handle) const {
  tf_shared_lock l(mu_);
  const auto& it = mdevice_data_.find(handle);
  if (it != mdevice_data_.end()) {
    return it->second.get();
  }
  return nullptr;
}

namespace {
// Returns the local tensors referred by `args`.
std::vector<Tensor> GetLocalArgs(gtl::ArraySlice<FunctionArg> args) {
  std::vector<Tensor> tensors;
  for (const auto& arg : args) {
    if (arg.index() == 0) {
      // NOLINTNEXTLINE
      tensors.push_back(absl::get<Tensor>(arg));
    }
  }
  return tensors;
}

// Update the done callback to push Tensors in `tensors` into `rets`.
FunctionLibraryRuntime::DoneCallback TensorsToFunctionRetsDoneCallback(
    std::vector<FunctionRet>* rets, std::vector<Tensor>* tensors,
    FunctionLibraryRuntime::DoneCallback done) {
  return [rets, tensors, done = std::move(done)](const Status& s) {
    if (s.ok()) {
      for (const auto& t : *tensors) {
        rets->push_back(t);
      }
    }
    delete tensors;
    done(s);
  };
}

// Push Tensors in `function_rets` into `tensors`.
Status FunctionRetsToTensors(const std::vector<FunctionRet>* function_rets,
                             std::vector<Tensor>* tensors) {
  for (const auto& ret : *function_rets) {
    if (ret.index() != 0) {
      return errors::Internal(
          "Expect a Tensor as a function output but got a TensorShape.");
    }
    // NOLINTNEXTLINE
    tensors->push_back(absl::get<Tensor>(ret));
  }
  return OkStatus();
}
}  // namespace

ProcessFunctionLibraryRuntime::AsyncAttributes::Summary
ProcessFunctionLibraryRuntime::AsyncAttributes::Summarize(const Graph* graph) {
  bool has_send_op = false;
  bool has_recv_op = false;
  bool has_unsafe_op = false;
  for (const Node* node : graph->nodes()) {
    if (node->IsSend() || node->IsHostSend()) {
      has_send_op = true;
    }
    if (node->IsRecv() || node->IsHostRecv()) {
      has_recv_op = true;
    }
    if (!ValidateOpIsSafeForSyncExecution(*node,
                                          allow_control_flow_sync_execution())
             .ok()) {
      has_unsafe_op = true;
    }
  }
  // (1) Anything completely unsupported?
  if (has_unsafe_op) {
    metrics::IncrementTestCounter("subgraph_async_summary", "unsafe_op");
    return AsyncAttributes::kAsyncRequired;
  }
  // (2) That only leaves send/recv.  If neither, then it's safe.
  if (!has_send_op && !has_recv_op) {
    metrics::IncrementTestCounter("subgraph_async_summary", "safe_for_sync");
    return AsyncAttributes::kSafeForSync;
  }
  // (3) If each subgraph has only send or only recv, then it's possible to
  // order them to run sequentially without deadlock.
  if (has_send_op && !has_recv_op) {
    metrics::IncrementTestCounter("subgraph_async_summary", "send_only");
    return AsyncAttributes::kSendOnly;
  }
  if (has_recv_op && !has_send_op) {
    metrics::IncrementTestCounter("subgraph_async_summary", "recv_only");
    return AsyncAttributes::kRecvOnly;
  }
  // Otherwise, assume it's unsupported.
  metrics::IncrementTestCounter("subgraph_async_summary", "other");
  return AsyncAttributes::kAsyncRequired;
}

void ProcessFunctionLibraryRuntime::PublishSubgraphs(
    const std::string& function_name,
    std::unique_ptr<std::unordered_map<std::string, std::unique_ptr<Graph>>>
        subgraphs) {
  // Use shared_ptr since std::function cannot capture move-only objects
  auto subgraphs_new =
      std::shared_ptr<std::unordered_map<std::string, std::unique_ptr<Graph>>>(
          subgraphs.release());
  auto completed = std::make_unique<tsl::Notification>();
  // Converting graphs to GraphDefs involves expensive copies. Delegate the work
  // to a separate thread to unblock the caller.
  std::function<void()> thread_fn = [this, function_name, n = completed.get(),
                                     subgraphs = subgraphs_new]() {
    std::unique_ptr<StatsPublisherInterface> stats_publisher =
        stats_publisher_factory_(function_name, BuildGraphOptions(),
                                 SessionOptions());
    std::vector<GraphDef> published_graph_defs;
    published_graph_defs.reserve(subgraphs->size());
    for (const auto& pair : *subgraphs) {
      Graph* subgraph = pair.second.get();
      GraphDef gd;
      subgraph->ToGraphDef(&gd);
      published_graph_defs.push_back(std::move(gd));
    }
    stats_publisher->PublishGraphProto(std::move(published_graph_defs));
    {
      mutex_lock l(mu_);
      stats_publishers_.push_back(std::move(stats_publisher));
    }
    n->Notify();
  };
  {
    mutex_lock l(mu_);
    stats_publisher_completed_.push_back(std::move(completed));
  }
  if (default_thread_pool_ != nullptr) {
    default_thread_pool_->Schedule(std::move(thread_fn));
  } else {
    env_->SchedClosure(std::move(thread_fn));
  }
}

Status ProcessFunctionLibraryRuntime::InstantiateMultiDevice(
    const string& function_name, AttrSlice attrs,
    const FunctionLibraryRuntime::InstantiateOptions& options,
    FunctionLibraryRuntime::Handle* handle) {
  // Check if this function has already been instantiated.
  const string& function_key = Canonicalize(function_name, attrs, options);

  {
    mutex_lock l(mu_);
    const auto& it = table_.find(function_key);
    if (it != table_.end()) {
      *handle = it->second;
      ++mdevice_data_[*handle]->instantiation_counter_;
      return OkStatus();
    }
  }

  VLOG(1) << "Instantiating MultiDevice function \"" << function_name
          << "\" on default device \"" << options.target << "\"";
  if (VLOG_IS_ON(3)) {
    int index = 0;
    VLOG(3) << "Requested input devices:";
    for (const string& device : options.input_devices) {
      VLOG(3) << "    [input " << index++ << "] " << device;
    }
    index = 0;
    VLOG(3) << "Requested output devices:";
    for (const string& device : options.output_devices) {
      VLOG(3) << "    [output " << index++ << "] " << device;
    }
  }

  const std::shared_ptr<DeviceSet> dev_set = device_set();
  // Get default device.
  Device* default_device = nullptr;
  if (options.default_device_to_target && !options.target.empty()) {
    // Make the `target` device the default device if nothing else is hard
    // coded. This allows the same function definition to be specialized to
    // different devices depending on the `PartitionedCallOp` device.
    FunctionLibraryRuntime* flr = GetFLR(options.target);
    if (flr == nullptr) {
      return errors::InvalidArgument(
          "Cannot instantiate multi-device function with target device ",
          options.target);
    }
    default_device = flr->device();
  }
  // Get composite devices.
  std::vector<CompositeDevice*> composite_devices;
  {
    tf_shared_lock l(mu_);
    for (auto* d : composite_devices_) composite_devices.push_back(d);
  }
  // Get cpu device.
  Device* cpu_device;
  TF_RETURN_IF_ERROR(device_mgr_->LookupDevice("CPU:0", &cpu_device));

  const uint64 optimization_start_time_usecs = Env::Default()->NowMicros();
  // Look up for optimized function graph in library. If found, skip
  // `OptimizeFunctionGraph` step.
  OptimizedFunctionGraph* optimized_graph_proto =
      options.lib_def != nullptr
          ? options.lib_def->FindOptimizedFunctionGraph(function_name)
          : lib_def_->FindOptimizedFunctionGraph(function_name);
  StatusOr<OptimizedFunctionGraphInfo> optimized_graph_info =
      optimized_graph_proto == nullptr
          ? OptimizeFunctionGraph(function_name, attrs, options, *dev_set,
                                  lib_def_, composite_devices, cpu_device,
                                  default_device, env_)
          : OptimizedFunctionGraphInfo::FromProto(*optimized_graph_proto);
  if (!optimized_graph_info.ok()) return optimized_graph_info.status();

  // Resets the library registration correctly.
  optimized_graph_info->function_graph->mutable_flib_def()
      ->set_default_registry(&(optimized_graph_info->lib_def));

  TF_ASSIGN_OR_RETURN(
      auto subgraphs,
      PreprocessAndPartitionGraph(*optimized_graph_info, options, *dev_set,
                                  lib_def_, composite_devices, env_));
  const uint64 optimization_end_time_usecs = Env::Default()->NowMicros();
  metrics::UpdateFunctionGraphOptimizationTime(optimization_end_time_usecs -
                                               optimization_start_time_usecs);
  VLOG(1) << "Finished graph optimizations for MultiDevice function \""
          << function_name << "\" with target device \"" << options.target
          << "\"";

  const FunctionLibraryDefinition* lib_def =
      options.lib_def == nullptr ? lib_def_ : options.lib_def;
  if (options.graph_collector != nullptr) {
    for (const auto& pair : *subgraphs) {
      GraphDef def;
      pair.second->ToGraphDef(&def);
      *def.mutable_library() = lib_def->ReachableDefinitions(def).ToProto();
      options.graph_collector->CollectPartitionedGraph(def);
    }
  }

  const auto& node_name_to_control_ret =
      optimized_graph_info->node_name_to_control_ret;
  // We must preserve control returns in each of the function components,
  // otherwise after function inlining we might prune side-effectful nodes.
  const auto control_ret =
      [&node_name_to_control_ret](const Node* n) -> absl::optional<string> {
    const auto it = node_name_to_control_ret.find(n->name());
    return it != node_name_to_control_ret.end()
               // NOLINTNEXTLINE
               ? absl::make_optional<string>(it->second)
               // NOLINTNEXTLINE
               : absl::nullopt;
  };

  auto data = std::make_unique<MultiDeviceFunctionData>(
      function_name, function_key, optimized_graph_info->num_return_nodes,
      std::move(optimized_graph_info->lib_def),
      std::move(optimized_graph_info->ret_types));

  int i = 0;
  // Generate a random function_name to avoid one function reuse the partition
  // function instantiated by another function.
  FunctionLibraryDefinition* data_lib_def = &data->lib_def_;
  FunctionNameGenerator name_generator(
      data_lib_def, absl::StrCat(function_name, "_", random::New64()));
  const int num_subgraphs = subgraphs->size();
  gtl::InlinedVector<Status, 4> instantiate_status(num_subgraphs);
  BlockingCounter counter(static_cast<int>(num_subgraphs));
  auto runner = [this, num_subgraphs](std::function<void()> fn) {
    // NOTE: Only use thread pool to instantiate sub-function when there are
    // more than 8 sub-functions. We want to avoid cost of switching thread when
    // there are only a few sub-functions.
    if (default_thread_pool_ != nullptr && num_subgraphs > 8) {
      default_thread_pool_->Schedule(fn);
    } else {
      fn();
    }
  };

  // Before instantiating component functions, determine synchronous execution.
  data->enable_sync_execution = false;
  if (options.allow_small_function_optimizations) {
    data->enable_sync_execution = true;
    for (const auto& pair : *subgraphs) {
      ComponentFunctionData* comp_data = &data->glue_[pair.first];
      const Graph* subgraph = pair.second.get();
      comp_data->async_attributes =
          AsyncAttributes(subgraph, options.allow_control_flow_sync_execution);
      if (comp_data->async_attributes.summary() ==
          AsyncAttributes::kAsyncRequired) {
        data->enable_sync_execution = false;
      }
    }
  }

  // Instantiate each component function (subgraph).
  for (const auto& pair : *subgraphs) {
    Status* status = &instantiate_status[i];
    string unique_name = name_generator.GetName();
    ComponentFunctionData* comp_data = &data->glue_[pair.first];
    runner([this, &pair, dev_set, comp_data, unique_name, data_lib_def,
            &control_ret, &options, status, &counter, &data] {
      const string& target = pair.first;

      const string& device_type =
          dev_set->FindDeviceByName(target)->device_type();
      Graph* subgraph = pair.second.get();

      bool ints_on_device =
          (device_type == "TPU" || device_type == "XLA_CPU" ||
           device_type == "XLA_GPU" || options.int_args_and_retvals_on_device);
      status->Update(UpdateArgAndRetvalMetadata(
          subgraph, &comp_data->arg_indices, &comp_data->ret_indices,
          &comp_data->arg_alloc_attrs, &comp_data->ret_alloc_attrs,
          ints_on_device));
      if (!status->ok()) {
        counter.DecrementCount();
        return;
      }
      FunctionDef shard;
      status->Update(
          GraphToFunctionDef(*subgraph, unique_name, control_ret, &shard));
      if (!status->ok()) {
        counter.DecrementCount();
        return;
      }
      status->Update(data_lib_def->AddFunctionDef(shard));
      if (!status->ok()) {
        counter.DecrementCount();
        return;
      }
      FunctionLibraryRuntime::InstantiateOptions opts;
      opts.executor_type = options.executor_type;
      opts.target = target;
      opts.lib_def = data_lib_def;
      opts.create_kernels_eagerly = options.create_kernels_eagerly;
      opts.state_handle = options.state_handle;
      opts.allow_small_function_optimizations = data->enable_sync_execution;
      opts.allow_control_flow_sync_execution =
          options.allow_control_flow_sync_execution;
      AttrValue ints_on_device_attr;
      ints_on_device_attr.set_b(options.int_args_and_retvals_on_device);
      shard.mutable_attr()->insert(
          {FunctionLibraryDefinition::kIntsOnDeviceAttr, ints_on_device_attr});
      auto attrs = AttrSlice(&shard.attr());
      VLOG(1) << "Start instantiating component function " << unique_name
              << " on device " << target;
      VLOG(4) << DebugString(shard);

      auto* component_handle = new FunctionLibraryRuntime::Handle;
      auto done = [this, status, unique_name, comp_data, component_handle,
                   &data, &counter](const Status& s) {
        status->Update(s);

        VLOG(1) << "Finished instantiating component function " << unique_name
                << " with handle " << *component_handle << " status: " << s;
        if (status->ok()) {
          {
            mutex_lock l(mu_);
            if (function_data_[*component_handle]->is_cross_process()) {
              data->is_cross_process_ = true;
            }
          }
          comp_data->handle = *component_handle;
        }
        delete component_handle;
        counter.DecrementCount();
      };

      FunctionLibraryRuntime* flr = GetFLR(opts.target);
      if (flr != nullptr) {
        // Initialize local function synchronously.
        Status s = flr->Instantiate(unique_name, attrs, opts, component_handle);
        done(s);
      } else {
        opts.ret_indices = comp_data->ret_indices;
        // Initialize remote function asynchronously.
        InstantiateRemote(unique_name, attrs, opts, component_handle, done);
      }
    });
    i += 1;
  }
  counter.Wait();
  StatusGroup group;
  for (auto& status : instantiate_status) {
    group.Update(status);
  }
  TF_RETURN_IF_ERROR(group.as_summary_status());

  *handle = AddMultiDeviceHandle(std::move(data), function_key);
  VLOG(1) << "Instantiated MultiDevice function \"" << function_name
          << "\" with handle " << *handle;

  PublishSubgraphs(function_name, std::move(subgraphs));
  return OkStatus();
}

Status ProcessFunctionLibraryRuntime::GetOutputDevices(
    FunctionLibraryRuntime::Handle handle,
    std::vector<Device*>* output_devices) const {
  MultiDeviceFunctionData* data = IsMultiDevice(handle);
  if (data == nullptr) {
    return errors::InvalidArgument(
        "Failed for find multi-device function handle ", handle);
  }

  for (const auto& pair : data->glue_) {
    const ComponentFunctionData& comp_data = pair.second;
    DCHECK(comp_data.ret_alloc_attrs.size() == comp_data.ret_indices.size());
    if (comp_data.ret_indices.empty()) {
      continue;
    }

    const string& target = pair.first;
    FunctionLibraryRuntime* target_flr = GetFLR(target);
    Device* target_device = nullptr;
    Device* host = nullptr;
    if (target_flr == nullptr) {
      if (!data->has_remote_outputs) {
        data->has_remote_outputs = true;
      }
      target_device = device_set()->FindDeviceByName(target);
      string remote_host;
      TF_RETURN_IF_ERROR(
          DeviceNameUtils::DeviceNameToCpuDeviceName(target, &remote_host));
      host = device_set()->FindDeviceByName(remote_host);
    } else {
      target_device = target_flr->device();
    }
    output_devices->resize(data->num_outputs_);
    for (int j = 0; j < comp_data.ret_indices.size(); ++j) {
      int ret_index = comp_data.ret_indices[j];
      if (data->ret_types_[ret_index] == DT_RESOURCE) {
        (*output_devices)[ret_index] = target_device;
      } else {
        (*output_devices)[ret_index] =
            comp_data.ret_alloc_attrs[j].on_host() ? host : target_device;
      }
    }
  }

  return OkStatus();
}

Status ProcessFunctionLibraryRuntime::PrepareRunMultiDevice(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle handle,
    const MultiDeviceFunctionData** data) const {
  if (opts.create_rendezvous) {
    // FLR->Run() is the default entry point. It checks for cancellation,
    // creates rendezvous, etc.
    // Letting create_rendezvous through will do the wrong thing - each
    // component function will get a separate rendezvous created by its FLR.
    return errors::Internal(
        "Cannot call ProcessFunctionLibraryRuntime::Run with "
        "create_rendezvous=true. Please run the function "
        "using FunctionLibraryRuntime::Run");
  }

  *data = IsMultiDevice(handle);
  if (*data == nullptr) {
    return errors::NotFound("Multi-device function handle ", handle,
                            "not found. Was the function instantiated?");
  }

  // Check whether we have the right rendezvous.
  if (opts.rendezvous && (*data)->is_cross_process_ &&
      !opts.rendezvous->is_cross_process()) {
    return errors::InvalidArgument(
        "Running a cross process function ", (*data)->function_name_,
        " without an appropriate cross process Rendezvous.");
  }

  return OkStatus();
}

std::vector<string> ProcessFunctionLibraryRuntime::GetOrderedSubgraphs(
    const MultiDeviceFunctionData* data) const {
  std::vector<string> subgraph_keys;
  subgraph_keys.reserve(data->glue_.size());
  for (const auto& pair : data->glue_) {
    subgraph_keys.push_back(pair.first);
  }
  auto send_first_ordering = [&](const string& a, const string& b) {
    auto a_summary = data->glue_.at(a).async_attributes.summary();
    auto b_summary = data->glue_.at(b).async_attributes.summary();
    if (a_summary == b_summary) {
      return false;
    }
    if (a_summary == AsyncAttributes::kSendOnly) {
      return true;
    }
    return false;
  };
  std::sort(subgraph_keys.begin(), subgraph_keys.end(), send_first_ordering);
  return subgraph_keys;
}

Status ProcessFunctionLibraryRuntime::RunMultiDeviceSync(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle outer_handle, std::vector<FunctionRet>* rets,
    std::function<Status(const ComponentFunctionData& comp_data,
                         InternalArgs* args)>
        get_component_args) const {
  const MultiDeviceFunctionData* data;
  Status prepare_status = PrepareRunMultiDevice(opts, outer_handle, &data);
  if (!prepare_status.ok()) {
    return prepare_status;
  }

  FunctionLibraryRuntime::Options opts_copy = opts;

  // Sort the subgraphs topologically before execution to avoid deadlock:
  //
  // Because subgraphs will not execute in parallel here, dependencies between
  // subgraphs cannot be resolved automatically. In contrast, with multi-
  // threaded execution, we launch all subgraphs at once, asynchronously, and
  // allow any to block mid-execution while its dependencies are resolved.
  //
  // In this synchronous execution path, currently supported ops with inter-
  // subgraph dependencies are send and receive.  As `_Send` and `_HostSend`
  // are non-blocking, we run subgraphs with those first, and those with
  // the blocking '_Recv' and '_HostRecv' ops will have their dependencies
  // resolved before execution.
  //
  // We assume that the partitioning has a valid deadlock-free ordering and the
  // safety of running synchronously has already been confirmed by this point.
  std::vector<string> subgraph_keys = GetOrderedSubgraphs(data);

  for (const string& target : subgraph_keys) {
    const ComponentFunctionData& comp_data = data->glue_.at(target);
    FunctionLibraryRuntime::Handle comp_handle = comp_data.handle;

    opts_copy.args_alloc_attrs = comp_data.arg_alloc_attrs;
    opts_copy.rets_alloc_attrs = comp_data.ret_alloc_attrs;

    InternalArgs comp_args;
    Status args_status = get_component_args(comp_data, &comp_args);
    if (!args_status.ok()) {
      VLOG(2) << "Failed to get component function arguments: " << args_status;
      return args_status;
    }
    rets->resize(data->num_outputs_);

    VLOG(1) << "Running component function on device " << target << " from "
            << data->function_name_ << " with handle " << comp_handle;
    FunctionLibraryRuntime* flr = GetFLR(target);
    if (flr != nullptr) {
      opts_copy.remote_execution = false;
      // When target device has private thread pool, use the target device
      // runner
      thread::ThreadPool* pool = flr->device()->tensorflow_device_thread_pool();
      opts_copy.runner = (pool == nullptr) ? opts.runner : flr->runner();
      VLOG(4) << "    with " << opts_copy.DebugString();

      std::vector<Tensor> comp_tensor_rets;
      Status run_status =
          flr->RunSync(opts_copy, comp_handle, GetLocalArgs(comp_args.args),
                       &comp_tensor_rets);
      if (!run_status.ok()) {
        VLOG(2) << "Component function execution failed: " << run_status;
        const string function_and_msg = strings::StrCat(
            errors::FormatFunctionForError(data->function_name_), " ",
            run_status.error_message());
        if (opts.rendezvous != nullptr) opts.rendezvous->StartAbort(run_status);
        return errors::CreateWithUpdatedMessage(run_status, function_and_msg);
      } else {
        VLOG(2) << "Component function execution succeeded.";
        for (int i = 0; i < comp_tensor_rets.size(); ++i) {
          (*rets)[comp_data.ret_indices[i]] = comp_tensor_rets[i];
        }
      }
    } else {
      // Fall back to DistributedFunctionLibraryRuntime for remote execution.
      opts_copy.remote_execution = true;
      VLOG(4) << "    with " << opts_copy.DebugString();

      std::vector<std::unique_ptr<CleanUpItem>> cleanup_items;
      Notification n;
      Status s;
      std::vector<FunctionRet> comp_rets;
      RunInternal(opts_copy, comp_handle, comp_args.args, &comp_rets,
                  &cleanup_items, [&n, &s](const Status& status) {
                    s.Update(status);
                    n.Notify();
                  });
      n.WaitForNotification();
      return s;
    }
  }
  return OkStatus();
}

void ProcessFunctionLibraryRuntime::RunMultiDeviceAsync(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle outer_handle, std::vector<FunctionRet>* rets,
    std::vector<std::unique_ptr<CleanUpItem>>* cleanup_items,
    FunctionLibraryRuntime::DoneCallback done,
    std::function<Status(const ComponentFunctionData& comp_data,
                         InternalArgs* args)>
        get_component_args) const {
  const MultiDeviceFunctionData* data;
  Status prepare_status = PrepareRunMultiDevice(opts, outer_handle, &data);
  if (!prepare_status.ok()) {
    done(prepare_status);
    return;
  }

  // A locally created cancellation manager, used only when the caller does not
  // provide one in argument.
  std::shared_ptr<CancellationManager> local_cm;
  CancellationManager* cm = opts.cancellation_manager;
  if (cm == nullptr) {
    local_cm = std::make_shared<CancellationManager>();
    cm = local_cm.get();
  }

  auto* refcounted_done = new ReffedStatusCallback(std::move(done));
  for (int i = 0; i < data->glue_.size(); ++i) {
    refcounted_done->Ref();
  }

  FunctionLibraryRuntime::Options opts_copy = opts;
  for (const auto& pair : data->glue_) {
    const string& target = pair.first;
    const ComponentFunctionData& comp_data = pair.second;
    FunctionLibraryRuntime::Handle comp_handle = pair.second.handle;

    opts_copy.args_alloc_attrs = comp_data.arg_alloc_attrs;
    opts_copy.rets_alloc_attrs = comp_data.ret_alloc_attrs;
    opts_copy.cancellation_manager = cm;

    InternalArgs comp_args;
    Status s = get_component_args(comp_data, &comp_args);
    if (!s.ok()) {
      VLOG(2) << "Failed to get component function arguments: " << s;
      refcounted_done->UpdateStatus(s);
      refcounted_done->Unref();
      cm->StartCancel();
      continue;
    }
    std::vector<FunctionRet>* comp_rets = new std::vector<FunctionRet>;
    rets->resize(data->num_outputs_);

    auto component_fn_callback = [comp_rets, rets, comp_data, refcounted_done,
                                  cm, local_cm, data, comp_handle,
                                  target](const Status& status) {
      if (!status.ok()) {
        VLOG(2) << "Component function execution on target " << target
                << " from " << data->function_name_ << " with handle "
                << comp_handle << " failed: " << status;
        const string function_and_msg = strings::StrCat(
            errors::FormatFunctionForError(data->function_name_), " ",
            status.error_message());
        refcounted_done->UpdateStatus(
            errors::CreateWithUpdatedMessage(status, function_and_msg));
        // Cancel the execution of other component functions.
        cm->StartCancel();
      } else {
        VLOG(2) << "Component function execution on target " << target
                << " from " << data->function_name_ << " with handle "
                << comp_handle << " succeeded.";
        for (int i = 0; i < comp_rets->size(); ++i) {
          (*rets)[comp_data.ret_indices[i]] = (*comp_rets)[i];
        }
      }
      delete comp_rets;
      // refcounted_done is thread-safe
      refcounted_done->Unref();
    };

    FunctionLibraryRuntime* flr = GetFLR(target);
    if (flr != nullptr) {
      opts_copy.remote_execution = false;
      // When target device has private thread pool, use the target device
      // runner
      thread::ThreadPool* pool = flr->device()->tensorflow_device_thread_pool();
      opts_copy.runner = (pool == nullptr) ? opts.runner : flr->runner();

      VLOG(1) << "Running component function on device " << target << " from "
              << data->function_name_ << " with handle " << comp_handle;
      VLOG(4) << "    with " << opts_copy.DebugString();

      std::vector<Tensor>* comp_tensor_rets = new std::vector<Tensor>;
      flr->Run(
          opts_copy, comp_handle, GetLocalArgs(comp_args.args),
          comp_tensor_rets,
          TensorsToFunctionRetsDoneCallback(comp_rets, comp_tensor_rets,
                                            std::move(component_fn_callback)));
    } else {
      opts_copy.remote_execution = true;

      VLOG(1) << "Running component function on device " << target << " from "
              << data->function_name_ << " with handle " << comp_handle;
      VLOG(4) << "    with " << opts_copy.DebugString();

      RunInternal(opts_copy, comp_handle, comp_args.args, comp_rets,
                  cleanup_items, std::move(component_fn_callback));
    }
  }
  refcounted_done->Unref();
}

Status ProcessFunctionLibraryRuntime::Instantiate(
    const string& function_name, AttrSlice attrs,
    const FunctionLibraryRuntime::InstantiateOptions& options,
    FunctionLibraryRuntime::Handle* handle) {
  if (options.is_multi_device_function) {
    return InstantiateMultiDevice(function_name, attrs, options, handle);
  }

  *handle = kInvalidHandle;
  FunctionLibraryRuntime* flr = GetFLR(options.target);
  if (flr != nullptr) {
    return flr->Instantiate(function_name, attrs, options, handle);
  }

  Status status;
  Notification notification;
  InstantiateRemote(function_name, attrs, options, handle,
                    [&status, &notification](const Status& s) {
                      status = s;
                      notification.Notify();
                    });
  notification.WaitForNotification();
  return status;
}

Status ProcessFunctionLibraryRuntime::IsCrossProcess(
    FunctionLibraryRuntime::Handle handle, bool* is_cross_process) const {
  tf_shared_lock l(mu_);
  const auto& mdevice_it = mdevice_data_.find(handle);
  if (mdevice_it != mdevice_data_.end()) {
    *is_cross_process = mdevice_it->second->is_cross_process_;
    return OkStatus();
  }
  const auto& it = function_data_.find(handle);
  if (it != function_data_.end()) {
    *is_cross_process = it->second->is_cross_process();
    return OkStatus();
  }
  return errors::InvalidArgument("Handle ", handle, " not found.");
}

void ProcessFunctionLibraryRuntime::InstantiateRemote(
    const string& function_name, AttrSlice attrs,
    const FunctionLibraryRuntime::InstantiateOptions& options,
    FunctionLibraryRuntime::Handle* handle,
    FunctionLibraryRuntime::DoneCallback done) {
  if (parent_ == nullptr) {
    done(errors::Internal(
        "Currently don't support instantiating functions on device: ",
        options.target));
    return;
  }
  auto target = options.target;
  VLOG(1) << "ProcessFLR Instantiate: " << function_name << " on: " << target;
  string function_key = Canonicalize(function_name, attrs, options);
  FunctionData* f;
  {
    mutex_lock l(mu_);
    FunctionLibraryRuntime::Handle h =
        gtl::FindWithDefault(table_, function_key, kInvalidHandle);
    if (h == kInvalidHandle || function_data_.count(h) == 0) {
      h = AddHandleLocked(function_key, target, kInvalidHandle);
    }
    f = function_data_[h].get();
    *handle = h;
  }
  f->DistributedInit(
      parent_, function_name,
      options.lib_def == nullptr ? *lib_def_ : *options.lib_def, attrs, options,
      [this, function_name, target, handle, done](const Status& s) {
        VLOG(1) << "ProcessFLR Instantiate [success]: " << function_name
                << " on: " << target << " with handle: " << *handle
                << " (this: " << this << ")";
        done(s);
      });
}

Status ProcessFunctionLibraryRuntime::RemoveHandle(
    FunctionLibraryRuntime::Handle handle) {
  mutex_lock l(mu_);
  table_.erase(function_data_[handle]->function_key());
  function_data_.erase(handle);
  return OkStatus();
}

Status ProcessFunctionLibraryRuntime::ReleaseMultiDeviceHandle(
    FunctionLibraryRuntime::Handle handle) {
  std::unique_ptr<MultiDeviceFunctionData> mdata;
  {
    mutex_lock l(mu_);
    auto it = mdevice_data_.find(handle);
    --it->second->instantiation_counter_;
    if (it->second->instantiation_counter_ != 0) {
      return OkStatus();
    }
    mdata = std::move(it->second);
    table_.erase(mdata->function_key_);
    mdevice_data_.erase(it);
  }

  // If we are here we are releasing the last instantiation of `handle`.
  // Release all component function handles.
  Status overall_status;
  for (const auto& it : mdata->glue_) {
    const string& device = it.first;
    FunctionLibraryRuntime::Handle flr_handle = it.second.handle;
    FunctionLibraryRuntime* flr = GetFLR(device);
    if (flr == nullptr) {
      // TODO(nareshmodi): Implement DeregisterGraph call to remote device if
      // parent is not null.
      if (parent_ != nullptr) {
        return errors::Unimplemented(
            "Releasing a multi-device component handle on a remote device is "
            "not yet implemented.");
      }
      return errors::InvalidArgument(
          "Failed to find FunctionLibraryRuntime for device ", device,
          " when releasing multi-device function handle ", handle);
    }
    Status status = flr->ReleaseHandle(flr_handle);
    if (!status.ok()) {
      overall_status = status;
    }
  }

  return overall_status;
}

Status ProcessFunctionLibraryRuntime::ReleaseHandle(
    FunctionLibraryRuntime::Handle handle) {
  // Return directly if all function handles has already been released.
  if (flr_map_ == nullptr) return OkStatus();

  if (IsMultiDevice(handle)) {
    return ReleaseMultiDeviceHandle(handle);
  }

  FunctionLibraryRuntime* flr = nullptr;
  string target_device;
  {
    mutex_lock l(mu_);
    CHECK_EQ(1, function_data_.count(handle)) << " handle: " << handle;
    target_device = function_data_[handle]->target_device();
  }
  flr = GetFLR(target_device);
  if (flr != nullptr) {
    return flr->ReleaseHandle(handle);
  }
  return errors::InvalidArgument("Handle not found: ", handle);
}

void ProcessFunctionLibraryRuntime::CleanupCreatedRendezvous(
    const Rendezvous* created_rendezvous, const int64_t step_id) const {
  if (created_rendezvous) {
    DCHECK(rendezvous_factory_);
    created_rendezvous->Unref();
    Status s = rendezvous_factory_.CleanUp(step_id);
    if (!s.ok()) {
      LOG(ERROR) << s;
    }
  }
}

FunctionLibraryRuntime::DoneCallback
ProcessFunctionLibraryRuntime::ApplyCleanUpToDoneCallback(
    std::vector<std::unique_ptr<CleanUpItem>>* items,
    FunctionLibraryRuntime::DoneCallback done, const int64_t step_id,
    const Rendezvous* created_rendezvous) const {
  return [this, items, done = std::move(done), step_id,
          created_rendezvous](const Status& status) {
    this->CleanupCreatedRendezvous(created_rendezvous, step_id);
    auto* local_status = new Status(status);
    CleanUp(items, [local_status, done](const Status& cleanup_status) {
      local_status->Update(cleanup_status);
      done(*local_status);
      delete local_status;
    });
    delete items;
  };
}

Status ProcessFunctionLibraryRuntime::CreateRendezvous(
    FunctionLibraryRuntime::Options& opts,
    Rendezvous** created_rendezvous) const {
  DCHECK(opts.rendezvous == nullptr);
  if (!rendezvous_factory_) {
    return errors::FailedPrecondition(
        "The caller does not provide a rendezvous and "
        "ProcessFunctionLibraryRuntime was created without a rendezvous "
        "factory.");
  }
  Status s = rendezvous_factory_(opts.step_id, device_mgr_, created_rendezvous);
  if (s.ok()) {
    opts.rendezvous = *created_rendezvous;
    opts.create_rendezvous = false;
  }
  return s;
}

Status ProcessFunctionLibraryRuntime::GetComponentArgs(
    const gtl::ArraySlice<Tensor> args,
    const ProcessFunctionLibraryRuntime::ComponentFunctionData& comp_data,
    ProcessFunctionLibraryRuntime::InternalArgs* comp_args) {
  // "Index"s of _Arg nodes are unique when all arguments are local Tensors.
  for (const auto& it : comp_data.arg_indices) {
    if (it.index >= args.size()) {
      return errors::InvalidArgument("index ", it.index,
                                     " is out of range [0, ", args.size(), ")");
    }
    if (it.sub_index >= 0) {
      const Tensor& t = args[it.index];
      if (t.dtype() != DT_RESOURCE) {
        return errors::InvalidArgument("Got unexpected sub_index ",
                                       it.sub_index, " for argument ",
                                       it.index);
      }
      const auto& handles = t.flat<ResourceHandle>();
      if (it.sub_index >= handles.size()) {
        return errors::InvalidArgument("Sub_index ", it.sub_index,
                                       "is out of range [0,", handles.size(),
                                       ") for argument ", it.index);
      }
      comp_args->args.push_back(Tensor(handles(it.sub_index)));
    } else {
      comp_args->args.push_back(args[it.index]);
    }
  }
  return OkStatus();
}

#if !defined(IS_MOBILE_PLATFORM)
Status ProcessFunctionLibraryRuntime::GetComponentArgs(
    const FunctionArgsInterface& args,
    const ProcessFunctionLibraryRuntime::ComponentFunctionData& comp_data,
    ProcessFunctionLibraryRuntime::InternalArgs* comp_args) {
  for (int i = 0; i < comp_data.arg_indices.size(); ++i) {
    const FunctionArgIndex index = comp_data.arg_indices.at(i);
    Tensor tensor;
    if (args.GetLocalArg(index, &tensor).ok()) {
      comp_args->args.push_back(std::move(tensor));
    } else {
      eager::RemoteTensorHandle remote_handle;
      TF_RETURN_IF_ERROR(args.GetRemoteArg(index, &remote_handle));
      comp_args->remote_args.emplace_back(
          std::make_unique<eager::RemoteTensorHandle>(
              std::move(remote_handle)));
      comp_args->args.push_back(comp_args->remote_args.back().get());
    }
  }
  return OkStatus();
}
#endif  // IS_MOBILE_PLATFORM

void ProcessFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle handle, gtl::ArraySlice<Tensor> args,
    std::vector<Tensor>* rets,
    FunctionLibraryRuntime::DoneCallback done) const {
  FunctionLibraryRuntime::Options new_opts = opts;
  Rendezvous* created_rendezvous = nullptr;
  if (!opts.rendezvous) {
    Status s = CreateRendezvous(new_opts, &created_rendezvous);
    if (!s.ok()) {
      done(s);
      return;
    }
  }

  auto* cleanup_items = new std::vector<std::unique_ptr<CleanUpItem>>;
  done = ApplyCleanUpToDoneCallback(cleanup_items, std::move(done),
                                    new_opts.step_id, created_rendezvous);
  std::vector<FunctionRet>* function_rets = new std::vector<FunctionRet>;
  done = [rets, function_rets, done = std::move(done)](const Status& s) {
    Status status = s;
    if (status.ok()) {
      status.Update(FunctionRetsToTensors(function_rets, rets));
    }
    delete function_rets;
    done(status);
  };
  bool multi_device = HasMultiDeviceHandle(handle);
  if (multi_device) {
    auto get_component_args = [&args](const ComponentFunctionData& comp_data,
                                      InternalArgs* comp_args) -> Status {
      return GetComponentArgs(args, comp_data, comp_args);
    };
    return RunMultiDeviceAsync(new_opts, handle, function_rets, cleanup_items,
                               std::move(done), std::move(get_component_args));
  }
  std::vector<FunctionArg> local_args;
  for (const auto& tensor : args) {
    local_args.push_back(tensor);
  }
  RunInternal(new_opts, handle, local_args, function_rets, cleanup_items,
              std::move(done));
}

// This method handles the simple remote call case (not multi-device).
void ProcessFunctionLibraryRuntime::RunInternal(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle handle, gtl::ArraySlice<FunctionArg> args,
    std::vector<FunctionRet>* rets,
    std::vector<std::unique_ptr<CleanUpItem>>* cleanup_items,
    FunctionLibraryRuntime::DoneCallback done) const {
  FunctionLibraryRuntime* flr = nullptr;
  string target_device;
  FunctionLibraryRuntime::LocalHandle local_handle;
  {
    tf_shared_lock l(mu_);
    auto iter = function_data_.find(handle);
    if (iter == function_data_.end()) {
      done(errors::NotFound("Handle: ", handle, " not found."));
      return;
    }
    FunctionData* function_data = iter->second.get();
    target_device = function_data->target_device();
    local_handle = function_data->local_handle();
  }

  if (!opts.remote_execution) {
    done(
        errors::InvalidArgument("ProcessFunctionLibraryRuntime::Run should "
                                "only be called for multi-device functions or "
                                "for remote execution."));
    return;
  }

  flr = GetFLR(target_device);
  if (flr != nullptr) {
    auto rendezvous = opts.rendezvous;
    string source_device = opts.source_device;
    DeviceContext* device_context;
    Status s = GetDeviceContext(source_device, &device_context);
    if (!s.ok()) {
      done(s);
      return;
    }
    int64_t src_incarnation, target_incarnation;
    s = GetDeviceIncarnation(source_device, &src_incarnation);
    s.Update(GetDeviceIncarnation(target_device, &target_incarnation));
    if (!s.ok()) {
      done(s);
      return;
    }

    std::vector<Tensor> local_args = GetLocalArgs(args);

    // Send the args over to the target device.
    s = SendTensors(source_device, target_device, "arg_", src_incarnation,
                    local_args, device_context, opts.args_alloc_attrs,
                    rendezvous);
    if (!s.ok()) {
      done(s);
      return;
    }
    const std::vector<AllocatorAttributes>& rets_alloc_attrs =
        opts.rets_alloc_attrs;
    std::vector<Tensor>* remote_rets = new std::vector<Tensor>;
    flr->Run(opts, handle, local_args, remote_rets,
             [source_device, target_device, target_incarnation, rendezvous,
              device_context, rets_alloc_attrs, remote_rets, rets,
              done = std::move(done)](const Status& status) mutable {
               if (!status.ok()) {
                 delete remote_rets;
                 done(status);
                 return;
               }
               int64_t num_returns = remote_rets->size();
               delete remote_rets;
               // Now receive the return values from the target.
               std::vector<Tensor>* recv_tensors = new std::vector<Tensor>;
               ReceiveTensorsAsync(target_device, source_device, "ret_",
                                   target_incarnation, num_returns,
                                   device_context, rets_alloc_attrs, rendezvous,
                                   recv_tensors,
                                   TensorsToFunctionRetsDoneCallback(
                                       rets, recv_tensors, std::move(done)));
             });
    return;
  }
  if (parent_ != nullptr) {
    auto cleanup_item = std::make_unique<CleanUpItem>();
    cleanup_item->device = target_device;
    cleanup_item->step_id = opts.step_id;
    cleanup_item->local_handle = local_handle;
    cleanup_items->emplace_back(std::move(cleanup_item));
    parent_->Run(opts, local_handle, args, rets, std::move(done));
    return;
  }
  done(errors::Internal("Could not find device"));
}

void ProcessFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle handle, CallFrameInterface* frame,
    FunctionLibraryRuntime::DoneCallback done) const {
  std::vector<Tensor> args;
  args.reserve(frame->num_args());
  for (size_t i = 0; i < frame->num_args(); ++i) {
    const Tensor* arg;
    Status s = frame->GetArg(i, &arg);
    args.emplace_back(*arg);
    if (!s.ok()) {
      done(s);
    }
  }
  std::vector<Tensor>* rets = new std::vector<Tensor>;
  rets->reserve(frame->num_retvals());

  Run(opts, handle, args, rets,

      [frame, rets, done = std::move(done)](const Status& status) {
        std::unique_ptr<std::vector<Tensor>> rets_releaser(rets);

        if (!status.ok()) {
          done(status);
          return;
        }

        if (rets->size() != frame->num_retvals()) {
          done(errors::Internal(
              "Number of return values from function (", rets->size(),
              ") did not match expected number of return values (",
              frame->num_retvals(), ")."));
          return;
        }

        for (size_t i = 0; i < frame->num_retvals(); ++i) {
          Status s = frame->SetRetval(i, (*rets)[i]);
          if (!s.ok()) {
            done(s);
            return;
          }
        }
        done(OkStatus());
      });
}

Status ProcessFunctionLibraryRuntime::RunSync(
    const FunctionLibraryRuntime::Options& orig_opts,
    FunctionLibraryRuntime::Handle handle, gtl::ArraySlice<Tensor> args,
    std::vector<Tensor>* rets) const {
  MultiDeviceFunctionData* multi_device_data = IsMultiDevice(handle);
  if (multi_device_data && multi_device_data->enable_sync_execution) {
    metrics::IncrementTestCounter("pflr_runsync", "sync");
    FunctionLibraryRuntime::Options new_opts = orig_opts;
    Rendezvous* created_rendezvous = nullptr;
    if (!new_opts.rendezvous) {
      TF_RETURN_IF_ERROR(CreateRendezvous(new_opts, &created_rendezvous));
    }

    std::vector<FunctionRet> function_rets;
    auto get_component_args = [&args](const ComponentFunctionData& comp_data,
                                      InternalArgs* comp_args) {
      return GetComponentArgs(args, comp_data, comp_args);
    };

    Status status = RunMultiDeviceSync(new_opts, handle, &function_rets,
                                       std::move(get_component_args));
    CleanupCreatedRendezvous(created_rendezvous, new_opts.step_id);
    status.Update(FunctionRetsToTensors(&function_rets, rets));
    return status;
  } else {
    // TODO(b/207484417): Either handle or avoid/delete this fallback path.
    metrics::IncrementTestCounter("pflr_runsync", "async");
    Notification n;
    Status s;
    Run(orig_opts, handle, args, rets, [&n, &s](const Status& status) {
      s.Update(status);
      n.Notify();
    });
    n.WaitForNotification();
    return s;
  }
}

Status ProcessFunctionLibraryRuntime::RunSync(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle handle, CallFrameInterface* frame) const {
  // TODO(b/207485199): Implement this as synchronous code.
  Notification n;
  Status s;
  Run(opts, handle, frame, [&n, &s](const Status& status) {
    s.Update(status);
    n.Notify();
  });
  n.WaitForNotification();
  return s;
}

void ProcessFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::Handle handle, const FunctionArgsInterface& args,
    std::vector<FunctionRet>* rets,
    FunctionLibraryRuntime::DoneCallback done) const {
  bool has_remote_outputs = false;
  const MultiDeviceFunctionData* data = IsMultiDevice(handle);
  if (data != nullptr) {
    has_remote_outputs = data->has_remote_outputs;
  }
  if (!args.HasRemoteOrPackedInputs() && !has_remote_outputs) {
    const std::vector<Tensor> local_inputs = args.GetLocalTensors();
    std::vector<Tensor>* tensor_rets = new std::vector<Tensor>;
    return Run(
        opts, handle, local_inputs, tensor_rets,
        TensorsToFunctionRetsDoneCallback(rets, tensor_rets, std::move(done)));
  }

  FunctionLibraryRuntime::Options new_opts = opts;
  Rendezvous* created_rendezvous = nullptr;
  if (!opts.rendezvous) {
    Status s = CreateRendezvous(new_opts, &created_rendezvous);
    if (!s.ok()) {
      done(s);
      return;
    }
  }

#if defined(IS_MOBILE_PLATFORM)
  done(errors::Unimplemented(
      "Remote inputs are not available on mobile devices."));
  return;
#else   // !IS_MOBILE_PLATFORM
  auto* cleanup_items = new std::vector<std::unique_ptr<CleanUpItem>>;
  done = ApplyCleanUpToDoneCallback(cleanup_items, done, opts.step_id,
                                    created_rendezvous);

  auto get_component_args = [&args](const ComponentFunctionData& comp_data,
                                    InternalArgs* comp_args) -> Status {
    return GetComponentArgs(args, comp_data, comp_args);
  };
  return RunMultiDeviceAsync(new_opts, handle, rets, cleanup_items,
                             std::move(done), std::move(get_component_args));
#endif  // !IS_MOBILE_PLATFORM
}

void ProcessFunctionLibraryRuntime::CleanUp(
    std::vector<std::unique_ptr<CleanUpItem>>* items,
    FunctionLibraryRuntime::DoneCallback done) const {
  auto* refcounted_done = new ReffedStatusCallback(std::move(done));
  for (auto& item : *items) {
    refcounted_done->Ref();
    auto* flr = GetFLR(item->device);
    if (flr != nullptr) {
      // TODO(fishx): cleanup state for local execution.
      refcounted_done->UpdateStatus(
          errors::Internal("Cleanup items shouldn't contain local item."));
      refcounted_done->Unref();
    } else if (parent_ != nullptr) {
      parent_->CleanUp(item->step_id, item->local_handle,
                       [refcounted_done](const Status& status) {
                         if (!status.ok()) {
                           refcounted_done->UpdateStatus(status);
                         }
                         // refcounted_done is thread-safe
                         refcounted_done->Unref();
                       });
    } else {
      refcounted_done->UpdateStatus(
          errors::Internal("Could not find device in cleanup."));
      refcounted_done->Unref();
    }
  }
  refcounted_done->Unref();
}

Status ProcessFunctionLibraryRuntime::Clone(
    Env* env, int graph_def_version, const OptimizerOptions& optimizer_options,
    std::unique_ptr<FunctionLibraryDefinition>* out_lib_def,
    std::unique_ptr<ProcessFunctionLibraryRuntime>* out_pflr,
    bool skip_flib_def) const {
  if (skip_flib_def) {
    *out_lib_def = std::make_unique<FunctionLibraryDefinition>(
        lib_def_->default_registry(), FunctionDefLibrary{});
  } else {
    *out_lib_def = std::make_unique<FunctionLibraryDefinition>(*lib_def_);
  }
  *out_pflr = std::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr_, env, config_ ? &(*config_) : nullptr, graph_def_version,
      out_lib_def->get(), optimizer_options, default_thread_pool_, parent_,
      session_metadata_, rendezvous_factory_);
  {
    tf_shared_lock l(mu_);
    for (auto* d : composite_devices_) (*out_pflr)->AddCompositeDevice(d);
  }
  return OkStatus();
}

}  // namespace tensorflow
