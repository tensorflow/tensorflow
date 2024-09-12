/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/tpu_global_init.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/tpu_configuration_ops.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/protobuf/tpu/topology.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_configuration_rewrite_pass.h"
#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_helpers.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace tensorflow {

namespace {

ABSL_CONST_INIT static absl::Mutex global_init_tpu_mutex(absl::kConstInit);
static tpu::TopologyProto* global_tpu_topology
    ABSL_GUARDED_BY(global_init_tpu_mutex) = nullptr;

constexpr char kTaskSpec[] = "/job:localhost/replica:0/task:0";

Status CreateDeviceMgr(Env* env, std::unique_ptr<DeviceMgr>* device_mgr) {
  SessionOptions session_options;
  session_options.env = env;
  std::vector<std::unique_ptr<Device>> devices;
  DeviceFactory* device_factory = DeviceFactory::GetFactory(DEVICE_TPU_SYSTEM);
  if (device_factory == nullptr) {
    return errors::Internal("Unable to initialize DeviceFactory.");
  }
  TF_RETURN_IF_ERROR(
      device_factory->CreateDevices(session_options, kTaskSpec, &devices));
  *device_mgr = std::make_unique<DynamicDeviceMgr>(std::move(devices));
  return absl::OkStatus();
}

void DeviceSetFromDeviceMgr(const DeviceMgr& device_mgr,
                            DeviceSet* device_set) {
  int devices_added = 0;
  for (auto d : device_mgr.ListDevices()) {
    device_set->AddDevice(d);
    if (devices_added == 0) {
      device_set->set_client_device(d);
    }
    ++devices_added;
  }
}

std::string GetTPUSystemDevice(absl::string_view job_name) {
  if (job_name.empty()) {
    return DeviceNameUtils::LocalName(DEVICE_TPU_SYSTEM, 0);
  } else {
    return absl::StrCat("/job:", job_name, "/device:TPU_SYSTEM:0");
  }
}

Status ConstructDistributedInitializationGraph(absl::string_view job_name,
                                               const DeviceSet& device_set,
                                               Graph* graph_to_run) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  options.device_set = &device_set;
  {
    Scope scope = Scope::NewRootScope();
    auto init_op = ops::ConfigureDistributedTPU(
        scope.WithOpName("InitializeTPUSystemGlobally")
            .WithDevice(GetTPUSystemDevice(job_name)),
        ops::ConfigureDistributedTPU::IsGlobalInit(true));
    TF_RETURN_IF_ERROR(scope.ToGraph(options.graph->get()));
  }
  DistributedTPUConfigurationRewritePass rewriter;
  TF_RETURN_IF_ERROR(rewriter.Run(options));

  // Graph doesn't update the node-def's after adding edges, which causes
  // node-def validation to fail in the executor. So we explicitly do a
  // round-trip through GraphDef, so that node-defs are updated.
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph({}, graph->ToGraphDefDebug(), graph_to_run));

  return absl::OkStatus();
}

Status InitializeFromSession(absl::string_view session_target,
                             const Graph* graph_to_run,
                             std::vector<Tensor>* outputs) {
  tensorflow::SessionOptions s_opts;
  s_opts.target = std::string(session_target);

  std::unique_ptr<tensorflow::Session> sess(tensorflow::NewSession(s_opts));

  GraphDef g_def;
  graph_to_run->ToGraphDef(&g_def);

  TF_RETURN_IF_ERROR(sess->Create(g_def));
  TF_RETURN_IF_ERROR(
      sess->Run({}, {"InitializeTPUSystemGlobally:0"}, {}, outputs));

  return absl::OkStatus();
}

}  // namespace

Status InitializeTPUSystemGlobally(absl::string_view job_name,
                                   absl::string_view session_target,
                                   const DeviceSet& device_set, Env* env,
                                   tpu::TopologyProto* tpu_topology) {
  VLOG(1) << "InitializeTpuSystemGlobally";

  absl::MutexLock lock(&global_init_tpu_mutex);
  if (global_tpu_topology != nullptr) {
    *tpu_topology = *global_tpu_topology;
    return absl::OkStatus();
  }

  std::unique_ptr<Graph> graph_to_run(new Graph(OpRegistry::Global()));

  DeviceNameUtils::ParsedName system_spec;
  Device* tpu_system_device;

  std::string task_spec =
      job_name.empty() ? kTaskSpec
                       : absl::StrCat("/job:", job_name, "/replica:0/task:0");
  // Placed here, much before usage, to get a sane error if TPU_SYSTEM_DEVICE
  // hasn't been linked in. Otherwise we may get a cryptic error down the line.
  TF_RETURN_IF_ERROR(DistributedTPURewriteHelpers::GetSystemDevice(
      task_spec, device_set, &system_spec, &tpu_system_device));

  TF_RETURN_IF_ERROR(ConstructDistributedInitializationGraph(
      job_name, device_set, graph_to_run.get()));

  std::vector<Tensor> outputs;
  // Being a bit conservative here to run non-distributed initialization with
  // graph runner.
  // TODO(hthu): Re-evaluate the choice of using session for running the
  // initialization graph given that we need to a session in distributed
  // initialization anyway.
  if (session_target.empty()) {
    GraphRunner graph_runner(tpu_system_device);
    TF_RETURN_IF_ERROR(graph_runner.Run(graph_to_run.get(), nullptr, {},
                                        {"InitializeTPUSystemGlobally:0"},
                                        &outputs));
  } else {
    TF_RETURN_IF_ERROR(
        InitializeFromSession(session_target, graph_to_run.get(), &outputs));
  }

  if (outputs.empty()) {
    return errors::Internal("No output from running TPU initialization.");
  }

  global_tpu_topology = new tpu::TopologyProto();
  if (!global_tpu_topology->ParseFromString(outputs[0].scalar<tstring>()())) {
    return errors::Internal(
        "Unable to parse output from running TPU initialization as "
        "TopologyProto proto.");
  }

  *tpu_topology = *global_tpu_topology;
  return absl::OkStatus();
}

// NOTE: Session would have been the obvious first choice to run the graph
// here, but instead we use a GraphRunner because Session creates a global
// EigenThreadPool based on the SessionOptions it receives the first time it
// runs. This means that we need to create the right options and pass it to this
// API to make it work correctly. We felt it was an onerous restriction to place
// on the API, so we went with the current approach.
Status InitializeTPUSystemGlobally(Env* env, tpu::TopologyProto* tpu_topology) {
  std::unique_ptr<DeviceMgr> device_mgr;
  TF_RETURN_IF_ERROR(CreateDeviceMgr(env, &device_mgr));
  DeviceSet device_set;
  DeviceSetFromDeviceMgr(*device_mgr, &device_set);

  return InitializeTPUSystemGlobally(/*job_name=*/absl::string_view(),
                                     /*session_target=*/absl::string_view(),
                                     device_set, env, tpu_topology);
}

Status InitializeTPUSystemGlobally() {
  tensorflow::tpu::TopologyProto tpu_topology;
  return InitializeTPUSystemGlobally(tensorflow::Env::Default(), &tpu_topology);
}

}  // namespace tensorflow
