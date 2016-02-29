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

#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"

#include "grpc++/grpc++.h"
#include "grpc++/security/credentials.h"
#include "grpc++/server_builder.h"

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/master_session.h"
#include "tensorflow/core/distributed_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/command_line_flags.h"

// This binary starts a TensorFlow server (master and worker) for test purposes.
namespace tensorflow {

struct GrpcTaskOptions {
  // This process belongs to the "job_name".
  string job_name;

  // This process is the task-th task within the replica. 0th, 1st,
  // 2nd, etc.
  int32 task = 0;

  // Specification of peers.
  GrpcChannelSpec channel_spec;

  SessionOptions default_session_options;
};

Status StartTensorFlowServer(const TaskOptions& task_options) {
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "server", 1);
  thread_pool->Schedule([argc, argv, task_options]() {
    // This process provides both the worker service and the master
    // service. We let these two services share the same channel cache
    // (rpc connections) and cpu devices (used by the master as the
    // client device). These client devices require a worker service
    // so that remote devices can copy the feeds from the client
    // device in the master.
    tensorflow::MasterEnv master_env;
    string name_prefix =
        strings::StrCat("/job:", task_optionss.job_name, "/replica:0", "/task:",
                        task_options.task);
    DeviceFactory::AddDevices(task_options.default_session_options, name_prefix,
                              &master_env.local_devices);

    // Create the DeviceMgr before initializing the RPC layer, because that
    // needs to know how many devices of each kind exist.
    WorkerEnv worker_env;
    worker_env.device_mgr = new DeviceMgr(master_env.local_devices);

    // Finish setting up Env for Worker service.
    string donotcare;
    CHECK(DeviceNameUtils::SplitDeviceName(master_env.local_devices[0]->name(),
                                           &worker_env.worker_name,
                                           &donotcare));
    worker_env.env = Env::Default();

    GrpcChannelCache* channel_cache =
        NewGrpcChannelCache(task_options.channel_spec);
    string server_address = channel_cache->TranslateTask(name_prefix);
    worker_env.worker_cache = NewGrpcWorkerCache(channel_cache);
    worker_env.graph_mgr = new GraphMgr(&worker_env);
    worker_env.rendezvous_mgr = new RpcRendezvousMgr(&worker_env);
    worker_env.compute_pool = ComputePool(task_options.default_session_options);

    // Finish setting up Env for Master service.
    master_env.env = Env::Default();
    master_env.ops = OpRegistry::Global();
    master_env.worker_cache = worker_env.worker_cache;
    master_env.master_session_factory = internal::NewMasterSession;

    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address,
                             ::grpc::InsecureServerCredentials());
    auto master_service = NewGrpcMasterService(&master_env, &builder);
    auto worker_service = NewGrpcWorkerService(&worker_env, &builder);
    // Finally assemble the server.
    auto server_ = builder.BuildAndStart();

    std::unique_ptr<Thread> master_thread(Env::Default()->StartThread(
        ThreadOptions(), "master_service_thread",
        [master_service]() { master_service->HandleRPCsLoop(); }));

    std::unique_ptr<Thread> worker_thread(Env::Default()->StartThread(
        ThreadOptions(), "worker_service_thread",
        [worker_service]() { worker_service->HandleRPCsLoop(); }));
  });

  // The ThreadPool destructor waits until all work is done (i.e. forever).
  delete thread_pool;
  return Status::OK();
}

}  // end namespace tensorflow
