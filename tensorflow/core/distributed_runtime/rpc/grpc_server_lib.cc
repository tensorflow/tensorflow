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

#include <memory>

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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {

void StartTensorFlowServer(const GrpcServerOptions& options) {
  // The Thread destructor waits until all the thread terminates is
  // done (i.e. forever).
  std::unique_ptr<Thread> launcher_thread(Env::Default()->StartThread(
      ThreadOptions(), "TF_service_launcher", [options]() {
        // Configure the MasterEnv and WorkerEnv, which provide service-global
        // context for the master and worker services, respectively.

        // The master and worker share the same worker cache (for RPC
        // connections to other workers) and devices (so that the master
        // may run some ops locally as a "client" device). The master
        // requires a device to serve as a "client device", so that remote
        // devices can copy the feeds from the master.
        MasterEnv master_env;
        WorkerEnv worker_env;
        master_env.env = Env::Default();
        worker_env.env = Env::Default();

        // Configure shared devices between master and worker.
        string name_prefix =
            strings::StrCat("/job:", options.job_name, "/replica:0", "/task:",
                            options.task_index);
        DeviceFactory::AddDevices(options.default_session_options, name_prefix,
                                  &master_env.local_devices);
        worker_env.device_mgr = new DeviceMgr(master_env.local_devices);
        string unused;
        CHECK(DeviceNameUtils::SplitDeviceName(
            master_env.local_devices[0]->name(), &worker_env.worker_name,
            &unused));

        GrpcChannelCache* channel_cache =
            NewGrpcChannelCache(options.channel_spec);
        int port;
        const std::vector<string> host_port =
            str_util::Split(channel_cache->TranslateTask(name_prefix), ':');
        CHECK(str_util::NumericParse32(host_port[1], &port));

        worker_env.worker_cache = NewGrpcWorkerCache(channel_cache);

        // Finish setting up master environment.
        master_env.ops = OpRegistry::Global();
        master_env.worker_cache = worker_env.worker_cache;
        master_env.master_session_factory = internal::NewMasterSession;

        // Finish setting up worker environment.
        worker_env.graph_mgr = new GraphMgr(&worker_env);
        worker_env.rendezvous_mgr = new RpcRendezvousMgr(&worker_env);
        worker_env.compute_pool = ComputePool(options.default_session_options);

        // Build the gRPC server that will host both the master and the
        // worker services.
        ::grpc::ServerBuilder builder;
        builder.AddListeningPort(strings::StrCat("0.0.0.0:", port),
                                 ::grpc::InsecureServerCredentials());
        auto master_service = NewGrpcMasterService(&master_env, &builder);
        auto worker_service = NewGrpcWorkerService(&worker_env, &builder);
        auto server_ = builder.BuildAndStart();

        // Start threads to handle the incoming RPCs for the master and worker.
        // NOTE(mrry): The Thread destructor waits until the thread terminates
        // (i.e. forever in this case).
        std::unique_ptr<Thread> master_thread(Env::Default()->StartThread(
            ThreadOptions(), "TF_master_service",
            [master_service]() { master_service->HandleRPCsLoop(); }));
        std::unique_ptr<Thread> worker_thread(Env::Default()->StartThread(
            ThreadOptions(), "TF_worker_service",
            [worker_service]() { worker_service->HandleRPCsLoop(); }));
      }));
}

}  // end namespace tensorflow
