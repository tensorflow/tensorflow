/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/rpc/profiler_server.h"
#include <memory>
#include <utility>
#include "grpcpp/grpcpp.h"
#include "tensorflow/core/platform/grpc_services.h"
#include "tensorflow/core/profiler/rpc/profiler_service_impl.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {

std::unique_ptr<Thread> StartProfilerServer(
    ProfilerContext* const profiler_context, int32 port) {
  Env* env = profiler_context->eager_context != nullptr
                 ? profiler_context->eager_context->TFEnv()
                 : Env::Default();
  // Starting the server in the child thread may be delay and user may already
  // delete the profiler context at that point. So we need to make a copy.
  ProfilerContext ctx = *profiler_context;
  return WrapUnique(env->StartThread({}, "profiler server", [ctx, port]() {
    string server_address = strings::StrCat("0.0.0.0:", port);
    std::unique_ptr<grpc::ProfilerService::Service> service =
        CreateProfilerService(ctx);
    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address,
                             ::grpc::InsecureServerCredentials());
    builder.RegisterService(service.get());
    std::unique_ptr<::grpc::Server> server(builder.BuildAndStart());
    LOG(INFO) << "Profiling Server listening on " << server_address;
    server->Wait();
  }));
}

}  // namespace tensorflow
