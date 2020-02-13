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
#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/profiler/profiler_service.grpc.pb.h"
#include "tensorflow/core/profiler/rpc/profiler_service_impl.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {

void ProfilerServer::StartProfilerServer(int32 port) {
  Env* env = Env::Default();
  auto start_server = [port, this]() {
    string server_address = absl::StrCat("0.0.0.0:", port);
    std::unique_ptr<grpc::ProfilerService::Service> service =
        CreateProfilerService();
    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address,
                             ::grpc::InsecureServerCredentials());
    builder.RegisterService(service.get());
    server_ = builder.BuildAndStart();
    LOG(INFO) << "Profiling Server listening on " << server_address;
    server_->Wait();
  };
  server_thread_ =
      WrapUnique(env->StartThread({}, "ProfilerServer", start_server));
}

void ProfilerServer::MaybeStartProfilerServer() {
  int64 profiler_port;
  // The implementation of ReadInt64FromEnvVar guaranteed that the output
  // argument will be set to default value failure.
  Status s = ReadInt64FromEnvVar("TF_PROFILER_PORT", -1, &profiler_port);
  if (!s.ok()) {
    LOG(WARNING) << "StartProfilerServer: " << s.error_message();
  }
  if (profiler_port < 1024 || profiler_port > 49151) {
    // Disable the log message if profiler_port is -1 to prevent spam the
    // terminal for TF user who doesn't set a profiler port.
    if (profiler_port == -1) return;
    LOG(WARNING)
        << "Profiler server not started. TF_PROFILER_PORT: " << profiler_port
        << " is out of the valid registered port range (1024 to 49151).";
    return;
  }
  StartProfilerServer(profiler_port);
}

ProfilerServer::~ProfilerServer() {
  if (server_) server_->Shutdown();
}

}  // namespace tensorflow
