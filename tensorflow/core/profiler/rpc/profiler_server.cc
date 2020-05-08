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
#include "tensorflow/core/profiler/rpc/profiler_service_impl.h"

namespace tensorflow {

void ProfilerServer::StartProfilerServer(int32 port) {
  string server_address = absl::StrCat("0.0.0.0:", port);
  service_ = CreateProfilerService();
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, ::grpc::InsecureServerCredentials());
  builder.RegisterService(service_.get());
  server_ = builder.BuildAndStart();
  LOG(INFO) << "Profiling Server listening on " << server_address;
}

ProfilerServer::~ProfilerServer() {
  if (server_) {
    server_->Shutdown();
    server_->Wait();
  }
}

}  // namespace tensorflow
