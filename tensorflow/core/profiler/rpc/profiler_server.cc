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
#include <string>

#include "grpcpp/grpcpp.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/profiler_service.grpc.pb.h"
#include "tensorflow/core/profiler/rpc/profiler_service_impl.h"

namespace tensorflow {
namespace profiler {

void ProfilerServer::StartProfilerServer(int32 port) {
  VLOG(1) << "Starting profiler server.";
  std::string server_address = absl::StrCat("[::]:", port);
  service_ = CreateProfilerService();
  ::grpc::ServerBuilder builder;

  int selected_port = 0;
  builder.AddListeningPort(server_address, ::grpc::InsecureServerCredentials(),
                           &selected_port);
  builder.RegisterService(service_.get());
  server_ = builder.BuildAndStart();
  if (!selected_port) {
    LOG(ERROR) << "Unable to bind to " << server_address
               << " selected port:" << selected_port;
  } else {
    LOG(INFO) << "Profiler server listening on " << server_address
              << " selected port:" << selected_port;
  }
}

ProfilerServer::~ProfilerServer() {
  if (server_) {
    server_->Shutdown();
    server_->Wait();
    LOG(INFO) << "Profiler server was shut down";
  }
}

}  // namespace profiler
}  // namespace tensorflow
