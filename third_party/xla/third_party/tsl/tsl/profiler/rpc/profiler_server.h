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
#ifndef TENSORFLOW_TSL_PROFILER_RPC_PROFILER_SERVER_H_
#define TENSORFLOW_TSL_PROFILER_RPC_PROFILER_SERVER_H_

#include <memory>

#include "grpcpp/grpcpp.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/protobuf/profiler_service.grpc.pb.h"

namespace tsl {
namespace profiler {

class ProfilerServer {
 public:
  ~ProfilerServer();
  // Starts a profiler server with a given port.
  void StartProfilerServer(int32_t port);

 private:
  std::unique_ptr<tensorflow::grpc::ProfilerService::Service> service_;
  std::unique_ptr<::grpc::Server> server_;
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_RPC_PROFILER_SERVER_H_
