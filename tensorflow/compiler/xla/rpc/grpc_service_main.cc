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

// Basic server binary that exposes a xla::Service through a GRPC interface
// on a configurable port.
#include "grpc++/security/server_credentials.h"
#include "grpc++/server.h"
#include "grpc++/server_builder.h"
#include "tensorflow/compiler/xla/rpc/grpc_service.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {
namespace {

int RealMain(int argc, char** argv) {
  int32 port = 1685;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("port", &port, "port to listen on"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parsed_values_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parsed_values_ok) {
    LOG(ERROR) << usage;
    return 2;
  }
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  std::unique_ptr<xla::GRPCService> service =
      xla::GRPCService::NewService().ConsumeValueOrDie();

  ::grpc::ServerBuilder builder;
  string server_address(tensorflow::strings::Printf("localhost:%d", port));

  builder.AddListeningPort(server_address, ::grpc::InsecureServerCredentials());
  builder.RegisterService(service.get());
  std::unique_ptr<::grpc::Server> server(builder.BuildAndStart());

  LOG(INFO) << "Server listening on " << server_address;
  server->Wait();

  return 0;
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) { return xla::RealMain(argc, argv); }
