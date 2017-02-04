/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "grpc++/grpc++.h"
#include "tensorflow/core/debug/debug_grpc_testlib.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"

// Usage: debug_test_server_main <port> <dump_root>
int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: debug_test_server_main <port> <dump_root>"
              << std::endl;
    return 1;
  }

  tensorflow::port::InitMain(argv[0], &argc, &argv);

  int port = 0;
  tensorflow::strings::safe_strto32(argv[1], &port);
  std::string test_server_addr = tensorflow::strings::StrCat("0.0.0.0:", port);

  tensorflow::test::TestEventListenerImpl debug_test_server(argv[2]);

  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(test_server_addr,
                           ::grpc::InsecureServerCredentials());
  builder.RegisterService(&debug_test_server);
  std::unique_ptr<::grpc::Server> test_server = builder.BuildAndStart();

  test_server->Wait();

  return 0;
}
