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

#ifndef TENSORFLOW_DEBUG_GRPC_TESTLIB_H_
#define TENSORFLOW_DEBUG_GRPC_TESTLIB_H_

#include "grpc++/grpc++.h"
#include "tensorflow/core/debug/debug_service.grpc.pb.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

namespace test {

class TestEventListenerImpl final : public EventListener::Service {
 public:
  TestEventListenerImpl(const string& dump_root) : dump_root(dump_root) {}

  ::grpc::Status SendEvents(
      ::grpc::ServerContext* context,
      ::grpc::ServerReaderWriter< ::tensorflow::EventReply,
                                  ::tensorflow::Event>* stream);

  string dump_root;
};

class GrpcTestServerClientPair {
 public:
  GrpcTestServerClientPair(const int server_port);
  virtual ~GrpcTestServerClientPair() {}

  // Keep sending requests to the test server until the first success.
  // This is necessary because the server may take a certain amount of time
  // to start up and become responsive.
  //
  // Returns: A boolean indicating whether a successful response is obtained
  //   within the limit of maximum number of attempts.
  bool PollTillFirstRequestSucceeds();

  string dump_root;

  int server_port;
  string test_server_url;

 private:
  std::unique_ptr<Tensor> prep_tensor_;

  const int kMaxAttempts = 100;
  const int kSleepDurationMicros = 100 * 1000;
};

}  // namespace test

}  // namespace tensorflow

#endif  // TENSORFLOW_DEBUG_GRPC_TESTLIB_H_
