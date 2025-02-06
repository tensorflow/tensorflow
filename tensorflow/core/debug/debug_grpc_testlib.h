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

#ifndef TENSORFLOW_CORE_DEBUG_DEBUG_GRPC_TESTLIB_H_
#define TENSORFLOW_CORE_DEBUG_DEBUG_GRPC_TESTLIB_H_

#include <atomic>
#include <unordered_set>

#include "grpcpp/grpcpp.h"
#include "tensorflow/core/debug/debug_io_utils.h"
#include "tensorflow/core/debug/debug_service.grpc.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

namespace test {

class TestEventListenerImpl final : public grpc::EventListener::Service {
 public:
  TestEventListenerImpl() : stop_requested_(false), stopped_(false) {}

  void RunServer(const int server_port);
  void StopServer();

  ::grpc::Status SendEvents(
      ::grpc::ServerContext* context,
      ::grpc::ServerReaderWriter< ::tensorflow::EventReply,
                                  ::tensorflow::Event>* stream) override;

  // Clear debug data (e.g., Tensors) received so far.
  void ClearReceivedDebugData();

  void RequestDebugOpStateChangeAtNextStream(
      const EventReply::DebugOpStateChange::State new_state,
      const DebugNodeKey& debug_node_key);

  std::vector<string> debug_metadata_strings;
  std::vector<string> encoded_graph_defs;
  std::vector<string> device_names;
  std::vector<string> node_names;
  std::vector<int32> output_slots;
  std::vector<string> debug_ops;
  std::vector<Tensor> debug_tensors;

 private:
  std::atomic_bool stop_requested_;
  std::atomic_bool stopped_;

  std::vector<DebugNodeKey> debug_node_keys_ TF_GUARDED_BY(states_mu_);
  std::vector<EventReply::DebugOpStateChange::State> new_states_
      TF_GUARDED_BY(states_mu_);

  std::unordered_set<DebugNodeKey> write_enabled_debug_node_keys_;

  mutex states_mu_;
};

// Poll a gRPC debug server by sending a small tensor repeatedly till success.
//
// Args:
//   server_url: gRPC URL of the server to poll, e.g., "grpc://foo:3333".
//   max_attempts: Maximum number of attempts.
//
// Returns:
//   Whether the polling succeeded within max_attempts.
bool PollTillFirstRequestSucceeds(const string& server_url,
                                  const size_t max_attempts);

}  // namespace test

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DEBUG_DEBUG_GRPC_TESTLIB_H_
