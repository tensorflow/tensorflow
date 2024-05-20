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

#include "tensorflow/core/debug/debug_grpc_testlib.h"

#include "tensorflow/core/debug/debug_graph_utils.h"
#include "tensorflow/core/debug/debugger_event_metadata.pb.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

namespace test {

::grpc::Status TestEventListenerImpl::SendEvents(
    ::grpc::ServerContext* context,
    ::grpc::ServerReaderWriter<::tensorflow::EventReply, ::tensorflow::Event>*
        stream) {
  Event event;

  while (stream->Read(&event)) {
    if (event.has_log_message()) {
      debug_metadata_strings.push_back(event.log_message().message());
      stream->Write(EventReply());
    } else if (!event.graph_def().empty()) {
      encoded_graph_defs.push_back(event.graph_def());
      stream->Write(EventReply());
    } else if (event.has_summary()) {
      const Summary::Value& val = event.summary().value(0);

      std::vector<string> name_items =
          tensorflow::str_util::Split(val.node_name(), ':');

      const string node_name = name_items[0];
      const string debug_op = name_items[2];

      const TensorProto& tensor_proto = val.tensor();
      Tensor tensor(tensor_proto.dtype());
      if (!tensor.FromProto(tensor_proto)) {
        return ::grpc::Status::CANCELLED;
      }

      // Obtain the device name, which is encoded in JSON.
      third_party::tensorflow::core::debug::DebuggerEventMetadata metadata;
      if (val.metadata().plugin_data().plugin_name() != "debugger") {
        // This plugin data was meant for another plugin.
        continue;
      }
      auto status = tensorflow::protobuf::util::JsonStringToMessage(
          val.metadata().plugin_data().content(), &metadata);
      if (!status.ok()) {
        // The device name could not be determined.
        continue;
      }

      device_names.push_back(metadata.device());
      node_names.push_back(node_name);
      output_slots.push_back(metadata.output_slot());
      debug_ops.push_back(debug_op);
      debug_tensors.push_back(tensor);

      // If the debug node is currently in the READ_WRITE mode, send an
      // EventReply to 1) unblock the execution and 2) optionally modify the
      // value.
      const DebugNodeKey debug_node_key(metadata.device(), node_name,
                                        metadata.output_slot(), debug_op);
      if (write_enabled_debug_node_keys_.find(debug_node_key) !=
          write_enabled_debug_node_keys_.end()) {
        stream->Write(EventReply());
      }
    }
  }

  {
    mutex_lock l(states_mu_);
    for (size_t i = 0; i < new_states_.size(); ++i) {
      EventReply event_reply;
      EventReply::DebugOpStateChange* change =
          event_reply.add_debug_op_state_changes();

      // State changes will take effect in the next stream, i.e., next debugged
      // Session.run() call.
      change->set_state(new_states_[i]);
      const DebugNodeKey& debug_node_key = debug_node_keys_[i];
      change->set_node_name(debug_node_key.node_name);
      change->set_output_slot(debug_node_key.output_slot);
      change->set_debug_op(debug_node_key.debug_op);
      stream->Write(event_reply);

      if (new_states_[i] == EventReply::DebugOpStateChange::READ_WRITE) {
        write_enabled_debug_node_keys_.insert(debug_node_key);
      } else {
        write_enabled_debug_node_keys_.erase(debug_node_key);
      }
    }

    debug_node_keys_.clear();
    new_states_.clear();
  }

  return ::grpc::Status::OK;
}

void TestEventListenerImpl::ClearReceivedDebugData() {
  debug_metadata_strings.clear();
  encoded_graph_defs.clear();
  device_names.clear();
  node_names.clear();
  output_slots.clear();
  debug_ops.clear();
  debug_tensors.clear();
}

void TestEventListenerImpl::RequestDebugOpStateChangeAtNextStream(
    const EventReply::DebugOpStateChange::State new_state,
    const DebugNodeKey& debug_node_key) {
  mutex_lock l(states_mu_);

  debug_node_keys_.push_back(debug_node_key);
  new_states_.push_back(new_state);
}

void TestEventListenerImpl::RunServer(const int server_port) {
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(strings::StrCat("localhost:", server_port),
                           ::grpc::InsecureServerCredentials());
  builder.RegisterService(this);
  std::unique_ptr<::grpc::Server> server = builder.BuildAndStart();

  while (!stop_requested_.load()) {
    Env::Default()->SleepForMicroseconds(200 * 1000);
  }
  server->Shutdown();
  stopped_.store(true);
}

void TestEventListenerImpl::StopServer() {
  stop_requested_.store(true);
  while (!stopped_.load()) {
  }
}

bool PollTillFirstRequestSucceeds(const string& server_url,
                                  const size_t max_attempts) {
  const int kSleepDurationMicros = 100 * 1000;
  size_t n_attempts = 0;
  bool success = false;

  // Try a number of times to send the Event proto to the server, as it may
  // take the server a few seconds to start up and become responsive.
  Tensor prep_tensor(DT_FLOAT, TensorShape({1, 1}));
  prep_tensor.flat<float>()(0) = 42.0f;

  while (n_attempts++ < max_attempts) {
    const uint64 wall_time = Env::Default()->NowMicros();
    Status publish_s = DebugIO::PublishDebugTensor(
        DebugNodeKey("/job:localhost/replica:0/task:0/cpu:0", "prep_node", 0,
                     "DebugIdentity"),
        prep_tensor, wall_time, {server_url});
    Status close_s = DebugIO::CloseDebugURL(server_url);

    if (publish_s.ok() && close_s.ok()) {
      success = true;
      break;
    } else {
      Env::Default()->SleepForMicroseconds(kSleepDurationMicros);
    }
  }

  return success;
}

}  // namespace test

}  // namespace tensorflow
