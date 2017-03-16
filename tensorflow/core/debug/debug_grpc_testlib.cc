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
#include "tensorflow/core/debug/debug_io_utils.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {

namespace test {

::grpc::Status TestEventListenerImpl::SendEvents(
    ::grpc::ServerContext* context,
    ::grpc::ServerReaderWriter< ::tensorflow::EventReply, ::tensorflow::Event>*
        stream) {
  Event event;

  while (stream->Read(&event)) {
    const Summary::Value& val = event.summary().value(0);

    std::vector<string> name_items =
        tensorflow::str_util::Split(val.node_name(), ':');

    const string node_name = name_items[0];
    int32 output_slot = 0;
    tensorflow::strings::safe_strto32(name_items[1], &output_slot);
    const string debug_op = name_items[2];

    const TensorProto& tensor_proto = val.tensor();
    Tensor tensor(tensor_proto.dtype());
    if (!tensor.FromProto(tensor_proto)) {
      return ::grpc::Status::CANCELLED;
    }

    string dump_path;
    DebugFileIO::DumpTensorToDir(node_name, output_slot, debug_op, tensor,
                                 event.wall_time(), dump_root, &dump_path)
        .IgnoreError();
  }

  return ::grpc::Status::OK;
}

GrpcTestServerClientPair::GrpcTestServerClientPair(const int server_port)
    : server_port(server_port) {
  const int kTensorSize = 2;
  prep_tensor_.reset(
      new Tensor(DT_FLOAT, TensorShape({kTensorSize, kTensorSize})));
  for (int i = 0; i < kTensorSize * kTensorSize; ++i) {
    prep_tensor_->flat<float>()(i) = static_cast<float>(i);
  }

  // Obtain server's gRPC url.
  test_server_url = strings::StrCat("grpc://localhost:", server_port);

  // Obtain dump directory for the stream server.
  string tmp_dir = port::Tracing::LogDir();
  dump_root =
      io::JoinPath(tmp_dir, strings::StrCat("tfdbg_dump_port", server_port, "_",
                                            Env::Default()->NowMicros()));
}

bool GrpcTestServerClientPair::PollTillFirstRequestSucceeds() {
  const std::vector<string> urls({test_server_url});
  int n_attempts = 0;
  bool success = false;

  // Try a number of times to send the Event proto to the server, as it may
  // take the server a few seconds to start up and become responsive.
  while (n_attempts++ < kMaxAttempts) {
    const uint64 wall_time = Env::Default()->NowMicros();

    Status publish_s = DebugIO::PublishDebugTensor(
        "prep_node:0", "DebugIdentity", *prep_tensor_, wall_time, urls);
    Status close_s = DebugIO::CloseDebugURL(test_server_url);

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
