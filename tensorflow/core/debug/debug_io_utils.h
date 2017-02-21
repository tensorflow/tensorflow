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

#ifndef TENSORFLOW_DEBUG_IO_UTILS_H_
#define TENSORFLOW_DEBUG_IO_UTILS_H_

#include <unordered_map>
#include <unordered_set>

#include "tensorflow/core/debug/debug_service.grpc.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

Status ReadEventFromFile(const string& dump_file_path, Event* event);

class DebugIO {
 public:
  static Status PublishDebugMetadata(
      const int64 global_step, const int64 session_run_count,
      const int64 executor_step_count, const std::vector<string>& input_names,
      const std::vector<string>& output_names,
      const std::vector<string>& target_nodes,
      const std::unordered_set<string>& debug_urls);

  // Publish a tensor to a debug target URL.
  //
  // Args:
  //   tensor_name: Name of the tensor being published: node_name followed by
  //     a colon, followed by the output slot index. E.g., "node_a:0".
  //     N.B.: Use the original tensor name, i.e., name of the input tensor to
  //     the debug op, even if the debug_op is not DebugIdentity.
  //   debug_op: Name of the debug op, e.g., "DebugIdentity".
  //   tensor: The Tensor object being published.
  //   wall_time_us: Time stamp for the Tensor. Unit: microseconds (us).
  //   debug_urls: An array of debug target URLs, e.g.,
  //     "file:///foo/tfdbg_dump", "grpc://localhost:11011"
  static Status PublishDebugTensor(const string& tensor_name,
                                   const string& debug_op, const Tensor& tensor,
                                   const uint64 wall_time_us,
                                   const gtl::ArraySlice<string>& debug_urls);

  // Publish a graph to a set of debug URLs.
  //
  // Args:
  //   graph: The graph to be published.
  //   debug_urls: The set of debug URLs to publish the graph to.
  static Status PublishGraph(const Graph& graph,
                             const std::unordered_set<string>& debug_urls);

  static Status CloseDebugURL(const string& debug_url);

  static const char* const kFileURLScheme;
  static const char* const kGrpcURLScheme;
};

// Helper class for debug ops.
class DebugFileIO {
 public:
  // Encapsulate the Tensor in an Event protobuf and write it to a directory.
  // The actual path of the dump file will be a contactenation of
  // dump_root_dir, tensor_name, along with the wall_time.
  //
  // For example:
  //   let dump_root_dir = "/tmp/tfdbg_dump",
  //       node_name = "foo/bar",
  //       output_slot = 0,
  //       debug_op = DebugIdentity,
  //       and wall_time_us = 1467891234512345,
  // the dump file will be generated at path:
  //   /tmp/tfdbg_dump/foo/bar_0_DebugIdentity_1467891234512345.
  //
  // Args:
  //   node_name: Name of the node from which the tensor is output.
  //   output_slot: Output slot index.
  //   debug_op: Name of the debug op, e.g., "DebugIdentity".
  //   tensor: The Tensor object to be dumped to file.
  //   wall_time_us: Wall time at which the Tensor is generated during graph
  //     execution. Unit: microseconds (us).
  //   dump_root_dir: Root directory for dumping the tensor.
  //   dump_file_path: The actual dump file path (passed as reference).
  static Status DumpTensorToDir(const string& node_name,
                                const int32 output_slot, const string& debug_op,
                                const Tensor& tensor, const uint64 wall_time_us,
                                const string& dump_root_dir,
                                string* dump_file_path);

  // Get the full path to the dump file.
  //
  // Args:
  //   dump_root_dir: The dump root directory, e.g., /tmp/tfdbg_dump
  //   node_name: Name of the node from which the dumped tensor is generated,
  //     e.g., foo/bar/node_a
  //   output_slot: Output slot index of the said node, e.g., 0.
  //   debug_op: Name of the debug op, e.g., DebugIdentity.
  //   wall_time_us: Time stamp of the dumped tensor, in microseconds (us).
  static string GetDumpFilePath(const string& dump_root_dir,
                                const string& node_name,
                                const int32 output_slot, const string& debug_op,
                                const uint64 wall_time_us);

  static Status DumpEventProtoToFile(const Event& event_proto,
                                     const string& dir_name,
                                     const string& file_name);

 private:
  // Encapsulate the Tensor in an Event protobuf and write it to file.
  static Status DumpTensorToEventFile(
      const string& node_name, const int32 output_slot, const string& debug_op,
      const Tensor& tensor, const uint64 wall_time_us, const string& file_path);

  // Implemented ad hoc here for now.
  // TODO(cais): Replace with shared implementation once http://b/30497715 is
  // fixed.
  static Status RecursiveCreateDir(Env* env, const string& dir);
};

class DebugGrpcChannel {
 public:
  // Constructor of DebugGrpcChannel.
  //
  // Args:
  //   server_stream_addr: Address (host name and port) of the debug stream
  //     server implementing the EventListener service (see
  //     debug_service.proto). E.g., "127.0.0.1:12345".
  DebugGrpcChannel(const string& server_stream_addr);

  virtual ~DebugGrpcChannel() {}

  // Query whether the gRPC channel is ready for use.
  bool is_channel_ready();

  // Write an Event proto to the debug gRPC stream.
  //
  // Thread-safety: Safe with respect to other calls to the same method and
  //   call to Close().
  // Args:
  //   event: The event proto to be written to the stream.
  //
  // Returns:
  //   True iff the write is successful.
  bool WriteEvent(const Event& event);

  // Close the stream and the channel.
  Status Close();

 private:
  ::grpc::ClientContext ctx_;
  std::shared_ptr<::grpc::Channel> channel_;
  std::unique_ptr<EventListener::Stub> stub_;
  std::unique_ptr<::grpc::ClientReaderWriterInterface<Event, EventReply>>
      reader_writer_;

  mutex mu_;
};

class DebugGrpcIO {
 public:
  // Send a tensor through a debug gRPC stream.
  static Status SendTensorThroughGrpcStream(const string& node_name,
                                            const int32 output_slot,
                                            const string& debug_op,
                                            const Tensor& tensor,
                                            const uint64 wall_time_us,
                                            const string& grpc_stream_url);

  // Send an Event proto through a debug gRPC stream.
  // Thread-safety: Safe with respect to other calls to the same method and
  // calls to CloseGrpcStream().
  static Status SendEventProtoThroughGrpcStream(const Event& event_proto,
                                                const string& grpc_stream_url);

  // Close a gRPC stream to the given address, if it exists.
  // Thread-safety: Safe with respect to other calls to the same method and
  // calls to SendTensorThroughGrpcStream().
  static Status CloseGrpcStream(const string& grpc_stream_url);

 private:
  static mutex streams_mu;
  static std::unordered_map<string, std::shared_ptr<DebugGrpcChannel>>
      stream_channels GUARDED_BY(streams_mu);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_DEBUG_IO_UTILS_H_
