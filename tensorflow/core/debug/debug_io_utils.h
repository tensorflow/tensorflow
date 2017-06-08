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

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {

Status ReadEventFromFile(const string& dump_file_path, Event* event);

struct DebugWatchAndURLSpec {
  DebugWatchAndURLSpec(const string& watch_key, const string& url,
                       const bool gated_grpc)
      : watch_key(watch_key), url(url), gated_grpc(gated_grpc) {}

  const string watch_key;
  const string url;
  const bool gated_grpc;
};

struct DebugNodeKey {
  DebugNodeKey(const string& device_name, const string& node_name,
               const int32 output_slot, const string& debug_op);

  static const string DeviceNameToDevicePath(const string& device_name);

  const string device_name;
  const string node_name;
  const int32 output_slot;
  const string debug_op;
  const string debug_node_name;
  const string device_path;
};

class DebugIO {
 public:
  static Status PublishDebugMetadata(
      const int64 global_step, const int64 session_run_index,
      const int64 executor_step_index, const std::vector<string>& input_names,
      const std::vector<string>& output_names,
      const std::vector<string>& target_nodes,
      const std::unordered_set<string>& debug_urls);

  // Publish a tensor to a debug target URL.
  //
  // Args:
  //   debug_node_key: A DebugNodeKey identifying the debug node.
  //   tensor: The Tensor object being published.
  //   wall_time_us: Time stamp for the Tensor. Unit: microseconds (us).
  //   debug_urls: An array of debug target URLs, e.g.,
  //     "file:///foo/tfdbg_dump", "grpc://localhost:11011"
  //   gated_grpc: Whether this call is subject to gRPC gating.
  static Status PublishDebugTensor(const DebugNodeKey& debug_node_key,
                                   const Tensor& tensor,
                                   const uint64 wall_time_us,
                                   const gtl::ArraySlice<string>& debug_urls,
                                   const bool gated_grpc);

  // Convenience overload of the method above for no gated_grpc by default.
  static Status PublishDebugTensor(const DebugNodeKey& debug_node_key,
                                   const Tensor& tensor,
                                   const uint64 wall_time_us,
                                   const gtl::ArraySlice<string>& debug_urls);

  // Publish a graph to a set of debug URLs.
  //
  // Args:
  //   graph: The graph to be published.
  //   debug_urls: The set of debug URLs to publish the graph to.
  static Status PublishGraph(const Graph& graph, const string& device_name,
                             const std::unordered_set<string>& debug_urls);

  // Determine whether a copy node needs to perform deep-copy of input tensor.
  //
  // The input arguments contain sufficient information about the attached
  // downstream debug ops for this method to determine whether all the said
  // ops are disabled given the current status of the gRPC gating.
  //
  // Args:
  //   specs: A vector of DebugWatchAndURLSpec carrying information about the
  //     debug ops attached to the Copy node, their debug URLs and whether
  //     they have the attribute value gated_grpc == True.
  //
  // Returns:
  //   Whether any of the attached downstream debug ops is enabled given the
  //   current status of the gRPC gating.
  static bool IsCopyNodeGateOpen(
      const std::vector<DebugWatchAndURLSpec>& specs);

  // Determine whether a debug node needs to proceed given the current gRPC
  // gating status.
  //
  // Args:
  //   watch_key: debug tensor watch key, in the format of
  //     tensor_name:debug_op, e.g., "Weights:0:DebugIdentity".
  //   debug_urls: the debug URLs of the debug node.
  //
  // Returns:
  //   Whether this debug op should proceed.
  static bool IsDebugNodeGateOpen(const string& watch_key,
                                  const std::vector<string>& debug_urls);

  // Determine whether debug information should be sent through a grpc://
  // debug URL given the current gRPC gating status.
  //
  // Args:
  //   watch_key: debug tensor watch key, in the format of
  //     tensor_name:debug_op, e.g., "Weights:0:DebugIdentity".
  //   debug_url: the debug URL, e.g., "grpc://localhost:3333",
  //     "file:///tmp/tfdbg_1".
  //
  // Returns:
  //   Whether the sending of debug data to the debug_url should
  //     proceed.
  static bool IsDebugURLGateOpen(const string& watch_key,
                                 const string& debug_url);

  static Status CloseDebugURL(const string& debug_url);

  static const char* const kMetadataFilePrefix;
  static const char* const kCoreMetadataTag;
  static const char* const kDeviceTag;
  static const char* const kGraphTag;

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
  //   debug_node_key: A DebugNodeKey identifying the debug node.
  //   wall_time_us: Wall time at which the Tensor is generated during graph
  //     execution. Unit: microseconds (us).
  //   dump_root_dir: Root directory for dumping the tensor.
  //   dump_file_path: The actual dump file path (passed as reference).
  static Status DumpTensorToDir(const DebugNodeKey& debug_node_key,
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
                                const DebugNodeKey& debug_node_key,
                                const uint64 wall_time_us);

  static Status DumpEventProtoToFile(const Event& event_proto,
                                     const string& dir_name,
                                     const string& file_name);

 private:
  // Encapsulate the Tensor in an Event protobuf and write it to file.
  static Status DumpTensorToEventFile(const DebugNodeKey& debug_node_key,
                                      const Tensor& tensor,
                                      const uint64 wall_time_us,
                                      const string& file_path);

  // Implemented ad hoc here for now.
  // TODO(cais): Replace with shared implementation once http://b/30497715 is
  // fixed.
  static Status RecursiveCreateDir(Env* env, const string& dir);
};

}  // namespace tensorflow

// TODO(cais): Support grpc:// debug URLs in open source once Python grpc
//   genrule becomes available. See b/23796275.
#if defined(PLATFORM_GOOGLE)
#include "tensorflow/core/debug/debug_service.grpc.pb.h"

namespace tensorflow {

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

  // Attempt to establish connection with server.
  //
  // Args:
  //   timeout_micros: Timeout (in microseconds) for the attempt to establish
  //     the connection.
  //
  // Returns:
  //   OK Status iff connection is successfully established before timeout,
  //   otherwise return an error Status.
  Status Connect(const int64 timeout_micros);

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

  // Receive EventReplies from server (if any) and close the stream and the
  // channel.
  Status ReceiveServerRepliesAndClose();

 private:
  string server_stream_addr_;
  string url_;
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
  static Status SendTensorThroughGrpcStream(const DebugNodeKey& debug_node_key,
                                            const Tensor& tensor,
                                            const uint64 wall_time_us,
                                            const string& grpc_stream_url,
                                            const bool gated);

  // Send an Event proto through a debug gRPC stream.
  // Thread-safety: Safe with respect to other calls to the same method and
  // calls to CloseGrpcStream().
  static Status SendEventProtoThroughGrpcStream(const Event& event_proto,
                                                const string& grpc_stream_url);

  // Check whether a debug watch key is allowed to send data to a given grpc://
  // debug URL given the current gating status.
  //
  // Args:
  //   watch_key: debug tensor watch key, in the format of
  //     tensor_name:debug_op, e.g., "Weights:0:DebugIdentity".
  //   grpc_debug_url: the debug URL, e.g., "grpc://localhost:3333",
  //
  // Returns:
  //   Whether the sending of debug data to grpc_debug_url should
  //     proceed.
  static bool IsGateOpen(const string& watch_key, const string& grpc_debug_url);

  // Close a gRPC stream to the given address, if it exists.
  // Thread-safety: Safe with respect to other calls to the same method and
  // calls to SendTensorThroughGrpcStream().
  static Status CloseGrpcStream(const string& grpc_stream_url);

  // Enable a debug watch key at a grpc:// debug URL.
  static void EnableWatchKey(const string& grpc_debug_url,
                             const string& watch_key);

  // Disable a debug watch key at a grpc:// debug URL.
  static void DisableWatchKey(const string& grpc_debug_url,
                              const string& watch_key);

 private:
  // Returns a global map from grpc debug URLs to the corresponding
  // DebugGrpcChannels.
  static std::unordered_map<string, std::shared_ptr<DebugGrpcChannel>>*
  GetStreamChannels();

  // Returns a global map from grpc debug URLs to the enabled gated debug nodes.
  // The keys are grpc:// URLs of the debug servers, e.g., "grpc://foo:3333".
  // Each value element of the value has the format
  // <node_name>:<output_slot>:<debug_op>", e.g.,
  // "Weights_1:0:DebugNumericSummary".
  static std::unordered_map<string, std::unordered_set<string>>*
  GetEnabledWatchKeys();

  static void ClearEnabledWatchKeys();
  static void CreateEmptyEnabledSet(const string& grpc_debug_url);

  static mutex streams_mu;
  static int64 channel_connection_timeout_micros;

  friend class GrpcDebugTest;
  friend class DebugNumericSummaryOpTest;
};

}  // namespace tensorflow
#endif  // #if defined(PLATFORM_GOOGLE)

#endif  // TENSORFLOW_DEBUG_IO_UTILS_H_
