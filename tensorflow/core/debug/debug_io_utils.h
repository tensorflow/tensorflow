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

#ifndef TENSORFLOW_CORE_DEBUG_DEBUG_IO_UTILS_H_
#define TENSORFLOW_CORE_DEBUG_DEBUG_IO_UTILS_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/debug/debug_node_key.h"
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

// TODO(cais): Put static functions and members in a namespace, not a class.
class DebugIO {
 public:
  static const char* const kDebuggerPluginName;

  static const char* const kCoreMetadataTag;
  static const char* const kGraphTag;
  static const char* const kHashTag;

  static const char* const kFileURLScheme;
  static const char* const kGrpcURLScheme;
  static const char* const kMemoryURLScheme;

  static Status PublishDebugMetadata(
      const int64 global_step, const int64 session_run_index,
      const int64 executor_step_index, const std::vector<string>& input_names,
      const std::vector<string>& output_names,
      const std::vector<string>& target_nodes,
      const std::unordered_set<string>& debug_urls);

  // Publishes a tensor to a debug target URL.
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

  // Publishes a graph to a set of debug URLs.
  //
  // Args:
  //   graph: The graph to be published.
  //   debug_urls: The set of debug URLs to publish the graph to.
  static Status PublishGraph(const Graph& graph, const string& device_name,
                             const std::unordered_set<string>& debug_urls);

  // Determines whether a copy node needs to perform deep-copy of input tensor.
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

  // Determines whether a debug node needs to proceed given the current gRPC
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

  // Determines whether debug information should be sent through a grpc://
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
};

// Helper class for debug ops.
class DebugFileIO {
 public:
  // Encapsulates the Tensor in an Event protobuf and write it to a directory.
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

  // Dumps an Event proto to a file.
  //
  // Args:
  //   event_prot: The Event proto to be dumped.
  //   dir_name: Directory path.
  //   file_name: Base file name.
  static Status DumpEventProtoToFile(const Event& event_proto,
                                     const string& dir_name,
                                     const string& file_name);

  // Request additional bytes to be dumped to the file system.
  //
  // Does not actually dump the bytes, but instead just performs the
  // bookkeeping necessary to prevent the total dumped amount of data from
  // exceeding the limit (default 100 GBytes or set customly through the
  // environment variable TFDBG_DISK_BYTES_LIMIT).
  //
  // Args:
  //   bytes: Number of bytes to request.
  //
  // Returns:
  //   Whether the request is approved given the total dumping
  //   limit.
  static bool requestDiskByteUsage(uint64 bytes);

  // Reset the disk byte usage to zero.
  static void resetDiskByteUsage();

  static uint64 globalDiskBytesLimit;

 private:
  // Encapsulates the Tensor in an Event protobuf and write it to file.
  static Status DumpTensorToEventFile(const DebugNodeKey& debug_node_key,
                                      const Tensor& tensor,
                                      const uint64 wall_time_us,
                                      const string& file_path);

  // Implemented ad hoc here for now.
  // TODO(cais): Replace with shared implementation once http://b/30497715 is
  // fixed.
  static Status RecursiveCreateDir(Env* env, const string& dir);

  // Tracks how much disk has been used so far.
  static uint64 diskBytesUsed;
  // Mutex for thread-safe access to diskBytesUsed.
  static mutex bytes_mu;
  // Default limit for the disk space.
  static const uint64 defaultGlobalDiskBytesLimit;

  friend class DiskUsageLimitTest;
};

}  // namespace tensorflow

namespace std {

template <>
struct hash<::tensorflow::DebugNodeKey> {
  size_t operator()(const ::tensorflow::DebugNodeKey& k) const {
    return ::tensorflow::Hash64(
        ::tensorflow::strings::StrCat(k.device_name, ":", k.node_name, ":",
                                      k.output_slot, ":", k.debug_op, ":"));
  }
};

}  // namespace std

// TODO(cais): Support grpc:// debug URLs in open source once Python grpc
//   genrule becomes available. See b/23796275.
#ifndef PLATFORM_WINDOWS
#include "grpcpp/channel.h"
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
  //   calls to ReadEventReply() and Close().
  //
  // Args:
  //   event: The event proto to be written to the stream.
  //
  // Returns:
  //   True iff the write is successful.
  bool WriteEvent(const Event& event);

  // Read an EventReply proto from the debug gRPC stream.
  //
  // This method blocks and waits for an EventReply from the server.
  // Thread-safety: Safe with respect to other calls to the same method and
  //   calls to WriteEvent() and Close().
  //
  // Args:
  //   event_reply: the to-be-modified EventReply proto passed as reference.
  //
  // Returns:
  //   True iff the read is successful.
  bool ReadEventReply(EventReply* event_reply);

  // Receive and process EventReply protos from the gRPC debug server.
  //
  // The processing includes setting debug watch key states using the
  // DebugOpStateChange fields of the EventReply.
  //
  // Args:
  //   max_replies: Maximum number of replies to receive. Will receive all
  //     remaining replies iff max_replies == 0.
  void ReceiveAndProcessEventReplies(size_t max_replies);

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
  static const size_t kGrpcMessageSizeLimitBytes;
  static const size_t kGrpcMaxVarintLengthSize;

  // Sends a tensor through a debug gRPC stream.
  static Status SendTensorThroughGrpcStream(const DebugNodeKey& debug_node_key,
                                            const Tensor& tensor,
                                            const uint64 wall_time_us,
                                            const string& grpc_stream_url,
                                            const bool gated);

  // Sends an Event proto through a debug gRPC stream.
  // Thread-safety: Safe with respect to other calls to the same method and
  // calls to CloseGrpcStream().
  //
  // Args:
  //   event_proto: The Event proto to be sent.
  //   grpc_stream_url: The grpc:// URL of the stream to use, e.g.,
  //     "grpc://localhost:11011", "localhost:22022".
  //   receive_reply: Whether an EventReply proto will be read after event_proto
  //     is sent and before the function returns.
  //
  // Returns:
  //   The Status of the operation.
  static Status SendEventProtoThroughGrpcStream(
      const Event& event_proto, const string& grpc_stream_url,
      const bool receive_reply = false);

  // Receive an EventReply proto through a debug gRPC stream.
  static Status ReceiveEventReplyProtoThroughGrpcStream(
      EventReply* event_reply, const string& grpc_stream_url);

  // Check whether a debug watch key is read-activated at a given gRPC URL.
  static bool IsReadGateOpen(const string& grpc_debug_url,
                             const string& watch_key);

  // Check whether a debug watch key is write-activated (i.e., read- and
  // write-activated) at a given gRPC URL.
  static bool IsWriteGateOpen(const string& grpc_debug_url,
                              const string& watch_key);

  // Closes a gRPC stream to the given address, if it exists.
  // Thread-safety: Safe with respect to other calls to the same method and
  // calls to SendTensorThroughGrpcStream().
  static Status CloseGrpcStream(const string& grpc_stream_url);

  // Set the gRPC state of a debug node key.
  // TODO(cais): Include device information in watch_key.
  static void SetDebugNodeKeyGrpcState(
      const string& grpc_debug_url, const string& watch_key,
      const EventReply::DebugOpStateChange::State new_state);

 private:
  using DebugNodeName2State =
      std::unordered_map<string, EventReply::DebugOpStateChange::State>;

  // Returns a global map from grpc debug URLs to the corresponding
  // DebugGrpcChannels.
  static std::unordered_map<string, std::unique_ptr<DebugGrpcChannel>>*
  GetStreamChannels();

  // Get a DebugGrpcChannel object at a given URL, creating one if necessary.
  //
  // Args:
  //   grpc_stream_url: grpc:// URL of the stream, e.g., "grpc://localhost:6064"
  //   debug_grpc_channel: A pointer to the DebugGrpcChannel object, passed as a
  //     a pointer to the pointer. The DebugGrpcChannel object is owned
  //     statically elsewhere, not by the caller of this function.
  //
  // Returns:
  //   Status of this operation.
  static Status GetOrCreateDebugGrpcChannel(
      const string& grpc_stream_url, DebugGrpcChannel** debug_grpc_channel);

  // Returns a map from debug URL to a map from debug op name to enabled state.
  static std::unordered_map<string, DebugNodeName2State>*
  GetEnabledDebugOpStates();

  // Returns a map from debug op names to enabled state, for a given debug URL.
  static DebugNodeName2State* GetEnabledDebugOpStatesAtUrl(
      const string& grpc_debug_url);

  // Clear enabled debug op state from all debug URLs (if any).
  static void ClearEnabledWatchKeys();

  static mutex streams_mu;
  static int64 channel_connection_timeout_micros;

  friend class GrpcDebugTest;
  friend class DebugNumericSummaryOpTest;
};

}  // namespace tensorflow
#endif  // #ifndef(PLATFORM_WINDOWS)

#endif  // TENSORFLOW_CORE_DEBUG_DEBUG_IO_UTILS_H_
