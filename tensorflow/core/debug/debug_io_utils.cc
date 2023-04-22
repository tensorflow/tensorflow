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

#include "tensorflow/core/debug/debug_io_utils.h"

#include <stddef.h>
#include <string.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#ifndef PLATFORM_WINDOWS
#include "grpcpp/create_channel.h"
#else
#endif  // #ifndef PLATFORM_WINDOWS

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "tensorflow/core/debug/debug_callback_registry.h"
#include "tensorflow/core/debug/debugger_event_metadata.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/event.pb.h"

#define GRPC_OSS_WINDOWS_UNIMPLEMENTED_ERROR \
  return errors::Unimplemented(              \
      kGrpcURLScheme, " debug URL scheme is not implemented on Windows yet.")

namespace tensorflow {

namespace {

// Creates an Event proto representing a chunk of a Tensor. This method only
// populates the field of the Event proto that represent the envelope
// information (e.g., timestamp, device_name, num_chunks, chunk_index, dtype,
// shape). It does not set the value.tensor field, which should be set by the
// caller separately.
Event PrepareChunkEventProto(const DebugNodeKey& debug_node_key,
                             const uint64 wall_time_us, const size_t num_chunks,
                             const size_t chunk_index,
                             const DataType& tensor_dtype,
                             const TensorShapeProto& tensor_shape) {
  Event event;
  event.set_wall_time(static_cast<double>(wall_time_us));
  Summary::Value* value = event.mutable_summary()->add_value();

  // Create the debug node_name in the Summary proto.
  // For example, if tensor_name = "foo/node_a:0", and the debug_op is
  // "DebugIdentity", the debug node_name in the Summary proto will be
  // "foo/node_a:0:DebugIdentity".
  value->set_node_name(debug_node_key.debug_node_name);

  // Tag by the node name. This allows TensorBoard to quickly fetch data
  // per op.
  value->set_tag(debug_node_key.node_name);

  // Store data within debugger metadata to be stored for each event.
  third_party::tensorflow::core::debug::DebuggerEventMetadata metadata;
  metadata.set_device(debug_node_key.device_name);
  metadata.set_output_slot(debug_node_key.output_slot);
  metadata.set_num_chunks(num_chunks);
  metadata.set_chunk_index(chunk_index);

  // Encode the data in JSON.
  string json_output;
  tensorflow::protobuf::util::JsonPrintOptions json_options;
  json_options.always_print_primitive_fields = true;
  auto status = tensorflow::protobuf::util::MessageToJsonString(
      metadata, &json_output, json_options);
  if (status.ok()) {
    // Store summary metadata. Set the plugin to use this data as "debugger".
    SummaryMetadata::PluginData* plugin_data =
        value->mutable_metadata()->mutable_plugin_data();
    plugin_data->set_plugin_name(DebugIO::kDebuggerPluginName);
    plugin_data->set_content(json_output);
  } else {
    LOG(WARNING) << "Failed to convert DebuggerEventMetadata proto to JSON. "
                 << "The debug_node_name is " << debug_node_key.debug_node_name
                 << ".";
  }

  value->mutable_tensor()->set_dtype(tensor_dtype);
  *value->mutable_tensor()->mutable_tensor_shape() = tensor_shape;

  return event;
}

// Translates the length of a string to number of bytes when the string is
// encoded as bytes in protobuf. Note that this makes a conservative estimate
// (i.e., an estimate that is usually too large, but never too small under the
// gRPC message size limit) of the Varint-encoded length, to workaround the lack
// of a portable length function.
const size_t StringValMaxBytesInProto(const string& str) {
#if defined(PLATFORM_GOOGLE)
  return str.size() + DebugGrpcIO::kGrpcMaxVarintLengthSize;
#else
  return str.size();
#endif
}

// Breaks a string Tensor (represented as a TensorProto) as a vector of Event
// protos.
Status WrapStringTensorAsEvents(const DebugNodeKey& debug_node_key,
                                const uint64 wall_time_us,
                                const size_t chunk_size_limit,
                                TensorProto* tensor_proto,
                                std::vector<Event>* events) {
  const protobuf::RepeatedPtrField<string>& strs = tensor_proto->string_val();
  const size_t num_strs = strs.size();
  const size_t chunk_size_ub = chunk_size_limit > 0
                                   ? chunk_size_limit
                                   : std::numeric_limits<size_t>::max();

  // E.g., if cutoffs is {j, k, l}, the chunks will have index ranges:
  //   [0:a), [a:b), [c:<end>].
  std::vector<size_t> cutoffs;
  size_t chunk_size = 0;
  for (size_t i = 0; i < num_strs; ++i) {
    // Take into account the extra bytes in proto buffer.
    if (StringValMaxBytesInProto(strs[i]) > chunk_size_ub) {
      return errors::FailedPrecondition(
          "string value at index ", i, " from debug node ",
          debug_node_key.debug_node_name,
          " does not fit gRPC message size limit (", chunk_size_ub, ")");
    }
    if (chunk_size + StringValMaxBytesInProto(strs[i]) > chunk_size_ub) {
      cutoffs.push_back(i);
      chunk_size = 0;
    }
    chunk_size += StringValMaxBytesInProto(strs[i]);
  }
  cutoffs.push_back(num_strs);
  const size_t num_chunks = cutoffs.size();

  for (size_t i = 0; i < num_chunks; ++i) {
    Event event = PrepareChunkEventProto(debug_node_key, wall_time_us,
                                         num_chunks, i, tensor_proto->dtype(),
                                         tensor_proto->tensor_shape());
    Summary::Value* value = event.mutable_summary()->mutable_value(0);

    if (cutoffs.size() == 1) {
      value->mutable_tensor()->mutable_string_val()->Swap(
          tensor_proto->mutable_string_val());
    } else {
      const size_t begin = (i == 0) ? 0 : cutoffs[i - 1];
      const size_t end = cutoffs[i];
      for (size_t j = begin; j < end; ++j) {
        value->mutable_tensor()->add_string_val(strs[j]);
      }
    }

    events->push_back(std::move(event));
  }

  return Status::OK();
}

// Encapsulates the tensor value inside a vector of Event protos. Large tensors
// are broken up to multiple protos to fit the chunk_size_limit. In each Event
// proto the field summary.tensor carries the content of the tensor.
// If chunk_size_limit <= 0, the tensor will not be broken into chunks, i.e., a
// length-1 vector will be returned, regardless of the size of the tensor.
Status WrapTensorAsEvents(const DebugNodeKey& debug_node_key,
                          const Tensor& tensor, const uint64 wall_time_us,
                          const size_t chunk_size_limit,
                          std::vector<Event>* events) {
  TensorProto tensor_proto;
  if (tensor.dtype() == DT_STRING) {
    // Treat DT_STRING specially, so that tensor_util.MakeNdarray in Python can
    // convert the TensorProto to string-type numpy array. MakeNdarray does not
    // work with strings encoded by AsProtoTensorContent() in tensor_content.
    tensor.AsProtoField(&tensor_proto);

    TF_RETURN_IF_ERROR(WrapStringTensorAsEvents(
        debug_node_key, wall_time_us, chunk_size_limit, &tensor_proto, events));
  } else {
    tensor.AsProtoTensorContent(&tensor_proto);

    const size_t total_length = tensor_proto.tensor_content().size();
    const size_t chunk_size_ub =
        chunk_size_limit > 0 ? chunk_size_limit : total_length;
    const size_t num_chunks =
        (total_length == 0)
            ? 1
            : (total_length + chunk_size_ub - 1) / chunk_size_ub;
    for (size_t i = 0; i < num_chunks; ++i) {
      const size_t pos = i * chunk_size_ub;
      const size_t len =
          (i == num_chunks - 1) ? (total_length - pos) : chunk_size_ub;
      Event event = PrepareChunkEventProto(debug_node_key, wall_time_us,
                                           num_chunks, i, tensor_proto.dtype(),
                                           tensor_proto.tensor_shape());
      event.mutable_summary()
          ->mutable_value(0)
          ->mutable_tensor()
          ->set_tensor_content(tensor_proto.tensor_content().substr(pos, len));
      events->push_back(std::move(event));
    }
  }

  return Status::OK();
}

// Appends an underscore and a timestamp to a file path. If the path already
// exists on the file system, append a hyphen and a 1-up index. Consecutive
// values of the index will be tried until the first unused one is found.
// TOCTOU race condition is not of concern here due to the fact that tfdbg
// sets parallel_iterations attribute of all while_loops to 1 to prevent
// the same node from between executed multiple times concurrently.
string AppendTimestampToFilePath(const string& in, const uint64 timestamp) {
  string out = strings::StrCat(in, "_", timestamp);

  uint64 i = 1;
  while (Env::Default()->FileExists(out).ok()) {
    out = strings::StrCat(in, "_", timestamp, "-", i);
    ++i;
  }
  return out;
}

#ifndef PLATFORM_WINDOWS
// Publishes encoded GraphDef through a gRPC debugger stream, in chunks,
// conforming to the gRPC message size limit.
Status PublishEncodedGraphDefInChunks(const string& encoded_graph_def,
                                      const string& device_name,
                                      const int64_t wall_time,
                                      const string& debug_url) {
  const uint64 hash = ::tensorflow::Hash64(encoded_graph_def);
  const size_t total_length = encoded_graph_def.size();
  const size_t num_chunks =
      static_cast<size_t>(std::ceil(static_cast<float>(total_length) /
                                    DebugGrpcIO::kGrpcMessageSizeLimitBytes));
  for (size_t i = 0; i < num_chunks; ++i) {
    const size_t pos = i * DebugGrpcIO::kGrpcMessageSizeLimitBytes;
    const size_t len = (i == num_chunks - 1)
                           ? (total_length - pos)
                           : DebugGrpcIO::kGrpcMessageSizeLimitBytes;
    Event event;
    event.set_wall_time(static_cast<double>(wall_time));
    // Prefix the chunk with
    //   <hash64>,<device_name>,<wall_time>|<index>|<num_chunks>|.
    // TODO(cais): Use DebuggerEventMetadata to store device_name, num_chunks
    // and chunk_index, instead.
    event.set_graph_def(strings::StrCat(hash, ",", device_name, ",", wall_time,
                                        "|", i, "|", num_chunks, "|",
                                        encoded_graph_def.substr(pos, len)));
    const Status s = DebugGrpcIO::SendEventProtoThroughGrpcStream(
        event, debug_url, num_chunks - 1 == i);
    if (!s.ok()) {
      return errors::FailedPrecondition(
          "Failed to send chunk ", i, " of ", num_chunks,
          " of encoded GraphDef of size ", encoded_graph_def.size(), " bytes, ",
          "due to: ", s.error_message());
    }
  }
  return Status::OK();
}
#endif  // #ifndef PLATFORM_WINDOWS

}  // namespace

const char* const DebugIO::kDebuggerPluginName = "debugger";

const char* const DebugIO::kCoreMetadataTag = "core_metadata_";

const char* const DebugIO::kGraphTag = "graph_";

const char* const DebugIO::kHashTag = "hash";

Status ReadEventFromFile(const string& dump_file_path, Event* event) {
  Env* env(Env::Default());

  string content;
  uint64 file_size = 0;

  Status s = env->GetFileSize(dump_file_path, &file_size);
  if (!s.ok()) {
    return s;
  }

  content.resize(file_size);

  std::unique_ptr<RandomAccessFile> file;
  s = env->NewRandomAccessFile(dump_file_path, &file);
  if (!s.ok()) {
    return s;
  }

  StringPiece result;
  s = file->Read(0, file_size, &result, &(content)[0]);
  if (!s.ok()) {
    return s;
  }

  event->ParseFromString(content);
  return Status::OK();
}

const char* const DebugIO::kFileURLScheme = "file://";
const char* const DebugIO::kGrpcURLScheme = "grpc://";
const char* const DebugIO::kMemoryURLScheme = "memcbk://";

// Publishes debug metadata to a set of debug URLs.
Status DebugIO::PublishDebugMetadata(
    const int64_t global_step, const int64_t session_run_index,
    const int64_t executor_step_index, const std::vector<string>& input_names,
    const std::vector<string>& output_names,
    const std::vector<string>& target_nodes,
    const std::unordered_set<string>& debug_urls) {
  std::ostringstream oss;

  // Construct a JSON string to carry the metadata.
  oss << "{";
  oss << "\"global_step\":" << global_step << ",";
  oss << "\"session_run_index\":" << session_run_index << ",";
  oss << "\"executor_step_index\":" << executor_step_index << ",";
  oss << "\"input_names\":[";
  for (size_t i = 0; i < input_names.size(); ++i) {
    oss << "\"" << input_names[i] << "\"";
    if (i < input_names.size() - 1) {
      oss << ",";
    }
  }
  oss << "],";
  oss << "\"output_names\":[";
  for (size_t i = 0; i < output_names.size(); ++i) {
    oss << "\"" << output_names[i] << "\"";
    if (i < output_names.size() - 1) {
      oss << ",";
    }
  }
  oss << "],";
  oss << "\"target_nodes\":[";
  for (size_t i = 0; i < target_nodes.size(); ++i) {
    oss << "\"" << target_nodes[i] << "\"";
    if (i < target_nodes.size() - 1) {
      oss << ",";
    }
  }
  oss << "]";
  oss << "}";

  const string json_metadata = oss.str();
  Event event;
  event.set_wall_time(static_cast<double>(Env::Default()->NowMicros()));
  LogMessage* log_message = event.mutable_log_message();
  log_message->set_message(json_metadata);

  Status status;
  for (const string& url : debug_urls) {
    if (absl::StartsWith(absl::AsciiStrToLower(url), kGrpcURLScheme)) {
#ifndef PLATFORM_WINDOWS
      Event grpc_event;

      // Determine the path (if any) in the grpc:// URL, and add it as a field
      // of the JSON string.
      const string address = url.substr(strlen(DebugIO::kFileURLScheme));
      const string path = address.find('/') == string::npos
                              ? ""
                              : address.substr(address.find('/'));
      grpc_event.set_wall_time(event.wall_time());
      LogMessage* log_message_grpc = grpc_event.mutable_log_message();
      log_message_grpc->set_message(
          strings::StrCat(json_metadata.substr(0, json_metadata.size() - 1),
                          ",\"grpc_path\":\"", path, "\"}"));

      status.Update(
          DebugGrpcIO::SendEventProtoThroughGrpcStream(grpc_event, url, true));
#else
      GRPC_OSS_WINDOWS_UNIMPLEMENTED_ERROR;
#endif
    } else if (absl::StartsWith(absl::AsciiStrToLower(url), kFileURLScheme)) {
      const string dump_root_dir = url.substr(strlen(kFileURLScheme));
      const string core_metadata_path = AppendTimestampToFilePath(
          io::JoinPath(dump_root_dir,
                       strings::StrCat(
                           DebugNodeKey::kMetadataFilePrefix,
                           DebugIO::kCoreMetadataTag, "sessionrun",
                           strings::Printf("%.14lld", static_cast<long long>(
                                                          session_run_index)))),
          Env::Default()->NowMicros());
      status.Update(DebugFileIO::DumpEventProtoToFile(
          event, string(io::Dirname(core_metadata_path)),
          string(io::Basename(core_metadata_path))));
    }
  }

  return status;
}

Status DebugIO::PublishDebugTensor(const DebugNodeKey& debug_node_key,
                                   const Tensor& tensor,
                                   const uint64 wall_time_us,
                                   const gtl::ArraySlice<string> debug_urls,
                                   const bool gated_grpc) {
  int32 num_failed_urls = 0;
  std::vector<Status> fail_statuses;
  for (const string& url : debug_urls) {
    if (absl::StartsWith(absl::AsciiStrToLower(url), kFileURLScheme)) {
      const string dump_root_dir = url.substr(strlen(kFileURLScheme));

      const int64_t tensorBytes =
          tensor.IsInitialized() ? tensor.TotalBytes() : 0;
      if (!DebugFileIO::requestDiskByteUsage(tensorBytes)) {
        return errors::ResourceExhausted(
            "TensorFlow Debugger has exhausted file-system byte-size "
            "allowance (",
            DebugFileIO::global_disk_bytes_limit_, "), therefore it cannot ",
            "dump an additional ", tensorBytes, " byte(s) of tensor data ",
            "for the debug tensor ", debug_node_key.node_name, ":",
            debug_node_key.output_slot, ". You may use the environment ",
            "variable TFDBG_DISK_BYTES_LIMIT to set a higher limit.");
      }

      Status s = DebugFileIO::DumpTensorToDir(
          debug_node_key, tensor, wall_time_us, dump_root_dir, nullptr);
      if (!s.ok()) {
        num_failed_urls++;
        fail_statuses.push_back(s);
      }
    } else if (absl::StartsWith(absl::AsciiStrToLower(url), kGrpcURLScheme)) {
#ifndef PLATFORM_WINDOWS
      Status s = DebugGrpcIO::SendTensorThroughGrpcStream(
          debug_node_key, tensor, wall_time_us, url, gated_grpc);

      if (!s.ok()) {
        num_failed_urls++;
        fail_statuses.push_back(s);
      }
#else
      GRPC_OSS_WINDOWS_UNIMPLEMENTED_ERROR;
#endif
    } else if (absl::StartsWith(absl::AsciiStrToLower(url), kMemoryURLScheme)) {
      const string dump_root_dir = url.substr(strlen(kMemoryURLScheme));
      auto* callback_registry = DebugCallbackRegistry::singleton();
      auto* callback = callback_registry->GetCallback(dump_root_dir);
      CHECK(callback) << "No callback registered for: " << dump_root_dir;
      (*callback)(debug_node_key, tensor);
    } else {
      return Status(error::UNAVAILABLE,
                    strings::StrCat("Invalid debug target URL: ", url));
    }
  }

  if (num_failed_urls == 0) {
    return Status::OK();
  } else {
    string error_message = strings::StrCat(
        "Publishing to ", num_failed_urls, " of ", debug_urls.size(),
        " debug target URLs failed, due to the following errors:");
    for (Status& status : fail_statuses) {
      error_message =
          strings::StrCat(error_message, " ", status.error_message(), ";");
    }

    return Status(error::INTERNAL, error_message);
  }
}

Status DebugIO::PublishDebugTensor(const DebugNodeKey& debug_node_key,
                                   const Tensor& tensor,
                                   const uint64 wall_time_us,
                                   const gtl::ArraySlice<string> debug_urls) {
  return PublishDebugTensor(debug_node_key, tensor, wall_time_us, debug_urls,
                            false);
}

Status DebugIO::PublishGraph(const Graph& graph, const string& device_name,
                             const std::unordered_set<string>& debug_urls) {
  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);

  string buf;
  graph_def.SerializeToString(&buf);

  const int64_t now_micros = Env::Default()->NowMicros();
  Event event;
  event.set_wall_time(static_cast<double>(now_micros));
  event.set_graph_def(buf);

  Status status = Status::OK();
  for (const string& debug_url : debug_urls) {
    if (absl::StartsWith(debug_url, kFileURLScheme)) {
      const string dump_root_dir =
          io::JoinPath(debug_url.substr(strlen(kFileURLScheme)),
                       DebugNodeKey::DeviceNameToDevicePath(device_name));
      const uint64 graph_hash = ::tensorflow::Hash64(buf);
      const string file_name =
          strings::StrCat(DebugNodeKey::kMetadataFilePrefix, DebugIO::kGraphTag,
                          DebugIO::kHashTag, graph_hash, "_", now_micros);

      status.Update(
          DebugFileIO::DumpEventProtoToFile(event, dump_root_dir, file_name));
    } else if (absl::StartsWith(debug_url, kGrpcURLScheme)) {
#ifndef PLATFORM_WINDOWS
      status.Update(PublishEncodedGraphDefInChunks(buf, device_name, now_micros,
                                                   debug_url));
#else
      GRPC_OSS_WINDOWS_UNIMPLEMENTED_ERROR;
#endif
    }
  }

  return status;
}

bool DebugIO::IsCopyNodeGateOpen(
    const std::vector<DebugWatchAndURLSpec>& specs) {
#ifndef PLATFORM_WINDOWS
  for (const DebugWatchAndURLSpec& spec : specs) {
    if (!spec.gated_grpc || spec.url.compare(0, strlen(DebugIO::kGrpcURLScheme),
                                             DebugIO::kGrpcURLScheme)) {
      return true;
    } else {
      if (DebugGrpcIO::IsReadGateOpen(spec.url, spec.watch_key)) {
        return true;
      }
    }
  }
  return false;
#else
  return true;
#endif
}

bool DebugIO::IsDebugNodeGateOpen(const string& watch_key,
                                  const std::vector<string>& debug_urls) {
#ifndef PLATFORM_WINDOWS
  for (const string& debug_url : debug_urls) {
    if (debug_url.compare(0, strlen(DebugIO::kGrpcURLScheme),
                          DebugIO::kGrpcURLScheme)) {
      return true;
    } else {
      if (DebugGrpcIO::IsReadGateOpen(debug_url, watch_key)) {
        return true;
      }
    }
  }
  return false;
#else
  return true;
#endif
}

bool DebugIO::IsDebugURLGateOpen(const string& watch_key,
                                 const string& debug_url) {
#ifndef PLATFORM_WINDOWS
  if (debug_url != kGrpcURLScheme) {
    return true;
  } else {
    return DebugGrpcIO::IsReadGateOpen(debug_url, watch_key);
  }
#else
  return true;
#endif
}

Status DebugIO::CloseDebugURL(const string& debug_url) {
  if (absl::StartsWith(debug_url, DebugIO::kGrpcURLScheme)) {
#ifndef PLATFORM_WINDOWS
    return DebugGrpcIO::CloseGrpcStream(debug_url);
#else
    GRPC_OSS_WINDOWS_UNIMPLEMENTED_ERROR;
#endif
  } else {
    // No-op for non-gRPC URLs.
    return Status::OK();
  }
}

Status DebugFileIO::DumpTensorToDir(const DebugNodeKey& debug_node_key,
                                    const Tensor& tensor,
                                    const uint64 wall_time_us,
                                    const string& dump_root_dir,
                                    string* dump_file_path) {
  const string file_path =
      GetDumpFilePath(dump_root_dir, debug_node_key, wall_time_us);

  if (dump_file_path != nullptr) {
    *dump_file_path = file_path;
  }

  return DumpTensorToEventFile(debug_node_key, tensor, wall_time_us, file_path);
}

string DebugFileIO::GetDumpFilePath(const string& dump_root_dir,
                                    const DebugNodeKey& debug_node_key,
                                    const uint64 wall_time_us) {
  return AppendTimestampToFilePath(
      io::JoinPath(dump_root_dir, debug_node_key.device_path,
                   strings::StrCat(debug_node_key.node_name, "_",
                                   debug_node_key.output_slot, "_",
                                   debug_node_key.debug_op)),
      wall_time_us);
}

Status DebugFileIO::DumpEventProtoToFile(const Event& event_proto,
                                         const string& dir_name,
                                         const string& file_name) {
  Env* env(Env::Default());

  Status s = RecursiveCreateDir(env, dir_name);
  if (!s.ok()) {
    return Status(error::FAILED_PRECONDITION,
                  strings::StrCat("Failed to create directory  ", dir_name,
                                  ", due to: ", s.error_message()));
  }

  const string file_path = io::JoinPath(dir_name, file_name);

  string event_str;
  event_proto.SerializeToString(&event_str);

  std::unique_ptr<WritableFile> f = nullptr;
  TF_CHECK_OK(env->NewWritableFile(file_path, &f));
  f->Append(event_str).IgnoreError();
  TF_CHECK_OK(f->Close());

  return Status::OK();
}

Status DebugFileIO::DumpTensorToEventFile(const DebugNodeKey& debug_node_key,
                                          const Tensor& tensor,
                                          const uint64 wall_time_us,
                                          const string& file_path) {
  std::vector<Event> events;
  TF_RETURN_IF_ERROR(
      WrapTensorAsEvents(debug_node_key, tensor, wall_time_us, 0, &events));
  return DumpEventProtoToFile(events[0], string(io::Dirname(file_path)),
                              string(io::Basename(file_path)));
}

Status DebugFileIO::RecursiveCreateDir(Env* env, const string& dir) {
  if (env->FileExists(dir).ok() && env->IsDirectory(dir).ok()) {
    // The path already exists as a directory. Return OK right away.
    return Status::OK();
  }

  string parent_dir(io::Dirname(dir));
  if (!env->FileExists(parent_dir).ok()) {
    // The parent path does not exist yet, create it first.
    Status s = RecursiveCreateDir(env, parent_dir);  // Recursive call
    if (!s.ok()) {
      return Status(
          error::FAILED_PRECONDITION,
          strings::StrCat("Failed to create directory  ", parent_dir));
    }
  } else if (env->FileExists(parent_dir).ok() &&
             !env->IsDirectory(parent_dir).ok()) {
    // The path exists, but it is a file.
    return Status(error::FAILED_PRECONDITION,
                  strings::StrCat("Failed to create directory  ", parent_dir,
                                  " because the path exists as a file "));
  }

  env->CreateDir(dir).IgnoreError();
  // Guard against potential race in creating directories by doing a check
  // after the CreateDir call.
  if (env->FileExists(dir).ok() && env->IsDirectory(dir).ok()) {
    return Status::OK();
  } else {
    return Status(error::ABORTED,
                  strings::StrCat("Failed to create directory  ", parent_dir));
  }
}

// Default total disk usage limit: 100 GBytes
const uint64 DebugFileIO::kDefaultGlobalDiskBytesLimit = 107374182400L;
uint64 DebugFileIO::global_disk_bytes_limit_ = 0;
uint64 DebugFileIO::disk_bytes_used_ = 0;

mutex DebugFileIO::bytes_mu_(LINKER_INITIALIZED);

bool DebugFileIO::requestDiskByteUsage(uint64 bytes) {
  mutex_lock l(bytes_mu_);
  if (global_disk_bytes_limit_ == 0) {
    const char* env_tfdbg_disk_bytes_limit = getenv("TFDBG_DISK_BYTES_LIMIT");
    if (env_tfdbg_disk_bytes_limit == nullptr ||
        strlen(env_tfdbg_disk_bytes_limit) == 0) {
      global_disk_bytes_limit_ = kDefaultGlobalDiskBytesLimit;
    } else {
      strings::safe_strtou64(string(env_tfdbg_disk_bytes_limit),
                             &global_disk_bytes_limit_);
    }
  }

  if (bytes == 0) {
    return true;
  }
  if (disk_bytes_used_ + bytes < global_disk_bytes_limit_) {
    disk_bytes_used_ += bytes;
    return true;
  } else {
    return false;
  }
}

void DebugFileIO::resetDiskByteUsage() {
  mutex_lock l(bytes_mu_);
  disk_bytes_used_ = 0;
}

#ifndef PLATFORM_WINDOWS
DebugGrpcChannel::DebugGrpcChannel(const string& server_stream_addr)
    : server_stream_addr_(server_stream_addr),
      url_(strings::StrCat(DebugIO::kGrpcURLScheme, server_stream_addr)) {}

Status DebugGrpcChannel::Connect(const int64_t timeout_micros) {
  ::grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH, std::numeric_limits<int32>::max());
  // Avoid problems where default reconnect backoff is too long (e.g., 20 s).
  args.SetInt(GRPC_ARG_MAX_RECONNECT_BACKOFF_MS, 1000);
  channel_ = ::grpc::CreateCustomChannel(
      server_stream_addr_, ::grpc::InsecureChannelCredentials(), args);
  if (!channel_->WaitForConnected(
          gpr_time_add(gpr_now(GPR_CLOCK_REALTIME),
                       gpr_time_from_micros(timeout_micros, GPR_TIMESPAN)))) {
    return errors::FailedPrecondition(
        "Failed to connect to gRPC channel at ", server_stream_addr_,
        " within a timeout of ", timeout_micros / 1e6, " s.");
  }
  stub_ = EventListener::NewStub(channel_);
  reader_writer_ = stub_->SendEvents(&ctx_);

  return Status::OK();
}

bool DebugGrpcChannel::WriteEvent(const Event& event) {
  mutex_lock l(mu_);
  return reader_writer_->Write(event);
}

bool DebugGrpcChannel::ReadEventReply(EventReply* event_reply) {
  mutex_lock l(mu_);
  return reader_writer_->Read(event_reply);
}

void DebugGrpcChannel::ReceiveAndProcessEventReplies(const size_t max_replies) {
  EventReply event_reply;
  size_t num_replies = 0;
  while ((max_replies == 0 || ++num_replies <= max_replies) &&
         ReadEventReply(&event_reply)) {
    for (const EventReply::DebugOpStateChange& debug_op_state_change :
         event_reply.debug_op_state_changes()) {
      string watch_key = strings::StrCat(debug_op_state_change.node_name(), ":",
                                         debug_op_state_change.output_slot(),
                                         ":", debug_op_state_change.debug_op());
      DebugGrpcIO::SetDebugNodeKeyGrpcState(url_, watch_key,
                                            debug_op_state_change.state());
    }
  }
}

Status DebugGrpcChannel::ReceiveServerRepliesAndClose() {
  reader_writer_->WritesDone();
  // Read all EventReply messages (if any) from the server.
  ReceiveAndProcessEventReplies(0);

  if (reader_writer_->Finish().ok()) {
    return Status::OK();
  } else {
    return Status(error::FAILED_PRECONDITION,
                  "Failed to close debug GRPC stream.");
  }
}

mutex DebugGrpcIO::streams_mu_(LINKER_INITIALIZED);

int64_t DebugGrpcIO::channel_connection_timeout_micros_ = 900 * 1000 * 1000;
// TODO(cais): Make this configurable?

const size_t DebugGrpcIO::kGrpcMessageSizeLimitBytes = 4000 * 1024;

const size_t DebugGrpcIO::kGrpcMaxVarintLengthSize = 6;

std::unordered_map<string, std::unique_ptr<DebugGrpcChannel>>*
DebugGrpcIO::GetStreamChannels() {
  static std::unordered_map<string, std::unique_ptr<DebugGrpcChannel>>*
      stream_channels =
          new std::unordered_map<string, std::unique_ptr<DebugGrpcChannel>>();
  return stream_channels;
}

Status DebugGrpcIO::SendTensorThroughGrpcStream(
    const DebugNodeKey& debug_node_key, const Tensor& tensor,
    const uint64 wall_time_us, const string& grpc_stream_url,
    const bool gated) {
  if (gated &&
      !IsReadGateOpen(grpc_stream_url, debug_node_key.debug_node_name)) {
    return Status::OK();
  } else {
    std::vector<Event> events;
    TF_RETURN_IF_ERROR(WrapTensorAsEvents(debug_node_key, tensor, wall_time_us,
                                          kGrpcMessageSizeLimitBytes, &events));
    for (const Event& event : events) {
      TF_RETURN_IF_ERROR(
          SendEventProtoThroughGrpcStream(event, grpc_stream_url));
    }
    if (IsWriteGateOpen(grpc_stream_url, debug_node_key.debug_node_name)) {
      DebugGrpcChannel* debug_grpc_channel = nullptr;
      TF_RETURN_IF_ERROR(
          GetOrCreateDebugGrpcChannel(grpc_stream_url, &debug_grpc_channel));
      debug_grpc_channel->ReceiveAndProcessEventReplies(1);
      // TODO(cais): Support new tensor value carried in the EventReply for
      // overriding the value of the tensor being published.
    }
    return Status::OK();
  }
}

Status DebugGrpcIO::ReceiveEventReplyProtoThroughGrpcStream(
    EventReply* event_reply, const string& grpc_stream_url) {
  DebugGrpcChannel* debug_grpc_channel = nullptr;
  TF_RETURN_IF_ERROR(
      GetOrCreateDebugGrpcChannel(grpc_stream_url, &debug_grpc_channel));
  if (debug_grpc_channel->ReadEventReply(event_reply)) {
    return Status::OK();
  } else {
    return errors::Cancelled(strings::StrCat(
        "Reading EventReply from stream URL ", grpc_stream_url, " failed."));
  }
}

Status DebugGrpcIO::GetOrCreateDebugGrpcChannel(
    const string& grpc_stream_url, DebugGrpcChannel** debug_grpc_channel) {
  const string addr_with_path =
      absl::StartsWith(grpc_stream_url, DebugIO::kGrpcURLScheme)
          ? grpc_stream_url.substr(strlen(DebugIO::kGrpcURLScheme))
          : grpc_stream_url;
  const string server_stream_addr =
      addr_with_path.substr(0, addr_with_path.find('/'));
  {
    mutex_lock l(streams_mu_);
    std::unordered_map<string, std::unique_ptr<DebugGrpcChannel>>*
        stream_channels = GetStreamChannels();
    if (stream_channels->find(grpc_stream_url) == stream_channels->end()) {
      std::unique_ptr<DebugGrpcChannel> channel(
          new DebugGrpcChannel(server_stream_addr));
      TF_RETURN_IF_ERROR(channel->Connect(channel_connection_timeout_micros_));
      stream_channels->insert(
          std::make_pair(grpc_stream_url, std::move(channel)));
    }
    *debug_grpc_channel = (*stream_channels)[grpc_stream_url].get();
  }
  return Status::OK();
}

Status DebugGrpcIO::SendEventProtoThroughGrpcStream(
    const Event& event_proto, const string& grpc_stream_url,
    const bool receive_reply) {
  DebugGrpcChannel* debug_grpc_channel;
  TF_RETURN_IF_ERROR(
      GetOrCreateDebugGrpcChannel(grpc_stream_url, &debug_grpc_channel));

  bool write_ok = debug_grpc_channel->WriteEvent(event_proto);
  if (!write_ok) {
    return errors::Cancelled(strings::StrCat("Write event to stream URL ",
                                             grpc_stream_url, " failed."));
  }

  if (receive_reply) {
    debug_grpc_channel->ReceiveAndProcessEventReplies(1);
  }

  return Status::OK();
}

bool DebugGrpcIO::IsReadGateOpen(const string& grpc_debug_url,
                                 const string& watch_key) {
  const DebugNodeName2State* enabled_node_to_state =
      GetEnabledDebugOpStatesAtUrl(grpc_debug_url);
  return enabled_node_to_state->find(watch_key) != enabled_node_to_state->end();
}

bool DebugGrpcIO::IsWriteGateOpen(const string& grpc_debug_url,
                                  const string& watch_key) {
  const DebugNodeName2State* enabled_node_to_state =
      GetEnabledDebugOpStatesAtUrl(grpc_debug_url);
  auto it = enabled_node_to_state->find(watch_key);
  if (it == enabled_node_to_state->end()) {
    return false;
  } else {
    return it->second == EventReply::DebugOpStateChange::READ_WRITE;
  }
}

Status DebugGrpcIO::CloseGrpcStream(const string& grpc_stream_url) {
  mutex_lock l(streams_mu_);

  std::unordered_map<string, std::unique_ptr<DebugGrpcChannel>>*
      stream_channels = GetStreamChannels();
  if (stream_channels->find(grpc_stream_url) != stream_channels->end()) {
    // Stream of the specified address exists. Close it and remove it from
    // record.
    Status s =
        (*stream_channels)[grpc_stream_url]->ReceiveServerRepliesAndClose();
    (*stream_channels).erase(grpc_stream_url);
    return s;
  } else {
    // Stream of the specified address does not exist. No action.
    return Status::OK();
  }
}

std::unordered_map<string, DebugGrpcIO::DebugNodeName2State>*
DebugGrpcIO::GetEnabledDebugOpStates() {
  static std::unordered_map<string, DebugNodeName2State>*
      enabled_debug_op_states =
          new std::unordered_map<string, DebugNodeName2State>();
  return enabled_debug_op_states;
}

DebugGrpcIO::DebugNodeName2State* DebugGrpcIO::GetEnabledDebugOpStatesAtUrl(
    const string& grpc_debug_url) {
  static mutex* debug_ops_state_mu = new mutex();
  std::unordered_map<string, DebugNodeName2State>* states =
      GetEnabledDebugOpStates();

  mutex_lock l(*debug_ops_state_mu);
  if (states->find(grpc_debug_url) == states->end()) {
    DebugNodeName2State url_enabled_debug_op_states;
    (*states)[grpc_debug_url] = url_enabled_debug_op_states;
  }
  return &(*states)[grpc_debug_url];
}

void DebugGrpcIO::SetDebugNodeKeyGrpcState(
    const string& grpc_debug_url, const string& watch_key,
    const EventReply::DebugOpStateChange::State new_state) {
  DebugNodeName2State* states = GetEnabledDebugOpStatesAtUrl(grpc_debug_url);
  if (new_state == EventReply::DebugOpStateChange::DISABLED) {
    if (states->find(watch_key) == states->end()) {
      LOG(ERROR) << "Attempt to disable a watch key that is not currently "
                 << "enabled at " << grpc_debug_url << ": " << watch_key;
    } else {
      states->erase(watch_key);
    }
  } else if (new_state != EventReply::DebugOpStateChange::STATE_UNSPECIFIED) {
    (*states)[watch_key] = new_state;
  }
}

void DebugGrpcIO::ClearEnabledWatchKeys() {
  GetEnabledDebugOpStates()->clear();
}

#endif  // #ifndef PLATFORM_WINDOWS

}  // namespace tensorflow
