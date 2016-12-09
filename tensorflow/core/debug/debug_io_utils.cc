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

#include <vector>

#include "grpc++/create_channel.h"

#if defined(PLATFORM_WINDOWS)
// winsock2.h is used in grpc, so Ws2_32.lib is needed
#pragma comment(lib,"Ws2_32.lib")
#endif

#include "tensorflow/core/debug/debug_service.grpc.pb.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {

namespace {

// Encapsulate the tensor value inside a Summary proto, and then inside an
// Event proto.
Event WrapTensorAsEvent(const string& tensor_name, const string& debug_op,
                        const Tensor& tensor, const uint64 wall_time_us) {
  Event event;
  event.set_wall_time(static_cast<double>(wall_time_us));

  Summary::Value* summ_val = event.mutable_summary()->add_value();

  // Create the debug node_name in the Summary proto.
  // For example, if tensor_name = "foo/node_a:0", and the debug_op is
  // "DebugIdentity", the debug node_name in the Summary proto will be
  // "foo/node_a:0:DebugIdentity".
  const string debug_node_name = strings::StrCat(tensor_name, ":", debug_op);
  summ_val->set_node_name(debug_node_name);

  if (tensor.dtype() == DT_STRING) {
    // Treat DT_STRING specially, so that tensor_util.MakeNdarray can convert
    // the TensorProto to string-type numpy array. MakeNdarray does not work
    // with strings encoded by AsProtoTensorContent() in tensor_content.
    tensor.AsProtoField(summ_val->mutable_tensor());
  } else {
    tensor.AsProtoTensorContent(summ_val->mutable_tensor());
  }

  return event;
}

}  // namespace

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

// static
const char* const DebugIO::kFileURLScheme = "file://";
// static
const char* const DebugIO::kGrpcURLScheme = "grpc://";

Status DebugIO::PublishDebugTensor(const string& tensor_name,
                                   const string& debug_op, const Tensor& tensor,
                                   const uint64 wall_time_us,
                                   const gtl::ArraySlice<string>& debug_urls) {
  // Split the tensor_name into node name and output slot index.
  std::vector<string> name_items = str_util::Split(tensor_name, ':');
  string node_name;
  int32 output_slot = 0;
  if (name_items.size() == 2) {
    node_name = name_items[0];
    if (!strings::safe_strto32(name_items[1], &output_slot)) {
      return Status(error::INVALID_ARGUMENT,
                    strings::StrCat("Invalid string value for output_slot: \"",
                                    name_items[1], "\""));
    }
  } else if (name_items.size() == 1) {
    node_name = name_items[0];
  } else {
    return Status(
        error::INVALID_ARGUMENT,
        strings::StrCat("Failed to parse tensor name: \"", tensor_name, "\""));
  }

  int num_failed_urls = 0;
  std::vector<Status> fail_statuses;
  for (const string& url : debug_urls) {
    if (str_util::Lowercase(url).find(kFileURLScheme) == 0) {
      const string dump_root_dir = url.substr(strlen(kFileURLScheme));

      Status s =
          DebugFileIO::DumpTensorToDir(node_name, output_slot, debug_op, tensor,
                                       wall_time_us, dump_root_dir, nullptr);
      if (!s.ok()) {
        num_failed_urls++;
        fail_statuses.push_back(s);
      }
    } else if (str_util::Lowercase(url).find(kGrpcURLScheme) == 0) {
      const string grpc_server_stream_addr = url.substr(strlen(kGrpcURLScheme));
      Status s = DebugGrpcIO::SendTensorThroughGrpcStream(
          node_name, output_slot, debug_op, tensor, wall_time_us,
          grpc_server_stream_addr);

      if (!s.ok()) {
        num_failed_urls++;
        fail_statuses.push_back(s);
      }
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

Status DebugIO::CloseDebugURL(const string& debug_url) {
  if (debug_url.find(DebugIO::kGrpcURLScheme) == 0) {
    return DebugGrpcIO::CloseGrpcStream(
        debug_url.substr(strlen(DebugIO::kGrpcURLScheme)));
  } else {
    // No-op for non-gRPC URLs.
    return Status::OK();
  }
}

// static
static Status CloseDebugURL(const string& debug_url) { return Status::OK(); }

// static
Status DebugFileIO::DumpTensorToDir(
    const string& node_name, const int32 output_slot, const string& debug_op,
    const Tensor& tensor, const uint64 wall_time_us,
    const string& dump_root_dir, string* dump_file_path) {
  const string file_path = GetDumpFilePath(dump_root_dir, node_name,
                                           output_slot, debug_op, wall_time_us);

  if (dump_file_path != nullptr) {
    *dump_file_path = file_path;
  }

  return DumpTensorToEventFile(node_name, output_slot, debug_op, tensor,
                               wall_time_us, file_path);
}

// static
string DebugFileIO::GetDumpFilePath(const string& dump_root_dir,
                                    const string& node_name,
                                    const int32 output_slot,
                                    const string& debug_op,
                                    const uint64 wall_time_us) {
  return io::JoinPath(
      dump_root_dir, strings::StrCat(node_name, "_", output_slot, "_", debug_op,
                                     "_", wall_time_us));
}

// static
Status DebugFileIO::DumpTensorToEventFile(
    const string& node_name, const int32 output_slot, const string& debug_op,
    const Tensor& tensor, const uint64 wall_time_us, const string& file_path) {
  Env* env(Env::Default());

  // Create the directory if necessary.
  string file_dir = io::Dirname(file_path).ToString();
  Status s = DebugFileIO::RecursiveCreateDir(env, file_dir);

  if (!s.ok()) {
    return Status(error::FAILED_PRECONDITION,
                  strings::StrCat("Failed to create directory  ", file_dir,
                                  ", due to: ", s.error_message()));
  }

  const string tensor_name = strings::StrCat(node_name, ":", output_slot);
  Event event = WrapTensorAsEvent(tensor_name, debug_op, tensor, wall_time_us);

  string event_str;
  event.SerializeToString(&event_str);

  std::unique_ptr<WritableFile> f = nullptr;
  TF_CHECK_OK(env->NewWritableFile(file_path, &f));
  f->Append(event_str);
  TF_CHECK_OK(f->Close());

  return Status::OK();
}

// static
Status DebugFileIO::RecursiveCreateDir(Env* env, const string& dir) {
  if (env->FileExists(dir).ok() && env->IsDirectory(dir).ok()) {
    // The path already exists as a directory. Return OK right away.
    return Status::OK();
  }

  string parent_dir = io::Dirname(dir).ToString();
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

  env->CreateDir(dir);
  // Guard against potential race in creating directories by doing a check
  // after the CreateDir call.
  if (env->FileExists(dir).ok() && env->IsDirectory(dir).ok()) {
    return Status::OK();
  } else {
    return Status(error::ABORTED,
                  strings::StrCat("Failed to create directory  ", parent_dir));
  }
}

DebugGrpcChannel::DebugGrpcChannel(const string& server_stream_addr)
    : ctx_(),
      channel_(::grpc::CreateCustomChannel(server_stream_addr,
                                           ::grpc::InsecureChannelCredentials(),
                                           ::grpc::ChannelArguments())),
      stub_(EventListener::NewStub(channel_)),
      reader_writer_(stub_->SendEvents(&ctx_)),
      mu_() {}
// TODO(cais): Set GRPC_ARG_MAX_MESSAGE_LENGTH to max if necessary.

bool DebugGrpcChannel::is_channel_ready() {
  return channel_->GetState(false) == GRPC_CHANNEL_READY;
}

bool DebugGrpcChannel::WriteEvent(const Event& event) {
  mutex_lock l(mu_);

  return reader_writer_->Write(event);
}

Status DebugGrpcChannel::Close() {
  mutex_lock l(mu_);

  reader_writer_->WritesDone();
  if (reader_writer_->Finish().ok()) {
    return Status::OK();
  } else {
    return Status(error::FAILED_PRECONDITION,
                  "Failed to close debug GRPC stream.");
  }
}

// static
mutex DebugGrpcIO::streams_mu;
std::unordered_map<string, std::shared_ptr<DebugGrpcChannel>>
    DebugGrpcIO::stream_channels;

// static
Status DebugGrpcIO::SendTensorThroughGrpcStream(
    const string& node_name, const int32 output_slot, const string& debug_op,
    const Tensor& tensor, const uint64 wall_time_us,
    const string& server_stream_addr) {
  const string tensor_name = strings::StrCat(node_name, ":", output_slot);

  // Prepare tensor Event data to be sent.
  Event event = WrapTensorAsEvent(tensor_name, debug_op, tensor, wall_time_us);

  std::shared_ptr<DebugGrpcChannel> debug_grpc_channel;
  {
    mutex_lock l(streams_mu);
    if (stream_channels.find(server_stream_addr) == stream_channels.end()) {
      debug_grpc_channel.reset(new DebugGrpcChannel(server_stream_addr));

      if (!debug_grpc_channel->is_channel_ready()) {
        return errors::FailedPrecondition(
            strings::StrCat("Channel at the following gRPC address is ",
                            "not ready: ", server_stream_addr));
      }

      stream_channels[server_stream_addr] = debug_grpc_channel;
    } else {
      debug_grpc_channel = stream_channels[server_stream_addr];
    }
  }

  bool write_ok = debug_grpc_channel->WriteEvent(event);
  if (!write_ok) {
    return errors::Cancelled(strings::StrCat("Write event to stream URL ",
                                             server_stream_addr, "failed."));
  }

  return Status::OK();
}

Status DebugGrpcIO::CloseGrpcStream(const string& server_stream_addr) {
  mutex_lock l(streams_mu);

  if (stream_channels.find(server_stream_addr) != stream_channels.end()) {
    // Stream of the specified address exists. Close it and remove it from
    // record.
    Status s;
    s = stream_channels[server_stream_addr]->Close();
    stream_channels.erase(server_stream_addr);
    return s;
  } else {
    // Stream of the specified address does not exist. No action.
    return Status::OK();
  }
}

}  // namespace tensorflow
