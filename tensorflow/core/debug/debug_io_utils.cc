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
  for (const string& url : debug_urls) {
    if (str_util::Lowercase(url).find(kFileURLScheme) == 0) {
      const string dump_root_dir = url.substr(strlen(kFileURLScheme));

      Status s =
          DebugFileIO::DumpTensorToDir(node_name, output_slot, debug_op, tensor,
                                       wall_time_us, dump_root_dir, nullptr);
      if (!s.ok()) {
        num_failed_urls++;
      }
    } else if (str_util::Lowercase(url).find(kGrpcURLScheme) == 0) {
      // TODO(cais): Implement PublishTensor with grpc urls.
      return Status(error::UNIMPLEMENTED,
                    strings::StrCat("Puslishing to GRPC debug target is not ",
                                    "implemented yet"));
    } else {
      return Status(error::UNAVAILABLE,
                    strings::StrCat("Invalid debug target URL: ", url));
    }
  }

  if (num_failed_urls == 0) {
    return Status::OK();
  } else {
    return Status(
        error::INTERNAL,
        strings::StrCat("Puslishing to ", num_failed_urls, " of ",
                        debug_urls.size(), " debug target URLs failed"));
  }
}

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
  if (env->FileExists(dir) && env->IsDirectory(dir).ok()) {
    // The path already exists as a directory. Return OK right away.
    return Status::OK();
  }

  string parent_dir = io::Dirname(dir).ToString();
  if (!env->FileExists(parent_dir)) {
    // The parent path does not exist yet, create it first.
    Status s = RecursiveCreateDir(env, parent_dir);  // Recursive call
    if (!s.ok()) {
      return Status(
          error::FAILED_PRECONDITION,
          strings::StrCat("Failed to create directory  ", parent_dir));
    }
  } else if (env->FileExists(parent_dir) &&
             !env->IsDirectory(parent_dir).ok()) {
    // The path exists, but it is a file.
    return Status(error::FAILED_PRECONDITION,
                  strings::StrCat("Failed to create directory  ", parent_dir,
                                  " because the path exists as a file "));
  }

  env->CreateDir(dir);
  // Guard against potential race in creating directories by doing a check
  // after the CreateDir call.
  if (env->FileExists(dir) && env->IsDirectory(dir).ok()) {
    return Status::OK();
  } else {
    return Status(error::ABORTED,
                  strings::StrCat("Failed to create directory  ", parent_dir));
  }
}

}  // namespace tensorflow
