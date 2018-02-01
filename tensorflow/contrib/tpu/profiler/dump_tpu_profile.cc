/* Copyright 2017 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/contrib/tpu/profiler/dump_tpu_profile.h"

#include <cstdio>
#include <ctime>
#include <vector>

#include "tensorflow/contrib/tpu/profiler/op_profile.pb.h"
#include "tensorflow/contrib/tpu/profiler/trace_events.pb.h"
#include "tensorflow/contrib/tpu/profiler/trace_events_to_json.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/event.pb.h"
#include "tensorflow/core/util/events_writer.h"

namespace tensorflow {
namespace tpu {
namespace {

using ::tensorflow::io::JoinPath;
using ::tensorflow::protobuf::util::JsonOptions;
using ::tensorflow::protobuf::util::MessageToJsonString;

constexpr char kGraphRunPrefix[] = "tpu_profiler.hlo_graph.";
constexpr char kJsonOpProfileFileName[] = "op_profile.json";
constexpr char kJsonTraceFileName[] = "trace.json.gz";
constexpr char kProfilePluginDirectory[] = "plugins/profile/";
constexpr char kProtoTraceFileName[] = "trace";

Status WriteGzippedDataToFile(const string& filename, const string& data) {
  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(Env::Default()->NewWritableFile(filename, &file));
  io::ZlibCompressionOptions options = io::ZlibCompressionOptions::GZIP();
  io::ZlibOutputBuffer buffer(file.get(), options.input_buffer_size,
                              options.output_buffer_size, options);
  TF_RETURN_IF_ERROR(buffer.Init());
  TF_RETURN_IF_ERROR(buffer.Append(data));
  TF_RETURN_IF_ERROR(buffer.Close());
  TF_RETURN_IF_ERROR(file->Close());
  return Status::OK();
}

Status DumpTraceToLogDirectory(StringPiece run_dir, const string& encoded_trace,
                               std::ostream* os) {
  string proto_path = JoinPath(run_dir, kProtoTraceFileName);
  TF_RETURN_IF_ERROR(
      WriteStringToFile(Env::Default(), proto_path, encoded_trace));
  LOG(INFO) << "Dumped raw-proto trace data to " << proto_path;

  string json_path = JoinPath(run_dir, kJsonTraceFileName);
  Trace trace;
  trace.ParseFromString(encoded_trace);
  *os << "Trace contains " << trace.trace_events_size() << " events."
      << std::endl;
  TF_RETURN_IF_ERROR(
      WriteGzippedDataToFile(json_path, TraceEventsToJson(trace)));
  *os << "Dumped JSON trace data to " << json_path << std::endl;
  return Status::OK();
}

Status DumpOpProfileToLogDirectory(StringPiece run_dir,
                                   const tpu::op_profile::Profile& profile,
                                   std::ostream* os) {
  string path = JoinPath(run_dir, kJsonOpProfileFileName);
  string json;
  JsonOptions options;
  options.always_print_primitive_fields = true;
  auto status = MessageToJsonString(profile, &json, options);
  if (!status.ok()) {
    return errors::Internal(
        "Failed to convert op profile to json. Skipping... ",
        string(status.error_message()));
  }
  TF_RETURN_IF_ERROR(WriteStringToFile(Env::Default(), path, json));
  *os << "Dumped json op profile data to " << path << std::endl;
  return Status::OK();
}

Status DumpToolDataToLogDirectory(StringPiece run_dir,
                                  const tensorflow::ProfileToolData& tool,
                                  std::ostream* os) {
  string path = JoinPath(run_dir, tool.name());
  TF_RETURN_IF_ERROR(WriteStringToFile(Env::Default(), path, tool.data()));
  *os << "Dumped tool data for " << tool.name() << " to " << path << std::endl;
  return Status::OK();
}

Status DumpGraphEvents(const string& logdir, const string& run,
                       const ProfileResponse& response, std::ostream* os) {
  int num_graphs = response.computation_graph_size();
  if (response.computation_graph_size() == 0) return Status::OK();
  // The server might generates multiple graphs for one program; we simply
  // pick the first one.
  if (num_graphs > 1) {
    *os << num_graphs
        << " TPU program variants observed over the profiling period. "
        << "One computation graph will be chosen arbitrarily." << std::endl;
  }
  // The graph plugin expects the graph in <logdir>/<run>/<event.file>.
  string run_dir = JoinPath(logdir, strings::StrCat(kGraphRunPrefix, run));
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(run_dir));
  EventsWriter event_writer(JoinPath(run_dir, "events"));
  Event event;
  // Add the computation graph.
  event.set_graph_def(response.computation_graph(0).SerializeAsString());
  event_writer.WriteEvent(event);
  *os << "Wrote a HLO graph to " << event_writer.FileName() << std::endl;

  if (response.has_hlo_metadata()) {
    tensorflow::TaggedRunMetadata tagged_run_metadata;
    tagged_run_metadata.set_tag(run);
    tagged_run_metadata.set_run_metadata(
        response.hlo_metadata().SerializeAsString());
    tensorflow::Event meta_event;
    *meta_event.mutable_tagged_run_metadata() = tagged_run_metadata;
    event_writer.WriteEvent(meta_event);
    *os << "Wrote HLO ops run metadata to " << event_writer.FileName()
        << std::endl;
  }
  return Status::OK();
}

}  // namespace

Status WriteTensorboardTPUProfile(const string& logdir, const string& run,
                                  const ProfileResponse& response,
                                  std::ostream* os) {
  // Dumps profile data to <logdir>/plugins/profile/<run>/.
  string profile_run_dir = JoinPath(logdir, kProfilePluginDirectory, run);
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(profile_run_dir));

  // Ignore computation_graph for now.
  const bool empty_trace = response.encoded_trace().empty();
  if (!empty_trace) {
    LOG(INFO) << "Converting trace events to TraceViewer JSON.";
    TF_RETURN_IF_ERROR(
        DumpTraceToLogDirectory(profile_run_dir, response.encoded_trace(), os));
  }
  if (response.has_op_profile() &&
      (response.op_profile().has_by_program_structure() ||
       response.op_profile().has_by_category())) {
    TF_RETURN_IF_ERROR(DumpOpProfileToLogDirectory(profile_run_dir,
                                                   response.op_profile(), os));
  }
  if (!empty_trace && !response.tool_data().empty()) {
    for (const auto& tool_data : response.tool_data()) {
      TF_RETURN_IF_ERROR(
          DumpToolDataToLogDirectory(profile_run_dir, tool_data, os));
    }
  }

  return Status::OK();
}

}  // namespace tpu
}  // namespace tensorflow
