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

// Usage: capture_tpu_profile --service_addr="localhost:8466" --logdir=/tmp/log
//
// Initiates a TPU profiling on the TPUProfiler service at service_addr,
// receives and dumps the profile data to a tensorboard log directory.

#include "grpc++/grpc++.h"

#include <cstdio>
#include <ctime>
#include <vector>

#include "tensorflow/contrib/tpu/profiler/tpu_profiler.grpc.pb.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/util/events_writer.h"

namespace tensorflow {
namespace tpu {
namespace {

using ::tensorflow::TPUProfiler;

using ::grpc::ClientContext;
using ::tensorflow::io::JoinPath;
using ::tensorflow::Env;
using ::tensorflow::WriteStringToFile;

constexpr char kProfilePluginDirectory[] = "plugins/profile/";
constexpr char kTraceFileName[] = "trace";
constexpr char kGraphRunPrefix[] = "tpu_profiler.hlo_graph.";

tensorflow::string GetCurrentTimeStampAsString() {
  char s[128];
  std::time_t t = std::time(nullptr);
  CHECK_NE(std::strftime(s, sizeof(s), "%F_%T", std::localtime(&t)), 0);
  return s;
}

// The trace will be stored in <logdir>/plugins/profile/<run>/trace.
void DumpTraceToLogDirectory(tensorflow::StringPiece logdir,
                             tensorflow::StringPiece run,
                             tensorflow::StringPiece trace) {
  tensorflow::string run_dir = JoinPath(logdir, kProfilePluginDirectory, run);
  TF_CHECK_OK(Env::Default()->RecursivelyCreateDir(run_dir));
  tensorflow::string path = JoinPath(run_dir, kTraceFileName);
  TF_CHECK_OK(WriteStringToFile(tensorflow::Env::Default(), path, trace));
  LOG(INFO) << "Dumped trace data to " << path;
}

ProfileResponse Profile(const tensorflow::string& service_addr,
                        int duration_ms) {
  ProfileRequest request;
  request.set_duration_ms(duration_ms);
  ProfileResponse response;
  ClientContext context;
  ::grpc::ChannelArguments channel_args;
  // TODO(ioeric): use `SetMaxReceiveMessageSize` instead once it's available.
  channel_args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH,
                      std::numeric_limits<int32>::max());
  std::unique_ptr<TPUProfiler::Stub> stub =
      TPUProfiler::NewStub(::grpc::CreateCustomChannel(
          service_addr, ::grpc::InsecureChannelCredentials(), channel_args));
  TF_QCHECK_OK(FromGrpcStatus(stub->Profile(&context, request, &response)));
  return response;
}

void DumpGraph(tensorflow::StringPiece logdir, tensorflow::StringPiece run,
               const tensorflow::string& graph_def) {
  // The graph plugin expects the graph in <logdir>/<run>/<event.file>.
  tensorflow::string run_dir =
      JoinPath(logdir, tensorflow::strings::StrCat(kGraphRunPrefix, run));
  TF_CHECK_OK(Env::Default()->RecursivelyCreateDir(run_dir));
  tensorflow::EventsWriter event_writer(JoinPath(run_dir, "events"));
  tensorflow::Event event;
  event.set_graph_def(graph_def);
  event_writer.WriteEvent(event);
}

}  // namespace
}  // namespace tpu
}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::string FLAGS_service_addr;
  tensorflow::string FLAGS_logdir;
  int FLAGS_duration_ms = 2000;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("service_addr", &FLAGS_service_addr,
                       "Address of TPU profiler service e.g. localhost:8466"),
      tensorflow::Flag("logdir", &FLAGS_logdir,
                       "Path of TensorBoard log directory e.g. /tmp/tb_log"),
      tensorflow::Flag("duration_ms", &FLAGS_duration_ms,
                       "Duration of tracing in ms. Default is 2000ms."),
  };

  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok || FLAGS_service_addr.empty() || FLAGS_logdir.empty()) {
    std::printf("%s", usage.c_str());
    return 2;
  }
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  int duration_ms = FLAGS_duration_ms;
  tensorflow::ProfileResponse response =
      tensorflow::tpu::Profile(FLAGS_service_addr, duration_ms);
  // Use the current timestamp as the run name.
  tensorflow::string run = tensorflow::tpu::GetCurrentTimeStampAsString();
  // Ignore computation_graph for now.
  if (response.encoded_trace().empty()) {
    LOG(WARNING) << "No trace event is collected during the " << duration_ms
                 << "ms interval.";
  } else {
    tensorflow::tpu::DumpTraceToLogDirectory(FLAGS_logdir, run,
                                             response.encoded_trace());
  }
  int num_graphs = response.computation_graph_size();
  if (num_graphs > 0) {
    // The server might generates multiple graphs for one program; we simply
    // pick the first one.
    if (num_graphs > 1) {
      LOG(INFO) << num_graphs
                << " TPU program variants observed over the profiling period. "
                << "One computation graph will be chosen arbitrarily.";
    }
    tensorflow::tpu::DumpGraph(
        FLAGS_logdir, run, response.computation_graph(0).SerializeAsString());
  }
  // Print this at the end so that it's not buried in irrelevant LOG messages.
  std::cout
      << "NOTE: using the trace duration " << duration_ms << "ms." << std::endl
      << "Set an appropriate duration (with --duration_ms) if you "
         "don't see a full step in your trace or the captured trace is too "
         "large."
      << std::endl;
}
