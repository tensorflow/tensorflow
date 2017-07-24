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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"

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

tensorflow::string GetCurrentTimeStampAsString() {
  char s[128];
  std::time_t t = std::time(nullptr);
  CHECK_NE(std::strftime(s, sizeof(s), "%F_%T", std::localtime(&t)), 0);
  return s;
}

// The trace will be stored in <logdir>/plugins/profile/<timestamp>/trace.
void DumpTraceToLogDirectory(const tensorflow::string& logdir,
                             tensorflow::StringPiece trace) {
  tensorflow::string run = GetCurrentTimeStampAsString();
  tensorflow::string run_dir = JoinPath(logdir, kProfilePluginDirectory, run);
  TF_CHECK_OK(Env::Default()->RecursivelyCreateDir(run_dir));
  tensorflow::string path = JoinPath(run_dir, kTraceFileName);
  TF_CHECK_OK(WriteStringToFile(tensorflow::Env::Default(), path, trace));
  LOG(INFO) << "Dumped trace data to " << path;
}

ProfileResponse Profile(const tensorflow::string& service_addr) {
  ProfileRequest request;
  ProfileResponse response;
  ClientContext context;
  ::grpc::ChannelArguments channel_args;
  // TODO(ioeric): use `SetMaxReceiveMessageSize` instead once it's available.
  channel_args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH,
                      std::numeric_limits<int32>::max());
  std::unique_ptr<TPUProfiler::Stub> stub =
      TPUProfiler::NewStub(::grpc::CreateCustomChannel(
          service_addr, ::grpc::InsecureChannelCredentials(), channel_args));
  TF_CHECK_OK(FromGrpcStatus(stub->Profile(&context, request, &response)));
  return response;
}

}  // namespace
}  // namespace tpu
}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::string FLAGS_service_addr;
  tensorflow::string FLAGS_logdir;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("service_addr", &FLAGS_service_addr,
                       "Address of TPU profiler service e.g. localhost:8466"),
      tensorflow::Flag("logdir", &FLAGS_logdir,
                       "Path of TensorBoard log directory e.g. /tmp/tb_log"),
  };
  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok || FLAGS_service_addr.empty() || FLAGS_logdir.empty()) {
    std::printf("%s", usage.c_str());
    return 2;
  }
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  tensorflow::ProfileResponse response =
      tensorflow::tpu::Profile(FLAGS_service_addr);
  // Ignore computation_graph for now.
  tensorflow::tpu::DumpTraceToLogDirectory(FLAGS_logdir,
                                           response.encoded_trace());
}
