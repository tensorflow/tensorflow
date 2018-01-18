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

#include "tensorflow/contrib/tpu/profiler/dump_tpu_profile.h"
#include "tensorflow/contrib/tpu/profiler/tpu_profiler.grpc.pb.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace tpu {
namespace {

using ::tensorflow::TPUProfiler;

constexpr uint64 kMaxEvents = 1000000;

string GetCurrentTimeStampAsString() {
  char s[128];
  std::time_t t = std::time(nullptr);
  CHECK_NE(std::strftime(s, sizeof(s), "%F_%T", std::localtime(&t)), 0);
  return s;
}

ProfileResponse Profile(const string& service_addr, int duration_ms) {
  ProfileRequest request;
  request.set_duration_ms(duration_ms);
  request.set_max_events(kMaxEvents);
  request.add_tools("input_pipeline");
  request.add_tools("overview_page");
  std::cout << "Limiting the number of trace events to " << kMaxEvents
            << std::endl;
  ::grpc::ClientContext context;
  ::grpc::ChannelArguments channel_args;
  // TODO(ioeric): use `SetMaxReceiveMessageSize` instead once it's available.
  channel_args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH,
                      std::numeric_limits<int32>::max());
  std::unique_ptr<TPUProfiler::Stub> stub =
      TPUProfiler::NewStub(::grpc::CreateCustomChannel(
          service_addr, ::grpc::InsecureChannelCredentials(), channel_args));
  ProfileResponse response;
  TF_QCHECK_OK(FromGrpcStatus(stub->Profile(&context, request, &response)));
  return response;
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
  TF_CHECK_OK(tensorflow::tpu::WriteTensorboardTPUProfile(
      FLAGS_logdir, run, response, &std::cout));
  // Print this at the end so that it's not buried in irrelevant LOG messages.
  std::cout
      << "NOTE: using the trace duration " << duration_ms << "ms." << std::endl
      << "Set an appropriate duration (with --duration_ms) if you "
         "don't see a full step in your trace or the captured trace is too "
         "large."
      << std::endl;
}
