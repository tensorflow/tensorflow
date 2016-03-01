/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "grpc++/grpc++.h"
#include "grpc++/security/credentials.h"
#include "grpc++/server_builder.h"

#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/command_line_flags.h"

// This binary starts a TensorFlow server (master and worker) for test purposes.
namespace tensorflow {
namespace {

Status ParseFlagsForTask(int argc, char* argv[], GrpcServerOptions* options) {
  string job_spec;
  int num_cpus = 1;
  int num_gpus = 0;
  const bool parse_result =
      ParseFlags(&argc, argv, {Flag("tf_jobs", &job_spec),             //
                               Flag("tf_job", &options->job_name),     //
                               Flag("tf_task", &options->task_index),  //
                               Flag("num_cpus", &num_cpus),            //
                               Flag("num_gpus", &num_gpus)});
  if (!parse_result) {
    return errors::InvalidArgument("Error parsing command-line flags");
  }

  uint32 my_tasks_per_replica = 0;
  for (const string& job_str : str_util::Split(job_spec, ',')) {
    // Split each entry in the flag into 3 pieces, separated by "|".
    const std::vector<string> job_pieces = str_util::Split(job_str, '|');
    CHECK_EQ(2, job_pieces.size()) << job_str;
    const string& job = job_pieces[0];
    // Does a bit more validation of the tasks_per_replica.
    const StringPiece spec = job_pieces[1];
    // job_str is of form <job_name>|<host_ports>.
    const std::vector<string> host_ports = str_util::Split(spec, ';');
    uint32 tasks_per_replica = host_ports.size();
    if (job == options->job_name) {
      my_tasks_per_replica = tasks_per_replica;
    }
    TF_RETURN_IF_ERROR(options->channel_spec.AddHostPortsJob(
        job, host_ports, tasks_per_replica));
    LOG(INFO) << "Peer " << job << " " << tasks_per_replica << " {"
              << str_util::Join(host_ports, ", ") << "}";
  }
  if (my_tasks_per_replica == 0) {
    return errors::InvalidArgument("Invalid job specification");
  }

  (*options->default_session_options.config.mutable_device_count())["CPU"] =
      num_cpus;
  (*options->default_session_options.config.mutable_device_count())["GPU"] =
      num_gpus;
  return Status::OK();
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  tensorflow::GrpcServerOptions options;
  tensorflow::Status s = tensorflow::ParseFlagsForTask(argc, argv, &options);
  if (!s.ok()) {
    LOG(ERROR) << "Could not parse flags: " << s.error_message();
    return -1;
  }
  tensorflow::StartTensorFlowServer(options);
  // NOTE(mrry): Unreachable code.
  return 0;
}
