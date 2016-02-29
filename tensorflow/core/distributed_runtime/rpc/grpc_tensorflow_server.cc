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

#include <iostream>

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

// This binary starts a TensorFlow server (master and worker).
namespace tensorflow {
namespace {

Status ParseFlagsForTask(int argc, char* argv[], GrpcServerOptions* options) {
  string cluster_spec;
  const bool parse_result =
      ParseFlags(&argc, argv, {Flag("cluster_spec", &cluster_spec),   //
                               Flag("job_name", &options->job_name),  //
                               Flag("task_id", &options->task_index)});
  if (!parse_result) {
    return errors::InvalidArgument("Error parsing command-line flags");
  }

  size_t my_num_tasks = 0;
  for (const string& job_str : str_util::Split(cluster_spec, ',')) {
    // Split each entry in the flag into 3 pieces, separated by "|".
    const std::vector<string> job_pieces = str_util::Split(job_str, '|');
    CHECK_EQ(2, job_pieces.size()) << job_str;
    const string& job = job_pieces[0];
    // Does a bit more validation of the tasks_per_replica.
    const StringPiece spec = job_pieces[1];
    // job_str is of form <job_name>|<host_ports>.
    const std::vector<string> host_ports = str_util::Split(spec, ';');
    size_t num_tasks = host_ports.size();
    if (job == options->job_name) {
      my_num_tasks = num_tasks;
    }
    TF_RETURN_IF_ERROR(
        options->channel_spec.AddHostPortsJob(job, host_ports, num_tasks));
    LOG(INFO) << "Peer " << job << " " << num_tasks << " {"
              << str_util::Join(host_ports, ", ") << "}";
  }
  if (my_num_tasks == 0) {
    return errors::InvalidArgument("Job name \"", options->job_name,
                                   "\" does not appear in the cluster spec");
  }
  if (options->task_index >= my_num_tasks) {
    return errors::InvalidArgument("Task index ", options->task_index,
                                   " is invalid (job \"", options->job_name,
                                   "\" contains ", my_num_tasks, " tasks");
  }
  return Status::OK();
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  tensorflow::GrpcServerOptions options;
  tensorflow::Status s = tensorflow::ParseFlagsForTask(argc, argv, &options);
  if (!s.ok()) {
    std::cerr << "ERROR: " << s.error_message() << std::endl;
    std::cerr << "Usage: " << argv[0]
              << " --cluster_spec=SPEC --job_name=NAME --task_id=ID"
              << std::endl;
    std::cerr << "Where:" << std::endl;
    std::cerr << "    SPEC is <JOB>(,<JOB>)*" << std::endl;
    std::cerr << "    JOB  is <NAME>|<HOST:PORT>(;<HOST:PORT>)*" << std::endl;
    std::cerr << "    NAME is a valid job name ([a-z][0-9a-z]*)" << std::endl;
    std::cerr << "    HOST is a hostname or IP address" << std::endl;
    std::cerr << "    PORT is a port number" << std::endl;
    return -1;
  }
  tensorflow::StartTensorFlowServer(options);
}
