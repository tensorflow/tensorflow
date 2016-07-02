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

#include <iostream>

#include "grpc++/grpc++.h"
#include "grpc++/security/credentials.h"
#include "grpc++/server_builder.h"

#include "tensorflow/core/distributed_runtime/server_lib.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/command_line_flags.h"

// This binary starts a TensorFlow server (master and worker).
//
// TODO(mrry): Replace with a py_binary that uses `tf.GrpcServer()`.
namespace tensorflow {
namespace {

Status ParseFlagsForTask(int argc, char* argv[], ServerDef* options) {
  options->set_protocol("grpc");
  string cluster_spec;
  int task_index = 0;
  const bool parse_result = ParseFlags(
      &argc, argv, {Flag("cluster_spec", &cluster_spec),            //
                    Flag("job_name", options->mutable_job_name()),  //
                    Flag("task_id", &task_index)});
  if (!parse_result) {
    return errors::InvalidArgument("Error parsing command-line flags");
  }
  options->set_task_index(task_index);

  size_t my_num_tasks = 0;

  ClusterDef* const cluster = options->mutable_cluster();

  for (const string& job_str : str_util::Split(cluster_spec, ',')) {
    JobDef* const job_def = cluster->add_job();
    // Split each entry in the flag into 2 pieces, separated by "|".
    const std::vector<string> job_pieces = str_util::Split(job_str, '|');
    CHECK_EQ(2, job_pieces.size()) << job_str;
    const string& job_name = job_pieces[0];
    job_def->set_name(job_name);
    // Does a bit more validation of the tasks_per_replica.
    const StringPiece spec = job_pieces[1];
    // job_str is of form <job_name>|<host_ports>.
    const std::vector<string> host_ports = str_util::Split(spec, ';');
    for (size_t i = 0; i < host_ports.size(); ++i) {
      (*job_def->mutable_tasks())[i] = host_ports[i];
    }
    size_t num_tasks = host_ports.size();
    if (job_name == options->job_name()) {
      my_num_tasks = host_ports.size();
    }
    LOG(INFO) << "Peer " << job_name << " " << num_tasks << " {"
              << str_util::Join(host_ports, ", ") << "}";
  }
  if (my_num_tasks == 0) {
    return errors::InvalidArgument("Job name \"", options->job_name(),
                                   "\" does not appear in the cluster spec");
  }
  if (options->task_index() >= my_num_tasks) {
    return errors::InvalidArgument("Task index ", options->task_index(),
                                   " is invalid (job \"", options->job_name(),
                                   "\" contains ", my_num_tasks, " tasks");
  }
  return Status::OK();
}

}  // namespace
}  // namespace tensorflow

void Usage(char* const argv_0) {
  std::cerr << "Usage: " << argv_0
            << " --cluster_spec=SPEC --job_name=NAME --task_id=ID" << std::endl;
  std::cerr << "Where:" << std::endl;
  std::cerr << "    SPEC is <JOB>(,<JOB>)*" << std::endl;
  std::cerr << "    JOB  is <NAME>|<HOST:PORT>(;<HOST:PORT>)*" << std::endl;
  std::cerr << "    NAME is a valid job name ([a-z][0-9a-z]*)" << std::endl;
  std::cerr << "    HOST is a hostname or IP address" << std::endl;
  std::cerr << "    PORT is a port number" << std::endl;
}

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  tensorflow::ServerDef server_def;
  tensorflow::Status s = tensorflow::ParseFlagsForTask(argc, argv, &server_def);
  if (!s.ok()) {
    std::cerr << "ERROR: " << s.error_message() << std::endl;
    Usage(argv[0]);
    return -1;
  }
  std::unique_ptr<tensorflow::ServerInterface> server;
  tensorflow::NewServer(server_def, &server);
  server->Start();
  server->Join();
}
