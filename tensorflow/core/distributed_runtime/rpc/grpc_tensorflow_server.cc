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
#include <vector>

#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server_builder.h"

#include "tensorflow/core/distributed_runtime/server_lib.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/command_line_flags.h"

// This binary starts a TensorFlow server (master and worker).
//
// TODO(mrry): Replace with a py_binary that uses `tf.GrpcServer()`.
namespace tensorflow {
namespace {

absl::Status FillServerDef(const string& cluster_spec, const string& job_name,
                           int task_index, ServerDef* options) {
  options->set_protocol("grpc");
  options->set_job_name(job_name);
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
    const absl::string_view spec = job_pieces[1];
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
              << absl::StrJoin(host_ports, ", ") << "}";
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
  return absl::OkStatus();
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
  tensorflow::string cluster_spec;
  tensorflow::string job_name;
  int task_index = 0;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("cluster_spec", &cluster_spec, "cluster spec"),
      tensorflow::Flag("job_name", &job_name, "job name"),
      tensorflow::Flag("task_id", &task_index, "task id"),
  };
  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (!parse_result || argc != 1) {
    std::cerr << usage << std::endl;
    Usage(argv[0]);
    return -1;
  }
  tensorflow::ServerDef server_def;
  absl::Status s = tensorflow::FillServerDef(cluster_spec, job_name, task_index,
                                             &server_def);
  if (!s.ok()) {
    std::cerr << "ERROR: " << s.message() << std::endl;
    Usage(argv[0]);
    return -1;
  }
  std::unique_ptr<tensorflow::ServerInterface> server;
  TF_QCHECK_OK(tensorflow::NewServer(server_def, &server));
  TF_QCHECK_OK(server->Start());
  TF_QCHECK_OK(server->Join());
}
