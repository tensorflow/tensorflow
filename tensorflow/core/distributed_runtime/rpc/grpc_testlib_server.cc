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

#include <string>
#include <vector>

#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server_builder.h"
#include "absl/strings/numbers.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/command_line_flags.h"

// This binary starts a TensorFlow server (master and worker) for test purposes.
namespace tensorflow {
namespace {

absl::Status FillServerDef(const string& job_spec, const string& job_name,
                           int num_cpus, int num_gpus, int task_index,
                           int replica, std::string host_port,
                           ServerDef* options) {
  options->set_protocol("grpc");
  options->set_job_name(job_name);
  options->set_task_index(task_index);
  options->set_replica(replica);

  uint32 my_tasks_per_replica = 0;
  // A job with a single task can have multiple "replicas" (multiple replicas).
  //
  // These replicas are unaware of each other during normal operation; when
  // we encounter a job with a replica configured, we select only the current
  // replica and ignore the others.
  for (const string& job_str : str_util::Split(job_spec, ',')) {
    JobDef* job_def = options->mutable_cluster()->add_job();
    // Split each entry in the flag into 3 pieces, separated by "|".
    const std::vector<string> job_pieces = str_util::Split(job_str, '|');
    CHECK_EQ(3, job_pieces.size()) << job_str;
    job_def->set_name(job_pieces[0]);

    int num_tasks;
    if (!absl::SimpleAtoi(job_pieces[2], &num_tasks)) {
      return errors::InvalidArgument("Invalid job string: ", job_str);
    }

    const StringPiece spec = job_pieces[1];

    // job_str is of form <job_name>|<host_ports>.
    const std::vector<string> host_ports = str_util::Split(spec, ';');
    uint32 tasks_per_replica = host_ports.size();
    auto& tasks = (*job_def->mutable_tasks());
    for (size_t i = 0; i < host_ports.size(); ++i) {
      int task_id = i % num_tasks;
      int replica_id = i / num_tasks;

      if (job_def->name() == options->job_name()) {
        if (replica_id == options->replica()) {
          tasks[task_id] = host_ports[i];
        }
      } else {
        if (tasks[task_id].empty()) {
          tasks[task_id] = host_ports[i];
        } else {
          tasks[task_id] = absl::StrCat(tasks[task_id], ",", host_ports[i]);
        }
      }
    }

    if (job_def->name() == options->job_name()) {
      my_tasks_per_replica = tasks_per_replica;
    }

    LOG(INFO) << "Peer " << job_def->name() << " " << tasks_per_replica << " {"
              << absl::StrJoin(host_ports, ", ") << "}";
  }
  if (my_tasks_per_replica == 0) {
    return errors::InvalidArgument("Invalid job specification");
  }

  std::vector<std::string> splits = absl::StrSplit(host_port, ':');
  int port = 0;
  if (!absl::SimpleAtoi(splits[1], &port)) {
    return errors::InvalidArgument("Invalid host port: ", host_port);
  }
  options->set_port(port);

  LOG(INFO) << options->DebugString();
  ConfigProto* config = options->mutable_default_session_config();
  (*config->mutable_device_count())["CPU"] = num_cpus;
  (*config->mutable_device_count())["GPU"] = num_gpus;
  return absl::OkStatus();
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  tensorflow::string job_spec;
  tensorflow::string job_name;
  tensorflow::string host_port;
  int num_cpus = 1;
  int num_gpus = 0;
  int task_index = 0;
  int replica = 0;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("tf_jobs", &job_spec, "job specification"),
      tensorflow::Flag("tf_job", &job_name, "job name"),
      tensorflow::Flag("tf_task", &task_index, "task index"),
      tensorflow::Flag("tf_replica", &replica, "task replica"),
      tensorflow::Flag("host_port", &host_port, "listen address"),
      tensorflow::Flag("num_cpus", &num_cpus, "number of CPUs"),
      tensorflow::Flag("num_gpus", &num_gpus, "number of GPUs"),
  };
  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  tensorflow::ServerDef def;
  absl::Status s =
      tensorflow::FillServerDef(job_spec, job_name, num_cpus, num_gpus,
                                task_index, replica, host_port, &def);
  if (!s.ok()) {
    LOG(ERROR) << "Could not parse job spec: " << s.message() << "\n" << usage;
    return -1;
  }

  std::unique_ptr<tensorflow::ServerInterface> svr;
  s = tensorflow::NewServer(def, &svr);

  if (!s.ok()) {
    LOG(ERROR) << "Could not create server: " << s.message();
    return -1;
  }
  TF_QCHECK_OK(svr->Start());
  TF_QCHECK_OK(svr->Join());

  // NOTE(mrry): Unreachable code.
  return 0;
}
