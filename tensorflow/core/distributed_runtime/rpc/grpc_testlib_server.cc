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

#include <vector>

#include "grpc++/grpc++.h"
#include "grpc++/security/credentials.h"
#include "grpc++/server_builder.h"

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

Status FillServerDef(const string& job_spec, const string& job_name,
                     int num_cpus, int num_gpus, int task_index,
                     ServerDef* options) {
  options->set_protocol("grpc");
  options->set_job_name(job_name);
  options->set_task_index(task_index);

  uint32 my_tasks_per_replica = 0;
  for (const string& job_str : str_util::Split(job_spec, ',')) {
    JobDef* job_def = options->mutable_cluster()->add_job();
    // Split each entry in the flag into 2 pieces, separated by "|".
    const std::vector<string> job_pieces = str_util::Split(job_str, '|');
    CHECK_EQ(2, job_pieces.size()) << job_str;
    job_def->set_name(job_pieces[0]);
    // Does a bit more validation of the tasks_per_replica.
    const StringPiece spec = job_pieces[1];
    // job_str is of form <job_name>|<host_ports>.
    const std::vector<string> host_ports = str_util::Split(spec, ';');
    uint32 tasks_per_replica = host_ports.size();
    for (size_t i = 0; i < host_ports.size(); ++i) {
      (*job_def->mutable_tasks())[i] = host_ports[i];
    }
    if (job_def->name() == options->job_name()) {
      my_tasks_per_replica = tasks_per_replica;
    }
    LOG(INFO) << "Peer " << job_def->name() << " " << tasks_per_replica << " {"
              << str_util::Join(host_ports, ", ") << "}";
  }
  if (my_tasks_per_replica == 0) {
    return errors::InvalidArgument("Invalid job specification");
  }
  ConfigProto* config = options->mutable_default_session_config();
  (*config->mutable_device_count())["CPU"] = num_cpus;
  (*config->mutable_device_count())["GPU"] = num_gpus;
  return Status::OK();
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  tensorflow::string job_spec;
  tensorflow::string job_name;
  int num_cpus = 1;
  int num_gpus = 0;
  int task_index = 0;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("tf_jobs", &job_spec, "job specification"),
      tensorflow::Flag("tf_job", &job_name, "job name"),
      tensorflow::Flag("tf_task", &task_index, "task index"),
      tensorflow::Flag("num_cpus", &num_cpus, "number of CPUs"),
      tensorflow::Flag("num_gpus", &num_gpus, "number of GPUs"),
  };
  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result || argc != 1) {
    LOG(ERROR) << usage;
    return -1;
  }

  tensorflow::ServerDef def;
  tensorflow::Status s = tensorflow::FillServerDef(job_spec, job_name, num_cpus,
                                                   num_gpus, task_index, &def);
  if (!s.ok()) {
    LOG(ERROR) << "Could not parse job spec: " << s.error_message() << "\n"
               << usage;
    return -1;
  }

  std::unique_ptr<tensorflow::ServerInterface> svr;
  s = tensorflow::NewServer(def, &svr);

  if (!s.ok()) {
    LOG(ERROR) << "Could not create server: " << s.error_message();
    return -1;
  }
  TF_QCHECK_OK(svr->Start());
  TF_QCHECK_OK(svr->Join());

  // NOTE(mrry): Unreachable code.
  return 0;
}
