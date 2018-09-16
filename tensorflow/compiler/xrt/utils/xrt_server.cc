/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/server_lib.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tensorflow/core/public/session_options.h"

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  tensorflow::ServerDef server_def;
  std::string job_name{"job"};
  server_def.set_protocol("grpc");
  server_def.set_job_name(job_name);
  server_def.set_task_index(0);
  tensorflow::ClusterDef* cluster_def = server_def.mutable_cluster();
  tensorflow::JobDef* job_def = cluster_def->add_job();
  job_def->set_name(job_name);
  job_def->mutable_tasks()->insert(
      {0, tensorflow::strings::StrCat("localhost:", 8470)});
  std::unique_ptr<tensorflow::ServerInterface> server;
  TF_QCHECK_OK(tensorflow::NewServer(server_def, &server));
  TF_QCHECK_OK(server->Start());
  TF_QCHECK_OK(server->Join());
}
