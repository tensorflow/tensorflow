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

#include "tensorflow/core/distributed_runtime/rpc/grpc_testlib.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/distributed_runtime/rpc/grpc_session.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace test {

Status TestCluster::MakeTestCluster(const TestClusterConfig& config,
                                    std::unique_ptr<TestCluster>* out_cluster) {
  std::string binary_path =
      !config.binary_path.empty()
          ? config.binary_path
          : strings::StrCat(
                testing::TensorFlowSrcRoot(),
                "/core/distributed_runtime/rpc/grpc_testlib_server");
  SessionOptions options = config.options;
  std::unique_ptr<TestCluster> ret(new TestCluster);

  std::vector<std::string> tf_job_args;
  for (const auto& job : config.jobs) {
    if (job.num_tasks % job.num_replicas != 0) {
      return errors::InvalidArgument(
          "Number of tasks must evenly divide replicas.");
    }

    std::vector<std::string>& job_targets = ret->targets_[job.name];
    for (int i = 0; i < job.num_tasks; ++i) {
      int port = testing::PickUnusedPortOrDie();
      job_targets.push_back(strings::StrCat("localhost:", port));
    }
    tf_job_args.push_back(strings::StrCat(job.name, "|",
                                          absl::StrJoin(job_targets, ";"), "|",
                                          job.num_tasks / job.num_replicas));
  }
  std::string tf_jobs =
      absl::StrCat("--tf_jobs=", absl::StrJoin(tf_job_args, ","));

  int num_cpus = 1;
  int num_gpus = 0;
  auto iter = options.config.device_count().find("CPU");
  if (iter != options.config.device_count().end()) {
    num_cpus = iter->second;
  }
  iter = options.config.device_count().find("GPU");
  if (iter != options.config.device_count().end()) {
    num_gpus = iter->second;
  }
  if (!options.env->FileExists(binary_path).ok()) {
    return errors::Internal("Could not find grpc_testlib_server");
  }

  for (const auto& job : config.jobs) {
    for (int i = 0; i < job.num_tasks; ++i) {
      const std::vector<string> argv(
          {binary_path, /* see grpc_testlib_server.cc for flags */
           tf_jobs, strings::StrCat("--tf_job=", job.name),
           strings::StrCat("--tf_task=", i / job.num_replicas),
           strings::StrCat("--tf_replica=", i % job.num_replicas),
           strings::StrCat("--num_cpus=", num_cpus),
           strings::StrCat("--host_port=", ret->targets_[job.name][i]),
           strings::StrCat("--num_gpus=", num_gpus)});
      LOG(INFO) << "Start: " << absl::StrJoin(argv, " ");
      auto subprocess = CreateSubProcess(argv);
      bool success = subprocess->Start();
      ret->subprocesses_.emplace_back(std::move(subprocess));
      if (!success) {
        return errors::Internal("Could not start subprocess");
      }
    }
  }

  SessionOptions options_copy(options);
  options_copy.target =
      strings::StrCat("grpc://", ret->targets(config.jobs[0].name)[0]);

  std::unique_ptr<GrpcSession> session;
  TF_RETURN_IF_ERROR(GrpcSession::Create(options_copy, &session));
  TF_RETURN_IF_ERROR(session->ListDevices(&ret->devices_));

  *out_cluster = std::move(ret);
  return OkStatus();
}

TestCluster::~TestCluster() {
  for (auto& subprocess : subprocesses_) {
    subprocess->Kill(9);
  }
}

}  // end namespace test
}  // end namespace tensorflow
