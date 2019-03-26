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

#include "tensorflow/core/distributed_runtime/rpc/grpc_session.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace test {

Status TestCluster::MakeTestCluster(const SessionOptions& options, int n,
                                    std::unique_ptr<TestCluster>* out_cluster) {
  string server_path =
      strings::StrCat(testing::TensorFlowSrcRoot(),
                      "/core/distributed_runtime/rpc/grpc_testlib_server");
  return MakeTestCluster(server_path, options, n, out_cluster);
}

Status TestCluster::MakeTestCluster(const string& binary_path,
                                    const SessionOptions& options, int n,
                                    std::unique_ptr<TestCluster>* out_cluster) {
  CHECK_GE(n, 1);
  std::unique_ptr<TestCluster> ret(new TestCluster);

  ret->targets_.resize(n);

  std::vector<int> port(n);
  for (int i = 0; i < n; ++i) {
    port[i] = testing::PickUnusedPortOrDie();
    ret->targets_[i] = strings::StrCat("localhost:", port[i]);
  }

  const string tf_jobs = strings::StrCat("--tf_jobs=localhost|",
                                         str_util::Join(ret->targets_, ";"));

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

  for (int i = 0; i < n; ++i) {
    if (!options.env->FileExists(binary_path).ok()) {
      return errors::Internal("Could not find grpc_testlib_server");
    }
    const std::vector<string> argv(
        {binary_path, /* see grpc_testlib_server.cc for flags */
         tf_jobs, "--tf_job=localhost", strings::StrCat("--tf_task=", i),
         strings::StrCat("--num_cpus=", num_cpus),
         strings::StrCat("--num_gpus=", num_gpus)});
    ret->subprocesses_.emplace_back(CreateSubProcess(argv));
    bool success = ret->subprocesses_[i]->Start();
    if (!success) {
      return errors::Internal("Could not start subprocess");
    }
  }

  SessionOptions options_copy(options);
  options_copy.target = strings::StrCat("grpc://", ret->targets_[0]);

  std::unique_ptr<GrpcSession> session;
  TF_RETURN_IF_ERROR(GrpcSession::Create(options_copy, &session));
  std::vector<DeviceAttributes> device_attributes;
  TF_RETURN_IF_ERROR(session->ListDevices(&ret->devices_));

  *out_cluster = std::move(ret);
  return Status::OK();
}

TestCluster::~TestCluster() {
  for (auto& subprocess : subprocesses_) {
    subprocess->Kill(9);
  }
}

}  // end namespace test
}  // end namespace tensorflow
