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

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_session.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

static const int kWorkers = 60;
static thread::ThreadPool* worker_threads;

void MakeGRPCCluster(const SessionOptions& options, int n,
                     std::vector<string>* workers,
                     std::vector<DeviceAttributes>* devices) {
  CHECK_GE(n, 1);

  workers->clear();
  std::vector<int> port(n);
  for (int i = 0; i < n; ++i) {
    port[i] = testing::PickUnusedPortOrDie();
    workers->push_back(strings::StrCat("grpc://localhost:", port[i]));
  }

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

  worker_threads = new thread::ThreadPool(Env::Default(), "worker_threads", n);
  for (int worker_idx = 0; worker_idx < n; ++worker_idx) {
    worker_threads->Schedule([worker_idx, n, num_cpus, num_gpus, &port] {
      ServerDef server;
      server.set_protocol("grpc");
      server.set_job_name("localhost");
      server.set_task_index(worker_idx);

      auto job_def = server.mutable_cluster()->add_job();
      job_def->set_name("localhost");
      for (int i = 0; i < n; i++) {
        (*(job_def->mutable_tasks()))[i] =
            strings::StrCat("localhost:", port[i]);
      }

      auto config = server.mutable_default_session_config();
      (*config->mutable_device_count())["CPU"] = num_cpus;
      (*config->mutable_device_count())["GPU"] = num_gpus;

      std::unique_ptr<ServerInterface> svr;
      TF_CHECK_OK(NewServer(server, &svr));
      TF_CHECK_OK(svr->Start());
      TF_CHECK_OK(svr->Join());
    });
  }

  // Get attributes for all devices.
  LOG(ERROR) << "W '" << (*workers)[0] << "'";
  SessionOptions options_copy(options);
  options_copy.target = (*workers)[0];
  std::unique_ptr<GrpcSession> session;
  TF_CHECK_OK(GrpcSession::Create(options_copy, &session));
  TF_CHECK_OK(session->ListDevices(devices));
}

struct Cluster {
  SessionOptions options;
  std::vector<string> workers;
  std::vector<DeviceAttributes> devices;  // One per process

  Cluster() {
    (*options.config.mutable_device_count())["CPU"] = 1;
    options.config.set_intra_op_parallelism_threads(1);
    options.config.set_inter_op_parallelism_threads(1);
    MakeGRPCCluster(options, kWorkers, &workers, &devices);
    LOG(ERROR) << "C " << workers.size() << " " << devices.size() << " "
               << workers[0] << " " << workers[1];
    options.target = workers[0];
  }
};

static const Cluster* GetCluster() {
  static Cluster* result = new Cluster;
  return result;
}

// Make a program with specified number of stages and "width" ops per stage.
GraphDef CreateGraphDef(int num_stages, int width, int tensor_size,
                        bool use_multiple_devices, const Cluster* cluster) {
  CHECK_GE(cluster->devices.size(), width);

  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  Scope s = Scope::NewRootScope();

  // x is from the feed.
  Output x = Const(s.WithOpName("x"), 0.0f, {tensor_size, 1});

  // Create stages.
  std::vector<Output> last_stage;
  last_stage.push_back(x);
  for (int i = 0; i < num_stages; i++) {
    std::vector<Output> this_stage;
    for (int j = 0; j < width; j++) {
      Output combine = AddN(
          s.WithDevice(cluster->devices[use_multiple_devices ? j : 0].name()),
          last_stage);
      this_stage.push_back(combine);
    }
    last_stage = this_stage;
  }

  // Create output.
  /* Output y =*/AddN give_me_a_name(s.WithOpName("y"), last_stage);

  GraphDef def;
  TF_CHECK_OK(s.ToGraphDef(&def));
  return def;
}

string DebugString(const Tensor& x, const Tensor& y, int tensor_size) {
  CHECK_EQ(x.NumElements(), tensor_size);
  CHECK_EQ(y.NumElements(), tensor_size);
  auto x_flat = x.flat<float>();
  auto y_flat = y.flat<float>();
  // Just print the first couple of elements of each tensor
  CHECK_GE(tensor_size, 2);
  return strings::Printf("x = [%8.6f %8.6f] y = [%8.6f %8.6f]", x_flat(0),
                         x_flat(1), y_flat(0), y_flat(1));
}

// TODO: Support sharding and depth.
static void BM_Helper(::testing::benchmark::State& state, int width,
                      int num_stages, int tensor_size,
                      bool use_multiple_devices) {
  const Cluster* cluster = GetCluster();

  // Creates a session.
  std::unique_ptr<Session> session(NewSession(cluster->options));
  GraphDef def = CreateGraphDef(num_stages, width, tensor_size,
                                use_multiple_devices, cluster);
  graph::SetDefaultDevice(cluster->devices[0].name(), &def);

  TF_CHECK_OK(session->Create(def));

  // Randomly initialize the input.
  Tensor x(DT_FLOAT, TensorShape({tensor_size, 1}));

  state.SetLabel(
      strings::StrCat(def.node_size(), " nodes; ",
                      use_multiple_devices ? "Multi device" : "Single device",
                      "; tensor bytes/send: ", tensor_size * sizeof(float)));

  std::vector<Tensor> outputs;

  // Do a few warmup iterations.
  for (int i = 0; i < 3; i++) {
    outputs.clear();
    TF_CHECK_OK(session->Run({{"x", x}}, {"y:0"}, {}, &outputs));
    CHECK_EQ(size_t{1}, outputs.size());

    if (i == 0) {
      // Print out x, and y.
      const Tensor& y = outputs[0];
      VLOG(1) << DebugString(x, y, tensor_size);
    }
  }

  // Iterations.
  for (auto s : state) {
    outputs.clear();
    TF_CHECK_OK(session->Run({{"x", x}}, {"y:0"}, {}, &outputs));
    CHECK_EQ(size_t{1}, outputs.size());
  }
  TF_CHECK_OK(session->Close());
}
static void BM_ShardedProgram(::testing::benchmark::State& state) {
  const int width = state.range(0);
  const int num_stages = state.range(1);

  BM_Helper(state, width, num_stages, 2 /*tensor_size*/, true /*multi-device*/);
}
BENCHMARK(BM_ShardedProgram)
    ->ArgPair(1, 1)
    ->ArgPair(1, 3)
    ->ArgPair(1, 5)
    ->ArgPair(1, 15)
    ->ArgPair(1, 60)
    ->ArgPair(15, 1)
    ->ArgPair(15, 3)
    ->ArgPair(15, 5)
    ->ArgPair(30, 1)
    ->ArgPair(30, 2)
    ->ArgPair(30, 3)
    ->ArgPair(30, 5)
    ->ArgPair(60, 1)
    ->ArgPair(60, 3)
    ->ArgPair(60, 5);

static void BM_RPC(::testing::benchmark::State& state) {
  const int width = state.range(0);
  const int tensor_size = state.range(1);

  BM_Helper(state, width, 2 /*num_stages*/, tensor_size, true /*multi-device*/);
}
BENCHMARK(BM_RPC)->ArgPair(30, 2)->ArgPair(30, 1000)->ArgPair(30, 100000);

static void BM_SingleDevice(::testing::benchmark::State& state) {
  const int width = state.range(0);
  const int num_stages = state.range(1);

  BM_Helper(state, width, num_stages, 2 /*tensor_size*/,
            false /*not multi-device*/);
}
BENCHMARK(BM_SingleDevice)
    ->ArgPair(1, 1)
    ->ArgPair(30, 2)
    ->ArgPair(60, 5)
    ->ArgPair(4, 10000)
    ->ArgPair(1, 1000000);

}  // namespace tensorflow
