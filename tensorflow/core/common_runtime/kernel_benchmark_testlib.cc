/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"

#include <vector>
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace test {

Benchmark::Benchmark(const string& device, Graph* g,
                     const SessionOptions* options, Graph* init,
                     Rendezvous* rendez) {
  SessionOptions default_options;
  if (!options) {
    options = &default_options;
  }

  testing::StopTiming();
  string t = str_util::Uppercase(device);
  // Allow NewDevice to allocate a new threadpool with different number of
  // threads for each new benchmark.
  LocalDevice::set_use_global_threadpool(false);
  device_ =
      DeviceFactory::NewDevice(t, *options, "/job:localhost/replica:0/task:0");
  CHECK(device_) << "Could not create a " << device << " device";

  pool_ = new thread::ThreadPool(options->env, "blocking",
                                 port::NumSchedulableCPUs());

  auto runner = [this](std::function<void()> closure) {
    pool_->Schedule(closure);
  };

  if (rendez == nullptr) {
    rendez_ = NewLocalRendezvous();
  } else {
    rendez_ = rendez;
  }

  const int graph_def_version = g->versions().producer();

  LocalExecutorParams params;
  params.device = device_;
  params.function_library = nullptr;
  params.create_kernel = [this, graph_def_version](const NodeDef& ndef,
                                                   OpKernel** kernel) {
    return CreateNonCachedKernel(device_, nullptr, ndef, graph_def_version,
                                 kernel);
  };
  params.delete_kernel = [](OpKernel* kernel) {
    DeleteNonCachedKernel(kernel);
  };

  if (init) {
    Executor* init_exec;
    TF_CHECK_OK(NewLocalExecutor(params, init, &init_exec));
    Executor::Args args;
    args.rendezvous = rendez_;
    args.runner = runner;
    TF_CHECK_OK(init_exec->Run(args));
    delete init_exec;
  }

  TF_CHECK_OK(NewLocalExecutor(params, g, &exec_));
}

Benchmark::~Benchmark() {
  if (device_) {
    rendez_->Unref();
    delete exec_;
    delete device_;
    delete pool_;
  }
}

void Benchmark::Run(int iters) { RunWithArgs({}, {}, iters); }

string GetRendezvousKey(const Node* node) {
  string send_device;
  TF_CHECK_OK(GetNodeAttr(node->attrs(), "send_device", &send_device));
  string recv_device;
  TF_CHECK_OK(GetNodeAttr(node->attrs(), "recv_device", &recv_device));
  string tensor_name;
  TF_CHECK_OK(GetNodeAttr(node->attrs(), "tensor_name", &tensor_name));
  uint64 send_device_incarnation;
  TF_CHECK_OK(GetNodeAttr(node->attrs(), "send_device_incarnation",
                          reinterpret_cast<int64*>(&send_device_incarnation)));
  return Rendezvous::CreateKey(send_device, send_device_incarnation,
                               recv_device, tensor_name, FrameAndIter(0, 0));
}

void Benchmark::RunWithArgs(
    const std::vector<std::pair<const Node*, Tensor>>& inputs,
    const std::vector<const Node*>& outputs, int iters) {
  if (!device_ || iters == 0) {
    return;
  }
  // Gets inputs' and outputs' rendezvous keys.
  std::vector<std::pair<string, Tensor>> in;
  in.reserve(inputs.size());
  for (const auto& p : inputs) {
    in.push_back({GetRendezvousKey(p.first), p.second});
  }
  std::vector<string> out;
  out.reserve(outputs.size());
  for (const auto& n : outputs) {
    out.push_back(GetRendezvousKey(n));
  }
  Tensor unused;  // In benchmark, we don't care the return value.
  bool is_dead;

  // Warm up
  Executor::Args args;
  args.rendezvous = rendez_;
  args.runner = [this](std::function<void()> closure) {
    pool_->Schedule(closure);
  };
  static const int kWarmupRuns = 3;
  for (int i = 0; i < kWarmupRuns; ++i) {
    for (const auto& p : in) {
      Rendezvous::ParsedKey parsed;
      TF_CHECK_OK(Rendezvous::ParseKey(p.first, &parsed));
      TF_CHECK_OK(rendez_->Send(parsed, Rendezvous::Args(), p.second, false));
    }
    TF_CHECK_OK(exec_->Run(args));
    for (const string& key : out) {
      Rendezvous::ParsedKey parsed;
      TF_CHECK_OK(Rendezvous::ParseKey(key, &parsed));
      TF_CHECK_OK(rendez_->Recv(parsed, Rendezvous::Args(), &unused, &is_dead));
    }
  }
  TF_CHECK_OK(device_->Sync());
  VLOG(3) << kWarmupRuns << " warmup runs done.";

  testing::StartTiming();
  while (iters-- > 0) {
    for (const auto& p : in) {
      Rendezvous::ParsedKey parsed;
      TF_CHECK_OK(Rendezvous::ParseKey(p.first, &parsed));
      TF_CHECK_OK(rendez_->Send(parsed, Rendezvous::Args(), p.second, false));
    }
    TF_CHECK_OK(exec_->Run(args));
    for (const string& key : out) {
      Rendezvous::ParsedKey parsed;
      TF_CHECK_OK(Rendezvous::ParseKey(key, &parsed));
      TF_CHECK_OK(rendez_->Recv(parsed, Rendezvous::Args(), &unused, &is_dead));
    }
  }

  TF_CHECK_OK(device_->Sync());
  testing::StopTiming();
}

}  // end namespace test
}  // end namespace tensorflow
