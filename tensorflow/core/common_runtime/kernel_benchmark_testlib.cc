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
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/executor_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace test {

// TODO(hongm): Convert `g` and `init` to using std::unique_ptr.
Benchmark::Benchmark(const string& device, Graph* g,
                     const SessionOptions* options, Graph* init,
                     Rendezvous* rendez, const char* executor_type,
                     bool old_benchmark_api) {
  auto cleanup = gtl::MakeCleanup([g, init]() {
    delete g;
    delete init;
  });

  SessionOptions default_options;
  if (!options) {
    options = &default_options;
  }

  old_benchmark_api_ = old_benchmark_api;
  CHECK(!old_benchmark_api) << "Expected new API only";
  if (old_benchmark_api_) testing::StopTiming();
  string t = absl::AsciiStrToUpper(device);
  // Allow NewDevice to allocate a new threadpool with different number of
  // threads for each new benchmark.
  LocalDevice::set_use_global_threadpool(false);

  device_mgr_ = absl::make_unique<StaticDeviceMgr>(
      DeviceFactory::NewDevice(t, *options, "/job:localhost/replica:0/task:0"));
  device_ = device_mgr_->ListDevices()[0];
  CHECK(device_) << "Could not create a " << device << " device";

  pool_ =
      new thread::ThreadPool(options->env, "blocking", port::MaxParallelism());

  auto runner = [this](std::function<void()> closure) {
    pool_->Schedule(closure);
  };

  if (rendez == nullptr) {
    rendez_ = NewLocalRendezvous();
  } else {
    rendez_ = rendez;
  }

  const int graph_def_version = g->versions().producer();

  flib_def_ = absl::make_unique<FunctionLibraryDefinition>(g->flib_def());

  pflr_ = std::unique_ptr<ProcessFunctionLibraryRuntime>(
      new ProcessFunctionLibraryRuntime(
          device_mgr_.get(), Env::Default(), nullptr, graph_def_version,
          flib_def_.get(), OptimizerOptions(), pool_, nullptr, nullptr,
          Rendezvous::Factory()));

  flr_ = pflr_->GetFLR(device_->name());

  LocalExecutorParams params;
  params.device = device_;
  params.function_library = flr_;
  params.create_kernel = [this, graph_def_version](
                             const std::shared_ptr<const NodeProperties>& props,
                             OpKernel** kernel) {
    return CreateNonCachedKernel(device_, flr_, props, graph_def_version,
                                 kernel);
  };
  params.delete_kernel = [](OpKernel* kernel) {
    DeleteNonCachedKernel(kernel);
  };

  if (init) {
    std::unique_ptr<Executor> init_exec;
    TF_CHECK_OK(NewExecutor(executor_type, params, *init, &init_exec));
    Executor::Args args;
    args.rendezvous = rendez_;
    args.runner = runner;
    TF_CHECK_OK(init_exec->Run(args));
  }

  TF_CHECK_OK(NewExecutor(executor_type, params, *g, &exec_));
}

Benchmark::Benchmark(const string& device, Graph* g, bool old_benchmark_api)
    : Benchmark(device, g, nullptr, nullptr, nullptr, "", old_benchmark_api) {}

Benchmark::~Benchmark() {
  if (device_) {
    rendez_->Unref();
    // We delete `exec_` before `device_mgr_` because the `exec_` destructor may
    // run kernel destructors that may attempt to access state borrowed from
    // `device_mgr_`, such as the resource manager.
    exec_.reset();
    pflr_.reset();
    device_mgr_.reset();
    delete pool_;
  }
}


void Benchmark::Run(::testing::benchmark::State& state) {
  RunWithRendezvousArgs({}, {}, state);
}

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

void Benchmark::RunWithRendezvousArgs(
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::vector<string>& outputs, ::testing::benchmark::State& state) {
  CHECK(!old_benchmark_api_)
      << "This method should only be called with new benchmark API";
  if (!device_ || state.max_iterations == 0) {
    return;
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
    for (const auto& p : inputs) {
      Rendezvous::ParsedKey parsed;
      TF_CHECK_OK(Rendezvous::ParseKey(p.first, &parsed));
      TF_CHECK_OK(rendez_->Send(parsed, Rendezvous::Args(), p.second, false));
    }
    TF_CHECK_OK(exec_->Run(args));
    for (const string& key : outputs) {
      Rendezvous::ParsedKey parsed;
      TF_CHECK_OK(Rendezvous::ParseKey(key, &parsed));
      TF_CHECK_OK(rendez_->Recv(parsed, Rendezvous::Args(), &unused, &is_dead));
    }
  }
  TF_CHECK_OK(device_->Sync());
  VLOG(3) << kWarmupRuns << " warmup runs done.";

  // Benchmark loop. Timer starts automatically at the beginning of the loop
  // and ends automatically after the last iteration.
  for (auto s : state) {
    for (const auto& p : inputs) {
      Rendezvous::ParsedKey parsed;
      TF_CHECK_OK(Rendezvous::ParseKey(p.first, &parsed));
      TF_CHECK_OK(rendez_->Send(parsed, Rendezvous::Args(), p.second, false));
    }
    TF_CHECK_OK(exec_->Run(args));
    for (const string& key : outputs) {
      Rendezvous::ParsedKey parsed;
      TF_CHECK_OK(Rendezvous::ParseKey(key, &parsed));
      TF_CHECK_OK(rendez_->Recv(parsed, Rendezvous::Args(), &unused, &is_dead));
    }
  }
  TF_CHECK_OK(device_->Sync());
}

}  // end namespace test
}  // end namespace tensorflow
