/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"

#include <memory>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

class TestEnv {
 public:
  TestEnv() : flib_def_(OpRegistry::Global(), {}) {
    Device* device =
        DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0");
    device_mgr_.reset(new DeviceMgr({device}));
    flib_runtime_ = NewFunctionLibraryRuntime(device_mgr_.get(), Env::Default(),
                                              device, TF_GRAPH_DEF_VERSION,
                                              &flib_def_, nullptr, {}, nullptr);
  }

  FunctionLibraryRuntime* function_library_runtime() const {
    return flib_runtime_.get();
  }

 private:
  FunctionLibraryDefinition flib_def_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryRuntime> flib_runtime_;
};

void BM_CreateGraph(int iters) {
  for (int i = 0; i < iters; ++i) {
    Scope root = Scope::NewRootScope();
    auto C = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}});
    auto M = ops::MatMul(root, C, C);
    TF_CHECK_OK(root.status());
  }
}
BENCHMARK(BM_CreateGraph);

void BM_RunGraph(int iters) {
  tensorflow::testing::StopTiming();
  Scope root = Scope::NewRootScope();
  auto C = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}});
  auto M = ops::MatMul(root, C, C);
  SessionOptions opts;
  opts.config.set_inter_op_parallelism_threads(1);
  opts.config.set_intra_op_parallelism_threads(1);
  ClientSession sess(root, opts);
  std::vector<Tensor> outputs;
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    outputs.clear();
    TF_CHECK_OK(sess.Run({M}, &outputs));
  }
}
BENCHMARK(BM_RunGraph);

void BM_CreateAndDestroySession(int iters) {
  tensorflow::testing::StopTiming();
  Scope root = Scope::NewRootScope();
  auto C = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}});
  auto M = ops::MatMul(root, C, C);
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    ClientSession sess(root);
  }
}
BENCHMARK(BM_CreateAndDestroySession);

void BM_KernelAndDeviceInit(int iters) {
  tensorflow::testing::StopTiming();
  NodeDef ndef(AttrBuilder("MatMul")
                   .Set("T", DT_FLOAT)
                   .Set("transpose_a", false)
                   .Set("transpose_b", false)
                   .NumInputs(2)
                   .BuildNodeDef());
  TestEnv env;
  KernelAndDevice k(nullptr, false);
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    TF_CHECK_OK(KernelAndDevice::Init(ndef, env.function_library_runtime(),
                                      nullptr, &k));
  }
}
BENCHMARK(BM_KernelAndDeviceInit);

void BM_KernelAndDeviceRun(int iters) {
  tensorflow::testing::StopTiming();
  Tensor t(Input({{1.0f, 2.0f}, {3.0f, 4.0f}}).tensor());
  std::vector<Tensor> inputs;
  inputs.push_back(t);
  inputs.push_back(t);
  std::vector<Tensor> outputs;
  NodeDef ndef(AttrBuilder("MatMul")
                   .Set("T", DT_FLOAT)
                   .Set("transpose_a", false)
                   .Set("transpose_b", false)
                   .NumInputs(inputs.size())
                   .BuildNodeDef());
  TestEnv env;
  KernelAndDevice kernel(nullptr, false);
  TF_CHECK_OK(KernelAndDevice::Init(ndef, env.function_library_runtime(),
                                    nullptr, &kernel));
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    TF_CHECK_OK(kernel.Run(&inputs, &outputs, nullptr));
  }
}
BENCHMARK(BM_KernelAndDeviceRun);
}  // namespace
}  // namespace tensorflow
