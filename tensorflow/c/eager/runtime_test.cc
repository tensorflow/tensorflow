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

#include "tensorflow/c/eager/runtime.h"

#include <memory>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

Device* CPUDevice() {
  return DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0");
}

TEST(AttrTypeMap, Lookup) {
  const AttrTypeMap* m = nullptr;
  Status s = AttrTypeMapForOp("ThisOpCannotPossiblyExist", &m);
  EXPECT_FALSE(s.ok());
  s = AttrTypeMapForOp("MatMul", &m);
  ASSERT_TRUE(s.ok()) << s;

  TF_AttrType t;
  unsigned char is_list = 1;
  s = AttrTypeByName(m, "ThisAttribyteCannotPossiblyExist", &t, &is_list);
  EXPECT_FALSE(s.ok());
  EXPECT_NE(is_list, 0);
  s = AttrTypeByName(m, "transpose_a", &t, &is_list);
  ASSERT_TRUE(s.ok()) << s;
  EXPECT_EQ(TF_ATTR_BOOL, t);
  EXPECT_EQ(is_list, 0);

  s = AttrTypeMapForOp("Squeeze", &m);
  ASSERT_TRUE(s.ok()) << s;
  s = AttrTypeByName(m, "squeeze_dims", &t, &is_list);
  ASSERT_TRUE(s.ok()) << s;
  EXPECT_EQ(TF_ATTR_INT, t);
  EXPECT_NE(is_list, 0);
}

TEST(KernelAndDevice, Run) {
  Tensor t(Input({{1.0f, 2.0f}, {3.0f, 4.0f}}).tensor());
  std::vector<Tensor> inputs;
  inputs.push_back(t);
  inputs.push_back(t);
  NodeDef ndef(AttrBuilder("MatMul")
                   .Set("T", DT_FLOAT)
                   .Set("transpose_a", false)
                   .Set("transpose_b", false)
                   .NumInputs(inputs.size())
                   .BuildNodeDef());
  std::unique_ptr<Device> device(CPUDevice());
  KernelAndDevice kernel(nullptr);
  Status s = KernelAndDevice::InitOp(device.get(), ndef, &kernel);
  ASSERT_TRUE(s.ok()) << s;
  std::vector<Tensor> outputs;
  s = kernel.Run(&inputs, &outputs);
  ASSERT_TRUE(s.ok()) << s;
  ASSERT_EQ(1, outputs.size());
  const Tensor& out = outputs[0];
  EXPECT_EQ(7, out.matrix<float>()(0, 0));
  EXPECT_EQ(10, out.matrix<float>()(0, 1));
  EXPECT_EQ(15, out.matrix<float>()(1, 0));
  EXPECT_EQ(22, out.matrix<float>()(1, 1));
}

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
  std::unique_ptr<Device> device(CPUDevice());
  KernelAndDevice k(nullptr);
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    TF_CHECK_OK(KernelAndDevice::InitOp(device.get(), ndef, &k));
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
  std::unique_ptr<Device> device(CPUDevice());
  KernelAndDevice kernel(nullptr);
  TF_CHECK_OK(KernelAndDevice::InitOp(device.get(), ndef, &kernel));
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    TF_CHECK_OK(kernel.Run(&inputs, &outputs));
  }
}
BENCHMARK(BM_KernelAndDeviceRun);
}  // namespace
}  // namespace tensorflow
