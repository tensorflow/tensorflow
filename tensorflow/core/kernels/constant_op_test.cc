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
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class ConstantOpTest : public OpsTestBase {
 protected:
  void PersistentMemoryTrackingTest(bool on_gpu);
};

void ConstantOpTest::PersistentMemoryTrackingTest(bool on_gpu) {
  DataType data_type = DT_INT32;
  std::initializer_list<int64> dims = {2, 3, 4, 5};
  Tensor tensor(data_type, TensorShape(dims));
  for (int i = 0; i < 2 * 3 * 4 * 5; ++i) {
    tensor.flat<int32>()(i) = i;
  }

  NodeDef const_node;
  TF_ASSERT_OK(NodeDefBuilder("some_node", "Const")
                   .Attr("dtype", data_type)
                   .Attr("value", tensor)
                   .Finalize(&const_node));

  string device_string = "CPU";
  DeviceType device_type = DEVICE_CPU;
  if (on_gpu) {
    device_string = "GPU";
    DeviceType device_type = DEVICE_GPU;
  }
  std::unique_ptr<Device> device(DeviceFactory::NewDevice(
      device_string, {}, "/job:worker/replica:0/task:0"));

  Status status;
  std::unique_ptr<OpKernel> op(CreateOpKernel(device_type, device.get(),
                                              cpu_allocator(), const_node,
                                              TF_GRAPH_DEF_VERSION, &status));
  TF_ASSERT_OK(status);

  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.op_kernel = op.get();
  params.track_allocations = true;

  OpKernelContext ctx(&params);
  op->Compute(&ctx);
  TF_EXPECT_OK(ctx.status());

  if (on_gpu) {
    EXPECT_EQ(ctx.persistent_memory_allocated(), 512);
  } else {
    EXPECT_EQ(ctx.persistent_memory_allocated(), 480);
  }

  // Remove memory leak errors.
  for (auto allocator_pair : ctx.ConsumeWrappedAllocators()) {
    allocator_pair.second->GetRecordsAndUnRef();
  }
}

TEST_F(ConstantOpTest, PersistentMemoryTracking) {
  PersistentMemoryTrackingTest(false);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  PersistentMemoryTrackingTest(true);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

// Returns graph containing "num" const nodes.  If 'sequential' is
// true, make sure all constants are executed sequentially in the
// graph by adding control dependencies.
static Graph* ManyConsts(int num, bool sequential) {
  Graph* g = new Graph(OpRegistry::Global());
  Node* prev = nullptr;
  for (int i = 0; i < num; ++i) {
    Tensor c(DT_FLOAT, TensorShape({}));
    c.scalar<float>()() = i;
    Node* curr = test::graph::Constant(g, c);
    if (sequential && prev != nullptr) {
      g->AddControlEdge(prev, curr);
    }
    prev = curr;
  }
  return g;
}

static void BM_ManyConsts_Parallel(int iters, int num) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num);
  test::Benchmark("cpu", ManyConsts(num, false /* !sequential */)).Run(iters);
}
BENCHMARK(BM_ManyConsts_Parallel)->Range(1, 1 << 10);

static void BM_ManyConsts_Sequential(int iters, int num) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num);
  test::Benchmark("cpu", ManyConsts(num, true /* sequential */)).Run(iters);
}
BENCHMARK(BM_ManyConsts_Sequential)->Range(1, 1 << 10);

}  // end namespace tensorflow
