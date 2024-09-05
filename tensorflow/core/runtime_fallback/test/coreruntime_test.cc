/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "testing/base/public/benchmark.h"
#include <gtest/gtest.h>
#include "tensorflow/core/runtime_fallback/test/coreruntime_driver.h"
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attr_type.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor_view.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_metadata.h"  // from @tf_runtime

namespace tfrt {
namespace {

// This test runs TFRT's native MatMul Op.
TEST(CoreRuntimeTest, Matmul) {
  auto driver = CoreRuntimeDriver();
  driver.InitializeCpuRuntimeFallbackOpHandler();

  tfrt::OpAttrs attrs1;
  tfrt::TensorHandle a1;
  attrs1.SetArray("shape", tfrt::ArrayRef<Index>{2, 2});
  attrs1.SetArray("values", tfrt::ArrayRef<float>{1.0, 1.0, 1.0, 1.0});
  driver.Execute("tfrt_test.create_dense_tensor", {}, attrs1.freeze(), a1,
                 __FILE__, __LINE__);

  tfrt::OpAttrs matmul_attrs;
  matmul_attrs.Set<bool>("transpose_a", false);
  matmul_attrs.Set<bool>("transpose_b", false);
  tfrt::OpAttrsRef matmul_attrs_ref = matmul_attrs.freeze();
  // This op will do a matrix multiply.
  tfrt::TensorHandle matmul_args[2] = {a1.CopyRef(), a1.CopyRef()};
  tfrt::TensorHandle a2;
  driver.Execute("tf.MatMul", matmul_args, matmul_attrs_ref, a2, __FILE__,
                 __LINE__);

  tfrt::OpAttrs empty_attrs;
  tfrt::OpAttrsRef empty_attrs_ref = empty_attrs.freeze();
  // This op will print the shape and the value of the result.
  tfrt::TensorHandle a2_ref = a2.CopyRef();
  driver.Execute("tfrt_test.print", a2_ref, empty_attrs_ref, {}, __FILE__,
                 __LINE__);

  // Check the output tensor.
  auto a2_metadata = a2.GetAvailableMetadata();
  ASSERT_EQ(a2_metadata.shape.GetRank(), 2);
  ASSERT_EQ(a2_metadata.shape.GetDimensionSize(0), 2);
  ASSERT_EQ(a2_metadata.shape.GetDimensionSize(1), 2);

  auto a2_view =
      DHTArrayView<float>(&a2.GetAsyncTensor()->get<DenseHostTensor>());
  ASSERT_EQ(a2_view.Elements()[0], 2.0);
  ASSERT_EQ(a2_view.Elements()[1], 2.0);
  ASSERT_EQ(a2_view.Elements()[2], 2.0);
  ASSERT_EQ(a2_view.Elements()[3], 2.0);
}

void BM_CoreRuntimeMatMul(benchmark::State& state) {
  auto driver = CoreRuntimeDriver();
  driver.InitializeCpuRuntimeFallbackOpHandler();

  tfrt::OpAttrs attrs1;
  tfrt::TensorHandle a1;
  attrs1.SetArray("shape", tfrt::ArrayRef<Index>{2, 2});
  attrs1.SetArray("values", tfrt::ArrayRef<float>{2.0, 2.0, 2.0, 2.0});
  driver.Execute("tfrt_test.create_dense_tensor", {}, attrs1.freeze(), a1,
                 __FILE__, __LINE__);

  for (auto _ : state) {
    tfrt::OpAttrs matmul_attrs;
    matmul_attrs.Set<bool>("transpose_a", false);
    matmul_attrs.Set<bool>("transpose_b", false);
    tfrt::OpAttrsRef matmul_attrs_ref = matmul_attrs.freeze();

    tfrt::TensorHandle matmul_args[2] = {a1.CopyRef(), a1.CopyRef()};
    tfrt::TensorHandle a4;
    driver.Execute("tfrt_test.matmul", matmul_args, matmul_attrs_ref, a4,
                   __FILE__, __LINE__);
  }
}
BENCHMARK(BM_CoreRuntimeMatMul);

void BM_CoreRuntimeMakeOpMatMul(benchmark::State& state) {
  auto driver = CoreRuntimeDriver();
  driver.InitializeCpuRuntimeFallbackOpHandler();

  tfrt::OpAttrs attrs1;
  tfrt::TensorHandle a1;
  attrs1.SetArray("shape", tfrt::ArrayRef<Index>{2, 2});
  attrs1.SetArray("values", tfrt::ArrayRef<float>{2.0, 2.0, 2.0, 2.0});
  driver.Execute("tfrt_test.create_dense_tensor", {}, attrs1.freeze(), a1,
                 __FILE__, __LINE__);

  auto matmul_op = driver.MakeOp("tfrt_test.matmul");

  for (auto _ : state) {
    tfrt::OpAttrs matmul_attrs;
    matmul_attrs.Set<bool>("transpose_a", false);
    matmul_attrs.Set<bool>("transpose_b", false);
    tfrt::OpAttrsRef matmul_attrs_ref = matmul_attrs.freeze();

    tfrt::TensorHandle matmul_args[2] = {a1.CopyRef(), a1.CopyRef()};
    tfrt::TensorHandle a4;

    matmul_op(driver.CreateExecutionContext(__FILE__, __LINE__), matmul_args,
              matmul_attrs_ref, a4, /*chain=*/nullptr);
  }
}
BENCHMARK(BM_CoreRuntimeMakeOpMatMul);

void BM_CreatFallbackCoreRuntimeOp(benchmark::State& state) {
  auto driver = CoreRuntimeDriver();
  driver.InitializeCpuRuntimeFallbackOpHandler();
  for (auto _ : state) {
    // Creates a fallback op.
    benchmark::DoNotOptimize(driver.MakeOp("tf.ParseExampleV2"));
  }
}
BENCHMARK(BM_CreatFallbackCoreRuntimeOp);

}  // namespace
}  // namespace tfrt
