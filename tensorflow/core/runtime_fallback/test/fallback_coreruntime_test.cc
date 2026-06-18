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
#include <utility>

#include "testing/base/public/benchmark.h"
#include <gtest/gtest.h>
#include "tensorflow/core/runtime_fallback/test/coreruntime_driver.h"
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attr_type.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor_view.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_serialize_utils.h"  // from @tf_runtime

namespace tfrt {
namespace {

class FallbackCoreRuntimeTest : public testing::TestWithParam<string_view> {};

void FallbackCoreRuntimeTestMatmulBase(CoreRuntimeDriver& driver) {
  tfrt::OpAttrs attrs1;
  tfrt::TensorHandle a1;
  attrs1.SetArray("shape", tfrt::ArrayRef<Index>{2, 2});
  attrs1.SetArray("values", tfrt::ArrayRef<float>{1.0, 1.0, 1.0, 1.0});
  driver.Execute("tfrt_test.create_dense_tensor", {}, attrs1.freeze(), a1,
                 __FILE__, __LINE__);

  tfrt::OpAttrs matmul_attrs;
  matmul_attrs.Set<bool>("transpose_a", false);
  matmul_attrs.Set<bool>("transpose_b", false);
  matmul_attrs.Set<OpAttrType>("T", OpAttrType::F32);
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

  // Convert fallback tensor to DenseHostTensor.
  tfrt::TensorHandle a3;
  tfrt::TensorHandle a2_ref2 = a2.CopyRef();
  driver.Execute("tfrt_test.identity", a2_ref2, empty_attrs_ref, a3, __FILE__,
                 __LINE__);

  // Fallback kernels are run on blocking work queue. Need to wait.
  driver.WaitForHostContextQuiesce();

  // Check the output tensor.
  auto a3_metadata = a2.GetAvailableMetadata();
  ASSERT_EQ(a3_metadata.shape.GetRank(), 2);
  ASSERT_EQ(a3_metadata.shape.GetDimensionSize(0), 2);
  ASSERT_EQ(a3_metadata.shape.GetDimensionSize(1), 2);

  auto a3_view =
      DHTArrayView<float>(&a3.GetAsyncTensor()->get<DenseHostTensor>());
  ASSERT_EQ(a3_view.Elements()[0], 2.0);
  ASSERT_EQ(a3_view.Elements()[1], 2.0);
  ASSERT_EQ(a3_view.Elements()[2], 2.0);
  ASSERT_EQ(a3_view.Elements()[3], 2.0);
}

TEST(FallbackCoreRuntimeTest, RuntimeFallbackMatmul) {
  auto runtime_driver = CoreRuntimeDriver();
  runtime_driver.InitializeCpuRuntimeFallbackOpHandler();
  FallbackCoreRuntimeTestMatmulBase(runtime_driver);
}

TEST(FallbackCoreRuntimeTest, KernelFallbackMatmul) {
  auto kernel_driver = CoreRuntimeDriver();
  kernel_driver.InitializeCpuKernelFallbackOpHandler();
  FallbackCoreRuntimeTestMatmulBase(kernel_driver);
}

void FallbackMatMulWithInputOutputTensorConversion(benchmark::State& state,
                                                   CoreRuntimeDriver& driver) {
  // Create input tensors as DHT.
  tfrt::OpAttrs attrs1;
  tfrt::TensorHandle a1;
  attrs1.SetArray("shape", tfrt::ArrayRef<Index>{2, 2});
  attrs1.SetArray("values", tfrt::ArrayRef<float>{1.0, 1.0, 1.0, 1.0});
  driver.Execute("tfrt_test.create_dense_tensor", {}, attrs1.freeze(), a1,
                 __FILE__, __LINE__);

  for (auto _ : state) {
    tfrt::OpAttrs matmul_attrs;
    matmul_attrs.Set<bool>("transpose_a", false);
    matmul_attrs.Set<bool>("transpose_b", false);
    tfrt::OpAttrsRef matmul_attrs_ref = matmul_attrs.freeze();
    // Matrix multiply using runtime fallback.
    tfrt::TensorHandle matmul_args[2] = {a1.CopyRef(), a1.CopyRef()};
    tfrt::TensorHandle a2;
    // Since inputs are DHT, they will first be copied to fallback tensors.
    driver.Execute("tf.MatMul", matmul_args, matmul_attrs_ref, a2, __FILE__,
                   __LINE__);
    // Copy fallback tensors to DHT.
    tfrt::TensorHandle a3;
    tfrt::OpAttrs empty_attrs;
    tfrt::OpAttrsRef empty_attrs_ref = empty_attrs.freeze();
    driver.Execute("tfrt_test.identity", a2, empty_attrs_ref, a3, __FILE__,
                   __LINE__);
    // Fallback kernels are run on blocking work queue. Need to wait.
    driver.WaitForHostContextQuiesce();
  }
}

void BM_RuntimeFallbackMatMulWithInputOutputTensorConversion(
    benchmark::State& state) {
  auto runtime_driver = CoreRuntimeDriver();
  runtime_driver.InitializeCpuRuntimeFallbackOpHandler();
  FallbackMatMulWithInputOutputTensorConversion(state, runtime_driver);
}

void BM_KernelFallbackMatMulWithInputOutputTensorConversion(
    benchmark::State& state) {
  auto kernel_driver = CoreRuntimeDriver();
  kernel_driver.InitializeCpuKernelFallbackOpHandler();
  FallbackMatMulWithInputOutputTensorConversion(state, kernel_driver);
}

BENCHMARK(BM_RuntimeFallbackMatMulWithInputOutputTensorConversion);
BENCHMARK(BM_KernelFallbackMatMulWithInputOutputTensorConversion);

void FallbackMatMulNoInputOutputTensorConversion(benchmark::State& state,
                                                 CoreRuntimeDriver& driver) {
  // Create a DHT to serialize to a DenseAttr, to be used as `value` attribute
  // of tf.Const.
  auto dht_create_res = tfrt::DenseHostTensor::CreateUninitialized<float>(
      TensorShape({2, 2}), driver.GetHost());
  ASSERT_TRUE(dht_create_res.has_value());
  DenseHostTensor dht(std::move(*dht_create_res));
  MutableDHTArrayView<float> tensor_view(&dht);
  tensor_view.Fill(1.0f);

  BefAttrEncoder encoder;
  const size_t offset = SerializeDenseHostTensorToDenseAttr(dht, &encoder);
  auto dense_attr_buffer = encoder.TakeResult();
  DenseAttr dense_attr(dense_attr_buffer.data() + offset);

  // Use tf.Const to create input tensors as fallback tensors.
  tfrt::OpAttrs const_attr;
  tfrt::TensorHandle a1;
  const_attr.Set("dtype", tfrt::OpAttrType::F32);
  const_attr.Set("value", dense_attr);
  driver.Execute("tf.Const", {}, const_attr.freeze(), a1, __FILE__, __LINE__);

  for (auto _ : state) {
    tfrt::OpAttrs matmul_attrs;
    matmul_attrs.Set<bool>("transpose_a", false);
    matmul_attrs.Set<bool>("transpose_b", false);
    tfrt::OpAttrsRef matmul_attrs_ref = matmul_attrs.freeze();
    // Matrix multiply using runtime fallback.
    tfrt::TensorHandle matmul_args[2] = {a1.CopyRef(), a1.CopyRef()};
    tfrt::TensorHandle a2;
    // Since inputs are fallback tensors, no input tensor conversion here.
    driver.Execute("tf.MatMul", matmul_args, matmul_attrs_ref, a2, __FILE__,
                   __LINE__);
    // Fallback kernels are run on blocking work queue. Need to wait.
    driver.WaitForHostContextQuiesce();
  }
}

void BM_RuntimeFallbackMatMulNoInputOutputTensorConversion(
    benchmark::State& state) {
  auto runtime_driver = CoreRuntimeDriver();
  runtime_driver.InitializeCpuRuntimeFallbackOpHandler();
  FallbackMatMulNoInputOutputTensorConversion(state, runtime_driver);
}

void BM_KernelFallbackMatMulNoInputOutputTensorConversion(
    benchmark::State& state) {
  auto kernel_driver = CoreRuntimeDriver();
  kernel_driver.InitializeCpuKernelFallbackOpHandler();
  FallbackMatMulNoInputOutputTensorConversion(state, kernel_driver);
}

BENCHMARK(BM_RuntimeFallbackMatMulNoInputOutputTensorConversion);
BENCHMARK(BM_KernelFallbackMatMulNoInputOutputTensorConversion);

}  // namespace
}  // namespace tfrt
