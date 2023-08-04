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
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/test/coreruntime_driver.h"
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attr_type.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor_view.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_serialize_utils.h"  // from @tf_runtime

using tensorflow::KernelFallbackTensor;

namespace tfrt {
namespace {

TEST(KernelFallbackCoreRuntimeTest, AddN) {
  auto driver = CoreRuntimeDriver();
  driver.InitializeCpuKernelFallbackOpHandler();

  tfrt::OpAttrs attrs1;
  tfrt::TensorHandle a1;
  attrs1.SetArray("shape", tfrt::ArrayRef<Index>{});
  attrs1.SetArray("values", tfrt::ArrayRef<int32_t>{4});
  driver.Execute("tfrt_test.create_dense_tensor", {}, attrs1.freeze(), a1,
                 __FILE__, __LINE__);

  tfrt::OpAttrs attrs2;
  tfrt::TensorHandle a2;
  attrs2.SetArray("shape", tfrt::ArrayRef<Index>{});
  attrs2.SetArray("values", tfrt::ArrayRef<int32_t>{5});
  driver.Execute("tfrt_test.create_dense_tensor", {}, attrs2.freeze(), a2,
                 __FILE__, __LINE__);

  tfrt::OpAttrs addn_attrs;
  addn_attrs.Set<int64_t>("N", 2);
  addn_attrs.Set("T", tfrt::OpAttrType::I32);
  tfrt::OpAttrsRef addn_attrs_ref = addn_attrs.freeze();
  // This op will do a matrix addition.
  tfrt::TensorHandle addn_args[2] = {a1.CopyRef(), a2.CopyRef()};
  tfrt::TensorHandle out;
  driver.Execute("tf.AddN", addn_args, addn_attrs_ref, out, __FILE__, __LINE__);

  // Wait for async value
  std::vector<RCReference<AsyncValue>> async_values;
  async_values.push_back(out.GetAsyncMetadata().CopyRef());
  async_values.push_back(tfrt::FormRef(out.GetAsyncTensor()));
  driver.GetHost()->Await(async_values);

  ASSERT_TRUE(out.IsValid());
  ASSERT_FALSE(out.GetAsyncTensor()->IsError());

  // Check the output tensor.
  auto out_metadata = out.GetAvailableMetadata();
  ASSERT_EQ(out_metadata.shape.GetRank(), 0);
  auto& out_tensor = out.GetAsyncTensor()->get<KernelFallbackTensor>();
  ASSERT_EQ(out_tensor.GetTensor()->DebugString(),
            "Tensor<type: int32 shape: [] values: 9>");
}

TEST(KernelFallbackCoreRuntimeTest, KernelNotFound) {
  auto driver = CoreRuntimeDriver();
  driver.InitializeCpuKernelFallbackOpHandler();

  tfrt::OpAttrs attrs;
  tfrt::OpAttrsRef attrs_ref = attrs.freeze();
  // This op will do a matrix addition.
  tfrt::TensorHandle out;
  driver.Execute("tf.UnavailableOp", {}, attrs_ref, out, __FILE__, __LINE__);

  // Wait for async value
  std::vector<RCReference<AsyncValue>> async_values;
  async_values.push_back(tfrt::FormRef(out.GetAsyncTensor()));
  driver.GetHost()->Await(async_values);

  ASSERT_TRUE(out.IsValid());
  ASSERT_TRUE(out.GetAsyncTensor()->IsError());
}

void BM_KernelFallbackAddNWithInputOutputTensorConversionTest(
    ::testing::benchmark::State& state) {
  auto driver = CoreRuntimeDriver();
  driver.InitializeCpuKernelFallbackOpHandler();

  // Create input tensors as DHT.
  tfrt::OpAttrs attrs1;
  tfrt::TensorHandle a1;
  attrs1.SetArray("shape", tfrt::ArrayRef<Index>{});
  attrs1.SetArray("values", tfrt::ArrayRef<int32_t>{4});
  driver.Execute("tfrt_test.create_dense_tensor", {}, attrs1.freeze(), a1,
                 __FILE__, __LINE__);

  tfrt::OpAttrs attrs2;
  tfrt::TensorHandle a2;
  attrs2.SetArray("shape", tfrt::ArrayRef<Index>{});
  attrs2.SetArray("values", tfrt::ArrayRef<int32_t>{5});
  driver.Execute("tfrt_test.create_dense_tensor", {}, attrs2.freeze(), a2,
                 __FILE__, __LINE__);

  for (auto _ : state) {
    tfrt::OpAttrs addn_attrs;
    addn_attrs.Set<int64_t>("N", 2);
    addn_attrs.Set("T", tfrt::OpAttrType::I32);
    tfrt::OpAttrsRef addn_attrs_ref = addn_attrs.freeze();
    tfrt::TensorHandle addn_args[2] = {a1.CopyRef(), a2.CopyRef()};
    tfrt::TensorHandle out1;
    // Since inputs are DHT, they will first be copied to fallback tensors.
    driver.Execute("tf.AddN", addn_args, addn_attrs_ref, out1, __FILE__,
                   __LINE__);

    // Wait for async value
    std::vector<RCReference<AsyncValue>> async_values;
    async_values.push_back(out1.GetAsyncMetadata().CopyRef());
    driver.GetHost()->Await(async_values);
  }
}

BENCHMARK(BM_KernelFallbackAddNWithInputOutputTensorConversionTest);

void BM_KernelFallbackAddNNoInputOutputTensorConversionTest(
    ::testing::benchmark::State& state) {
  auto driver = CoreRuntimeDriver();
  driver.InitializeCpuKernelFallbackOpHandler();

  // Create input tensors as DHT.
  tfrt::OpAttrs attrs1;
  tfrt::TensorHandle a1;
  attrs1.SetArray("shape", tfrt::ArrayRef<Index>{});
  attrs1.SetArray("values", tfrt::ArrayRef<int32_t>{4});
  driver.Execute("tfrt_test.create_dense_tensor", {}, attrs1.freeze(), a1,
                 __FILE__, __LINE__);

  tfrt::OpAttrs attrs2;
  tfrt::TensorHandle a2;
  attrs2.SetArray("shape", tfrt::ArrayRef<Index>{});
  attrs2.SetArray("values", tfrt::ArrayRef<int32_t>{5});
  driver.Execute("tfrt_test.create_dense_tensor", {}, attrs2.freeze(), a2,
                 __FILE__, __LINE__);

  tfrt::OpAttrs attrs_zero;
  tfrt::TensorHandle zero;
  attrs_zero.SetArray("shape", tfrt::ArrayRef<Index>{});
  attrs_zero.SetArray("values", tfrt::ArrayRef<int32_t>{0});
  driver.Execute("tfrt_test.create_dense_tensor", {}, attrs_zero.freeze(), zero,
                 __FILE__, __LINE__);

  // Generate output tensors that don't need conversions by first
  // running preliminary additions.
  tfrt::OpAttrs addn_attrs1;
  addn_attrs1.Set<int64_t>("N", 2);
  addn_attrs1.Set("T", tfrt::OpAttrType::I32);
  tfrt::TensorHandle addn_args1[2] = {a1.CopyRef(), zero.CopyRef()};
  tfrt::TensorHandle converted_arg1;
  driver.Execute("tf.AddN", addn_args1, addn_attrs1.freeze(), converted_arg1,
                 __FILE__, __LINE__);

  tfrt::OpAttrs addn_attrs2;
  addn_attrs2.Set<int64_t>("N", 2);
  addn_attrs2.Set("T", tfrt::OpAttrType::I32);
  tfrt::TensorHandle addn_args2[2] = {a2.CopyRef(), zero.CopyRef()};
  tfrt::TensorHandle converted_arg2;
  driver.Execute("tf.AddN", addn_args2, addn_attrs2.freeze(), converted_arg2,
                 __FILE__, __LINE__);

  std::vector<RCReference<AsyncValue>> async_values;
  async_values.push_back(converted_arg1.GetAsyncMetadata().CopyRef());
  async_values.push_back(converted_arg2.GetAsyncMetadata().CopyRef());
  driver.GetHost()->Await(async_values);

  // Now run AddN for already converted tensors.
  for (auto _ : state) {
    tfrt::TensorHandle addn_args[2] = {converted_arg1.CopyRef(),
                                       converted_arg2.CopyRef()};
    tfrt::OpAttrs addn_attrs;
    addn_attrs.Set<int64_t>("N", 2);
    addn_attrs.Set("T", tfrt::OpAttrType::I32);
    tfrt::TensorHandle out;
    driver.Execute("tf.AddN", addn_args, addn_attrs.freeze(), out, __FILE__,
                   __LINE__);
    assert(out.IsValid());

    // Wait for async value
    std::vector<RCReference<AsyncValue>> async_values2;
    async_values2.push_back(out.GetAsyncMetadata().CopyRef());
    driver.GetHost()->Await(async_values2);
  }
}

BENCHMARK(BM_KernelFallbackAddNNoInputOutputTensorConversionTest);

}  // namespace
}  // namespace tfrt
