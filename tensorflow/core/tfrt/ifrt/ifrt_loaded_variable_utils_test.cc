/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_utils.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_matcher.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_config.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace tensorflow {
namespace ifrt_serving {
namespace {

using tensorflow::test::TensorEq;
using tsl::testing::StatusIs;

TEST(ShardingUtilsTest, ShardTensorToIfrtLoadedVariableNotFoundWrongName) {
  auto input_tensor =
      test::AsTensor<int32_t>({1, 2, 3, 4}, tensorflow::TensorShape({2, 2}));

  Tensor variable_handle(DT_RESOURCE, TensorShape({}));
  ResourceHandle resource_handle;
  resource_handle.set_name("var_x");
  resource_handle.set_dtypes_and_shapes({{
      DT_INT32,
      TensorShape({2, 2}),
  }});
  variable_handle.flat<ResourceHandle>()(0) = std::move(resource_handle);

  IfrtRestoreTensorRegistry restored_tensor_registry;
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  constexpr int kMaxParallelism = 16;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "Resharding", kMaxParallelism);
  IfrtLoadedVariableRegistry loaded_variable_registry;
  auto restore_work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);

  VariableDeviceShardingConfigProto sharding_config;
  sharding_config.add_device_ids(0);

  auto promise =
      xla::ifrt::Future<absl::StatusOr<tensorflow::Tensor>>::CreatePromise();
  auto future = xla::ifrt::Future<absl::StatusOr<tensorflow::Tensor>>(promise);

  IfrtRestoreTensorRegistry::RestoredTensorInfo restored_tensor_info = {
      GetDtypeAndShape(variable_handle.scalar<ResourceHandle>()()).value(),
      future};
  TF_ASSERT_OK(restored_tensor_registry.TryRegister("var_x_wrong",
                                                    restored_tensor_info));
  promise.Set(input_tensor);
  EXPECT_THAT(
      AsyncLoadRestoredTensorAsIfrtLoadedVariable(
          "var_x", client, thread_pool, restored_tensor_registry,
          loaded_variable_registry, restore_work_queue.get(), sharding_config),
      StatusIs(absl::StatusCode::kNotFound));
}

TEST(ShardingUtilsTest, ShardTensorToIfrtLoadedVariableSucceed) {
  auto input_tensor =
      test::AsTensor<int32_t>({1, 2, 3, 4}, TensorShape({2, 2}));

  Tensor variable_handle(DT_RESOURCE, TensorShape({}));
  ResourceHandle resource_handle;
  resource_handle.set_name("var_x");
  resource_handle.set_dtypes_and_shapes({{
      DT_INT32,
      TensorShape({2, 2}),
  }});
  variable_handle.flat<ResourceHandle>()(0) = std::move(resource_handle);

  IfrtRestoreTensorRegistry restored_tensor_registry;
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  constexpr int kMaxParallelism = 16;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "Resharding", kMaxParallelism);
  IfrtLoadedVariableRegistry loaded_variable_registry;
  auto restore_work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);

  VariableDeviceShardingConfigProto sharding_config;
  sharding_config.add_device_ids(0);

  auto promise =
      xla::ifrt::Future<absl::StatusOr<tensorflow::Tensor>>::CreatePromise();
  auto future = xla::ifrt::Future<absl::StatusOr<tensorflow::Tensor>>(promise);

  IfrtRestoreTensorRegistry::RestoredTensorInfo restored_tensor_info = {
      GetDtypeAndShape(variable_handle.scalar<ResourceHandle>()()).value(),
      future};

  TF_ASSERT_OK(
      restored_tensor_registry.TryRegister("var_x", restored_tensor_info));
  TF_ASSERT_OK(AsyncLoadRestoredTensorAsIfrtLoadedVariable(
      "var_x", client, thread_pool, restored_tensor_registry,
      loaded_variable_registry, restore_work_queue.get(), sharding_config));
  promise.Set(input_tensor);
  IfrtLoadedVariableRegistry::Key key{
      .device_ids = {0},
      .input_name = "var_x",
  };
  TF_ASSERT_OK_AND_ASSIGN(auto v,
                          loaded_variable_registry.GetLoadedVariable(key));
  TF_ASSERT_OK_AND_ASSIGN(auto assembled_array, v.array.Await());

  TF_ASSERT_OK_AND_ASSIGN(auto disassembled_arrays,
                          assembled_array->DisassembleIntoSingleDeviceArrays(
                              xla::ifrt::ArrayCopySemantics::kAlwaysCopy));
  ASSERT_EQ(disassembled_arrays.size(), 1);
  for (int i = 0; i < disassembled_arrays.size(); ++i) {
    tensorflow::Tensor host_tensor(input_tensor.dtype(), input_tensor.shape());
    TF_ASSERT_OK(
        disassembled_arrays[i]
            ->CopyToHostBuffer(host_tensor.data(), /*byte_strides=*/{},
                               xla::ifrt::ArrayCopySemantics::kAlwaysCopy)
            .Await());
    EXPECT_THAT(host_tensor, TensorEq(input_tensor));
  }
}
}  // namespace
}  // namespace ifrt_serving

}  // namespace tensorflow
