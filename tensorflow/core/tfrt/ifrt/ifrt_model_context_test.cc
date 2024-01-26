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
// Enable definition of Eigen::ThreadPoolDevice instead of just declaration.
#define EIGEN_USE_THREADS

#include "tensorflow/core/tfrt/ifrt/ifrt_model_context.h"

#include <memory>
#include <numeric>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/concurrency/ref_count.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

Eigen::ThreadPoolDevice GetThreadPoolDevice() {
  constexpr int kMaxParallelism = 16;
  static tsl::thread::ThreadPool* thread_pool = []() {
    return new tsl::thread::ThreadPool(tsl::Env::Default(),
                                       tsl::ThreadOptions(), "IfrtSharding",
                                       kMaxParallelism);
  }();
  return Eigen::ThreadPoolDevice(thread_pool->AsEigenThreadPool(),
                                 kMaxParallelism);
}

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> CreateDummyArray(
    xla::ifrt::Client& client) {
  xla::ifrt::DType dtype(xla::ifrt::DType::kF32);
  xla::ifrt::Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);

  return client.MakeArrayFromHostBuffer(
      data.data(), dtype, shape,
      /*byte_strides=*/std::nullopt,
      xla::ifrt::SingleDeviceSharding::Create(client.devices()[0],
                                              xla::ifrt::MemoryKind()),
      xla::ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
      /*on_done_with_host_buffer=*/{});
}

TEST(IfrtModelContext, ReRegisterShallFail) {
  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  Eigen::ThreadPoolDevice thread_pool_device = GetThreadPoolDevice();
  IfrtModelContext ifrt_model_context(client, &thread_pool_device);

  TF_ASSERT_OK_AND_ASSIGN(tsl::RCReference<xla::ifrt::Array> loaded_variable,
                          CreateDummyArray(*client));

  absl::string_view variable_name = "variable";

  TF_ASSERT_OK(ifrt_model_context.RegisterLoadedVariable(variable_name,
                                                         loaded_variable));

  auto re_register_status =
      ifrt_model_context.RegisterLoadedVariable(variable_name, loaded_variable);

  EXPECT_THAT(re_register_status,
              tsl::testing::StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(IfrtModelContext, GetUnregisterVariableShallFail) {
  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  Eigen::ThreadPoolDevice thread_pool_device = GetThreadPoolDevice();
  IfrtModelContext ifrt_model_context(client, &thread_pool_device);

  absl::string_view variable_name = "variable";

  auto statusor = ifrt_model_context.GetLoadedVariable(variable_name);

  EXPECT_THAT(statusor.status(),
              tsl::testing::StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
