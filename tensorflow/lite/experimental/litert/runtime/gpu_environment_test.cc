// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/runtime/gpu_environment.h"

#include <any>
#include <array>
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "third_party/ml_drift/cl/environment.h"
#include "third_party/ml_drift/cl/opencl_wrapper.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_any.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/opencl_wrapper.h"

namespace litert {
namespace {

TEST(EnvironmentSingletonTest, OpenClEnvironment) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif

  if (!ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }
  if (!litert::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for litert";
  }

  ml_drift::cl::Environment env;
  ASSERT_OK(ml_drift::cl::CreateEnvironment(&env));

  const std::array<LiteRtEnvOption, 2> environment_options = {
      LiteRtEnvOption{
          /*.tag=*/kLiteRtEnvOptionTagOpenClContext,
          /*.value=*/
          *ToLiteRtAny(
              std::any(reinterpret_cast<int64_t>(env.context().context()))),
      },
      LiteRtEnvOption{
          /*.tag=*/kLiteRtEnvOptionTagOpenClCommandQueue,
          /*.value=*/
          *ToLiteRtAny(
              std::any(reinterpret_cast<int64_t>(env.queue()->queue()))),
      },
  };
  auto litert_envt = LiteRtEnvironmentT::CreateWithOptions(environment_options);
  ASSERT_TRUE(litert_envt);
  auto singleton_env =
      litert::internal::GpuEnvironmentSingleton::Create(litert_envt->get());
  ASSERT_TRUE(singleton_env);
  EXPECT_EQ((*singleton_env)->getContext()->context(), env.context().context());
  EXPECT_EQ((*singleton_env)->getCommandQueue()->queue(), env.queue()->queue());

  // Create another singleton environment should fail.
  auto another_singleton_env =
      litert::internal::GpuEnvironmentSingleton::Create(litert_envt->get());
  EXPECT_FALSE(another_singleton_env);
}

}  // namespace
}  // namespace litert
