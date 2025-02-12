// Copyright 2024 The ML Drift Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/runtime/opencl/buffer.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_command_queue.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_context.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_device.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/opencl_wrapper.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace litert {
namespace internal {

TEST(OpenCLTest, BufferTestFloat) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported In msan";
#endif

  if (!litert::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL buffers are not supported on this platform; "
                    "skipping the test";
  }
  const std::vector<float> data = {1.0, 2.0, 3.0, -4.0, 5.1};
  litert::cl::Buffer buffer;
  litert::cl::ClContext context;
  litert::cl::ClDevice device;
  litert::cl::ClCommandQueue queue;
  ASSERT_TRUE(CreateDefaultGPUDevice(&device).ok());
  ASSERT_TRUE(CreateClContext(device, &context).ok());
  ASSERT_TRUE(CreateClCommandQueue(device, context, &queue).ok());
  ASSERT_TRUE(CreateReadWriteBuffer(sizeof(float) * 5, &context, &buffer).ok());
  ASSERT_TRUE(
      buffer.WriteData(&queue, absl::MakeConstSpan(data.data(), data.size()))
          .ok());
  std::vector<float> gpu_data;
  ASSERT_TRUE(buffer.ReadData<float>(&queue, &gpu_data).ok());

  EXPECT_THAT(gpu_data, Pointwise(FloatNear(0.0f), data));
}
}  // namespace internal
}  // namespace litert
