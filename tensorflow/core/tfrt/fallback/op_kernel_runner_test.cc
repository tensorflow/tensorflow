/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"

#include <string>

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

using ::testing::IsNull;
using ::testing::SizeIs;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
constexpr const char* kDeviceType = "GPU";
#else
constexpr const char* kDeviceType = "CPU";
#endif

TEST(OpKernelRunnerTest, OpKernelRunState) {
  SessionOptions options;
  auto* device_count = options.config.mutable_device_count();
  device_count->insert({kDeviceType, 1});
  std::vector<std::unique_ptr<Device>> devices;
  TF_ASSERT_OK(DeviceFactory::GetFactory(kDeviceType)
                   ->CreateDevices(options,
                                   /*name_prefix=*/"/job:a/replica:0/task:0",
                                   &devices));
  ASSERT_EQ(devices.size(), 1);

  OpKernelContext::Params params;
  params.device = devices[0].get();
  params.ensure_eigen_gpu_device();
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  ASSERT_THAT(params.eigen_gpu_device, ::testing::NotNull());
#endif

  Tensor a(DT_FLOAT, TensorShape({}));
  Tensor b(DT_INT32, TensorShape({}));
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&a), TensorValue(&b)};
  params.inputs = inputs;

  Tensor c(DT_UINT8, TensorShape({}));
  gtl::InlinedVector<TensorValue, 4> new_inputs{TensorValue(&c)};

  tfrt_stub::OpKernelRunState run_state(new_inputs, params);

  EXPECT_THAT(run_state.input_tf_tensors, SizeIs(1));
  EXPECT_THAT(run_state.input_tf_tensor_values, SizeIs(1));
  EXPECT_EQ(run_state.params.inputs.data(),
            run_state.input_tf_tensor_values.data());
  EXPECT_THAT(run_state.params.eigen_gpu_device, IsNull());
}

}  // namespace
}  // namespace tensorflow
