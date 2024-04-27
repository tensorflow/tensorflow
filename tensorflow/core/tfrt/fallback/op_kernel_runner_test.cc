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

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner_cache.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

using ::testing::IsNull;
using ::testing::SizeIs;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
constexpr const char* kDeviceType = "GPU";
#else
constexpr const char* kDeviceType = "CPU";
#endif

class TestOpKernel : public OpKernel {
 public:
  using OpKernel::OpKernel;

  ~TestOpKernel() override = default;

  void Compute(OpKernelContext* context) override {
    context->set_output(0, context->input(0));
  }
};

REGISTER_KERNEL_BUILDER(Name("TestOp").Device(DEVICE_CPU), TestOpKernel);

// Identical to BatchFunction except it has 2 extra TFRT attributes and it does
// not have `f` attribute. Users will not invoke this op directly.
REGISTER_OP("TestOp").Input("x: int32").Output("y: int32");

TEST(OpKernelRunnerTest, Create) {
  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state,
                          FallbackState::Create(session_options, fdef_lib));

  TF_ASSERT_OK_AND_ASSIGN(
      auto runner,
      OpKernelRunner::Create(
          /*op_name=*/
          "TestOp", /*node_name=*/"TestOp_node_name",
          /*device_name=*/"/job:localhost/replica:0/task:0/device:CPU:0",
          /*num_args=*/1,
          /*attr_builder=*/
          [](tensorflow::AttrValueMap*) { return absl::OkStatus(); },
          fallback_state->device_manager(),
          fallback_state->process_function_library_runtime()));

  ASSERT_TRUE(runner);

  EXPECT_EQ(runner.op_kernel()->name(), "TestOp_node_name");
}

TEST(OpKernelRunnerTest, OpKernelRunnerCache) {
  tensorflow::SessionOptions session_options;
  tensorflow::FunctionDefLibrary fdef_lib;
  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state,
                          FallbackState::Create(session_options, fdef_lib));

  OpKernelRunnerCache cache;

  tfrt::Location loc(/*handler=*/nullptr, /*data=*/100);

  TF_ASSERT_OK_AND_ASSIGN(
      auto* runner,
      cache.GetOrCreate(
          loc,
          /*op_name=*/"TestOp",
          /*device_name=*/"/job:localhost/replica:0/task:0/device:CPU:0",
          /*num_args=*/1,
          /*attr_builder=*/
          [](tensorflow::AttrValueMap*) { return absl::OkStatus(); },
          fallback_state->device_manager(),
          fallback_state->process_function_library_runtime()));

  ASSERT_TRUE(runner);

  EXPECT_EQ(runner->op_kernel()->name(), "TestOp_100_0");

  TF_ASSERT_OK_AND_ASSIGN(
      runner,
      cache.GetOrCreate(
          loc,
          /*op_name=*/"TestOp",
          /*device_name=*/"/job:localhost/replica:0/task:0/device:CPU:0",
          /*num_args=*/1,
          /*attr_builder=*/
          [](tensorflow::AttrValueMap*) { return absl::OkStatus(); },
          fallback_state->device_manager(),
          fallback_state->process_function_library_runtime()));

  ASSERT_TRUE(runner);

  EXPECT_EQ(runner->op_kernel()->name(), "TestOp_100_0");
}

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
  absl::InlinedVector<TensorValue, 4UL> inputs{TensorValue(&a),
                                               TensorValue(&b)};
  params.inputs = inputs;

  Tensor c(DT_UINT8, TensorShape({}));
  absl::InlinedVector<TensorValue, 4UL> new_inputs{TensorValue(&c)};

  OpKernelRunState run_state(new_inputs, params);

  EXPECT_THAT(run_state.input_tf_tensors, SizeIs(1));
  EXPECT_THAT(run_state.input_tf_tensor_values, SizeIs(1));
  EXPECT_EQ(run_state.params.inputs.data(),
            run_state.input_tf_tensor_values.data());
  EXPECT_THAT(run_state.params.eigen_gpu_device, IsNull());
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
