/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"

namespace tensorflow {
namespace {

static bool Initialized = [] {
  auto& rollout_config = GetXlaOpsCommonFlags()->tf_xla_use_device_api;
  rollout_config.enabled_for_xla_launch_ = true;
  rollout_config.enabled_for_compile_on_demand_ = true;

  tensorflow::GetXlaDeviceFlags()->tf_xla_enable_xla_devices = true;
  return true;
}();

class DeviceContextTest : public ::testing::Test {
 public:
  void SetDevice(const string& device_type) {
    auto& rollout_config = GetXlaOpsCommonFlags()->tf_xla_use_device_api;
    rollout_config.AllowForDeviceInXlaLaunch(DeviceType(device_type));
    rollout_config.AllowForDeviceInXlaCompileOnDemand(DeviceType(device_type));

    auto device_factory = DeviceFactory::GetFactory(device_type);
    SessionOptions options;
    std::vector<std::unique_ptr<Device>> devices;
    absl::Status s = device_factory->CreateDevices(
        options, "/job:worker/replica:0/task:0", &devices);
    device_ = std::move(devices[0]);

    tensorflow::AllocatorAttributes host_alloc_attr;
    host_alloc_attr.set_on_host(true);
    host_allocator_ = device_->GetAllocator(host_alloc_attr);

    tensorflow::AllocatorAttributes device_alloc_attr;
    device_alloc_attr.set_on_host(false);
    device_allocator_ = device_->GetAllocator(device_alloc_attr);

    tensorflow::DeviceContext* device_context;
    auto status = device_->TryGetDeviceContext(&device_context);
    TF_EXPECT_OK(status);
    device_context_.reset(device_context);
  }

  std::unique_ptr<Device> device_;
  tensorflow::core::RefCountPtr<DeviceContext> device_context_;
  tensorflow::Allocator* host_allocator_;
  tensorflow::Allocator* device_allocator_;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TEST_F(DeviceContextTest, TestXlaGpuRoundTripTransferWithDeviceApi) {
  SetDevice(DEVICE_XLA_GPU);
  tensorflow::Tensor origin_cpu_tensor(host_allocator_, tensorflow::DT_FLOAT,
                                       tensorflow::TensorShape({2, 2}));
  tensorflow::test::FillValues<float>(&origin_cpu_tensor, {1.2, 2.3, 3.4, 4.5});
  tensorflow::Tensor device_tensor(device_allocator_, tensorflow::DT_FLOAT,
                                   tensorflow::TensorShape({2, 2}));
  tensorflow::Tensor dest_cpu_tensor(host_allocator_, tensorflow::DT_FLOAT,
                                     tensorflow::TensorShape({2, 2}));

  TF_ASSERT_OK(device_context_->CopyCPUTensorToDeviceSync(
      &origin_cpu_tensor, device_.get(), &device_tensor));
  TF_ASSERT_OK(device_context_->CopyDeviceTensorToCPUSync(
      &device_tensor, "", device_.get(), &dest_cpu_tensor));
  LOG(INFO) << "H2D - D2H roundtrip completes. tensor: "
            << dest_cpu_tensor.DebugString(4);

  tensorflow::test::ExpectClose(origin_cpu_tensor, dest_cpu_tensor);
}
#endif

}  // namespace
}  // namespace tensorflow
