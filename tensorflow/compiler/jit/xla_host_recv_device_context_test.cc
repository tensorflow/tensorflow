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
#include "tensorflow/compiler/jit/xla_host_recv_device_context.h"

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/multi_platform_manager.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

class XlaHostRecvDeviceContextTest : public ::testing::Test {
 public:
  void SetDevice(const string& device_type) {
    auto device_factory = DeviceFactory::GetFactory(device_type);
    SessionOptions options;
    std::vector<std::unique_ptr<Device>> devices;
    Status s = device_factory->CreateDevices(
        options, "/job:worker/replica:0/task:0", &devices);
    device_ = std::move(devices[0]);

    AllocatorAttributes host_alloc_attr;
    host_alloc_attr.set_on_host(true);
    host_allocator_ = device_->GetAllocator(host_alloc_attr);

    AllocatorAttributes device_alloc_attr;
    device_alloc_attr.set_on_host(false);
    device_allocator_ = device_->GetAllocator(device_alloc_attr);
  }

 protected:
  std::unique_ptr<Device> device_;
  Allocator* host_allocator_;
  Allocator* device_allocator_;
};

TEST_F(XlaHostRecvDeviceContextTest, CopyDeviceTensorToCPU) {
  SetDevice("GPU");
  Tensor origin_cpu_tensor(host_allocator_, DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&origin_cpu_tensor, {1.2, 2.3, 3.4, 4.5});
  Tensor device_tensor(device_allocator_, DT_FLOAT, TensorShape({2, 2}));
  Tensor dest_cpu_tensor(host_allocator_, DT_FLOAT, TensorShape({2, 2}));

  stream_executor::Platform* platform =
      stream_executor::MultiPlatformManager::PlatformWithName("CUDA").value();
  stream_executor::StreamExecutor* executor =
      platform->ExecutorForDevice(0).value();
  stream_executor::Stream stream(executor);
  stream.Init();
  ASSERT_TRUE(stream.ok());

  se::DeviceMemoryBase gpu_dst{device_tensor.data(), 4 * sizeof(float)};
  xla::Shape shape;
  TF_ASSERT_OK(TensorShapeToXLAShape(DT_FLOAT, TensorShape({2, 2}), &shape));

  // Copy the cpu_tensor to the GPU first before trying to copy it back.
  stream.ThenMemcpy(&gpu_dst, origin_cpu_tensor.data(), gpu_dst.size());
  TF_ASSERT_OK(stream.BlockHostUntilDone());

  tsl::AsyncValueRef<se::Event> done_event =
      tsl::MakeConstructedAsyncValueRef<se::Event>(stream.parent());
  done_event->Init();
  XlaHostRecvDeviceContext* device_context =
      new XlaHostRecvDeviceContext(&stream, gpu_dst, shape, done_event);
  TF_ASSERT_OK(device_context->CopyDeviceTensorToCPUSync(
      &device_tensor, "", device_.get(), &dest_cpu_tensor));

  tensorflow::test::ExpectClose(origin_cpu_tensor, dest_cpu_tensor);
  device_context->Unref();
}

}  // namespace
}  // namespace tensorflow
