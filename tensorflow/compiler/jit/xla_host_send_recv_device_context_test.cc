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
#include "tensorflow/compiler/jit/xla_host_recv_device_context.h"
#include "tensorflow/compiler/jit/xla_host_send_device_context.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"

namespace tensorflow {
namespace {

class XlaHostSendRecvDeviceContextTest : public ::testing::Test {
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

TEST_F(XlaHostSendRecvDeviceContextTest, CopyDeviceTensorToCPU) {
  SetDevice("GPU");
  Tensor origin_cpu_tensor(host_allocator_, DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&origin_cpu_tensor, {1.2, 2.3, 3.4, 4.5});
  Tensor device_tensor(device_allocator_, DT_FLOAT, TensorShape({2, 2}));
  Tensor dest_cpu_tensor(host_allocator_, DT_FLOAT, TensorShape({2, 2}));

  stream_executor::Platform* platform =
      stream_executor::PlatformManager::PlatformWithName("CUDA").value();
  stream_executor::StreamExecutor* executor =
      platform->ExecutorForDevice(0).value();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceMemoryBase gpu_dst{device_tensor.data(), 4 * sizeof(float)};
  xla::Shape shape;
  TF_ASSERT_OK(TensorShapeToXLAShape(DT_FLOAT, TensorShape({2, 2}), &shape));

  // Copy the cpu_tensor to the GPU first before trying to copy it back.
  TF_ASSERT_OK(
      stream->Memcpy(&gpu_dst, origin_cpu_tensor.data(), gpu_dst.size()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  TF_ASSERT_OK_AND_ASSIGN(auto se_event, executor->CreateEvent());
  tsl::AsyncValueRef<std::unique_ptr<se::Event>> done_event =
      tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(
          std::move(se_event));
  XlaHostRecvDeviceContext* device_context =
      new XlaHostRecvDeviceContext(stream.get(), gpu_dst, shape, done_event);
  TF_ASSERT_OK(device_context->CopyDeviceTensorToCPUSync(
      &device_tensor, "", device_.get(), &dest_cpu_tensor));

  tensorflow::test::ExpectClose(origin_cpu_tensor, dest_cpu_tensor);
  device_context->Unref();
}

TEST_F(XlaHostSendRecvDeviceContextTest, CopyCPUTensorToDevice) {
  SetDevice("GPU");
  Tensor origin_cpu_tensor(host_allocator_, DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&origin_cpu_tensor, {1.2, 2.3, 3.4, 4.5});
  Tensor device_tensor(device_allocator_, DT_FLOAT, TensorShape({2, 2}));
  Tensor dest_cpu_tensor(host_allocator_, DT_FLOAT, TensorShape({2, 2}));

  stream_executor::Platform* platform =
      stream_executor::PlatformManager::PlatformWithName("CUDA").value();
  stream_executor::StreamExecutor* executor =
      platform->ExecutorForDevice(0).value();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceMemoryBase gpu_dst{device_tensor.data(), 4 * sizeof(float)};
  xla::Shape shape;
  TF_ASSERT_OK(TensorShapeToXLAShape(DT_FLOAT, TensorShape({2, 2}), &shape));

  TF_ASSERT_OK_AND_ASSIGN(auto se_event, executor->CreateEvent());
  tsl::AsyncValueRef<std::unique_ptr<se::Event>> done_event =
      tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(
          std::move(se_event));
  XlaHostSendDeviceContext* device_context =
      new XlaHostSendDeviceContext(stream.get(), &gpu_dst, shape, done_event);
  TF_ASSERT_OK(device_context->CopyCPUTensorToDeviceSync(
      &origin_cpu_tensor, device_.get(), &device_tensor));

  // Copy the GPU tensor back to CPU to check that copy worked.
  TF_ASSERT_OK(stream->Memcpy(dest_cpu_tensor.data(), gpu_dst, gpu_dst.size()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  tensorflow::test::ExpectClose(origin_cpu_tensor, dest_cpu_tensor);
  device_context->Unref();
}

TEST_F(XlaHostSendRecvDeviceContextTest, RoundTrip) {
  SetDevice("GPU");
  Tensor origin_cpu_tensor(host_allocator_, DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&origin_cpu_tensor, {1.2, 2.3, 3.4, 4.5});
  Tensor device_tensor(device_allocator_, DT_FLOAT, TensorShape({2, 2}));
  Tensor dest_cpu_tensor(host_allocator_, DT_FLOAT, TensorShape({2, 2}));

  stream_executor::Platform* platform =
      stream_executor::PlatformManager::PlatformWithName("CUDA").value();
  stream_executor::StreamExecutor* executor =
      platform->ExecutorForDevice(0).value();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceMemoryBase gpu_dst{device_tensor.data(), 4 * sizeof(float)};
  xla::Shape shape;
  TF_ASSERT_OK(TensorShapeToXLAShape(DT_FLOAT, TensorShape({2, 2}), &shape));

  TF_ASSERT_OK_AND_ASSIGN(auto se_event, executor->CreateEvent());
  tsl::AsyncValueRef<std::unique_ptr<se::Event>> send_done_event =
      tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(
          std::move(se_event));
  XlaHostSendDeviceContext* send_device_context = new XlaHostSendDeviceContext(
      stream.get(), &gpu_dst, shape, send_done_event);
  TF_ASSERT_OK(send_device_context->CopyCPUTensorToDeviceSync(
      &origin_cpu_tensor, device_.get(), &device_tensor));

  TF_ASSERT_OK_AND_ASSIGN(auto recv_se_event, executor->CreateEvent());
  tsl::AsyncValueRef<std::unique_ptr<se::Event>> recv_done_event =
      tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(
          std::move(recv_se_event));
  XlaHostRecvDeviceContext* recv_device_context = new XlaHostRecvDeviceContext(
      stream.get(), gpu_dst, shape, recv_done_event);
  TF_ASSERT_OK(recv_device_context->CopyDeviceTensorToCPUSync(
      &device_tensor, "", device_.get(), &dest_cpu_tensor));

  tensorflow::test::ExpectClose(origin_cpu_tensor, dest_cpu_tensor);
  send_device_context->Unref();
  recv_device_context->Unref();
}

}  // namespace
}  // namespace tensorflow
