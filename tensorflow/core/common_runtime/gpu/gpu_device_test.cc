/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA

#include "tensorflow/core/common_runtime/gpu/gpu_device.h"

#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/gpu/process_state.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {
const char* kDeviceNamePrefix = "/job:localhost/replica:0/task:0";

int64 GetTotalGPUMemory(CudaGpuId gpu_id) {
  se::StreamExecutor* se =
      GpuIdUtil::ExecutorForCudaGpuId(GPUMachineManager(), gpu_id).ValueOrDie();

  int64 total_memory, available_memory;
  CHECK(se->DeviceMemoryUsage(&available_memory, &total_memory));
  return total_memory;
}

Status GetComputeCapability(CudaGpuId gpu_id, int* cc_major, int* cc_minor) {
  se::StreamExecutor* se =
      GpuIdUtil::ExecutorForCudaGpuId(GPUMachineManager(), gpu_id).ValueOrDie();
  if (!se->GetDeviceDescription().cuda_compute_capability(cc_major, cc_minor)) {
    *cc_major = 0;
    *cc_minor = 0;
    return errors::Internal("Failed to get compute capability for device.");
  }
  return Status::OK();
}

void ExpectErrorMessageSubstr(const Status& s, StringPiece substr) {
  EXPECT_TRUE(str_util::StrContains(s.ToString(), substr))
      << s << ", expected substring " << substr;
}
}  // namespace

class GPUDeviceTest : public ::testing::Test {
 public:
  void TearDown() override { ProcessState::singleton()->TestOnlyReset(); }

 protected:
  static SessionOptions MakeSessionOptions(
      const string& visible_device_list = "",
      double per_process_gpu_memory_fraction = 0, int gpu_device_count = 1,
      const std::vector<std::vector<float>>& memory_limit_mb = {}) {
    SessionOptions options;
    ConfigProto* config = &options.config;
    (*config->mutable_device_count())["GPU"] = gpu_device_count;
    GPUOptions* gpu_options = config->mutable_gpu_options();
    gpu_options->set_visible_device_list(visible_device_list);
    gpu_options->set_per_process_gpu_memory_fraction(
        per_process_gpu_memory_fraction);
    for (const auto& v : memory_limit_mb) {
      auto virtual_devices =
          gpu_options->mutable_experimental()->add_virtual_devices();
      for (float mb : v) {
        virtual_devices->add_memory_limit_mb(mb);
      }
    }
    return options;
  }
};

TEST_F(GPUDeviceTest, FailedToParseVisibleDeviceList) {
  SessionOptions opts = MakeSessionOptions("0,abc");
  std::vector<tensorflow::Device*> devices;
  Status status = DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  ExpectErrorMessageSubstr(status, "Could not parse entry");
}

TEST_F(GPUDeviceTest, InvalidGpuId) {
  SessionOptions opts = MakeSessionOptions("100");
  std::vector<tensorflow::Device*> devices;
  Status status = DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  ExpectErrorMessageSubstr(status,
                           "'visible_device_list' listed an invalid GPU id");
}

TEST_F(GPUDeviceTest, DuplicateEntryInVisibleDeviceList) {
  SessionOptions opts = MakeSessionOptions("0,0");
  std::vector<tensorflow::Device*> devices;
  Status status = DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  ExpectErrorMessageSubstr(status,
                           "visible_device_list contained a duplicate entry");
}

TEST_F(GPUDeviceTest, VirtualDeviceConfigConflictsWithMemoryFractionSettings) {
  SessionOptions opts = MakeSessionOptions("0", 0.1, 1, {{}});
  std::vector<tensorflow::Device*> devices;
  Status status = DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  ExpectErrorMessageSubstr(
      status, "It's invalid to set per_process_gpu_memory_fraction");
}

TEST_F(GPUDeviceTest, GpuDeviceCountTooSmall) {
  // device_count is 0, but with one entry in visible_device_list and one
  // (empty) VirtualDevices messages.
  SessionOptions opts = MakeSessionOptions("0", 0, 0, {{}});
  std::vector<tensorflow::Device*> devices;
  Status status = DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices);
  EXPECT_EQ(status.code(), error::UNKNOWN);
  ExpectErrorMessageSubstr(status,
                           "Not enough GPUs to create virtual devices.");
}

TEST_F(GPUDeviceTest, NotEnoughGpuInVisibleDeviceList) {
  // Single entry in visible_device_list with two (empty) VirtualDevices
  // messages.
  SessionOptions opts = MakeSessionOptions("0", 0, 8, {{}, {}});
  std::vector<tensorflow::Device*> devices;
  Status status = DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices);
  EXPECT_EQ(status.code(), error::UNKNOWN);
  ExpectErrorMessageSubstr(status,
                           "Not enough GPUs to create virtual devices.");
}

TEST_F(GPUDeviceTest, VirtualDeviceConfigConflictsWithVisibleDeviceList) {
  // This test requires at least two visible GPU hardware.
  if (GPUMachineManager()->VisibleDeviceCount() < 2) return;
  // Three entries in visible_device_list with two (empty) VirtualDevices
  // messages.
  SessionOptions opts = MakeSessionOptions("0,1", 0, 8, {{}});
  std::vector<tensorflow::Device*> devices;
  Status status = DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  ExpectErrorMessageSubstr(
      status,
      "The number of GPUs in visible_device_list doesn't "
      "match the number of elements in the virtual_devices "
      "list.");
}

TEST_F(GPUDeviceTest, EmptyVirtualDeviceConfig) {
  // It'll create single virtual device when the virtual device config is empty.
  SessionOptions opts = MakeSessionOptions("0");
  std::vector<tensorflow::Device*> devices;
  TF_CHECK_OK(DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices));
  EXPECT_EQ(1, devices.size());
  EXPECT_GE(devices[0]->attributes().memory_limit(), 0);
  gtl::STLDeleteElements(&devices);
}

TEST_F(GPUDeviceTest, SingleVirtualDeviceWithNoMemoryLimit) {
  // It'll create single virtual device for the gpu in question when
  // memory_limit_mb is unset.
  SessionOptions opts = MakeSessionOptions("0", 0, 1, {{}});
  std::vector<tensorflow::Device*> devices;
  TF_CHECK_OK(DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices));
  EXPECT_EQ(1, devices.size());
  EXPECT_GE(devices[0]->attributes().memory_limit(), 0);
  gtl::STLDeleteElements(&devices);
}

TEST_F(GPUDeviceTest, SingleVirtualDeviceWithMemoryLimit) {
  SessionOptions opts = MakeSessionOptions("0", 0, 1, {{123}});
  std::vector<tensorflow::Device*> devices;
  TF_CHECK_OK(DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices));
  EXPECT_EQ(1, devices.size());
  EXPECT_EQ(123 << 20, devices[0]->attributes().memory_limit());
  gtl::STLDeleteElements(&devices);
}

TEST_F(GPUDeviceTest, MultipleVirtualDevices) {
  SessionOptions opts = MakeSessionOptions("0", 0, 1, {{123, 456}});
  std::vector<tensorflow::Device*> devices;
  TF_CHECK_OK(DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices));
  EXPECT_EQ(2, devices.size());
  EXPECT_EQ(123 << 20, devices[0]->attributes().memory_limit());
  EXPECT_EQ(456 << 20, devices[1]->attributes().memory_limit());
  ASSERT_EQ(1, devices[0]->attributes().locality().links().link_size());
  ASSERT_EQ(1, devices[1]->attributes().locality().links().link_size());
  EXPECT_EQ(1, devices[0]->attributes().locality().links().link(0).device_id());
  EXPECT_EQ("SAME_DEVICE",
            devices[0]->attributes().locality().links().link(0).type());
  EXPECT_EQ(BaseGPUDeviceFactory::InterconnectMap::kSameDeviceStrength,
            devices[0]->attributes().locality().links().link(0).strength());
  EXPECT_EQ(0, devices[1]->attributes().locality().links().link(0).device_id());
  EXPECT_EQ("SAME_DEVICE",
            devices[1]->attributes().locality().links().link(0).type());
  EXPECT_EQ(BaseGPUDeviceFactory::InterconnectMap::kSameDeviceStrength,
            devices[1]->attributes().locality().links().link(0).strength());
  gtl::STLDeleteElements(&devices);
}

// Enabling unified memory on pre-Pascal GPUs results in an initialization
// error.
TEST_F(GPUDeviceTest, UnifiedMemoryUnavailableOnPrePascalGpus) {
  int cc_major, cc_minor;
  TF_ASSERT_OK(GetComputeCapability(CudaGpuId(0), &cc_major, &cc_minor));
  // Exit early while running on Pascal or later GPUs.
  if (cc_major >= 6) {
    return;
  }

  SessionOptions opts = MakeSessionOptions("0", /*memory_fraction=*/1.2);
  opts.config.mutable_gpu_options()
      ->mutable_experimental()
      ->set_use_unified_memory(true);
  std::vector<tensorflow::Device*> devices;
  Status status = DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices);
  EXPECT_EQ(status.code(), error::INTERNAL);
  ExpectErrorMessageSubstr(status, "does not support oversubscription.");
}

// Enabling unified memory on Pascal or later GPUs makes it possible to allocate
// more memory than what is available on the device.
TEST_F(GPUDeviceTest, UnifiedMemoryAllocation) {
  static constexpr double kGpuMemoryFraction = 1.2;
  static constexpr CudaGpuId kCudaGpuId(0);

  int cc_major, cc_minor;
  TF_ASSERT_OK(GetComputeCapability(kCudaGpuId, &cc_major, &cc_minor));
  // Exit early if running on pre-Pascal GPUs.
  if (cc_major < 6) {
    LOG(INFO)
        << "Unified memory allocation is not supported with pre-Pascal GPUs.";
    return;
  }

  SessionOptions opts = MakeSessionOptions("0", kGpuMemoryFraction);
  std::vector<tensorflow::Device*> devices;
  TF_ASSERT_OK(DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices));
  ASSERT_EQ(1, devices.size());

  int64 memory_limit = devices[0]->attributes().memory_limit();
  ASSERT_EQ(memory_limit, static_cast<int64>(GetTotalGPUMemory(kCudaGpuId) *
                                             kGpuMemoryFraction));

  AllocatorAttributes allocator_attributes = AllocatorAttributes();
  allocator_attributes.set_gpu_compatible(true);
  Allocator* allocator = devices[0]->GetAllocator(allocator_attributes);

  // Try to allocate all the available memory after rounding down to the nearest
  // multiple of MB.
  void* ptr = allocator->AllocateRaw(Allocator::kAllocatorAlignment,
                                     (memory_limit >> 20) << 20);
  EXPECT_NE(ptr, nullptr);
  allocator->DeallocateRaw(ptr);

  gtl::STLDeleteElements(&devices);
}

}  // namespace tensorflow

#endif
