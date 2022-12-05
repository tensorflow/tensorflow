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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/common_runtime/gpu/gpu_device.h"

#include "tensorflow/compiler/xla/stream_executor/device_id_utils.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_cudamallocasync_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/tsl/framework/device_id.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

using ::testing::SizeIs;

const char* kDeviceNamePrefix = "/job:localhost/replica:0/task:0";

int64_t GetTotalGPUMemory(tsl::PlatformDeviceId gpu_id) {
  se::StreamExecutor* se = se::DeviceIdUtil::ExecutorForPlatformDeviceId(
                               se::GPUMachineManager(), gpu_id)
                               .value();

  int64_t total_memory, available_memory;
  CHECK(se->DeviceMemoryUsage(&available_memory, &total_memory));
  return total_memory;
}

se::CudaComputeCapability GetComputeCapability() {
  return se::DeviceIdUtil::ExecutorForPlatformDeviceId(se::GPUMachineManager(),
                                                       tsl::PlatformDeviceId(0))
      .value()
      ->GetDeviceDescription()
      .cuda_compute_capability();
}

void ExpectErrorMessageSubstr(const Status& s, StringPiece substr) {
  EXPECT_TRUE(absl::StrContains(s.ToString(), substr))
      << s << ", expected substring " << substr;
}

}  // namespace

class GPUDeviceTest : public ::testing::Test {
 public:
  void TearDown() override {
    BaseGPUDevice::TestOnlyReset();
    GPUProcessState::singleton()->TestOnlyReset();
  }

 protected:
  static SessionOptions MakeSessionOptions(
      const string& visible_device_list = "",
      double per_process_gpu_memory_fraction = 0, int gpu_device_count = 1,
      const std::vector<std::vector<float>>& memory_limit_mb = {},
      const std::vector<std::vector<int32>>& priority = {},
      const std::vector<std::vector<int32>>& device_ordinal = {},
      const bool use_cuda_malloc_async = false) {
    SessionOptions options;
    ConfigProto* config = &options.config;
    (*config->mutable_device_count())["GPU"] = gpu_device_count;
    GPUOptions* gpu_options = config->mutable_gpu_options();
    gpu_options->set_visible_device_list(visible_device_list);
    gpu_options->set_per_process_gpu_memory_fraction(
        per_process_gpu_memory_fraction);
    gpu_options->mutable_experimental()->set_use_cuda_malloc_async(
        use_cuda_malloc_async);
    for (int i = 0; i < memory_limit_mb.size(); ++i) {
      auto virtual_devices =
          gpu_options->mutable_experimental()->add_virtual_devices();
      for (float mb : memory_limit_mb[i]) {
        virtual_devices->add_memory_limit_mb(mb);
      }
      if (i < device_ordinal.size()) {
        for (int o : device_ordinal[i]) {
          virtual_devices->add_device_ordinal(o);
        }
      }
      if (i < priority.size()) {
        for (int p : priority[i]) {
          virtual_devices->add_priority(p);
        }
      }
    }
    return options;
  }

  void InitCPUTensor(Tensor* cpu_tensor, int num_elements, float value) {
    auto tensor = cpu_tensor->tensor<float, 1>();
    for (int i = 0; i < num_elements; ++i) {
      tensor(i) = value;
    }
  }

  void CopyCPUToGPU(Tensor* cpu_tensor, Tensor* gpu_tensor, Device* device,
                    DeviceContext* device_context) {
    TF_ASSERT_OK(device_context->CopyCPUTensorToDeviceSync(cpu_tensor, device,
                                                           gpu_tensor));
  }

  void CopyGPUToCPU(Tensor* gpu_tensor, Tensor* cpu_tensor, Device* device,
                    DeviceContext* device_context) {
    TF_ASSERT_OK(device_context->CopyDeviceTensorToCPUSync(
        gpu_tensor, /*tensor_name=*/"", device, cpu_tensor));
  }
};

TEST_F(GPUDeviceTest, DISABLED_ON_GPU_ROCM(CudaMallocAsync)) {
  // cudaMallocAsync supported only when cuda toolkit and driver supporting
  // CUDA 11.2+
#ifndef GOOGLE_CUDA
  return;
#elif CUDA_VERSION < 11020
  LOG(INFO) << "CUDA toolkit too old, skipping this test: " << CUDA_VERSION;
  return;
#else
  // cudaMallocAsync supported only for driver supporting CUDA 11.2+
  int driverVersion;
  cuDriverGetVersion(&driverVersion);
  if (driverVersion < 11020) {
    LOG(INFO) << "Driver version too old, skipping this test: "
              << driverVersion;
    return;
  }
#endif

  SessionOptions opts = MakeSessionOptions("0", 0, 1, {}, {}, {},
                                           /*use_cuda_malloc_async=*/true);
  std::vector<std::unique_ptr<Device>> devices;
  Status status;
  int number_instantiated =
      se::GpuCudaMallocAsyncAllocator::GetInstantiatedCountTestOnly();
  {  // The new scope is to trigger the destruction of the object.
    status = DeviceFactory::GetFactory("GPU")->CreateDevices(
        opts, kDeviceNamePrefix, &devices);
    EXPECT_THAT(devices, SizeIs(1));
    Device* device = devices[0].get();
    auto* device_info = device->tensorflow_accelerator_device_info();
    EXPECT_NE(device_info, nullptr);

    AllocatorAttributes allocator_attributes = AllocatorAttributes();
    allocator_attributes.set_gpu_compatible(true);
    Allocator* allocator = devices[0]->GetAllocator(allocator_attributes);
    void* ptr = allocator->AllocateRaw(Allocator::kAllocatorAlignment, 1024);
    EXPECT_NE(ptr, nullptr);
    allocator->DeallocateRaw(ptr);
  }
  EXPECT_EQ(number_instantiated + 1,
            se::GpuCudaMallocAsyncAllocator::GetInstantiatedCountTestOnly());
  EXPECT_EQ(status.code(), error::OK);
}

TEST_F(GPUDeviceTest, DISABLED_ON_GPU_ROCM(CudaMallocAsyncPreallocate)) {
  SessionOptions opts = MakeSessionOptions("0", 0, 1, {}, {}, {},
                                           /*use_cuda_malloc_async=*/true);
  setenv("TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC", "2048", 1);
  std::vector<std::unique_ptr<Device>> devices;
  Status status;

  int number_instantiated =
      se::GpuCudaMallocAsyncAllocator::GetInstantiatedCountTestOnly();
  {  // The new scope is to trigger the destruction of the object.
    status = DeviceFactory::GetFactory("GPU")->CreateDevices(
        opts, kDeviceNamePrefix, &devices);
    EXPECT_THAT(devices, SizeIs(1));
    Device* device = devices[0].get();
    auto* device_info = device->tensorflow_accelerator_device_info();
    CHECK(device_info);

    AllocatorAttributes allocator_attributes = AllocatorAttributes();
    allocator_attributes.set_gpu_compatible(true);
    Allocator* allocator = devices[0]->GetAllocator(allocator_attributes);
    void* ptr = allocator->AllocateRaw(Allocator::kAllocatorAlignment, 1024);
    EXPECT_NE(ptr, nullptr);
    allocator->DeallocateRaw(ptr);
  }

  unsetenv("TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC");

  EXPECT_EQ(number_instantiated + 1,
            se::GpuCudaMallocAsyncAllocator::GetInstantiatedCountTestOnly());

  EXPECT_EQ(status.code(), error::OK);
}

TEST_F(GPUDeviceTest, FailedToParseVisibleDeviceList) {
  SessionOptions opts = MakeSessionOptions("0,abc");
  std::vector<std::unique_ptr<Device>> devices;
  Status status = DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  ExpectErrorMessageSubstr(status, "Could not parse entry");
}

TEST_F(GPUDeviceTest, InvalidGpuId) {
  SessionOptions opts = MakeSessionOptions("100");
  std::vector<std::unique_ptr<Device>> devices;
  Status status = DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  ExpectErrorMessageSubstr(status,
                           "'visible_device_list' listed an invalid Device id");
}

TEST_F(GPUDeviceTest, DuplicateEntryInVisibleDeviceList) {
  SessionOptions opts = MakeSessionOptions("0,0");
  std::vector<std::unique_ptr<Device>> devices;
  Status status = DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  ExpectErrorMessageSubstr(status,
                           "visible_device_list contained a duplicate entry");
}

TEST_F(GPUDeviceTest, VirtualDeviceConfigConflictsWithMemoryFractionSettings) {
  SessionOptions opts = MakeSessionOptions("0", 0.1, 1, {{}});
  std::vector<std::unique_ptr<Device>> devices;
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
  std::vector<std::unique_ptr<Device>> devices;
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
  std::vector<std::unique_ptr<Device>> devices;
  Status status = DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices);
  EXPECT_EQ(status.code(), error::UNKNOWN);
  ExpectErrorMessageSubstr(status,
                           "Not enough GPUs to create virtual devices.");
}

TEST_F(GPUDeviceTest, VirtualDeviceConfigConflictsWithVisibleDeviceList) {
  // This test requires at least two visible GPU hardware.
  if (se::GPUMachineManager()->VisibleDeviceCount() < 2) return;
  // Three entries in visible_device_list with two (empty) VirtualDevices
  // messages.
  SessionOptions opts = MakeSessionOptions("0,1", 0, 8, {{}});
  std::vector<std::unique_ptr<Device>> devices;
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
  std::vector<std::unique_ptr<Device>> devices;
  TF_CHECK_OK(DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices));
  EXPECT_THAT(devices, SizeIs(1));
  EXPECT_GE(devices[0]->attributes().memory_limit(), 0);
  EXPECT_EQ(static_cast<BaseGPUDevice*>(devices[0].get())->priority(), 0);
}

TEST_F(GPUDeviceTest, SingleVirtualDeviceWithNoMemoryLimit) {
  // It'll create single virtual device for the gpu in question when
  // memory_limit_mb is unset.
  SessionOptions opts = MakeSessionOptions("0", 0, 1, {{}});
  std::vector<std::unique_ptr<Device>> devices;
  TF_CHECK_OK(DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices));
  EXPECT_THAT(devices, SizeIs(1));
  EXPECT_GE(devices[0]->attributes().memory_limit(), 0);
  EXPECT_EQ(static_cast<BaseGPUDevice*>(devices[0].get())->priority(), 0);
}

TEST_F(GPUDeviceTest, SingleVirtualDeviceWithMemoryLimitAndNoPriority) {
  SessionOptions opts = MakeSessionOptions("0", 0, 1, {{123}});
  std::vector<std::unique_ptr<Device>> devices;
  TF_CHECK_OK(DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices));
  EXPECT_THAT(devices, SizeIs(1));
  EXPECT_EQ(devices[0]->attributes().memory_limit(), 123 << 20);
  EXPECT_EQ(static_cast<BaseGPUDevice*>(devices[0].get())->priority(), 0);
}

TEST_F(GPUDeviceTest, SingleVirtualDeviceWithInvalidPriority) {
  {
#if TENSORFLOW_USE_ROCM
    // Priority outside the range (-1, 1) for AMD GPUs
    SessionOptions opts =
        MakeSessionOptions("0", 0, 1, {{123, 456}}, {{-2, 1}});
#else
    // Priority outside the range (-2, 0) for NVidia GPUs
    SessionOptions opts =
        MakeSessionOptions("0", 0, 1, {{123, 456}}, {{-9999, 0}});
#endif
    std::vector<std::unique_ptr<Device>> devices;
    Status status = DeviceFactory::GetFactory("GPU")->CreateDevices(
        opts, kDeviceNamePrefix, &devices);
    EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
#if TENSORFLOW_USE_ROCM
    ExpectErrorMessageSubstr(
        status,
        "Priority -2 is outside the range of supported priorities [-1,1] for"
        " virtual device 0 on GPU# 0");
#else
    ExpectErrorMessageSubstr(
        status, "Priority -9999 is outside the range of supported priorities");
#endif
  }
  {
#if TENSORFLOW_USE_ROCM
    // Priority outside the range (-1, 1) for AMD GPUs
    SessionOptions opts =
        MakeSessionOptions("0", 0, 1, {{123, 456}}, {{-1, 2}});
#else
    // Priority outside the range (-2, 0) for NVidia GPUs
    SessionOptions opts = MakeSessionOptions("0", 0, 1, {{123, 456}}, {{0, 1}});
#endif
    std::vector<std::unique_ptr<Device>> devices;
    Status status = DeviceFactory::GetFactory("GPU")->CreateDevices(
        opts, kDeviceNamePrefix, &devices);
    EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
#if TENSORFLOW_USE_ROCM
    ExpectErrorMessageSubstr(
        status,
        "Priority 2 is outside the range of supported priorities [-1,1] for"
        " virtual device 0 on GPU# 0");
#else
    ExpectErrorMessageSubstr(
        status, "Priority 1 is outside the range of supported priorities");
#endif
  }
}

TEST_F(GPUDeviceTest, SingleVirtualDeviceWithMemoryLimitAndPriority) {
  // 0 is a valid priority value for both AMD and NVidia GPUs
  SessionOptions opts = MakeSessionOptions("0", 0, 1, {{123}}, {{0}});
  std::vector<std::unique_ptr<Device>> devices;
  TF_CHECK_OK(DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices));
  EXPECT_THAT(devices, SizeIs(1));
  EXPECT_EQ(devices[0]->attributes().memory_limit(), 123 << 20);
  EXPECT_EQ(static_cast<BaseGPUDevice*>(devices[0].get())->priority(), 0);
}

TEST_F(GPUDeviceTest, MultipleVirtualDevices) {
  // Valid range for priority values on AMD GPUs in (-1,1)
  // Valid range for priority values on NVidia GPUs in (-2, 0)
  SessionOptions opts = MakeSessionOptions("0", 0, 1, {{123, 456}}, {{0, -1}});
  std::vector<std::unique_ptr<Device>> devices;
  TF_CHECK_OK(DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices));
  EXPECT_THAT(devices, SizeIs(2));
  EXPECT_EQ(devices[0]->attributes().memory_limit(), 123 << 20);
  EXPECT_EQ(devices[1]->attributes().memory_limit(), 456 << 20);
  EXPECT_EQ(static_cast<BaseGPUDevice*>(devices[0].get())->priority(), 0);
  EXPECT_EQ(-1, static_cast<BaseGPUDevice*>(devices[1].get())->priority());
  ASSERT_EQ(devices[0]->attributes().locality().links().link_size(), 1);
  ASSERT_EQ(devices[1]->attributes().locality().links().link_size(), 1);
  EXPECT_EQ(devices[0]->attributes().locality().links().link(0).device_id(), 1);
  EXPECT_EQ(devices[0]->attributes().locality().links().link(0).type(),
            "SAME_DEVICE");
  EXPECT_EQ(BaseGPUDeviceFactory::InterconnectMap::kSameDeviceStrength,
            devices[0]->attributes().locality().links().link(0).strength());
  EXPECT_EQ(devices[1]->attributes().locality().links().link(0).device_id(), 0);
  EXPECT_EQ(devices[1]->attributes().locality().links().link(0).type(),
            "SAME_DEVICE");
  EXPECT_EQ(BaseGPUDeviceFactory::InterconnectMap::kSameDeviceStrength,
            devices[1]->attributes().locality().links().link(0).strength());
}

TEST_F(GPUDeviceTest, MultipleVirtualDevicesWithPriority) {
  {
    // Multile virtual devices with fewer priorities.
    // 0 is a valid priority value for both AMD and NVidia GPUs
    SessionOptions opts = MakeSessionOptions("0", 0, 1, {{123, 456}}, {{0}});
    std::vector<std::unique_ptr<Device>> devices;
    Status status = DeviceFactory::GetFactory("GPU")->CreateDevices(
        opts, kDeviceNamePrefix, &devices);
    EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
    ExpectErrorMessageSubstr(
        status,
        "Number of virtual device priorities specified doesn't "
        "match with number of memory_limit_mb specified for GPU# 0"
        " memory_limit_mb size: 2 and priority size: 1");
  }
  {
    // Multile virtual devices with matching priority.
    // Valid range for priority values on AMD GPUs in (-1,1)
    // Valid range for priority values on NVidia GPUs in (-2, 0)
    SessionOptions opts =
        MakeSessionOptions("0", 0, 1, {{123, 456}}, {{-1, 0}});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::GetFactory("GPU")->CreateDevices(
        opts, kDeviceNamePrefix, &devices));
    EXPECT_THAT(devices, SizeIs(2));
    EXPECT_EQ(devices[0]->attributes().memory_limit(), 123 << 20);
    EXPECT_EQ(devices[1]->attributes().memory_limit(), 456 << 20);
    EXPECT_EQ(-1, static_cast<BaseGPUDevice*>(devices[0].get())->priority());
    EXPECT_EQ(static_cast<BaseGPUDevice*>(devices[1].get())->priority(), 0);
  }
}

TEST_F(GPUDeviceTest, MultipleVirtualDevicesWithDeviceOrdinal) {
  SessionOptions opts = MakeSessionOptions("0", 0, 1, {{1, 2}}, {}, {{2, 1}});
  std::vector<std::unique_ptr<Device>> devices;
  TF_CHECK_OK(DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices));
  EXPECT_THAT(devices, SizeIs(2));
  // Order is flipped due to ordinal.
  EXPECT_EQ(devices[0]->attributes().memory_limit(), 2 << 20);
  EXPECT_EQ(devices[1]->attributes().memory_limit(), 1 << 20);
}

TEST_F(GPUDeviceTest,
       MultipleVirtualDevicesWithDeviceOrdinalOnMultipleDevices) {
  // This test requires at least two visible GPU hardware.
  if (se::GPUMachineManager()->VisibleDeviceCount() < 2) return;

  SessionOptions opts =
      MakeSessionOptions("0,1", 0, 2, {{1, 2}, {3, 4}}, {}, {{1, 2}, {1, 2}});
  std::vector<std::unique_ptr<Device>> devices;
  TF_CHECK_OK(DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices));
  EXPECT_THAT(devices, SizeIs(4));
  EXPECT_EQ(devices[0]->attributes().memory_limit(), 1 << 20);
  EXPECT_EQ(devices[1]->attributes().memory_limit(), 3 << 20);
  EXPECT_EQ(devices[2]->attributes().memory_limit(), 2 << 20);
  EXPECT_EQ(devices[3]->attributes().memory_limit(), 4 << 20);
}

// Enabling unified memory on pre-Pascal GPUs results in an initialization
// error.
TEST_F(GPUDeviceTest, UnifiedMemoryUnavailableOnPrePascalGpus) {
  if (GetComputeCapability().IsAtLeast(se::CudaComputeCapability::PASCAL_)) {
    return;
  }

  SessionOptions opts = MakeSessionOptions("0", /*memory_fraction=*/1.2);
  opts.config.mutable_gpu_options()
      ->mutable_experimental()
      ->set_use_unified_memory(true);
  std::vector<std::unique_ptr<Device>> devices;
  Status status = DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices);
  EXPECT_EQ(status.code(), error::INTERNAL);
  ExpectErrorMessageSubstr(status, "does not support oversubscription.");
}

// Enabling unified memory on Pascal or later GPUs makes it possible to allocate
// more memory than what is available on the device.
TEST_F(GPUDeviceTest, UnifiedMemoryAllocation) {
  static constexpr double kGpuMemoryFraction = 1.2;
  static constexpr tsl::PlatformDeviceId kPlatformDeviceId(0);

  // Exit early if running on pre-Pascal GPUs.
  if (!GetComputeCapability().IsAtLeast(se::CudaComputeCapability::PASCAL_)) {
    LOG(INFO)
        << "Unified memory allocation is not supported with pre-Pascal GPUs.";
    return;
  }

  SessionOptions opts = MakeSessionOptions("0", kGpuMemoryFraction);
  std::vector<std::unique_ptr<Device>> devices;
  TF_ASSERT_OK(DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices));
  ASSERT_THAT(devices, SizeIs(1));

  int64_t memory_limit = devices[0]->attributes().memory_limit();
  ASSERT_EQ(memory_limit,
            static_cast<int64_t>(GetTotalGPUMemory(kPlatformDeviceId) *
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
}

TEST_F(GPUDeviceTest, CopyTensorInSameDevice) {
  SessionOptions opts = MakeSessionOptions("0");
  std::vector<std::unique_ptr<Device>> devices;
  TF_ASSERT_OK(DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices));
  Device* device = devices[0].get();
  auto* device_info = device->tensorflow_accelerator_device_info();
  CHECK(device_info);
  DeviceContext* device_context = device_info->default_context;
  Allocator* allocator = device->GetAllocator(AllocatorAttributes());

  constexpr int kNumElements = 4;
  Tensor input_tensor(allocator, DT_FLOAT, TensorShape({kNumElements}));
  Tensor output_tensor(allocator, DT_FLOAT, TensorShape({kNumElements}));
  Tensor cpu_tensor(cpu_allocator(), DT_FLOAT, TensorShape({kNumElements}));
  // Initialize input as {1, 1, 1, 1} and output as {0, 0, 0, 0}.  After copy,
  // both should become {1, 1, 1, 1}.
  InitCPUTensor(&cpu_tensor, kNumElements, 0);
  CopyCPUToGPU(&cpu_tensor, &output_tensor, device, device_context);
  InitCPUTensor(&cpu_tensor, kNumElements, 1);
  CopyCPUToGPU(&cpu_tensor, &input_tensor, device, device_context);
  Notification note;
  device->CopyTensorInSameDevice(&input_tensor, &output_tensor, device_context,
                                 [&note](const Status& s) {
                                   TF_ASSERT_OK(s);
                                   note.Notify();
                                 });
  note.WaitForNotification();

  Tensor output_cpu_tensor(cpu_allocator(), DT_FLOAT,
                           TensorShape({kNumElements}));
  CopyGPUToCPU(&output_tensor, &output_cpu_tensor, device, device_context);
  auto input = cpu_tensor.tensor<float, 1>();
  auto output = output_cpu_tensor.tensor<float, 1>();
  for (int i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(input(i), output(i)) << " for index " << i;
  }
}

TEST_F(GPUDeviceTest, DeviceDetails) {
  DeviceFactory* factory = DeviceFactory::GetFactory("GPU");
  std::vector<string> devices;
  TF_ASSERT_OK(factory->ListPhysicalDevices(&devices));
  EXPECT_GE(devices.size(), 1);
  for (int i = 0; i < devices.size(); i++) {
    std::unordered_map<string, string> details;
    TF_ASSERT_OK(factory->GetDeviceDetails(i, &details));
    EXPECT_NE(details["device_name"], "");
#if TENSORFLOW_USE_ROCM
    EXPECT_EQ(details.count("compute_capability"), 0);
#else
    EXPECT_NE(details["compute_capability"], "");
#endif
  }
}

class GPUKernelTrackerTest : public ::testing::Test {
 protected:
  void Init(const GPUKernelTracker::Params& params) {
    timing_counter_.reset(new SharedCounter);
    kernel_tracker_.reset(new GPUKernelTracker(params, Env::Default(), nullptr,
                                               timing_counter_.get(), nullptr,
                                               nullptr));
  }

  void RecordQueued(uint64 v) {
    mutex_lock l(kernel_tracker_->mu_);
    kernel_tracker_->RecordQueued(v, 1);
  }

  std::unique_ptr<GPUKernelTracker> kernel_tracker_;
  std::unique_ptr<SharedCounter> timing_counter_;
};

TEST_F(GPUKernelTrackerTest, CappingOnly) {
  Init({0 /*max_interval*/, 0 /*max_bytes*/, 32 /*max_pending*/});
  EXPECT_EQ(kernel_tracker_->NumPending(), 0);
  // 1 is the expected value when no kernels have yet terminated.
  EXPECT_EQ(kernel_tracker_->LastTerminatedCount(0), 1);

  std::deque<int64_t> queued_counts;
  for (int i = 0; i < 32; ++i) {
    uint64 queued_count = timing_counter_->next();
    queued_counts.push_back(queued_count);
    RecordQueued(queued_count);
  }
  EXPECT_EQ(kernel_tracker_->NumPending(), 32);
  EXPECT_EQ(kernel_tracker_->LastTerminatedCount(0), 1);

  // Mature the kernels in order until empty.
  while (!queued_counts.empty()) {
    int64_t x = queued_counts.front();
    queued_counts.pop_front();
    kernel_tracker_->RecordTerminated(x);
    EXPECT_THAT(queued_counts, SizeIs(kernel_tracker_->NumPending()));
    EXPECT_EQ(x, kernel_tracker_->LastTerminatedCount(0));
  }
  EXPECT_EQ(timing_counter_->get(), kernel_tracker_->LastTerminatedCount(0));

  // Next inject so many kernel events that the ring buffer needs
  // to grow a couple of times, while maturing a few in random order
  // to introduce gaps between last_completed_ and first_available_.
  int64_t lower_bound = timing_counter_->get();
  for (int i = 0; i < 1111; ++i) {
    uint64 queued_count = timing_counter_->next();
    queued_counts.push_back(queued_count);
    RecordQueued(queued_count);
    int64_t upper_bound = timing_counter_->get();
    if (0 == (i % 16)) {
      size_t index = (random::New64() % queued_counts.size());
      kernel_tracker_->RecordTerminated(queued_counts[index]);
      queued_counts.erase(queued_counts.begin() + index);
      EXPECT_LE(lower_bound, kernel_tracker_->LastTerminatedCount(0));
      EXPECT_GE(upper_bound, kernel_tracker_->LastTerminatedCount(0));
    }
  }

  // Next mature the remaining kernels in order until empty.
  while (!queued_counts.empty()) {
    int64_t x = queued_counts.front();
    queued_counts.pop_front();
    kernel_tracker_->RecordTerminated(x);
    EXPECT_THAT(queued_counts, SizeIs(kernel_tracker_->NumPending()));
    // There may be a gap here where we find a kernel that got terminated
    // out of order, earlier, so the LastTerminatedCount can actually
    // jump past x.
    EXPECT_LE(x, kernel_tracker_->LastTerminatedCount(0));
  }
  EXPECT_EQ(timing_counter_->get(), kernel_tracker_->LastTerminatedCount(0));
}

}  // namespace tensorflow

#endif
