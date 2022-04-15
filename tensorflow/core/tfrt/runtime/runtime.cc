/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/runtime/runtime.h"

#include <string>
#include <utility>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.h"
#include "tfrt/cpu/core_runtime/cpu_op_handler.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/tensor/scalar_host_tensor.h"  // from @tf_runtime

#ifdef GOOGLE_CUDA
#include "tfrt/gpu/core_runtime/gpu_op_handler.h"  // from @tf_runtime
#include "tfrt/gpu/device/device.h"  // from @tf_runtime
#include "tfrt/gpu/device/device_util.h"  // from @tf_runtime
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"  // from @tf_runtime
#endif  // GOOGLE_CUDA

constexpr char const kDefaultHostDeviceName[] =
    "/job:localhost/replica:0/task:0/device:CPU:0";

namespace tensorflow {
namespace tfrt_stub {
namespace {

tensorflow::Status InitializeOpHandlers(tfrt::CoreRuntime* corert) {
  // TODO(b/196962112): Make default device configurable through Runtime.
  std::string default_device = kDefaultHostDeviceName;

  DeviceNameUtils::ParsedName device_parsed_name;
  if (!DeviceNameUtils::ParseFullName(default_device, &device_parsed_name) ||
      !device_parsed_name.has_type) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Invalid device name");
  }

  if (device_parsed_name.type == DEVICE_CPU) {
    default_device = kDefaultHostDeviceName;
  } else if (device_parsed_name.type == DEVICE_GPU &&
             (!device_parsed_name.has_job || !device_parsed_name.has_id ||
              !device_parsed_name.has_replica ||
              !device_parsed_name.has_task)) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Device name must be fully specified");
  }

  tfrt::OpHandler* op_handler = nullptr;

  if (device_parsed_name.type == DEVICE_GPU) {
#ifdef GOOGLE_CUDA
    auto fallback_op_handler = tensorflow::tfd::CreateRuntimeFallbackOpHandler(
        corert, /*tf_device_name=*/"");
    corert->RegisterOpHandler("tf", fallback_op_handler.get());
    op_handler = fallback_op_handler.get();
#endif  // GOOGLE_CUDA
  } else {
    auto fallback_op_handler = tensorflow::tfd::CreateKernelFallbackOpHandler(
        corert, corert->GetHostContext()->GetHostDeviceRef());
    corert->RegisterOpHandler("tfkernel", fallback_op_handler.get());
    op_handler = fallback_op_handler.get();
  }

  if (device_parsed_name.type == DEVICE_CPU) {
    auto cpu_device = corert->GetHostContext()->GetHostDeviceRef();
    auto cpu_op_handler =
        tfrt::CreateCpuOpHandler(corert, std::move(cpu_device), op_handler);

    cpu_op_handler.get()->AddImplicitConversion(
        tensorflow::tfd::RuntimeFallbackTensor::kTensorType,
        tfrt::DenseHostTensor::kTensorType);
    cpu_op_handler.get()->AddImplicitConversion(
        tensorflow::tfd::RuntimeFallbackTensor::kTensorType,
        tfrt::AnyScalarHostTensor::kTensorType);
    cpu_op_handler.get()->AddImplicitConversion(
        tensorflow::tfd::RuntimeFallbackTensor::kTensorType,
        tfrt::StringHostTensor::kTensorType);
    cpu_op_handler.get()->AddImplicitConversion(
        tensorflow::KernelFallbackTensor::kTensorType,
        tfrt::DenseHostTensor::kTensorType);
    cpu_op_handler.get()->AddImplicitConversion(
        tensorflow::KernelFallbackTensor::kTensorType,
        tfrt::AnyScalarHostTensor::kTensorType);
    cpu_op_handler.get()->AddImplicitConversion(
        tensorflow::KernelFallbackTensor::kTensorType,
        tfrt::StringHostTensor::kTensorType);

    op_handler = cpu_op_handler.get();
#ifdef GOOGLE_CUDA
  } else if (device_parsed_name.type == DEVICE_GPU) {
    const int gpu_ordinal = 0;
    auto gpu_device = tfrt::gpu::GetOrCreateGpuDevice(
        default_device, gpu_ordinal, corert->GetHostContext());
    auto gpu_op_handler = tfrt::gpu::CreateGpuOpHandler(
        corert, std::move(gpu_device.get()), op_handler);
    op_handler = gpu_op_handler.get();
#endif  // GOOGLE_CUDA
  } else {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Unknown device type");
  }

  corert->RegisterOpHandler(default_device, op_handler);

  return tensorflow::Status::OK();
}

}  // namespace

std::unique_ptr<Runtime> Runtime::Create(
    std::unique_ptr<WorkQueueInterface> work_queue) {
  auto* work_queue_ptr = work_queue.get();
  auto expected_core_runtime = tfrt::CoreRuntime::Create(
      [](const tfrt::DecodedDiagnostic& diag) { LOG(ERROR) << diag.message; },
      tfrt::CreateMallocAllocator(), std::move(work_queue),
      kDefaultHostDeviceName);
  DCHECK(expected_core_runtime);
  const auto& status = InitializeOpHandlers(expected_core_runtime.get().get());
  if (!status.ok()) {
    LOG(ERROR) << "Failed to initialize op handlers: " << status;
    return {};
  }

  // We don't use std::make_unique here because the constructor should better be
  // private.
  return std::unique_ptr<Runtime>(
      new Runtime(std::move(expected_core_runtime.get()), work_queue_ptr));
}

// TODO(b/196962112): Remove this overload.
std::unique_ptr<Runtime> Runtime::Create() {
  static constexpr int kDefaultNumInterOpThreads = 4;
  // Let system pick the number of intra op threads.
  static constexpr int kDefaultNumIntraOpThreads = 0;
  return Runtime::Create(kDefaultNumInterOpThreads, kDefaultNumIntraOpThreads);
}

std::unique_ptr<Runtime> Runtime::Create(int num_inter_op_threads,
                                         int num_intra_op_threads) {
  if (num_intra_op_threads <= 0)
    num_intra_op_threads = tensorflow::port::MaxParallelism();
  return Runtime::Create(
      WrapDefaultWorkQueue(tfrt::CreateMultiThreadedWorkQueue(
          num_intra_op_threads, num_inter_op_threads)));
}

Runtime::Runtime(std::unique_ptr<tfrt::CoreRuntime> core_runtime,
                 WorkQueueInterface* work_queue)
    : core_runtime_(std::move(core_runtime)), work_queue_(work_queue) {
  DCHECK(work_queue_);
}

Runtime::~Runtime() = default;

}  // namespace tfrt_stub
}  // namespace tensorflow
