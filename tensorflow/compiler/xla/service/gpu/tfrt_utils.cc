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

#include "tensorflow/compiler/xla/service/gpu/tfrt_utils.h"

#include <utility>

#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/errors.h"
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/gpu/gpu_types.h"
#endif

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#endif

namespace xla {
namespace gpu {

static const char kDefaultHostDeviceName[] =
    "/job:localhost/replica:0/task:0/device:CPU:0";

StatusOr<CoreRuntimeAndWorkQueue> GetCoreRuntimeAndWorkQueue() {
  // TODO(hanbinyoon): Make these configurable.
  int tfrt_num_threads = tensorflow::port::MaxParallelism();
  int tfrt_num_blocking_threads = 16;

  static StatusOr<CoreRuntimeAndWorkQueue>* runtime_and_queue_or =
      [&](int num_threads, int num_blocking_threads) {
        // Create work queue.
        auto work_queue = tensorflow::tfrt_stub::WrapDefaultWorkQueue(
            tfrt::CreateMultiThreadedWorkQueue(num_threads,
                                               num_blocking_threads));
        if (work_queue == nullptr) {
          auto status =
              tensorflow::errors::Internal("Failed to create TFRT work queue.");
          return new StatusOr<CoreRuntimeAndWorkQueue>(status);
        }
        auto* work_queue_ptr = work_queue.get();

        // Create core runtime.
        auto expected_core_runtime = tfrt::CoreRuntime::Create(
            [](const tfrt::DecodedDiagnostic& diag) {
              LOG(ERROR) << diag.message;
            },
            tfrt::CreateMallocAllocator(), std::move(work_queue),
            kDefaultHostDeviceName);
        if (!expected_core_runtime) {
          auto error = expected_core_runtime.takeError();
          auto status =
              tensorflow::errors::Internal(llvm::toString(std::move(error)));
          return new StatusOr<CoreRuntimeAndWorkQueue>(status);
        }

        auto runtime_and_queue = CoreRuntimeAndWorkQueue{
            expected_core_runtime->release(), work_queue_ptr};
        return new StatusOr<CoreRuntimeAndWorkQueue>(runtime_and_queue);
      }(tfrt_num_threads, tfrt_num_blocking_threads);

  TF_RETURN_IF_ERROR(runtime_and_queue_or->status());
  return runtime_and_queue_or->ValueOrDie();
}

StatusOr<std::unique_ptr<tfrt::gpu::BorrowedGpuStream>> CreateGpuStream(
    stream_executor::Stream* stream) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  auto se_gpu_executor = static_cast<stream_executor::gpu::GpuExecutor*>(
      stream->parent()->implementation());
  auto se_gpu_stream =
      static_cast<stream_executor::gpu::GpuStream*>(stream->implementation());
  stream_executor::gpu::GpuContextHandle context_handle =
      stream_executor::gpu::GpuDriver::GetContextHandle(
          se_gpu_executor->gpu_context());
  return absl::make_unique<tfrt::gpu::BorrowedGpuStream>(
      tfrt::gpu::wrapper::Context(context_handle),
      tfrt::gpu::wrapper::Stream(se_gpu_stream->gpu_stream()));
#else
  return tensorflow::errors::Unimplemented("GPU is not configured.");
#endif
}

tfrt::RCReference<tfrt::AsyncValue> CreateGpuBuffer(
    stream_executor::DeviceMemoryBase* data) {
  tfrt::gpu::wrapper::Pointer<void> pointer(data->opaque(),
                                            tfrt::gpu::wrapper::Platform::CUDA);
  auto allocator =
      tfrt::MakeAvailableAsyncValueRef<tfrt::gpu::GpuOneShotAllocator<void>>(
          pointer);
  auto buffer =
      tfrt::gpu::GpuBuffer::Allocate(std::move(allocator), data->size());
  if (!buffer)
    return tfrt::MakeErrorAsyncValueRef(tfrt::StrCat(buffer.takeError()));
  return tfrt::MakeAvailableAsyncValueRef<tfrt::gpu::GpuBuffer>(
      std::move(*buffer));
}

}  // namespace gpu
}  // namespace xla
