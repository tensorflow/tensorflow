/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_kernel.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "tsl/platform/errors.h"

namespace stream_executor {
namespace gpu {

namespace {

absl::Status GetCudaAttribute(CUfunction_attribute attribute, CUfunction func,
                              int* attribute_value) {
  return cuda::ToStatus(
      cuFuncGetAttribute(attribute_value, attribute, func),
      absl::StrCat("Failed to query kernel attribute: ", attribute));
}

}  // namespace

absl::StatusOr<int32_t> CudaKernel::GetMaxOccupiedBlocksPerCore(
    ThreadDim threads, size_t dynamic_shared_memory_bytes) const {
  int32_t threads_per_block = threads.x * threads.y * threads.z;
  VLOG(3) << "Get kernel block occupancy: " << name()
          << "; threads_per_block: " << threads_per_block
          << "; dynamic_shared_memory_bytes: " << dynamic_shared_memory_bytes;
  std::unique_ptr<ActivateContext> activation = executor_->Activate();

  int max_blocks;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
          &max_blocks, gpu_function_, threads_per_block,
          dynamic_shared_memory_bytes, CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE),
      absl::StrFormat("Failed to calculate occupancy of kernel %p",
                      gpu_function_)));
  return max_blocks;
}

absl::StatusOr<KernelMetadata> CudaKernel::GetKernelMetadata() {
  KernelMetadata kernel_metadata;
  int value;
  TF_RETURN_IF_ERROR(
      GetCudaAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS, gpu_function_, &value));
  kernel_metadata.set_registers_per_thread(value);

  TF_RETURN_IF_ERROR(GetCudaAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                      gpu_function_, &value));
  kernel_metadata.set_shared_memory_bytes(value);
  return kernel_metadata;
}

}  // namespace gpu
}  // namespace stream_executor
