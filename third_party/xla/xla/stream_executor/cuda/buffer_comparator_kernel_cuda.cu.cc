/* Copyright 2018 The OpenXLA Authors.

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

#include "third_party/gpus/cuda/include/device_launch_parameters.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/gpu/buffer_comparator_kernel_lib.cu.h"
#include "xla/stream_executor/platform/initialize.h"

namespace stream_executor::cuda {

// Comparison kernel code: compare two buffers of
// fp8/bf16/fp16/fp32/fp64/int8_t/int32_t of length buffer_length where the
// relative error does not exceed the passed rel_error_threshold. Write the
// number of mismatches into out parameter mismatch_count.

namespace {

static void RegisterBufferComparatorKernelCudaImpl() {
  auto register_kernel = [&](auto primitive_type_constant) {
    gpu::RegisterBufferComparatorKernelParametrized<
        xla::primitive_util::NativeTypeOf<primitive_type_constant()>>(
        stream_executor::cuda::kCudaPlatformId);
  };
  xla::primitive_util::IntegralTypeForEach(register_kernel);
  xla::primitive_util::FloatingPointTypeForEach(register_kernel);
}

}  // namespace
}  // namespace stream_executor::cuda

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    RegisterBufferComparatorKernelCuda,
    stream_executor::cuda::RegisterBufferComparatorKernelCudaImpl());
