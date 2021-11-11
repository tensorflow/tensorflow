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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_BEF_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_BEF_THUNK_H_

#include "mlir/IR/Operation.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/custom_call_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/core/platform/statusor.h"

namespace tfrt {
class ExecutionContext;
}

namespace xla {
namespace gpu {

// Return whether --//...:enable_bef_thunk was specified on the command line.
bool IsBefThunkEnabled();

// Creates a Thunk that uses TFRT BEF execution to perform the work of various
// Thunk types. A BefThunk is not restricted to a particular op function, unlike
// GemmThunk, ConvolutionThunk, etc. Rather, a BefThunk is to stand in place of
// an other Thunk type.
StatusOr<std::unique_ptr<Thunk>> CreateBefThunk(
    Thunk::ThunkInfo thunk_info, mlir::Operation* op,
    std::vector<BufferAllocation::Slice> buffers);

// Creates a Thunk that uses TFRT BEF execution to perform Collective Permute.
StatusOr<std::unique_ptr<Thunk>> CreateBefCollectivePermuteThunk(
    Thunk::ThunkInfo thunk_info, mlir::Operation* op,
    std::vector<BufferAllocation::Slice> buffers, int64_t replica_count,
    int64_t partition_count);

// Creates a Thunk that uses TFRT BEF execution to perform the work of a kernel.
StatusOr<std::unique_ptr<Thunk>> CreateBefKernelThunk(
    Thunk::ThunkInfo thunk_info, absl::Span<const BufferAllocation* const> args,
    const std::string& kernel_name, const LaunchDimensions& launch_dimensions);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_BEF_THUNK_H_
