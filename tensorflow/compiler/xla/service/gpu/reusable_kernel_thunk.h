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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REUSABLE_KERNEL_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REUSABLE_KERNEL_THUNK_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace gpu {

class GpuExecutable;

// Allows StreamExecutor to launch so-called "reusable kernels".
//
// We have 2 calling conventions in use for passing parameters to kernels:
// - Old "kernels" take a pointer to each *allocation* and the code of the
// kernel calculates the pointer to each *argument* based on the allocation
// pointer + some hardcoded offsets. The pointers to constant arguments are
// hardcoded.
// - "Reusable kernels" take pointers to each *argument* separately, these
// pointers are often not pointing to the first byte of an allocation, but an
// offset is already added in the host code. Pointers to constant arguments are
// also passed normally. This calling convention often allows reusing a kernel
// for multiple operations.
//
// Note: The thunk itself is not reusable: we need a separate thunk for each
// kernel launch in the code.
//
// TODO(b/272252440): Fully replace the old KernelThunk with this and rename
// this to KernelThunk.
//
// This is thread-compatible.
class ReusableKernelThunk : public Thunk {
 public:
  // Constructs a thunk for the given "reusable kernel".
  //
  // ReusableKernelThunk takes args as BufferAllocation::Slice's as opposed
  // to BufferAllocation's. Each slice directly corresponds to an argument or
  // output of the computation. Also, the values must correspond to each arg
  // directly, not to their base allocation (e.g. they can be the result of an
  // mlir::memref::ViewOp).
  ReusableKernelThunk(ThunkInfo thunk_info,
                      std::vector<BufferAllocation::Slice> args,
                      const std::string& kernel_name,
                      const LaunchDimensions& launch_dimensions,
                      std::vector<mlir::Value> values);
  ReusableKernelThunk(const ReusableKernelThunk&) = delete;
  ReusableKernelThunk& operator=(const ReusableKernelThunk&) = delete;
  ~ReusableKernelThunk() override = default;

  std::string ToStringExtra(int indent) const override;

  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;
  Status ExecuteOnStream(const ExecuteParams& params) override;

  void ClearCompileTimeInfo() override {
    Thunk::ClearCompileTimeInfo();
    for (auto& value : values_) {
      value = nullptr;
    }
  }

  const std::vector<BufferAllocation::Slice>& arguments() const {
    return args_;
  }
  const std::string& kernel_name() const { return kernel_name_; }
  const LaunchDimensions& launch_dimensions() const {
    return launch_dimensions_;
  }
  absl::Span<const mlir::Value> values() const { return values_; }

 private:
  // Buffer slices passed to the kernel as arguments.
  const std::vector<BufferAllocation::Slice> args_;

  // Entry kernel name for the computation.
  const std::string kernel_name_;

  // The thread and block dimension used to launch the kernel.
  const LaunchDimensions launch_dimensions_;

  // mlir::Value(s) corresponding to the buffer slice arguments.
  std::vector<mlir::Value> values_;

  mutable absl::Mutex mutex_;

  // Loaded kernels for each `StreamExecutor`.  Requires pointer stability of
  // values.
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<se::KernelBase>>
      kernel_cache_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REUSABLE_KERNEL_THUNK_H_
