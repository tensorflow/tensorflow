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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_KERNEL_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_KERNEL_THUNK_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace xla {
namespace gpu {

class GpuExecutable;

// This class stores everything that StreamExecutor needs for launching a
// kernel. It implements the ExecuteOnStream interface for GpuExecutable to
// invoke the corresponding kernel.
//
// This is thread-compatible.
class KernelThunk : public Thunk {
 public:
  // Constructs a thunk for the given kernel.
  //
  // `hlo_instruction` is as in Thunk. Other arguments are as the class members.
  KernelThunk(tensorflow::gtl::ArraySlice<const BufferAllocation*> args,
              const string& kernel_name, const HloInstruction* hlo_instruction);
  KernelThunk(const KernelThunk&) = delete;
  KernelThunk& operator=(const KernelThunk&) = delete;
  ~KernelThunk() override = default;

  const string& kernel_name() const { return kernel_name_; }
  void SetLaunchDimensions(const LaunchDimensions& launch_dims);

  tensorflow::Status Initialize(const GpuExecutable& executable) override;

  // Executes the kernel for the thunk on "stream", which must be non-null.
  tensorflow::Status ExecuteOnStream(
      const BufferAllocations& buffer_allocations,
      perftools::gputools::Stream* stream) override;

 private:
  // Buffers passed to the kernel as arguments.
  const std::vector<const BufferAllocation*> args_;

  // Entry kernel name for the computation.
  const string kernel_name_;

  // The thread and block dimension used to launch the kernel.
  // Will be set by IrEmitterUnnested.
  LaunchDimensions launch_dimensions_;

  // Describes how to load this kernel. ExecuteOnStream reuses this loader
  // specification for all executions.
  mutable tensorflow::mutex mutex_;
  std::unique_ptr<perftools::gputools::MultiKernelLoaderSpec> loader_spec_
      GUARDED_BY(mutex_);

  // Loaded kernels for each `StreamExecutor`
  std::unordered_map<perftools::gputools::StreamExecutor*,
                     perftools::gputools::KernelBase>
      kernel_cache_ GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_KERNEL_THUNK_H_
