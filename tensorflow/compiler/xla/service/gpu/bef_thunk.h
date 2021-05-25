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
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/statusor.h"

#if BEF_THUNKS
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/gpu/system/system.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#endif  // BEF_THUNKS

namespace xla {
namespace gpu {

// A Thunk that uses TFRT BEF execution to perform the work of various Thunk
// types. A BefThunk is not restricted to a particular op function, unlike
// GemmThunk, ConvolutionThunk, etc. Rather, a BefThunk is to stand in place of
// any other Thunk type (eventually).
// See BefThunk::GetThunkKind() to see the list of op functions currently
// supported.
class BefThunk : public Thunk {
 public:
  // Returns true if a BefThunk can be constructed to perform the work of 'op'.
  static bool SupportsOp(mlir::Operation* op);

  // Returns a Thunk that performs the work of 'op' using 'inputs' to give
  // 'outputs'.
  static StatusOr<std::unique_ptr<BefThunk>> Create(
      ThunkInfo thunk_info, mlir::Operation* op,
      std::vector<BufferAllocation::Slice> inputs,
      std::vector<BufferAllocation::Slice> outputs);

  BefThunk(const BefThunk&) = delete;
  BefThunk& operator=(const BefThunk&) = delete;

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  explicit BefThunk(ThunkInfo thunk_info, Thunk::Kind kind,
                    std::vector<BufferAllocation::Slice> inputs,
                    std::vector<BufferAllocation::Slice> outputs);

  static StatusOr<Thunk::Kind> GetThunkKind(mlir::Operation* op);
#if BEF_THUNKS
  // TODO(hanbinyoon): Pass in Runtime and the ExecutionContext at construction
  // time when TF/XLA can depend on TFRT in OSS.
  static tensorflow::tfrt_stub::Runtime& runtime();
  static tfrt::ExecutionContext& exec_ctx();

  std::vector<BufferAllocation::Slice> inputs_;
  std::vector<BufferAllocation::Slice> outputs_;
  std::unique_ptr<tfrt::gpu::Program> gpu_program_;
#endif  // BEF_THUNKS
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_BEF_THUNK_H_
